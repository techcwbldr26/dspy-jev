"""The CLI is a production surface: the Pi skill shells out to `dspy-jev decide`."""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest

from dspy_jev import cli
from dspy_jev.config import Settings
from tests._support import gate_evidence, stub_lm, triage_evidence

pytestmark = pytest.mark.unit


def run(capsys, argv: list[str]) -> tuple[int, dict]:
    code = cli.main(argv)
    captured = capsys.readouterr().out
    return code, json.loads(captured) if captured.strip() else {}


@pytest.fixture
def cli_env(monkeypatch, settings: Settings):
    """Point the CLI's own settings lookup at the test configuration."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("DSPY_JEV_HARNESS", "pi")
    monkeypatch.setenv("DSPY_JEV_MLFLOW_ENABLED", "false")
    monkeypatch.setenv("DSPY_JEV_ARTIFACT_DIR", str(settings.artifact_dir))
    monkeypatch.setenv("DSPY_JEV_LOG_FORMAT", "text")
    from dspy_jev.config import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "dspy-jev" in capsys.readouterr().out


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_doctor_offline_reports_every_check(capsys, cli_env):
    code, body = run(capsys, ["doctor", "--offline"])
    names = {check["check"] for check in body["checks"]}
    assert {"harness", "provider", "credentials", "model.decision", "calibration"} <= names
    assert code in (cli.EXIT_OK, cli.EXIT_ERROR)


def test_doctor_fails_without_credentials(capsys, monkeypatch):
    monkeypatch.setenv("DSPY_JEV_HARNESS", "pi")
    from dspy_jev.config import reset_settings_cache

    reset_settings_cache()
    code, body = run(capsys, ["doctor", "--offline"])
    assert code == cli.EXIT_ERROR
    credentials = next(c for c in body["checks"] if c["check"] == "credentials")
    assert credentials["ok"] is False
    assert "OLLAMA_API_KEY" in credentials["detail"]


def test_doctor_does_not_reach_the_network_when_offline(capsys, cli_env, monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("--offline must not call the network")

    monkeypatch.setattr(cli, "_fetch_ollama_tags", explode)
    run(capsys, ["doctor", "--offline"])


def test_models_lists_the_registry_and_role_defaults(capsys, cli_env):
    code, body = run(capsys, ["models"])
    assert code == cli.EXIT_OK
    assert body["roles"]["decision"] == "glm-5.3"
    assert any(row["provider"] == "ollama_cloud" and row["open_weight"] for row in body["models"])


def test_harness_flag_switches_the_provider(capsys, cli_env):
    _, body = run(capsys, ["--harness", "claude-code", "models"])
    assert body["roles"]["decision"].startswith("claude-")


def test_decide_exits_zero_and_prints_the_record(capsys, cli_env, monkeypatch, configured_dspy):
    monkeypatch.setattr(cli, "_prepare_runtime", lambda settings, calibration=None: _gate(settings))
    code, body = run(capsys, ["decide", "--task", "fix a test", "--action", "read a file"])
    assert code == cli.EXIT_OK
    assert body["allow"] is True
    assert body["decisions"]["safe_to_proceed"]["probability"] == pytest.approx(0.95)


def test_decide_exits_ten_when_the_gate_holds(capsys, cli_env, monkeypatch, configured_dspy):
    """A distinct exit code lets a shell wrapper branch without parsing JSON."""
    monkeypatch.setattr(
        cli,
        "_prepare_runtime",
        lambda settings, calibration=None: _gate(settings, gate_evidence(safe=0.02, route="block")),
    )
    code, body = run(capsys, ["decide", "--task", "t", "--action", "rm -rf /"])
    assert code == cli.EXIT_HOLD
    assert body["allow"] is False
    assert body["reasons"]


def test_decide_reads_context_from_a_file(capsys, cli_env, monkeypatch, tmp_path: Path, configured_dspy):
    seen: dict = {}

    class Recorder:
        def decide(self, task, proposed_action, context):
            seen["context"] = context
            return _gate(Settings(harness="pi", OLLAMA_API_KEY="k")).decide(
                task=task, proposed_action=proposed_action, context=context
            )

    monkeypatch.setattr(cli, "_prepare_runtime", lambda settings, calibration=None: Recorder())
    context_file = tmp_path / "context.txt"
    context_file.write_text("branch: main\nreviewers: 2", encoding="utf-8")
    run(capsys, ["decide", "--task", "t", "--action", "a", "--context-file", str(context_file)])
    assert "reviewers: 2" in seen["context"]


def test_decide_surfaces_a_missing_credential_as_an_error(capsys, monkeypatch):
    monkeypatch.setenv("DSPY_JEV_HARNESS", "pi")
    from dspy_jev.config import reset_settings_cache

    reset_settings_cache()
    code, body = run(capsys, ["decide", "--task", "t", "--action", "a"])
    assert code == cli.EXIT_ERROR
    assert body["error"] == "MissingCredentialError"


def test_triage_prints_a_record(capsys, cli_env, monkeypatch, configured_dspy):
    import dspy_jev.lm as lm_module

    monkeypatch.setattr(lm_module, "build_lm", lambda *a, **k: stub_lm(triage_evidence(category="account")))
    code, body = run(capsys, ["triage", "--ticket", "cannot log in"])
    assert code == cli.EXIT_OK
    assert body["decisions"]["category"]["value"] == "account"


def test_compact_output_is_single_line(capsys, cli_env, monkeypatch, configured_dspy):
    monkeypatch.setattr(cli, "_prepare_runtime", lambda settings, calibration=None: _gate(settings))
    cli.main(["decide", "--task", "t", "--action", "a", "--compact"])
    assert capsys.readouterr().out.strip().count("\n") == 0


def _gate(settings: Settings, evidence=None):
    from dspy_jev.program import ActionGateProgram

    dspy.configure(lm=stub_lm(evidence or gate_evidence()))
    return ActionGateProgram(settings=settings)
