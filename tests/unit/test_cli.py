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


# --- live model availability -----------------------------------------------------

LIVE = ["glm-5.3", "glm-5.3-flash", "deepseek-v4-pro:0813", "gpt-oss:120b"]


def test_an_exact_tag_is_available():
    assert cli._availability("glm-5.3", LIVE) == (True, "glm-5.3")


def test_a_family_near_miss_fails_and_names_the_served_tag():
    """Passing on a near miss is how a 404 at the first real request gets missed."""
    ok, detail = cli._availability("deepseek-v4-pro", LIVE)
    assert ok is False
    assert "deepseek-v4-pro:0813" in detail


def test_a_retired_model_fails():
    ok, detail = cli._availability("kimi-k1", LIVE)
    assert ok is False
    assert "not in the live cloud listing" in detail


def test_doctor_reports_a_retired_model_as_a_failure(capsys, cli_env, monkeypatch):
    monkeypatch.setattr(cli, "_fetch_ollama_tags", lambda *a, **k: ["some-other-model"])
    code, body = run(capsys, ["doctor"])
    assert code == cli.EXIT_ERROR
    availability = [c for c in body["checks"] if c["check"].startswith("ollama.available.")]
    assert availability and all(c["ok"] is False for c in availability)


def test_doctor_passes_when_every_configured_model_is_served(capsys, cli_env, monkeypatch):
    from dspy_jev import models as registry

    served = [registry.resolve(registry.default_for("ollama_cloud", r)).name for r in ("decision", "fast", "judge")]
    monkeypatch.setattr(cli, "_fetch_ollama_tags", lambda *a, **k: served)
    _, body = run(capsys, ["doctor"])
    availability = [c for c in body["checks"] if c["check"].startswith("ollama.available.")]
    assert len(availability) == 3
    assert all(c["ok"] for c in availability)


def test_models_live_marks_only_exact_matches(capsys, cli_env, monkeypatch):
    monkeypatch.setattr(cli, "_fetch_ollama_tags", lambda *a, **k: ["glm-5.3"])
    _, body = run(capsys, ["models", "--live"])
    by_name = {row["name"]: row for row in body["models"] if row["provider"] == "ollama_cloud"}
    assert by_name["glm-5.3"]["live"] is True
    assert by_name["deepseek-v4-pro:0813"]["live"] is False


# --- guard -----------------------------------------------------------------------


def test_guard_without_a_command_is_an_error(capsys, cli_env):
    code, body = run(capsys, ["guard", "--task", "t"])
    assert code == cli.EXIT_ERROR
    assert body["error"] == "NothingToRun"


def test_guard_strips_the_argparse_separator(capsys, cli_env, monkeypatch):
    """`--` reaches REMAINDER as a literal, and must not become argv[0]."""
    seen = {}

    class Recorder:
        returncode = 0

    def fake_run(command, *a, **k):
        seen["command"] = command
        return Recorder()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    code = cli.main(["guard", "--task", "t", "--", "echo", "hi"])
    assert code == 0
    assert seen["command"] == ["echo", "hi"]
