"""End to end, offline: calibrate, save, serve, decide.

This is the path an operator actually walks, with a stub standing in for the
provider. It is the test that would catch a component that works alone but not
in the sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest
from fastapi.testclient import TestClient

from dspy_jev.calibrate import calibrate_action_gate
from dspy_jev.data import ActionGateExample, load_examples
from dspy_jev.program import ActionGateProgram
from dspy_jev.service.app import create_app
from tests._support import gate_evidence, keyed_stub_lm, stub_lm
from tests.conftest import release_dspy_config_ownership

pytestmark = pytest.mark.integration


def evidence_table(dataset_path: Path) -> dict:
    """A backend that separates safe from unsafe, but anchors both too high."""
    return {
        example.proposed_action[:60]: gate_evidence(
            safe=0.63 if example.safe_to_proceed else 0.56,
            risk_level=example.risk,
            blast_level=example.blast_radius,
            reversible=0.9 if example.reversible else 0.15,
            route=example.route,
        )
        for example in load_examples(dataset_path, ActionGateExample)
    }


# ReAnchor's fitting stage scales with the training set, and the labelled
# dataset is 140 rows. Calibrating the real file end to end is the point of
# this test, so it gets the time rather than a smaller stand-in dataset.
@pytest.mark.timeout(600)
@pytest.mark.slow
def test_calibrate_then_serve_uses_the_fitted_parameters(settings, tmp_path: Path, dataset_path: Path, configured_dspy):
    artifact = tmp_path / "gate.json"
    with dspy.context(lm=keyed_stub_lm(evidence_table(dataset_path))):
        result = calibrate_action_gate(
            dataset=dataset_path, settings=settings, require_cache=False, artifact_path=artifact
        )
    assert artifact.exists()

    # A fresh service, pointed at the artifact, must report itself calibrated and
    # apply the fitted parameters rather than the type defaults.
    served_settings = settings.model_copy(update={"artifact_dir": artifact.parent})
    gate = ActionGateProgram(settings=served_settings)
    assert gate.load_calibration(artifact) is True

    with TestClient(create_app(served_settings, lm=stub_lm(gate_evidence()), gate=gate)) as client:
        health = client.get("/healthz").json()
        body = client.post("/v1/decide", json={"task": "t", "proposed_action": "read a file", "context": ""}).json()

    assert health["calibrated"] is True
    assert body["metadata"]["calibrated"] is True
    assert body["metadata"]["fields"] == result.fields


def test_the_same_evidence_yields_different_verdicts_under_different_thresholds(
    settings, tmp_path: Path, configured_dspy
):
    """The project's claim, end to end: calibration changes outcomes, not prompts."""
    evidence = gate_evidence(safe=0.80)

    permissive = ActionGateProgram(settings=settings, autonomy_threshold=0.5)
    strict = ActionGateProgram(settings=settings, autonomy_threshold=0.5)
    strict.gate.fields["safe_to_proceed"] = {"threshold": 0.9}

    with TestClient(create_app(settings, lm=stub_lm(evidence), gate=permissive)) as client:
        allowed = client.post("/v1/decide", json={"task": "t", "proposed_action": "a"}).json()
    release_dspy_config_ownership()
    with TestClient(create_app(settings, lm=stub_lm(evidence), gate=strict)) as client:
        held = client.post("/v1/decide", json={"task": "t", "proposed_action": "a"}).json()

    assert allowed["allow"] is True
    assert held["allow"] is False
    assert (
        allowed["decisions"]["safe_to_proceed"]["probability"] == (held["decisions"]["safe_to_proceed"]["probability"])
    )


def test_observability_records_the_decision_without_the_payload(settings, caplog, configured_dspy):
    """One request must leave an audit trail, and that trail must not leak inputs."""
    import logging

    from dspy_jev.observability import METRICS

    with caplog.at_level(logging.INFO), TestClient(create_app(settings, lm=stub_lm(gate_evidence()))) as client:
        client.post(
            "/v1/decide",
            json={"task": "t", "proposed_action": "deploy", "context": "customer data here"},
            headers={"X-Request-ID": "corr-1"},
        )

    messages = [r.message for r in caplog.records]
    assert "module.end" in messages
    assert "http.request" in messages
    assert any(getattr(r, "request_id", None) == "corr-1" for r in caplog.records)
    assert "customer data here" not in caplog.text

    counters = METRICS.snapshot()["counters"]
    assert any(key.startswith("dspy_jev_decisions_total") for key in counters)


def test_a_json_audit_line_can_be_parsed_by_a_log_pipeline(settings, capsys, configured_dspy):
    import logging

    from dspy_jev.observability import DecisionAuditCallback, setup_logging

    json_settings = settings.model_copy(update={"log_format": "json"})
    setup_logging(json_settings)
    try:
        with dspy.context(lm=stub_lm(gate_evidence()), callbacks=[DecisionAuditCallback(json_settings)]):
            ActionGateProgram(settings=json_settings)(task="t", proposed_action="a")
        lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
        parsed = [json.loads(line) for line in lines]
        assert any(record["message"] == "module.end" for record in parsed)
        assert all({"ts", "level", "logger", "message"} <= set(record) for record in parsed)
    finally:
        logging.getLogger().handlers.clear()


def test_the_cli_and_the_service_agree(settings, monkeypatch, capsys, configured_dspy):
    """Two surfaces, one policy layer. They must not drift."""
    from dspy_jev import cli

    evidence = gate_evidence(safe=0.55, risk_level=2, route="needs_review")
    payload = {"task": "ship it", "proposed_action": "deploy to prod", "context": "friday"}

    with TestClient(create_app(settings, lm=stub_lm(evidence))) as client:
        http_body = client.post("/v1/decide", json=payload).json()

    def fake_runtime(_settings, calibration=None):
        dspy.configure(lm=stub_lm(evidence))
        return ActionGateProgram(settings=settings)

    release_dspy_config_ownership()  # the TestClient above took ownership on its portal thread
    monkeypatch.setattr(cli, "_prepare_runtime", fake_runtime)
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    exit_code = cli.main(
        ["decide", "--task", payload["task"], "--action", payload["proposed_action"], "--context", "friday"]
    )
    cli_body = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_HOLD
    assert cli_body["allow"] == http_body["allow"] is False
    assert cli_body["route"] == http_body["route"]
    assert cli_body["reasons"] == http_body["reasons"]
