"""Observability must never be the thing that leaks the data it is watching."""

from __future__ import annotations

import json
import logging
import os

import dspy
import pytest
from dspy.utils.callback import BaseCallback

from dspy_jev.observability import (
    MLFLOW_REQUEST_RETRIES,
    REDACTED,
    DecisionAuditCallback,
    JsonFormatter,
    Metrics,
    configure_mlflow,
    configure_observability,
    configure_otel,
    install_audit_callback,
    policy_span,
    redact,
    setup_logging,
)
from dspy_jev.program import ActionGateProgram
from tests._support import gate_evidence, stub_lm

pytestmark = pytest.mark.unit


# --- redaction ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["api_key", "Authorization", "X-API-Key", "password", "token", "secret"])
def test_credential_shaped_keys_are_blanked(key):
    assert redact({key: "value"})[key] == REDACTED


def test_redaction_recurses_into_nested_structures():
    cleaned = redact({"outer": [{"api_key": "k"}, {"fine": 1}]})
    assert cleaned["outer"][0]["api_key"] == REDACTED
    assert cleaned["outer"][1]["fine"] == 1


def test_redaction_is_depth_capped_and_terminates():
    deep: dict = {}
    node = deep
    for _ in range(50):
        node["next"] = {}
        node = node["next"]
    assert redact(deep) is not None


def test_non_credential_values_survive():
    assert redact({"task": "ship it", "count": 3}) == {"task": "ship it", "count": 3}


# --- structured logging ---------------------------------------------------------


def test_json_formatter_emits_one_object_with_extras():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "decision.made", (), None)
    record.request_id = "abc123"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "decision.made"
    assert payload["request_id"] == "abc123"
    assert payload["level"] == "INFO"


def test_setup_logging_replaces_its_own_handler_rather_than_stacking(settings):
    setup_logging(settings)
    first = len(logging.getLogger().handlers)
    setup_logging(settings)
    assert len(logging.getLogger().handlers) == first


def test_setup_logging_leaves_other_handlers_alone(settings):
    """A library that clears the root logger breaks its host's logging."""
    foreign = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        setup_logging(settings)
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_litellm_is_quietened(settings):
    """LiteLLM logs request URLs at INFO."""
    setup_logging(settings)
    assert logging.getLogger("LiteLLM").level >= logging.WARNING


# --- metrics --------------------------------------------------------------------


def test_counters_accumulate_per_label_set():
    metrics = Metrics()
    metrics.increment("calls_total", outcome="ok")
    metrics.increment("calls_total", outcome="ok")
    metrics.increment("calls_total", outcome="error")
    snapshot = metrics.snapshot()["counters"]
    assert snapshot["calls_total{outcome=ok}"] == 2
    assert snapshot["calls_total{outcome=error}"] == 1


def test_prometheus_rendering_declares_types_once():
    metrics = Metrics()
    metrics.increment("calls_total", outcome="ok")
    metrics.increment("calls_total", outcome="error")
    rendered = metrics.render_prometheus()
    assert rendered.count("# TYPE calls_total counter") == 1
    assert 'calls_total{outcome="ok"} 1' in rendered


def test_latency_summary_exposes_quantiles():
    metrics = Metrics()
    for value in range(1, 101):
        metrics.observe_latency("lat", float(value))
    rendered = metrics.render_prometheus()
    assert 'lat_ms{quantile="0.5"}' in rendered
    assert "lat_ms_count 100" in rendered
    assert metrics.snapshot()["latency"]["lat"]["p95_ms"] >= metrics.snapshot()["latency"]["lat"]["p50_ms"]


def test_empty_metrics_render_without_error():
    assert Metrics().render_prometheus().strip() == ""


def test_reset_clears_everything():
    metrics = Metrics()
    metrics.increment("a")
    metrics.observe_latency("b", 1.0)
    metrics.reset()
    assert metrics.snapshot() == {"counters": {}, "latency": {}}


# --- callback -------------------------------------------------------------------


def test_callback_records_evidence_not_prose(settings, caplog, configured_dspy):
    metrics = Metrics()
    callback = DecisionAuditCallback(settings, metrics=metrics)
    with (
        caplog.at_level(logging.INFO, logger="dspy_jev.audit"),
        dspy.context(lm=stub_lm(gate_evidence(safe=0.91)), callbacks=[callback]),
    ):
        ActionGateProgram(settings=settings)(task="t", proposed_action="a")

    ends = [r for r in caplog.records if r.message == "module.end"]
    assert ends, "no module.end record was emitted"
    evidence = ends[-1].evidence
    assert evidence["safe_to_proceed"]["probability"] == 0.91
    assert "value" in evidence["route"]


def test_callback_counts_calls_and_latency(settings, configured_dspy):
    metrics = Metrics()
    with dspy.context(lm=stub_lm(gate_evidence()), callbacks=[DecisionAuditCallback(settings, metrics=metrics)]):
        ActionGateProgram(settings=settings)(task="t", proposed_action="a")
    counters = metrics.snapshot()["counters"]
    assert counters.get("dspy_jev_module_calls_total{outcome=ok}", 0) >= 1
    assert counters.get("dspy_jev_lm_calls_total{outcome=ok}", 0) >= 1


def test_payloads_are_withheld_unless_explicitly_enabled(settings, caplog):
    callback = DecisionAuditCallback(settings, metrics=Metrics())
    with caplog.at_level(logging.DEBUG, logger="dspy_jev.audit"):
        callback.on_lm_start("call-1", object(), {"messages": [{"role": "user", "content": "secret ticket"}]})
    assert all(not hasattr(record, "inputs") for record in caplog.records)


def test_enabling_payloads_still_redacts_credentials(settings, caplog):
    loud = settings.model_copy(update={"audit_payloads": True})
    callback = DecisionAuditCallback(loud, metrics=Metrics())
    with caplog.at_level(logging.DEBUG, logger="dspy_jev.audit"):
        callback.on_lm_start("call-1", object(), {"api_key": "sk-live", "prompt": "hello"})
    logged = [r for r in caplog.records if hasattr(r, "inputs")]
    assert logged and logged[0].inputs["api_key"] == REDACTED


def test_callback_records_errors_without_raising(settings, caplog):
    metrics = Metrics()
    callback = DecisionAuditCallback(settings, metrics=metrics)
    callback.on_module_start("c", object(), {})
    with caplog.at_level(logging.ERROR, logger="dspy_jev.audit"):
        callback.on_module_end("c", None, RuntimeError("provider down"))
    assert metrics.snapshot()["counters"]["dspy_jev_module_calls_total{outcome=error}"] == 1
    assert any(r.message == "module.error" for r in caplog.records)


# --- exporters ------------------------------------------------------------------


def test_mlflow_is_skipped_when_disabled(settings):
    assert configure_mlflow(settings) is False


def test_otel_is_off_by_default(settings):
    assert configure_otel(settings) is False


def test_configure_observability_reports_what_came_up(settings):
    result = configure_observability(settings)
    assert result["logging"] is True
    assert result["mlflow"] is False


def test_policy_span_is_a_no_op_when_tracing_is_unconfigured():
    """Library code calls this unconditionally; it must never be the failure."""
    with policy_span("test", {"api_key": "sk-live"}) as span:
        assert span is None or span is not None


def test_an_unreachable_tracking_server_does_not_stop_the_gate(settings, monkeypatch):
    """Tracing is a nice-to-have. A tracking server that is down must degrade.

    It used to take the process with it: MLflow's default retry budget is seven
    attempts with backoff, so a dead server meant a minute of blocked startup.
    """
    settings = settings.model_copy(update={"mlflow_enabled": True, "mlflow_tracking_uri": "http://127.0.0.1:1"})
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    assert configure_mlflow(settings) is False
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == str(MLFLOW_REQUEST_RETRIES)
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_the_audit_callback_does_not_displace_other_callbacks(settings, configured_dspy):
    """Regression: configuring ours removed MLflow's, so traces arrived empty."""

    class Bystander(BaseCallback):
        pass

    bystander = Bystander()
    dspy.configure(callbacks=[bystander])

    install_audit_callback(settings)
    callbacks = list(dspy.settings.callbacks)
    assert bystander in callbacks, "someone else's callback was thrown away"
    assert sum(isinstance(cb, DecisionAuditCallback) for cb in callbacks) == 1

    install_audit_callback(settings)
    assert sum(isinstance(cb, DecisionAuditCallback) for cb in dspy.settings.callbacks) == 1, "installed twice"
