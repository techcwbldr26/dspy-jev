"""Observability: tracing, structured audit logs, and metrics.

Three independent layers, each usable on its own:

1. **MLflow tracing** (``mlflow.dspy.autolog()``) -- spans for every module, LM
   call, adapter format/parse and tool call, with no code changes. This is the
   layer the DSPy observability tutorial recommends, and it needs no signup and
   no API key.
2. **A DSPy callback** (:class:`DecisionAuditCallback`) -- one structured JSON
   line per module and LM call, for environments that ship logs rather than
   traces. Probability evidence is recorded; prompt and response bodies are not,
   unless ``DSPY_JEV_AUDIT_PAYLOADS=true``.
3. **Counters** (:class:`Metrics`) -- in-process counters and a Prometheus text
   exposition, so the service has a ``/metrics`` endpoint without a new
   dependency.

Per the MLflow guidance, autolog is *not* supplemented with decorators over the
DSPy modules it already instruments; the one span this module adds is the policy
layer, which DSPy does not see.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import Counter as _Counter
from contextlib import contextmanager, suppress
from typing import Any

from dspy.utils.callback import BaseCallback

from dspy_jev.config import Settings, get_settings

logger = logging.getLogger("dspy_jev.audit")

#: Name given to the handler this module installs, so it can be replaced in place.
_HANDLER_NAME = "dspy_jev"

#: Fields never written to logs or traces, whatever the payload setting.
REDACTED_KEYS = frozenset({"api_key", "authorization", "apikey", "x-api-key", "password", "secret", "token", "bearer"})
REDACTED = "***"


# --- structured logging ---------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """One JSON object per log record, with extras merged in."""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(settings: Settings | None = None) -> None:
    """Install a root handler matching the configured format and level."""
    settings = settings or get_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
    )
    handler.set_name(_HANDLER_NAME)
    root = logging.getLogger()
    # Replace only our own handler. A host application's handlers -- and pytest's
    # capture handler -- are none of this library's business.
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    # LiteLLM is extremely chatty at INFO and leaks request URLs.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Recursively blank out credential-shaped keys. Depth-capped."""
    if _depth > 6:
        return REDACTED
    if isinstance(value, dict):
        return {
            k: (REDACTED if str(k).lower() in REDACTED_KEYS else redact(v, _depth=_depth + 1)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, _depth=_depth + 1) for v in value]
    return value


# --- metrics --------------------------------------------------------------------


class Metrics:
    """Thread-safe counters plus a Prometheus text exposition.

    Deliberately dependency-free: an enterprise deployment can scrape this, or
    ignore it and use the MLflow traces.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: _Counter[tuple[str, tuple[tuple[str, str], ...]]] = _Counter()
        self._latency_ms: dict[str, list[float]] = {}

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] += amount

    def observe_latency(self, name: str, milliseconds: float) -> None:
        with self._lock:
            self._latency_ms.setdefault(name, []).append(float(milliseconds))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {
                (name if not labels else f"{name}{{{','.join(f'{k}={v}' for k, v in labels)}}}"): count
                for (name, labels), count in self._counters.items()
            }
            latency = {
                name: {
                    "count": len(values),
                    "p50_ms": _percentile(values, 50),
                    "p95_ms": _percentile(values, 95),
                    "max_ms": max(values) if values else 0.0,
                }
                for name, values in self._latency_ms.items()
            }
        return {"counters": counters, "latency": latency}

    def render_prometheus(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        lines: list[str] = []
        with self._lock:
            names_seen: set[str] = set()
            for (name, labels), count in sorted(self._counters.items()):
                if name not in names_seen:
                    lines.append(f"# TYPE {name} counter")
                    names_seen.add(name)
                label_str = "" if not labels else "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
                lines.append(f"{name}{label_str} {count}")
            for name, values in sorted(self._latency_ms.items()):
                if not values:
                    continue
                lines.append(f"# TYPE {name}_ms summary")
                lines.append(f'{name}_ms{{quantile="0.5"}} {_percentile(values, 50):.3f}')
                lines.append(f'{name}_ms{{quantile="0.95"}} {_percentile(values, 95):.3f}')
                lines.append(f"{name}_ms_count {len(values)}")
                lines.append(f"{name}_ms_sum {sum(values):.3f}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latency_ms.clear()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 3)


#: Process-wide metrics, shared by the service and the CLI.
METRICS = Metrics()


# --- DSPy callback --------------------------------------------------------------


class DecisionAuditCallback(BaseCallback):
    """Emit one structured record per module / LM / tool call.

    The interesting part for a decision system is not the text: it is the
    probability evidence and the derived value. Those are always recorded.
    Prompts and completions are recorded only when ``audit_payloads`` is on.
    """

    def __init__(self, settings: Settings | None = None, *, metrics: Metrics | None = None) -> None:
        self._settings = settings or get_settings()
        self._metrics = metrics if metrics is not None else METRICS
        self._started: dict[str, float] = {}

    # module ---------------------------------------------------------------
    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._started[call_id] = time.perf_counter()
        logger.debug(
            "module.start",
            extra={
                "call_id": call_id,
                "module": type(instance).__name__,
                "input_fields": sorted(inputs.get("kwargs", inputs) or {}),
            },
        )

    def on_module_end(self, call_id: str, outputs: Any, exception: Exception | None) -> None:
        elapsed_ms = self._elapsed(call_id)
        module_metric = "dspy_jev_module_calls_total"
        if exception is not None:
            self._metrics.increment(module_metric, outcome="error")
            logger.error(
                "module.error",
                extra={"call_id": call_id, "latency_ms": elapsed_ms, "error": type(exception).__name__},
            )
            return
        self._metrics.increment(module_metric, outcome="ok")
        self._metrics.observe_latency("dspy_jev_module_latency", elapsed_ms)
        logger.info(
            "module.end",
            extra={
                "call_id": call_id,
                "latency_ms": elapsed_ms,
                "evidence": _evidence_summary(outputs),
            },
        )

    # language model -------------------------------------------------------
    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._started[call_id] = time.perf_counter()
        record: dict[str, Any] = {"call_id": call_id, "model": getattr(instance, "model", None)}
        if self._settings.audit_payloads:
            record["inputs"] = redact(inputs)
        logger.debug("lm.start", extra=record)

    def on_lm_end(self, call_id: str, outputs: Any, exception: Exception | None) -> None:
        elapsed_ms = self._elapsed(call_id)
        if exception is not None:
            self._metrics.increment("dspy_jev_lm_calls_total", outcome="error")
            logger.error(
                "lm.error",
                extra={"call_id": call_id, "latency_ms": elapsed_ms, "error": type(exception).__name__},
            )
            return
        self._metrics.increment("dspy_jev_lm_calls_total", outcome="ok")
        self._metrics.observe_latency("dspy_jev_lm_latency", elapsed_ms)
        record: dict[str, Any] = {"call_id": call_id, "latency_ms": elapsed_ms}
        if self._settings.audit_payloads:
            record["outputs"] = redact(outputs)
        logger.info("lm.end", extra=record)

    # tools ----------------------------------------------------------------
    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._started[call_id] = time.perf_counter()

    def on_tool_end(self, call_id: str, outputs: Any, exception: Exception | None) -> None:
        self._metrics.increment("dspy_jev_tool_calls_total", outcome="error" if exception is not None else "ok")
        logger.debug("tool.end", extra={"call_id": call_id, "latency_ms": self._elapsed(call_id)})

    def _elapsed(self, call_id: str) -> float:
        started = self._started.pop(call_id, None)
        return 0.0 if started is None else round((time.perf_counter() - started) * 1000, 3)


def _evidence_summary(outputs: Any) -> dict[str, Any]:
    """Pull probability evidence out of a prediction without importing it whole."""
    summary: dict[str, Any] = {}
    items = outputs.items() if hasattr(outputs, "items") else []
    for name, value in items:
        entry: dict[str, Any] = {}
        for attribute in ("value", "level", "probability", "confidence"):
            found = getattr(value, attribute, None)
            if found is not None:
                entry[attribute] = found
        if entry:
            summary[name] = entry
    return summary


# --- MLflow ---------------------------------------------------------------------


def configure_mlflow(settings: Settings | None = None) -> bool:
    """Turn on MLflow autologging for DSPy. Returns ``True`` when it is active.

    Pre-set ``MLFLOW_TRACKING_URI`` / ``MLFLOW_EXPERIMENT_ID`` win: a platform
    team that already points the process at a tracking server should not have it
    overridden by this library's defaults.
    """
    settings = settings or get_settings()
    if not settings.mlflow_enabled:
        return False
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow.unavailable", extra={"hint": "pip install 'dspy-jev[observability]'"})
        return False

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if not os.environ.get("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(settings.mlflow_experiment)

    mlflow.dspy.autolog(log_traces=True, log_traces_from_eval=True, log_compiles=True, silent=True)
    logger.info(
        "mlflow.enabled",
        extra={"tracking_uri": mlflow.get_tracking_uri(), "experiment": settings.mlflow_experiment},
    )
    return True


def configure_otel(settings: Settings | None = None) -> bool:
    """Export OpenTelemetry spans over OTLP/HTTP. Returns ``True`` when active.

    MLflow's tracing is itself OTel-based; this is for sites that want the spans
    in their existing collector rather than in an MLflow server.
    """
    settings = settings or get_settings()
    if not settings.otel_enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("otel.unavailable", extra={"hint": "pip install 'dspy-jev[otel]'"})
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": "dspy-jev", "dspy_jev.harness": settings.harness})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint)))
    trace.set_tracer_provider(provider)
    logger.info("otel.enabled", extra={"endpoint": settings.otel_endpoint})
    return True


@contextmanager
def policy_span(name: str, attributes: dict[str, Any] | None = None):
    """A span around app-specific work DSPy's autolog does not see.

    Falls through silently when MLflow is not installed or not configured, so
    library code can use it unconditionally.
    """
    span_cm = None
    with suppress(Exception):
        import mlflow

        span_cm = mlflow.start_span(name=name)
    if span_cm is None:
        yield None
        return
    with span_cm as span:
        with suppress(Exception):
            span.set_inputs(redact(attributes or {}))
        yield span


def configure_observability(settings: Settings | None = None) -> dict[str, bool]:
    """Set up logging, MLflow and OTel in one call. Returns what came up."""
    settings = settings or get_settings()
    setup_logging(settings)
    return {
        "logging": True,
        "mlflow": configure_mlflow(settings),
        "otel": configure_otel(settings),
    }
