# Observability

Three layers, each usable alone. Following the
[DSPy observability tutorial](https://dspy.ai/tutorials/observability/): start
with `inspect_history`, move to tracing when one call is not enough, and add
callbacks when you need something the framework does not capture.

## 0. The console — start here

```bash
dspy-jev serve            # then open http://127.0.0.1:8080
```

Served by the gate itself at `/`, so it shares an origin with the API and needs
no build step, no package install and no network. It answers the three questions
the other layers cannot:

| Panel | What it answers |
|---|---|
| Threshold lens | Where is the cut, and which labelled actions fall on the wrong side of it? |
| Policy chain | For *this* decision, which of the six conditions failed, and by how much? |
| Recent decisions | What has the gate been doing? |
| Calibration in force | Which fitted parameters are live, and what did fitting them buy? |

Dragging the threshold recomputes every verdict in the browser from probabilities
already fetched — no model call. That is the point: it makes the local, reviewable
nature of calibration something you can feel rather than read about.

The feed is in memory only and bounded. The action text is the user's data; the
durable record is the audit log below, which redacts.

## 1. `inspect_history` — the first thing to try

```python
import dspy
dspy.inspect_history(n=3)
```

Prints the last LM calls: system message, formatted inputs, raw completion. For
a decision program the useful part is the raw evidence — if
`decisions.safe_to_proceed.probability` came back `null`, this shows whether the
model returned evidence at all or the adapter fell back to plain generation.

Its limits are why the next two layers exist: it shows LM calls only, it has no
structure across multiple calls, and it records no latency or relationships.

## 2. MLflow tracing — the default

No signup, no API key.

```bash
./scripts/observability.sh        # mlflow server, SQLite store, :5000
```

SQLite is required; the default file store does not support tracing.

```bash
export DSPY_JEV_MLFLOW_ENABLED=true
export DSPY_JEV_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dspy-jev decide --task "ship it" --action "deploy to prod" --context "friday"
```

Open http://127.0.0.1:5000 → experiment `dspy-jev` → **Traces**. One trace per
decision, containing the module span, the LM call (with token usage and
latency), the adapter format and parse steps, and a `dspy_jev.decide` span for
the policy layer.

`configure_mlflow()` calls `mlflow.dspy.autolog(log_traces=True,
log_traces_from_eval=True, log_compiles=True)`. `log_compiles` is on so a
`ReAnchor` run is traced too — which is how you see *why* a threshold moved.

Pre-set `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_ID` win over this project's
defaults, so a platform team pointing the process at its own tracking server is
not overridden.

**One span, not two.** Per MLflow's guidance, the DSPy modules autolog already
instruments are not decorated again — that produces duplicate spans. The only
manual span is the policy layer, which DSPy never sees.

## 3. Structured audit log — for log pipelines

```bash
export DSPY_JEV_LOG_FORMAT=json
```

One JSON object per line on stderr:

```json
{"ts":"2026-09-28T20:41:07.412Z","level":"INFO","logger":"dspy_jev.audit",
 "message":"module.end","call_id":"a1b2","latency_ms":842.7,
 "evidence":{"safe_to_proceed":{"value":false,"probability":0.31,"confidence":0.63},
             "risk":{"value":2.8,"level":3,"confidence":0.71},
             "route":{"value":"needs_review","confidence":0.66}}}
```

The evidence is always recorded; that is what makes a decision auditable months
later. Prompt and completion bodies are **not**, unless you opt in:

```bash
export DSPY_JEV_AUDIT_PAYLOADS=true   # decisions carry user data — think first
```

Even then, credential-shaped keys (`api_key`, `authorization`, `token`,
`password`, `secret`, `x-api-key`) are redacted, recursively, at any depth.

Events: `module.start` / `module.end` / `module.error`, `lm.start` / `lm.end` /
`lm.error`, `tool.end`, and from the service `http.request`. `X-Request-ID` is
accepted, generated when absent, echoed on the response, and attached to every
log line, so a harness transcript, a service log and an MLflow trace can be
joined.

## 4. Metrics

```bash
curl -s localhost:8080/metrics
```

```
# TYPE dspy_jev_decisions_total counter
dspy_jev_decisions_total{outcome="allow",route="auto_execute"} 214
dspy_jev_decisions_total{outcome="hold",route="needs_review"} 37
dspy_jev_decisions_total{outcome="hold",route="block"} 9
# TYPE dspy_jev_http_latency_ms summary
dspy_jev_http_latency_ms{quantile="0.95"} 1840.220
```

| Metric | Watch for |
|---|---|
| `dspy_jev_decisions_total{outcome="error"}` | Anything above zero. A gate that errors must be treated as a hold, so this is user-visible friction. |
| `dspy_jev_decisions_total{route="block"}` | A sudden rise means either a new class of risky action, or a mis-anchored threshold. |
| ratio of `allow` to `hold` | Drifting towards `allow` over weeks is the signal to recalibrate. |
| `dspy_jev_lm_calls_total{outcome="error"}` | Provider trouble, or a retired model. |
| p95 `dspy_jev_http_latency_ms` | The gate is on the critical path of every risky step. |

## 5. OpenTelemetry

MLflow's tracing is itself OTel-based. If you would rather have the spans in
your own collector:

```bash
pip install 'dspy-jev[otel]'
export DSPY_JEV_OTEL_ENABLED=true
export DSPY_JEV_OTEL_ENDPOINT=http://collector:4318/v1/traces
```

Resource attributes carry `service.name=dspy-jev` and `dspy_jev.harness`.

## 6. Custom callbacks

`DecisionAuditCallback` subclasses `dspy.utils.callback.BaseCallback`. Subclass
it, or write your own, for anything project-specific:

```python
from dspy.utils.callback import BaseCallback

class BlockAlarm(BaseCallback):
    def on_module_end(self, call_id, outputs, exception):
        route = getattr(outputs.get("route"), "value", None) if outputs else None
        if route == "block":
            pager.notify(f"gate blocked an action: {call_id}")

dspy.configure(callbacks=[DecisionAuditCallback(settings), BlockAlarm()])
```

Available handlers: `on_module_*`, `on_lm_*`, `on_adapter_format_*`,
`on_adapter_parse_*`, `on_tool_*`, `on_evaluate_*`, `on_compile_*`.

**Copy before you mutate.** Callbacks receive the live inputs and outputs;
changing them in place changes what the program returns.

## What to check after a change

1. `curl -s localhost:8080/metrics | grep decisions_total` — counters moving.
2. MLflow → Traces — a span tree per decision, with the LM call inside it.
3. `DSPY_JEV_LOG_FORMAT=json … 2>&1 | jq -c 'select(.message=="module.end")'` —
   evidence present.
4. The same command with real user text in `--context`: that text must **not**
   appear anywhere in the log.
