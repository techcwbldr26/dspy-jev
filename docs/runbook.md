# Runbook

## Daily

Nothing. If the gate needs daily attention, its thresholds are wrong.

## Weekly

1. **Sample held decisions.**
   ```bash
   curl -s localhost:8080/metrics | grep dspy_jev_decisions_total
   ```
   Pull ~20 held decisions from the audit log. For each, ask: was the hold right?
2. **Every wrong verdict becomes a row** in `data/action_gate.jsonl`. Wrong holds
   matter as much as wrong allows — a gate people route around is worse than none.
3. **Check the weekly calibration workflow.** It verifies every configured model
   is still served and re-scores the gate against the safety floor.

## When the dataset has ~10 new rows

```bash
dspy-jev calibrate --dataset data/action_gate.jsonl
cat artifacts/action_gate.*.report.json | jq .summary
```

Accept only if `improved` is `true` and `val_score` beats `val_score_before`.
Review the parameter diff like code: it is a handful of numbers.

```bash
dspy-jev evaluate --dataset data/action_gate.jsonl --min-safety-recall 0.9
```

With a warm cache this costs no new inference.

---

## Symptoms

### `probability` is `null` on every decision

The adapter fell back to plain generation. Check:

```bash
dspy-jev doctor
dspy-jev decide --task t --action a 2>&1 | head -50   # then dspy.inspect_history(n=1)
```

Usual causes: an output field lost its `desc=` (DSPy raises before the call), or
the model returned malformed JSON. Try the `judge` role — a stronger model —
and if that fixes it, the decision model is not following the evidence format.

### Every action is held

```bash
cat artifacts/action_gate.*.json | jq .fields
```

Either the fitted `threshold` is very high, or the dataset was mostly unsafe
examples and calibration learned "say no". Check the balance:

```bash
jq -s 'group_by(.safe_to_proceed) | map({safe: .[0].safe_to_proceed, n: length})' \
  data/action_gate.jsonl
```

Aim for no worse than 25/75 either way. Delete the artifact to fall back to the
conservative defaults while you fix it.

### Unsafe actions are getting through

This is the failure that matters.

```bash
dspy-jev evaluate --dataset data/action_gate.jsonl | jq '{safety_recall, false_allow_rate}'
```

1. Raise the policy floor immediately: `DSPY_JEV_AUTONOMY_THRESHOLD=0.95`. It
   takes effect on the next request, needs no calibration, and only tightens.
2. Add the specific action that got through to the dataset, labelled.
3. Recalibrate and re-evaluate.
4. Ask whether the harness honoured the hold at all — check the preamble and the
   profile's `on_hold`.

### The gate is slow

```bash
curl -s localhost:8080/metrics | grep latency
```

Switch the decision role to a `flash` model, or reserve the gate for genuinely
risky steps: every preamble says not to call it for reads and in-scope edits.
Keep `DSPY_JEV_CACHE=true` — repeated identical actions then cost nothing.

### `doctor` says a model is no longer listed

Ollama retired it.

```bash
dspy-jev models --live | jq '.models[] | select(.live == false)'
```

Pick a replacement from `src/dspy_jev/models.py`, or override:

```bash
export DSPY_JEV_DECISION_MODEL=glm-5.3-flash
```

Then **recalibrate**. A threshold fitted against one model's probability
distribution is not valid for another's.

### The service is down

Every harness must treat that as a hold. Verify:

```bash
# hermes-agent profile
grep on_error harnesses/hermes-agent/profiles/dspy-jev.yaml   # halt_and_report
```

If an agent proceeded anyway, that is the bug — not the outage.

### MLflow traces are missing

```bash
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

Tracing needs the SQLite store (`./scripts/observability.sh` uses it). Note that
pre-set `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_ID` override this project's
settings, by design. To flush before checking:

```python
import mlflow; mlflow.flush_trace_async_logging()
print(len(mlflow.search_traces()))
```

---

## Changing a threshold in a hurry

Fastest to slowest, all reversible:

| Change | Scope | Takes effect |
|---|---|---|
| `autonomy_threshold` on one request | That call | Immediately |
| `DSPY_JEV_AUTONOMY_THRESHOLD` | The service | Next restart |
| Edit `artifacts/*.json` `fields` | That harness | Next restart |
| Recalibrate | That harness | After the run |

The first three only ever tighten, or are a one-line revert. Prefer them during
an incident; recalibrate afterwards, with the incident as a new row.

## Rolling back

```bash
git checkout HEAD~1 -- artifacts/action_gate.pi.json   # if you version artifacts
# or, to fall back to the conservative type defaults:
mv artifacts/action_gate.pi.json /tmp/
```

An uncalibrated gate is more cautious, not less. That is the safe direction.
