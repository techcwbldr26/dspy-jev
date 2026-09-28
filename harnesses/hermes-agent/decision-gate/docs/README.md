# decision-gate — docs guide

This folder holds the operating material for the `decision-gate` profile. The
rubric, the thresholds and the calibration procedure are versioned in the
dspy-jev repository — this profile points at them rather than duplicating them,
so a recalibration does not need a profile release.

## Quickstart for users

```bash
# From your clone of this repo:
hermes profile install ./decision-gate --alias
printf 'OLLAMA_API_KEY=<your key from ollama.com -> Settings -> Keys>\n' > decision-gate/.env
```

Then start the gate service the profile talks to:

```bash
pip install 'dspy-jev[service,observability]'
dspy-jev serve            # 127.0.0.1:8080
```

## Reference material

| Topic | Where |
|---|---|
| Rubric and signature | `src/dspy_jev/decisions.py` in dspy-jev |
| Policy thresholds | `src/dspy_jev/program.py`, and `artifacts/*.json` once calibrated |
| Calibration procedure | `docs/runbook.md` in dspy-jev |
| Labelled examples | `data/action_gate.jsonl` in dspy-jev |
| Observability | `docs/observability.md` in dspy-jev |

## Model note

`config.yaml` pins `glm-5.3:cloud` through the `ollama-cloud` provider — an
open-weight model, and no `base_url` line, per repo convention. Bump
`distribution.yaml` `version:` and name the switch in the commit message when
this changes.

The model the *gate itself* runs on is configured separately, in the dspy-jev
service (`DSPY_JEV_DECISION_MODEL`). They can differ, and often should: this
profile's model writes the report, the service's model makes the judgement.
