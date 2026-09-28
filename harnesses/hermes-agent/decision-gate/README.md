# decision-gate — Action Gating Agent Profile

A Hermes agent profile that decides whether a proposed action should run, and
enforces the answer. Built on [dspy-jev](https://github.com/techcwbldr26/dspy-jev):
DSPy Jev decision types (`Noul`, `Score`, `Choice`) calibrated with `ReAnchor`.

## Who this profile is for

- **Agents about to do something irreversible** — a second opinion with a number
  attached, and a command wrapper that makes the answer binding.
- **Engineers** who want autonomy levels set by a reviewable threshold rather
  than by a paragraph of prompt.
- **Leads** who need an audit trail: every verdict carries its probability, the
  failing conditions by name, and the thresholds in force at the time.

## Installation

```bash
git clone https://github.com/techcwbldr26/hermes-agent-profiles.git
cd hermes-agent-profiles
hermes profile install ./decision-gate --alias
printf 'OLLAMA_API_KEY=<your key from ollama.com -> Settings -> Keys>\n' > decision-gate/.env
```

Then run the service the profile calls:

```bash
pip install 'dspy-jev[service,observability]'
dspy-jev calibrate --dataset data/action_gate.jsonl
dspy-jev serve &
export DSPY_JEV_SERVICE_URL=http://127.0.0.1:8080
```

## Capabilities

| Skill | What it does |
|---|---|
| `decision-gate` | Gate and enforce one action (`dspy-jev guard`), or judge without running (`dspy-jev decide`) |

## Example asks

- "I want to force-push this branch — check it first."
- "Run the production migration, through the gate."
- "Why was the last deploy held?"
- "What would it take for this action to be allowed automatically?"

## How enforcement works

The skill wraps commands in `dspy-jev guard`, which gates the command and
executes it only if the gate allows. A hold means the command never runs — there
is no window in which the verdict is merely advice.

Failure is a hold: an unreachable service, a timeout or a missing credential all
refuse. A gate that fails open is not a gate.

## Model

`config.yaml` pins `glm-5.3:cloud` via the `ollama-cloud` provider — open
weights, and no `base_url` line, per repo convention. The model the *gate* runs
on is separate, configured in the dspy-jev service via
`DSPY_JEV_DECISION_MODEL`; that one defaults to `glm-5.3` on Ollama Cloud.

Bump `distribution.yaml` `version:` and name the switch in the commit message
when either changes.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every action is held | Service unreachable | `curl localhost:8080/healthz`; start `dspy-jev serve` |
| `doctor` fails on a model | Ollama retired it | `dspy-jev models --live`, then pick a served tag and recalibrate |
| Holds feel arbitrary | No calibration artifact | `dspy-jev calibrate --dataset data/action_gate.jsonl` |
| Too many false holds | Threshold too high for your work | Add the wrongly-held actions to the dataset as labelled rows, recalibrate |

## Profile contents

```
decision-gate/
├── distribution.yaml       name, version, hermes_requires, env_requires
├── config.yaml             model + provider (no base_url)
├── SOUL.md                 scope, routes, working style, out of scope
├── README.md               this file
├── docs/README.md          where the rubric and thresholds live
├── skills/decision-gate/   the skill the agent loads
└── cron/                   (none yet)
```
