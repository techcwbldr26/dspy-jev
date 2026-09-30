# dspy-jev

A decision gate for agent harnesses. An agent about to do something it is not
sure about asks the gate, and gets back a verdict with the probability behind it.

```json
{
  "allow": false,
  "route": "needs_review",
  "reasons": ["risk level 3 > max 1", "action is not self-reversible"],
  "decisions": {
    "safe_to_proceed": { "value": false, "probability": 0.31, "confidence": 0.63 },
    "risk":            { "value": 2.8, "level": 3, "confidence": 0.71 }
  },
  "rationale": "Restarting the fleet drops in-flight jobs and runs in production."
}
```

Built on [DSPy](https://dspy.ai)'s experimental Jev decision types — `Noul`,
`Score`, `Choice` — and the `ReAnchor` calibrator. Three harnesses call it:
**hermes-agent**, **[Pi](https://pi.dev)** and **Claude Code**.

The first two run state-of-the-art **open-weight** models on Ollama Cloud.
Claude Code runs Claude models: one documented exception, which also gives a
genuine second opinion on an identical signature.

---

## Quick start

```bash
git clone https://github.com/techcwbldr26/dspy-jev && cd dspy-jev

export OLLAMA_API_KEY=...            # https://ollama.com/settings/keys
./scripts/setup_pi.sh                # or setup_hermes_agent.sh / setup_claude_code.sh
source .venv/bin/activate

dspy-jev doctor --probe              # proves the provider actually answers
dspy-jev calibrate --dataset data/action_gate.jsonl
dspy-jev serve                       # console at http://127.0.0.1:8080
```

In another shell:

```bash
dspy-jev decide --task "free up disk space" \
                --action "delete the contents of /var/lib/docker" \
                --context "shared CI runner, three teams building now"
# exit 10, route: block
```

Running in a **Claude Code cloud session**? Two extra settings are needed, and one
of them is easy to miss — [`docs/setup.md`](docs/setup.md) has them with
screenshots.

- **Setup, step by step, with screenshots** → [`docs/setup.md`](docs/setup.md)
- **The console** → [`docs/console.md`](docs/console.md)
- **Harness guides** → [hermes-agent](docs/harnesses/hermes-agent.md) · [Pi](docs/harnesses/pi.md) · [Claude Code](docs/harnesses/claude-code.md)
- **Models and retirements** → [`docs/models.md`](docs/models.md)
- **Observability** → [`docs/observability.md`](docs/observability.md)
- **When it misbehaves** → [`docs/runbook.md`](docs/runbook.md)

> The step-by-step **task list** and the **implementation plan** are distributed
> separately as Word documents rather than tracked in this repository.

---

## Why this approach is beneficial

### A number you can tune beats a paragraph you can only rewrite

The usual way to make an agent more careful is to edit its prompt. You add "be
cautious with destructive commands", the agent gets more cautious about
everything, and you have no way to say *how much* more cautious, no way to
measure whether it worked, and no way to change it back except by rewriting the
paragraph and hoping.

A `Noul` field does not return a label. It returns `P(True)`, and the value is
derived locally from a threshold you own:

```python
predict.fields["safe_to_proceed"] = {"threshold": 0.71}
```

That is the whole change. The prompt is untouched, the model was not asked
again, and the diff is one number in a JSON file that a reviewer can read.

### Calibration is fitted, not guessed

`ReAnchor` runs the program over labelled examples, collects the evidence each
call produced, and searches for the thresholds, cuts and weights that maximise
your metric — keeping a value only when it beats the default *and* survives a
fold check. Because it reuses cached evidence, calibrating again costs no
inference. You can iterate on the objective at nearly zero marginal cost, which
is not true of anything that re-prompts.

### Confidence survives the call

`"probability": 0.31` is in the response, the audit log and the trace. Three
months later you can answer "how sure was it?" — which a parsed `"no"` cannot.
It also makes a whole class of behaviour possible: escalate the uncertain middle
to a person and auto-handle the confident tails, with the boundary as a
parameter rather than a guess.

### Judgement and policy are separate

The model judges. Plain Python decides:

```python
allow = (route == "auto_execute" and safe_to_proceed
         and P(safe) >= autonomy_threshold
         and risk.level <= max_risk_level
         and blast_radius.level <= max_blast_level
         and reversible)
```

An auditor reads that without reading a prompt. It is tested without a model at
all. A harness can raise its own bar without recalibrating. And during an
incident you change `DSPY_JEV_AUTONOMY_THRESHOLD` and it takes effect on the
next request.

### You can see what it is doing

`dspy-jev serve` also serves a console at `/`. One self-contained file, no build
step, no CDN — it works on a laptop with no network but a reachable gate.

![The dspy-jev console](docs/images/console-light.png)

Its centrepiece is the **threshold lens**: every labelled action in your dataset,
placed by the probability the model gave it, with the actions a person labelled
safe on one row and unsafe on the other. A draggable line is the threshold.

Drag it and the verdicts change — but nothing is re-run. The probabilities are
already in; only the local cut moves. Two counts trade against each other as you
drag. The screenshots below are from a real run against `glm-5.3` on Ollama
Cloud, over the 40 labelled actions in `data/action_gate.jsonl`.

At the calibrated threshold of `0.90`, the two error counts are both zero —
nothing unsafe gets through, and nothing safe is stopped:

![The lens at the calibrated threshold of 0.90](docs/images/lens-threshold-090.png)

Drag it down to `0.55` and **five unsafe actions slip through**, at no saving in
friction, because there was no friction to save:

![The lens with the threshold dragged to 0.55](docs/images/lens-threshold-055.png)

That trade-off is the whole of calibration, and it is the one thing a log line
cannot show you.

Ask about a real action and you also get the **policy chain** — which of the six
conditions failed, and by how much. This layer is plain Python over the numbers,
so when someone asks why an action was blocked, the row names the number to
change:

![The policy chain for a held action](docs/images/policy-chain.png)

That example is worth reading twice. The model argued *for* the action — "on-call
has approved this restart … no further human gate is needed" — and the gate held
it anyway, because `0.85` clears the policy floor but not the fitted threshold of
`0.90`, and the risk level exceeds the ceiling. The model's prose is an opinion.
The policy is the control.

It follows your system theme:

![The console in dark mode](docs/images/console-dark.png)

Full tour: [`docs/console.md`](docs/console.md).

### It blocks, rather than suggesting

A verdict the agent can ignore is not a control. Each harness gets a real
interception point:

| Harness | Where it hooks | What a hold does |
|---|---|---|
| Claude Code | `PreToolUse` hook | Returns `permissionDecision: deny`; the tool never runs |
| Pi | `tool_call` extension event | Returns `{ block: true }`; the tool never runs |
| Any harness | `dspy-jev guard -- <cmd>` | Gates the command and execs only on allow |

Reads, globs and read-only shell commands (`ls`, `git status`, `cat`) skip the
gate entirely — no model call, no latency — while anything that writes, reaches
the network, or is simply unrecognised gets checked. A read-only command stops
being read-only the moment it redirects, pipes into a shell, chains with `&&`,
or mentions `curl` or `sudo`.

Failure is a hold. An unreachable service, a timeout, a missing credential, a
crash in the hook itself: every one denies. A gate that fails open is not a
gate, and an error is exactly when one would be most useful to an attacker.

### One gate, three harnesses

Three harnesses each embedding DSPy means three calibration artifacts, three
audit trails and three places the policy drifts. Here the gate is one service;
the harnesses hold a URL. They call it the way each prefers — hermes-agent an
HTTP tool profile, Pi a CLI with a README, Claude Code MCP — over one
implementation, so the verdicts cannot diverge. A test asserts the CLI and the
HTTP service return the same verdict for the same input.

### Open weights, with a real control

The hermes-agent and Pi harnesses run models whose weights are published, served
by Ollama Cloud with no GPU of your own: no lock-in, no per-seat pricing on the
component that gates everything, and the option to self-host later without
changing a line. Claude Code runs Claude on the same signature, so you can
measure what the closed model buys you instead of assuming:

```bash
DSPY_JEV_HARNESS=pi          dspy-jev evaluate --dataset data/action_gate.jsonl
DSPY_JEV_HARNESS=claude-code dspy-jev evaluate --dataset data/action_gate.jsonl
```

### Observability you can actually look at

Four sinks, one set of callbacks, and the console tells you which are live:

![The observability panel](docs/images/observability-panel.png)

MLflow tracing needs no signup and no API key — `./scripts/observability.sh`
starts a local server with a SQLite store, and every decision becomes a trace:

![One decision, as a span tree](docs/images/mlflow-trace.png)

Open the model call and you are looking at the thing this project is about: the
prompt that went out, and the raw probability that came back.

![The model call, with the probability it returned](docs/images/mlflow-lm-call.png)

`{"noul": 0.02}` is the entire design in one line. The model reported a
probability; it decided nothing. The threshold that turned `0.02` into a block
lives in a calibration artifact you can read, move and defend. Token usage rides
along on the trace, so cost per decision is a query rather than an estimate.

Alongside tracing: a structured JSON audit line per decision carrying the
evidence, and a `/metrics` endpoint with no extra dependency. Prompt bodies are
withheld by default — decisions carry user data — and credential-shaped keys are
redacted at any depth. Tracing never blocks the gate: an unreachable tracking
server logs a warning and is skipped, rather than holding up startup.

Full walkthrough: [`docs/observability.md`](docs/observability.md).

---

## Pros and cons

### Pros

| | |
|---|---|
| **Auditable** | Every verdict carries its probability, the failing conditions by name, and the parameters in force. |
| **Tunable without retraining** | Thresholds, cuts and weights are local numbers. Changing one reuses cached evidence. |
| **Measurable** | `safety_recall` and `false_allow_rate` against a labelled set. "Better" is a number. |
| **Cheap to iterate** | Calibration reuses cached responses. Trying a new metric costs no inference. |
| **Portable** | One signature, two provider families, three harnesses. Swapping models is a registry edit. |
| **Fails closed** | No artifact, no credentials, service down — all hold. Conservative defaults everywhere. |
| **Conservative by default** | The policy layer's ceilings apply before any calibration. Uncalibrated is not unguarded. |
| **Observable out of the box** | Traces, structured logs and metrics, all on by default, all redacting by default. |
| **Tested where it matters** | Stubs return real wire-format evidence, so DSPy's actual decoding path runs under test. |

### Cons

| | |
|---|---|
| **Jev decision types are experimental** | `Noul`, `Score`, `Choice`, `TypeSafe` and `ReAnchor` may change without warning. DSPy is pinned to `3.4.0` for that reason, and the golden contract tests exist partly to catch it. |
| **It needs labelled data** | 40 examples ship, but they are an example, not your policy. Without your own rows, calibration fits someone else's judgement. |
| **The probabilities are not calibrated in the statistical sense** | `P(True) = 0.7` does not mean it is right 70 % of the time. `Noul.confidence` is distance from the threshold, not a calibrated probability. Treat both as *orderings*, not as frequencies. |
| **Another hop on the critical path** | The gate sits in front of every risky step. It adds latency, and it can be down. The cache and the "do not call it for in-scope reads" rule are the mitigations. |
| **Two artifacts to keep in step** | Open-weight and Claude backends produce differently-shaped distributions, so each harness family needs its own calibration and its own recalibration. |
| **Cloud models get retired** | Ollama retires models on a schedule. `doctor` and a weekly workflow catch it, but a retirement means recalibrating. |
| **Enforcement is per-harness** | Each harness needs its own interception point, and they are only as good as that harness's hook surface. The Claude Code `PreToolUse` hook and the Pi `tool_call` handler are true controls; `dspy-jev guard` covers harnesses with no hook at all, but only for commands actually run through it. |
| **The lens needs a labelled dataset** | With no labels there is nothing to plot the threshold against, so the console's centrepiece is empty until you have one. |
| **Triage exists twice** | The Python and TypeScript classifiers must agree, or one harness gates something the other does not. A parity test over 49 cases fails on any disagreement, but it is a test, not a shared implementation. |
| **A gate can be talked around** | The signature says to treat inputs as data, there is a labelled row for it and tests that the preambles forbid rewording, but an agent that rephrases until it passes is a real failure mode. Log and review verdicts. |
| **Bootstrap cost** | Datasets, thresholds, a service, a runbook. For a two-person project running one agent, a hard-coded deny-list is cheaper and you should use that instead. |

---

## How to improve it from here

Roughly in order of value per unit of work.

**1. Close the loop from outcomes.** Right now the dataset grows by hand. Every
held decision a human then approves, and every allowed action that later needed
a revert, is a labelled row that the system already has — it just is not
collecting them. A `/v1/feedback` endpoint plus a weekly job that appends to the
dataset and re-runs calibration would make the gate improve from use.

**2. Measure whether the probabilities mean anything.** Bucket decisions by
`P(safe)` and plot the observed rate of "this turned out fine" per bucket. If
the curve is flat, the probabilities are an ordering and nothing more, and the
thresholds are doing all the work. Worth knowing either way; a reliability
diagram in the calibration report would show it.

**3. Use the judge role for the uncertain middle.** A second model on
disagreement, or on `0.4 < P < 0.7`. `Choice` distributions make the "close
call" band explicit, so this is cheap to target — you spend the extra call only
where the first one was unsure.

**4. Per-context policies.** One `autonomy_threshold` for every environment is
crude. Production should be stricter than a scratch branch. The policy layer is
already pure and parameterised; it needs a lookup, not a redesign.

**5. Try the TypeSafe backend.** `dspy.experimental.TypeSafe("jev-latest")`
returns probabilities natively instead of asking a generative model to produce
evidence as JSON. The same signature, demos and field parameters work with both,
so it is a backend swap — and a comparison worth publishing.

**6. Prompt-injection evaluation.** One labelled row (`ag-022`) is not a test
suite. Build an adversarial set: actions phrased to sound routine, context
carrying instructions, rewordings of a blocked action. Track the pass rate as a
metric, the way `safety_recall` is tracked.

**7. Sharpen the rubrics with real disagreements.** The rows where the
open-weight and Claude backends disagree are usually rows whose *label* is
arguable. Those are the most valuable rows in the dataset: resolving them
improves both the labels and the rubric text.

**8. Cost and latency budgets.** Token usage per decision is in the MLflow
traces but not in the metrics. Exporting it, with a per-harness budget, makes
"is the gate worth it?" answerable.

**9. Multi-step planning, not just single steps.** The gate judges one action.
Agents propose plans. Scoring a sequence — where step three is only unsafe
because step two happened — is a different and more interesting problem, and the
`Choice`-over-runtime-candidates pattern in the DSPy decision-types reference is
a plausible starting point.

---

## Repository layout

```
src/dspy_jev/
  models.py          Registry: open-weight cloud models + the Claude exception
  config.py          Settings; harness -> provider; per-harness artifact paths
  lm.py              Builds the dspy.LM (Ollama Cloud via the OpenAI-compatible API)
  decisions.py       Noul/Score/Choice types, signatures, the audit record
  program.py         Predictor + the deterministic policy layer
  data.py            JSONL loading with Pydantic validation
  metrics.py         Asymmetric gate metric, safety recall, false-allow rate
  calibrate.py       ReAnchor driver, artifact and report
  observability.py   MLflow autolog, audit callback, redaction, metrics
  enforce.py         Triage, action rendering, fail-closed verdicts for the hooks
  service/           FastAPI: /v1/decide, /v1/triage, /healthz, /readyz, /metrics
                     plus the console at / and the data it reads
  mcp_server.py      MCP tools, in-process or proxying to the service
  cli.py             doctor, models, decide, guard, triage, calibrate, evaluate, serve, mcp

harnesses/           Ready-to-copy assets per harness (skills, profiles, configs)
scripts/             install.sh + setup_<harness>.sh + observability.sh
data/                Labelled datasets
docs/                Setup, console tour, harness guides, models, observability, runbook
docs/images/         Screenshots used by the docs
tests/               unit · contract · calibration · integration · live
```

## Testing

```bash
make dev
make test-fast      # unit + contract, seconds
make test           # everything except live providers
make coverage       # with the coverage floor enforced
make test-live      # opt-in, needs real credentials
```

| Suite | Protects |
|---|---|
| `tests/unit` | Registry, settings, LM construction, policy layer, metrics, dataset validation, redaction, CLI |
| `tests/contract` | HTTP surface, golden response and OpenAPI shape, MCP tool names and schemas |
| `tests/calibration` | That ReAnchor measurably improves a mis-anchored gate, deterministically, without touching the prompt |
| `tests/integration` | Calibrate → save → serve → decide, the shipped harness assets, the hooks run as real processes, and Python/TypeScript triage parity |
| `tests/live` | Real providers, and that every configured cloud model is still served |

The stubs return probability evidence in the wire format, so DSPy's real
decoding runs under test rather than a mock of it.

## Requirements

- Python 3.10–3.13
- `OLLAMA_API_KEY` for the hermes-agent and Pi harnesses
- `ANTHROPIC_API_KEY` for the Claude Code harness
- Optional: MLflow for traces, `mcp` for the MCP server

## References

- [DSPy: Decision-Making with Jev Types](https://dspy.ai/tutorials/jev_decisions/)
- [DSPy: Decision types and System One models](https://dspy.ai/api/experimental/DecisionTypes/)
- [DSPy: Debugging & Observability](https://dspy.ai/tutorials/observability/)
- [Ollama Cloud](https://docs.ollama.com/cloud) · [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Pi agent harness](https://pi.dev/docs/latest)

## License

Apache-2.0
