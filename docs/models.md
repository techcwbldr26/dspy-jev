# Models

## The rule

**Open weights everywhere, except the Claude Code harness.**

| Harness | Provider | Why |
|---|---|---|
| hermes-agent | Ollama Cloud | Open weights, no local GPU, one key |
| Pi | Ollama Cloud | Same |
| Claude Code | Anthropic | The harness is Claude's; using Claude here also gives a genuine second opinion on the same signature |

## How Ollama Cloud is reached

Through its OpenAI-compatible endpoint:

```python
dspy.LM(
    "openai/glm-5.3",                 # NOT "ollama/..." — that targets a local daemon
    api_base="https://ollama.com/v1/",
    api_key=os.environ["OLLAMA_API_KEY"],
)
```

No Ollama installation, no model download. A test asserts that no registry entry
uses the `ollama/` prefix, because getting this wrong fails with a connection
error to `localhost:11434` that reads like a network problem.

## The registry

`src/dspy_jev/models.py`. Roles, not model names, at the call sites, so a
retirement is a one-line change.

| Role | Ollama Cloud | Anthropic |
|---|---|---|
| `decision` | `glm-5.3` | `claude-sonnet-5-5` |
| `fast` | `glm-5.3-flash` | `claude-haiku-4-5-20251001` |
| `judge` | `deepseek-v4-pro:0813` | `claude-opus-5-5` |

Open-weight models in the registry: `glm-5.3`, `deepseek-v4-pro:0813`,
`kimi-k3`, `minimax-m3`, `nemotron-3-ultra`, `glm-5.3-flash`,
`deepseek-v4.1-flash`, `gpt-oss:120b`, `gpt-oss:20b`, `mistral-large-3:675b`,
`nemotron-3-super`.

### Three naming conventions, and which to use where

The same model is spelled three ways depending on how you reach it. Getting this
wrong produces a 404 that reads like a network problem.

| Reached via | Spelling | Used by |
|---|---|---|
| Direct cloud API | the exact tag from `/api/tags`, e.g. `deepseek-v4-pro:0813` | this registry, `dspy-jev` |
| Signed-in local Ollama server | `<model>:cloud`, e.g. `glm-5.3:cloud` | the hermes-agent profile's `ollama-cloud` provider |
| LiteLLM provider prefix | `openai/<exact tag>` plus `api_base` | what `build_lm` constructs |

The registry stores exact tags and keeps the bare spelling as an alias, so
`DSPY_JEV_DECISION_MODEL=deepseek-v4-pro` still resolves — but what goes on the
wire is the suffixed name.

Override per role without touching the code:

```bash
export DSPY_JEV_DECISION_MODEL=kimi-k3
export DSPY_JEV_FAST_MODEL=gpt-oss:20b
```

## Retirements

Ollama retires cloud models, and your account's
[usage settings](https://ollama.com/settings) show the schedule for models you
have used. The registry is therefore never trusted on its own:

```bash
dspy-jev models --live    # reconciles against https://ollama.com/api/tags
dspy-jev doctor           # fails if a configured model is no longer served
```

`doctor` requires an **exact** match. A family-level near miss —
`deepseek-v4-pro` when the listing offers `deepseek-v4-pro:0813` — fails and
names the served tag, rather than passing and leaving the 404 for the first real
request.

`.github/workflows/calibration.yml` runs `doctor` weekly, so a retirement
surfaces as a red build rather than as a production error.

## Choosing a decision model

Decision work is unusual: the output is a probability, not prose. That changes
what matters.

- **Prefer a model that follows a rubric over one that writes well.** None of
  the output is read by a person except `rationale`.
- **`temperature=0.0`, always.** The default. Sampled probabilities are not
  reproducible, and `ReAnchor` fits against the evidence it recorded.
- **Keep the cache on.** `ReAnchor` needs it to try candidate thresholds without
  re-billing; it refuses to run against an uncached client unless you pass
  `--allow-uncached`.
- **Calibration matters more than the model.** A well-anchored mid-size model
  beats a badly-anchored frontier one on the only number that counts — how often
  an unsafe action gets through.

## Comparing backends

Because both harness families run the identical signature, the disagreement
between them is measurable:

```bash
DSPY_JEV_HARNESS=pi          dspy-jev evaluate --dataset data/action_gate.jsonl
DSPY_JEV_HARNESS=claude-code dspy-jev evaluate --dataset data/action_gate.jsonl
```

Compare `safety_recall` and `false_allow_rate`. Agreement on the clear cases is
asserted by a live test; disagreement in the middle of the distribution is
expected, and is precisely what calibration is for.
