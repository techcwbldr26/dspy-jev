---
name: dspy-jev
description: Gate a risky action before running it. Use before any step that writes outside the current task's scope, leaves the machine, spends money, touches shared or production state, or cannot be undone — force-push, delete, deploy, migrate, send, publish, install, rotate a secret. Returns a calibrated allow/hold verdict with probabilities. Do not use for ordinary reads or edits already inside the task.
license: Apache-2.0
---

# dspy-jev decision gate (Claude Code)

This harness runs the gate on **Claude models**, unlike the hermes-agent and Pi
harnesses, which use open-weight models on Ollama Cloud. The signature, the
policy layer and the calibration artifact format are identical; only the backend
differs, which is what makes the two comparable.

## Calling it

Prefer the MCP tool when `.mcp.json` is installed:

```
jev_decide(task=..., proposed_action=..., context=...)
```

Otherwise shell out:

```bash
DSPY_JEV_HARNESS=claude-code dspy-jev decide \
  --task "<the ask>" --action "<the next step>" --context "<branch, env, grants>"
```

Exit `0` = allowed, `10` = held, `1` = the gate itself failed. A failed gate is
not permission.

## Acting on the answer

| `route` | What you do |
|---|---|
| `auto_execute` | Proceed. |
| `needs_review` | Stop. Ask the user to approve this specific action, quoting `reasons`. |
| `clarify` | Stop. Ask the requester the specific question the ambiguity raises. |
| `block` | Do not do it. Say what you would have done and why it was refused. |

`allow: false` is binding. Report `reasons` verbatim: they name policy
conditions (`risk level 3 > max 1`), not opinions, so the user can change a
threshold rather than argue with a model. Do not reword the action and ask
again — the gate reads the action text, and rephrasing to get past it is exactly
the behaviour it exists to catch.

## Configuration

`ANTHROPIC_API_KEY` and `DSPY_JEV_HARNESS=claude-code`. Run `dspy-jev doctor` to
confirm. The calibration artifact for this harness is
`artifacts/action_gate.claude-code.json`; without it the gate falls back to type
defaults, which are deliberately conservative.
