---
name: decision-gate
description: >
  Gate a risky action before running it, and enforce the verdict. Use before any step that
  writes outside the current task's scope, leaves the machine, spends money, touches shared or
  production state, or cannot be undone — force-push, delete, deploy, migrate, send, publish,
  install, rotate a secret. Wrap the command in `dspy-jev guard` so a held action cannot run.
  Also use to explain why an earlier action was held. Do not use for reads, or for edits already
  inside the task.
license: Proprietary
---

# decision-gate

A calibrated gate stands in front of risky actions. It returns a probability and
a verdict derived from thresholds fitted against labelled outcomes — not another
model to argue with.

## Enforce it, do not just consult it

Run the command **through** the gate rather than asking first and then running
it yourself. `guard` gates the command and executes it only if the gate allows:

```bash
dspy-jev guard --task "$TASK" -- git push --force origin main
```

| Exit | Meaning |
|------|---------|
| `0` | Allowed, and the command ran. Its own exit code is returned. |
| `10` | Held. The command did **not** run. Read `route` and `reasons`. |
| `1` | The gate failed. The command did **not** run. This is not permission. |

Asking first and running separately leaves a gap where the verdict is advice.
`guard` closes it: there is no path from a hold to an execution.

## When to reach for it

Before an action that is any of:

- outside the scope the task named
- irreversible without someone else's help (force-push, `rm -rf`, `DROP`, deploy, migrate)
- outward-facing (email, publish, post, page someone)
- money-spending or quota-consuming at scale
- security-relevant (secrets, permissions, TLS, authn/authz)

Not for reading files you may already read, running the project's own tests, or
editing files the task named. Those are in scope; gating them wastes a call and
teaches you to ignore the answer.

## Judging without running

When you need the verdict but not the execution — explaining a hold, or checking
a plan before writing it out:

```bash
dspy-jev decide --task "$TASK" --action "$ACTION" --context "$CONTEXT"
```

## Reading the verdict

```json
{
  "allow": false,
  "route": "needs_review",
  "reasons": ["risk level 3 > max 1", "action is not self-reversible"],
  "decisions": { "safe_to_proceed": { "value": false, "probability": 0.21 } }
}
```

| `route` | What you do |
|---|---|
| `auto_execute` | Proceed. |
| `needs_review` | Stop. Ask a person to approve this specific action, quoting `reasons`. |
| `clarify` | Stop. Ask the requester the specific question the ambiguity raises. |
| `block` | Do not do it. Say what you would have done and why it was refused. |

`allow: false` is binding. Report `reasons` verbatim — they name policy
thresholds, so a person can change a number instead of arguing with a model.
Never reword the action and try again: the gate reads the action text, and
rephrasing to get past it is exactly the behaviour it exists to catch.

## Setup

The profile talks to a running dspy-jev service:

```bash
pip install 'dspy-jev[service,observability]'
dspy-jev serve &
export DSPY_JEV_SERVICE_URL=http://127.0.0.1:8080
dspy-jev doctor
```

The profile holds no model credentials of its own — `OLLAMA_API_KEY` belongs to
the service. If `doctor` reports no calibration artifact, the gate still works on
conservative defaults; it will simply hold more than it needs to.
