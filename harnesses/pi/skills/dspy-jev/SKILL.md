---
name: dspy-jev
description: Gate a risky action before running it. Use before any step that writes outside the current task's scope, leaves the machine, spends money, touches shared or production state, or cannot be undone — force-push, delete, deploy, migrate, send, publish, install, rotate a secret. Returns a calibrated allow/hold verdict with probabilities. Do not use for ordinary reads or edits already inside the task.
license: Apache-2.0
metadata:
  project: dspy-jev
  surface: cli
---

# dspy-jev decision gate

You have a calibrated decision gate. It is not another model you argue with: it
returns a probability and a verdict derived from thresholds that were fitted
against labelled outcomes.

## When to call it

Call it **before** an action that is any of:

- outside the scope the task named
- irreversible without someone else's help (force-push, `rm -rf`, `DROP`, deploy, migrate)
- outward-facing (email, publish, post, page someone)
- money-spending or quota-consuming at scale
- security-relevant (secrets, permissions, TLS, auth)

Do **not** call it for reading files you already have permission to read, running
the project's own tests, or edits to files the task named. Those are already in
scope; asking wastes a call and trains you to ignore the answer.

## How to call it

```bash
dspy-jev decide \
  --task "<what you were asked to do>" \
  --action "<the single next step>" \
  --context "<branch, environment, permissions already granted>"
```

Exit codes, so you can branch without parsing:

| Exit | Meaning |
|------|---------|
| `0`  | Allowed. Proceed. |
| `10` | Held. Read `route` and `reasons` in the JSON. |
| `1`  | The gate itself failed. Do **not** treat this as permission. |

For long context, use `--context-file path.txt`. For a stricter bar on one call,
add `--autonomy-threshold 0.95` — it can only tighten the gate, never loosen it.

## How to read the answer

```json
{
  "allow": false,
  "route": "needs_review",
  "reasons": ["risk level 3 > max 1", "action is not self-reversible"],
  "decisions": { "safe_to_proceed": { "value": false, "probability": 0.21 } }
}
```

- `route: auto_execute` — proceed.
- `route: needs_review` — stop and ask a person to approve *this specific action*.
- `route: clarify` — the task under-specifies the step; ask the requester what they meant.
- `route: block` — do not do it. Say what you would have done and why it was refused.

`allow: false` is binding. Do not retry with softer wording to get a different
answer; the gate reads the action, and rephrasing to get past it is exactly the
behaviour it exists to catch.

## Checking the setup

`dspy-jev doctor` verifies credentials, that the configured cloud model still
exists, and whether a calibration artifact is loaded. Run it once per machine,
and again if decisions start looking wrong.
