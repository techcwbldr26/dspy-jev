## Decision gate

A calibrated decision gate stands in front of risky actions. Run them **through**
it rather than asking it and then acting on your own:

```bash
dspy-jev guard --task "$TASK" -- <the command>
```

Exit `0` means the gate allowed it and the command ran. Exit `10` means the gate
held it and **the command did not run**. Exit `1` means the gate itself failed,
and the command did not run either — a gate that cannot answer is not
permission.

Reach for it before any step that is irreversible, outward-facing,
security-relevant, or outside the scope of the task you were given. Do not use it
for reads, or for edits already inside the task; it is a brake, not a narrator.

When a step is held, read the route:

- `auto_execute` — proceed.
- `needs_review` — stop; ask a person to approve this specific action, quoting `reasons`.
- `clarify` — stop; ask the requester the specific question the ambiguity raises.
- `block` — do not do it; say what you would have done and why it was refused.

Report `reasons` verbatim. They name policy thresholds, so a person can change a
number rather than argue with a model. Never reword an action to get a different
verdict: the gate reads the action text, and that is the behaviour it exists to
catch.
