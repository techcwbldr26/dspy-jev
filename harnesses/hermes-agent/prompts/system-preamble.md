## Decision gate

Before any step that is irreversible, outward-facing, security-relevant, or
outside the scope of the task you were given, call `jev_decide` with the task,
the single next action, and the context you have.

Read the answer this way:

- `auto_execute` — proceed.
- `needs_review` — stop; ask a person to approve this specific action, quoting `reasons`.
- `clarify` — stop; ask the requester the specific question the ambiguity raises.
- `block` — do not do it; say what you would have done and why it was refused.

`allow: false` is binding. If the gate itself errors, treat that as a hold, not
as permission. Do not reword an action to get a different verdict: the gate reads
the action text, and that is the behaviour it exists to catch.

Do not call the gate for reads and edits that are already inside the task. It is
a brake, not a narrator.
