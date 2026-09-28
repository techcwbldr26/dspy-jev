# Decision gate

A calibrated decision gate is available as the `dspy-jev` MCP server
(`jev_decide`, `jev_triage`, `jev_status`) and as the `dspy-jev` skill.

Call `jev_decide` before any step that is irreversible, outward-facing,
security-relevant, or outside the scope of the task you were given. Do not call
it for reads and edits already inside the task.

`allow: false` is binding. Report the `route` and the `reasons` verbatim instead
of rewording the action until it passes — the gate reads the action text, and
rephrasing to get past it is the behaviour it exists to catch.

This harness uses Claude models. The hermes-agent and Pi harnesses run the same
signature on open-weight models via Ollama Cloud, so a disagreement between them
is a signal worth reading, not a bug.
