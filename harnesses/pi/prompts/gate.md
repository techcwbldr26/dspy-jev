Before I continue, gate the next step.

Run:

```bash
dspy-jev decide --task "$TASK" --action "$ACTION" --context "$CONTEXT"
```

Then report, in this order:

1. The verdict (`allow`) and the `route`.
2. Every entry in `reasons`, verbatim.
3. `P(safe_to_proceed)` from `decisions.safe_to_proceed.probability`, and the
   `risk` and `blast_radius` levels.
4. What you will actually do now, given the route.

If the route is `needs_review` or `block`, do not run the action. Say what you
would have run and stop.
