# Claude Code harness

The one documented exception to "open weights only": this harness runs the gate
on **Claude models**. Same signature, same policy layer, same artifact format —
only the backend differs, which is what makes the two families comparable.

## Setup

```bash
export ANTHROPIC_API_KEY=...       # https://console.anthropic.com/settings/keys
./scripts/setup_claude_code.sh /path/to/your/project
```

That installs into the target project:

| File | Purpose |
|---|---|
| `.claude/skills/dspy-jev/SKILL.md` | The gate's instructions, loaded on demand |
| `.mcp.json` | Runs `dspy-jev mcp --transport stdio` |
| `CLAUDE.md` | When to call it, and that a hold is binding |

Restart Claude Code, then:

```
/mcp            # dspy-jev should be connected
jev_status      # harness: claude-code, model: anthropic/claude-sonnet-5-5
```

`.mcp.json` launches the `dspy-jev` command, so the venv's `bin` directory must
be on `PATH` in the environment that starts Claude Code.

## Tools

| Tool | Use |
|---|---|
| `jev_decide(task, proposed_action, context, autonomy_threshold?)` | Gate one action |
| `jev_triage(ticket)` | Score a support ticket |
| `jev_status()` | Model, harness, calibration state, policy parameters |

`autonomy_threshold` can only ever tighten the gate; a value below the service's
own floor is ignored.

## Calibrate this harness separately

```bash
DSPY_JEV_HARNESS=claude-code dspy-jev calibrate --dataset data/action_gate.jsonl
```

It writes `artifacts/action_gate.claude-code.json`. A separate artifact is not
bookkeeping: a different backend produces differently-shaped probability
distributions, so a threshold fitted against one is wrong for the other.

## Comparing the two backends

```bash
DSPY_JEV_HARNESS=pi          dspy-jev evaluate --dataset data/action_gate.jsonl
DSPY_JEV_HARNESS=claude-code dspy-jev evaluate --dataset data/action_gate.jsonl
```

Compare `safety_recall` and `false_allow_rate`. Where they disagree, read the
rows: the ones both get right need no work, and the ones they split on are
usually rows whose label is genuinely arguable — which makes them the most
valuable rows in the dataset.

## Using the HTTP service instead

To share one artifact across every harness, run the service and point the MCP
server at it:

```json
{
  "mcpServers": {
    "dspy-jev": {
      "command": "dspy-jev",
      "args": ["mcp", "--transport", "stdio", "--service-url", "http://127.0.0.1:8080"],
      "env": { "DSPY_JEV_HARNESS": "claude-code" }
    }
  }
}
```

Claude Code then holds no provider credential at all.

## Without a calibration artifact

The gate still works, on the type defaults: `threshold 0.5`, evenly spaced
`cuts`, equal `weights`, and the policy layer's conservative ceilings
(`risk ≤ 1`, `blast ≤ 1`, reversible, `P(safe) ≥ 0.85`). It will be more
cautious than it needs to be. That is the intended failure mode — uncalibrated
must not mean unguarded.
