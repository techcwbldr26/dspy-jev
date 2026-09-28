# Pi harness

[Pi](https://pi.dev) is a minimal agent harness: no MCP, no sub-agents and no
permission popups out of the box, on the principle that you build what you need.
That fits a decision gate well — the gate *is* the permission layer, and it is
one you can read and calibrate.

Pi's own guidance is to prefer **CLI tools with READMEs** over MCP servers, so
the primary integration is a skill that shells out to `dspy-jev decide`.

## Setup

```bash
export OLLAMA_API_KEY=...          # https://ollama.com/settings/keys
./scripts/setup_pi.sh
```

That installs:

| From | To | What it does |
|---|---|---|
| `harnesses/pi/skills/dspy-jev/` | `~/.pi/agent/skills/dspy-jev/` | The gate, loaded on demand |
| `harnesses/pi/prompts/gate.md` | `~/.pi/agent/prompts/gate.md` | `/gate` template |
| `harnesses/pi/config/models.json` | `~/.pi/agent/models.json` | Ollama Cloud provider for Pi itself |

Then, in the shell that starts Pi:

```bash
export PATH="$PWD/.venv/bin:$PATH"   # the skill shells out to `dspy-jev`
```

## Using it

Pi advertises the skill's name and description at startup and loads the full
instructions only when a task matches — progressive disclosure, without spending
the prompt cache.

```
/skill:dspy-jev            force-load the instructions
/skill:dspy-jev check the force-push I am about to do
/gate                      run the prompt template
```

Under the hood:

```bash
dspy-jev decide --task "..." --action "..." --context "..."
```

| Exit | Meaning |
|---|---|
| `0` | Allowed |
| `10` | Held — read `route` and `reasons` |
| `1` | The gate failed. Not permission. |

The distinct exit code is deliberate: a Pi extension or a shell wrapper can
branch on it without parsing JSON.

## If Pi does not load the skill on its own

Pi routes on the description alone. The shipped one states both what the skill
does and **when not to use it**, because a description that only says what it
does gets loaded for everything and then ignored. If routing is still wrong, edit
the `description` in `SKILL.md` — that, not the body, is the routing surface.

## Pi's own models

`config/models.json` adds Ollama Cloud as an `openai-completions` provider with
`apiKey: "${OLLAMA_API_KEY}"`, so Pi and the gate run on the same open-weight
models. Opening `/model` reloads the file.

Before pinning a model, check it is still served:

```bash
dspy-jev models --live
```

## Optional: MCP instead

If you have `pi-mcp-adapter` installed:

```bash
dspy-jev serve &
dspy-jev mcp --transport streamable-http     # :8081
cp harnesses/pi/config/mcp.json .mcp.json
```

The adapter takes an HTTP `url` and exposes `jev_decide`, `jev_triage` and
`jev_status` as direct tools. The CLI route is simpler and is what the skill
uses; this exists for setups that already standardise on MCP.

## Extending it

Pi's own answer to a missing feature is to build it. Two worth having:

- A **permission-gate extension** that calls `dspy-jev decide` before `bash`
  runs, so the gate is enforced rather than advisory. See Pi's
  `examples/extensions/permission-gate.ts`.
- A **status-bar item** showing the last verdict and `P(safe)`, so the current
  posture is visible without scrolling.
