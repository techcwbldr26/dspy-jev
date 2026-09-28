# Working in this project with Pi

A calibrated decision gate is installed as the `dspy-jev` skill. Consult it
before any step that is irreversible, outward-facing, security-relevant, or
outside the scope of the task you were given. `/skill:dspy-jev` forces it to
load if you need it explicitly.

`allow: false` from the gate is binding. Report the `route` and `reasons` rather
than rewording the action until it passes.

Setup lives in `harnesses/pi/` of the dspy-jev repository:

- `skills/dspy-jev/SKILL.md` → your user or project skills directory
- `config/models.json` → `~/.pi/agent/models.json` (Ollama Cloud, open weights)
- `config/mcp.json` → only if you use `pi-mcp-adapter` instead of the CLI
- `prompts/gate.md` → your prompt templates directory, then type `/gate`
