#!/usr/bin/env bash
# Pi harness (https://pi.dev): open-weight models on Ollama Cloud.
# Pi prefers CLI tools with READMEs over MCP, so the primary integration is the
# `dspy-jev` skill, which shells out to `dspy-jev decide`.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HARNESS=pi
PI_AGENT_DIR="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"

step "setting up the $HARNESS harness (Ollama Cloud, open weights)"

"$DSPY_JEV_ROOT/scripts/install.sh" "service,observability,mcp"

AUTH_MODE="$(detect_auth_mode OLLAMA_API_KEY)"
require_key OLLAMA_API_KEY "Create one at https://ollama.com/settings/keys" "$AUTH_MODE" || true
network_note

write_env_file "$HARNESS" \
  "OLLAMA_API_KEY=${OLLAMA_API_KEY:-}" \
  "DSPY_JEV_AUTH_MODE=$AUTH_MODE" \
  "DSPY_JEV_DECISION_MODEL=glm-5.3" \
  "DSPY_JEV_FAST_MODEL=glm-5.3-flash" \
  "DSPY_JEV_JUDGE_MODEL=deepseek-v4-pro" \
  "DSPY_JEV_AUTONOMY_THRESHOLD=0.85" \
  "DSPY_JEV_MLFLOW_ENABLED=true"

step "installing the Pi skill, prompt template and enforcement extension"
mkdir -p "$PI_AGENT_DIR/skills" "$PI_AGENT_DIR/prompts" "$PI_AGENT_DIR/extensions"
cp -R "$DSPY_JEV_ROOT/harnesses/pi/skills/dspy-jev" "$PI_AGENT_DIR/skills/"
cp "$DSPY_JEV_ROOT/harnesses/pi/prompts/gate.md" "$PI_AGENT_DIR/prompts/"
cp -R "$DSPY_JEV_ROOT/harnesses/pi/extensions/dspy-jev-gate" "$PI_AGENT_DIR/extensions/"
ok "skill     -> $PI_AGENT_DIR/skills/dspy-jev/SKILL.md"
ok "prompt    -> $PI_AGENT_DIR/prompts/gate.md  (type /gate in Pi)"
ok "extension -> $PI_AGENT_DIR/extensions/dspy-jev-gate/  (blocks held tool calls)"

if [[ -f "$PI_AGENT_DIR/models.json" ]]; then
  warn "$PI_AGENT_DIR/models.json exists; merge harnesses/pi/config/models.json by hand"
else
  cp "$DSPY_JEV_ROOT/harnesses/pi/config/models.json" "$PI_AGENT_DIR/models.json"
  ok "models -> $PI_AGENT_DIR/models.json (Ollama Cloud provider)"
fi

run_doctor

cat <<NEXT

Pi wiring
---------
The skill calls the CLI, so make sure the venv's bin directory is on PATH in the
shell that starts Pi:

    export PATH="$VENV_DIR/bin:\$PATH"

Enforcement is on: the extension handles Pi's tool_call event, so a held tool
call is blocked rather than reported. Start the service it talks to:

    dspy-jev serve &
    export DSPY_JEV_SERVICE_URL=http://127.0.0.1:8080

Then, in Pi:
    /gate-status         where the gate is, and whether it answers
    /skill:dspy-jev      force-load the gate's instructions
    /gate                run the gate prompt template
    /gate-off            stop enforcing for this session

Optional MCP route (needs pi-mcp-adapter):
    dspy-jev serve &
    dspy-jev mcp --transport streamable-http
    cp harnesses/pi/config/mcp.json .mcp.json

Confirm the provider really answers, then calibrate, then look at it:
    dspy-jev doctor --probe
    dspy-jev calibrate --dataset data/action_gate.jsonl
    dspy-jev serve            # console at http://127.0.0.1:8080

To see inside a single decision -- the prompt, the raw probability, the tokens:
    scripts/observability.sh                       # tracking server on :5000
    DSPY_JEV_MLFLOW_ENABLED=true dspy-jev serve    # restart the gate onto it
Then open http://127.0.0.1:5000, pick the 'dspy-jev' experiment, Traces tab.
The console's own Observability panel says which sinks are live either way.
NEXT
