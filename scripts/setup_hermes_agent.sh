#!/usr/bin/env bash
# hermes-agent harness: open-weight models on Ollama Cloud, gate exposed as an
# HTTP service that hermes-agent calls as a tool.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HARNESS=hermes-agent
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
  "DSPY_JEV_MLFLOW_ENABLED=true" \
  "DSPY_JEV_MLFLOW_TRACKING_URI=http://127.0.0.1:5000" \
  "DSPY_JEV_SERVICE_HOST=127.0.0.1" \
  "DSPY_JEV_SERVICE_PORT=8080"

run_doctor

cat <<NEXT

hermes-agent wiring
-------------------
1. Start the gate:
       dspy-jev serve

2. Install the profile distribution:
       cp -R harnesses/hermes-agent/decision-gate /path/to/hermes-agent-profiles/
       cd /path/to/hermes-agent-profiles
       hermes profile install ./decision-gate --alias
       printf 'OLLAMA_API_KEY=<your key>\n' > decision-gate/.env

3. Add to the system prompt:  harnesses/hermes-agent/prompts/system-preamble.md
4. Or use MCP instead:        harnesses/hermes-agent/.mcp.json

Enforcement runs at the process boundary, since Hermes has no tool-call hook
this project can rely on:

       dspy-jev guard --task "$TASK" -- <the command>

Exit 0 = allowed and ran; 10 = held and did NOT run; 1 = the gate failed and the
command did NOT run.

The profile holds no model credentials: only DSPY_JEV_SERVICE_URL and, if you
set one, DSPY_JEV_SERVICE_API_KEY. The service holds the Ollama key.

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
