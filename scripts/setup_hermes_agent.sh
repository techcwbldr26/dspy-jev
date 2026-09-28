#!/usr/bin/env bash
# hermes-agent harness: open-weight models on Ollama Cloud, gate exposed as an
# HTTP service that hermes-agent calls as a tool.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HARNESS=hermes-agent
step "setting up the $HARNESS harness (Ollama Cloud, open weights)"

"$DSPY_JEV_ROOT/scripts/install.sh" "service,observability,mcp"

require_key OLLAMA_API_KEY "Create one at https://ollama.com/settings/keys" || true

write_env_file "$HARNESS" \
  "OLLAMA_API_KEY=${OLLAMA_API_KEY:-}" \
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
1. Start the gate:            dspy-jev serve
2. Copy the tool profile:     harnesses/hermes-agent/profiles/dspy-jev.yaml
                              -> your hermes-agent profiles directory
3. Add to the system prompt:  harnesses/hermes-agent/prompts/system-preamble.md
4. Or use MCP instead:        harnesses/hermes-agent/.mcp.json

The harness holds no model credentials: only DSPY_JEV_SERVICE_URL and, if you
set one, DSPY_JEV_SERVICE_API_KEY. The service holds the Ollama key.

Calibrate before relying on it:
    dspy-jev calibrate --dataset data/action_gate.jsonl
NEXT
