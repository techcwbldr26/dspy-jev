#!/usr/bin/env bash
# Claude Code harness: the one documented exception to "open weights only".
# Same signature, same policy layer, same artifact format -- Claude models.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HARNESS=claude-code
TARGET="${1:-$PWD}"

step "setting up the $HARNESS harness (Claude models)"

"$DSPY_JEV_ROOT/scripts/install.sh" "service,observability,mcp"

require_key ANTHROPIC_API_KEY "Create one at https://console.anthropic.com/settings/keys" || true

write_env_file "$HARNESS" \
  "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
  "DSPY_JEV_DECISION_MODEL=claude-sonnet-5-5" \
  "DSPY_JEV_FAST_MODEL=claude-haiku-4-5-20251001" \
  "DSPY_JEV_JUDGE_MODEL=claude-opus-5-5" \
  "DSPY_JEV_AUTONOMY_THRESHOLD=0.85" \
  "DSPY_JEV_MLFLOW_ENABLED=true"

step "installing the skill, the PreToolUse hook and the MCP config into $TARGET"
mkdir -p "$TARGET/.claude/skills" "$TARGET/.claude/hooks"
cp -R "$DSPY_JEV_ROOT/harnesses/claude-code/.claude/skills/dspy-jev" "$TARGET/.claude/skills/"
cp "$DSPY_JEV_ROOT/harnesses/claude-code/.claude/hooks/dspy_jev_gate.py" "$TARGET/.claude/hooks/"
chmod +x "$TARGET/.claude/hooks/dspy_jev_gate.py"
ok "skill -> $TARGET/.claude/skills/dspy-jev/SKILL.md"
ok "hook  -> $TARGET/.claude/hooks/dspy_jev_gate.py"

if [[ -f "$TARGET/.claude/settings.json" ]]; then
  warn "$TARGET/.claude/settings.json exists; merge the PreToolUse block from"
  warn "  harnesses/claude-code/.claude/settings.json by hand -- without it the hook never runs"
else
  cp "$DSPY_JEV_ROOT/harnesses/claude-code/.claude/settings.json" "$TARGET/.claude/settings.json"
  ok "hooks -> $TARGET/.claude/settings.json (PreToolUse enforcement)"
fi

if [[ -f "$TARGET/.mcp.json" ]]; then
  warn "$TARGET/.mcp.json exists; merge harnesses/claude-code/.mcp.json by hand"
else
  cp "$DSPY_JEV_ROOT/harnesses/claude-code/.mcp.json" "$TARGET/.mcp.json"
  ok "mcp   -> $TARGET/.mcp.json"
fi

if [[ -f "$TARGET/CLAUDE.md" ]]; then
  warn "$TARGET/CLAUDE.md exists; append harnesses/claude-code/CLAUDE.md by hand"
else
  cp "$DSPY_JEV_ROOT/harnesses/claude-code/CLAUDE.md" "$TARGET/CLAUDE.md"
  ok "notes -> $TARGET/CLAUDE.md"
fi

run_doctor

cat <<NEXT

Claude Code wiring
------------------
Enforcement is on: the PreToolUse hook denies a held tool call, so it does not
run. Start the service it talks to (or unset DSPY_JEV_SERVICE_URL in
.claude/settings.json to run the gate in-process):

    dspy-jev serve &

Restart Claude Code in $TARGET, then:
    /hooks            confirm the PreToolUse hook is registered
    /mcp              confirm the dspy-jev server is connected
    jev_status        confirm the model and calibration state

Make sure the venv's bin directory is on PATH, since .mcp.json launches the
'dspy-jev' command:
    export PATH="$VENV_DIR/bin:\$PATH"

Calibrate this harness separately -- it has its own artifact:
    DSPY_JEV_HARNESS=claude-code dspy-jev calibrate
NEXT
