#!/usr/bin/env bash
# Base install: virtual environment, dependencies, and a health check.
# Harness-specific wiring lives in setup_<harness>.sh, which calls this first.
#
#   ./scripts/install.sh [extras]     # default: service,observability,mcp
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EXTRAS="${1:-service,observability,mcp}"

step "dspy-jev base install"
PYTHON_BIN="$(require_python)"
ok "using $PYTHON_BIN ($("$PYTHON_BIN" -V))"
create_venv "$PYTHON_BIN"
install_project "$EXTRAS"

step "verifying the import graph"
"$VENV_DIR/bin/python" - <<'PY'
import warnings
warnings.filterwarnings("ignore")
import dspy
from dspy.experimental import Choice, Noul, ReAnchor, Score  # noqa: F401
import dspy_jev
print(f"  dspy {dspy.__version__}, dspy-jev {dspy_jev.__version__}, decision types available")
PY
ok "base install complete"

cat <<'NEXT'

Next:
  source .venv/bin/activate
  ./scripts/setup_hermes_agent.sh     # or setup_pi.sh / setup_claude_code.sh
NEXT
