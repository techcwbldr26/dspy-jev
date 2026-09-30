#!/usr/bin/env bash
# Start a local MLflow tracking server so you can SEE what the gate did.
#
# Every decision is already traced. This gives the traces somewhere to live and
# a browser page to read them in. No signup and no API key; nothing leaves the
# machine, since the store is a SQLite file under ./mlruns.
#
#   scripts/observability.sh               # start it in the background, print the URL
#   scripts/observability.sh --foreground  # run it in this terminal instead
#   scripts/observability.sh --stop        # stop the background one
#
# Then run the gate with tracing pointed at it:
#   DSPY_JEV_MLFLOW_ENABLED=true dspy-jev serve

set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

PORT="${MLFLOW_PORT:-5000}"
STORE="${MLFLOW_STORE:-$DSPY_JEV_ROOT/mlruns}"
PID_FILE="$STORE/mlflow.pid"

if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"; rm -f "$PID_FILE"
    ok "stopped the tracking server"
  else
    warn "no tracking server recorded at $PID_FILE"
  fi
  exit 0
fi

[[ -x "$VENV_DIR/bin/mlflow" ]] || die "mlflow is not installed. Run scripts/setup_<harness>.sh; it installs the observability extra."

if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  ok "a tracking server is already answering on port $PORT"
  echo "    open http://127.0.0.1:$PORT and pick the 'dspy-jev' experiment"
  exit 0
fi

mkdir -p "$STORE"
# SQLite, not the default file store: tracing needs a database backend.
# Telemetry off, so this works on an air-gapped box too.
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
MLFLOW_ARGS=(
  server --host 127.0.0.1 --port "$PORT"
  --backend-store-uri "sqlite:///$STORE/mlflow.db"
  --artifacts-destination "$STORE/artifacts"
)

next_steps() {
  cat <<EOF

  Open  http://127.0.0.1:$PORT  and choose the 'dspy-jev' experiment, then the
  Traces tab. Each decision is one trace: the policy step, the DSPy chain, the
  model call inside it, the prompt, the raw probabilities, tokens and latency.

  Point the gate at it:
      DSPY_JEV_MLFLOW_ENABLED=true dspy-jev serve
  Walkthrough with screenshots: docs/observability.md
EOF
}

if [[ "${1:-}" == "--foreground" ]]; then
  step "starting MLflow on http://127.0.0.1:$PORT (store: $STORE)"
  next_steps
  exec "$VENV_DIR/bin/mlflow" "${MLFLOW_ARGS[@]}"
fi

step "starting the MLflow tracking server on port $PORT"
nohup "$VENV_DIR/bin/mlflow" "${MLFLOW_ARGS[@]}" > "$STORE/mlflow.log" 2>&1 &
echo $! > "$PID_FILE"

for _ in $(seq 1 40); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ok "tracking server is up (pid $(cat "$PID_FILE"))"
    next_steps
    echo "  Stop it again:"
    echo "      scripts/observability.sh --stop"
    echo "  Logs: $STORE/mlflow.log"
    exit 0
  fi
  sleep 1
done

rm -f "$PID_FILE"
die "the tracking server did not come up in 40s. See $STORE/mlflow.log"
