#!/usr/bin/env bash
# Start a local MLflow tracking server for DSPy traces.
# No signup and no API key: MLflow's DSPy autolog works against a local server.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PORT="${MLFLOW_PORT:-5000}"
STORE="${MLFLOW_STORE:-$DSPY_JEV_ROOT/.mlflow/mlflow.sqlite}"

mkdir -p "$(dirname "$STORE")"
step "starting MLflow on http://127.0.0.1:$PORT (store: $STORE)"
warn "SQLite is required for tracing; the default file store does not support it."

exec "$VENV_DIR/bin/mlflow" server \
  --backend-store-uri "sqlite:///$STORE" \
  --host 127.0.0.1 \
  --port "$PORT"
