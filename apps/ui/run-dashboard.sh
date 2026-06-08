#!/usr/bin/env bash
# Launch the Streamlit dashboard (thin client of the FastAPI backend).
# The API must be running first (server/run.sh). Point at it via API_BASE.
set -euo pipefail
cd "$(dirname "$0")"

export API_BASE="${API_BASE:-http://localhost:7860}"
PORT="${UI_PORT:-8501}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q -U pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

exec ./.venv/bin/streamlit run dashboard.py \
  --server.address 0.0.0.0 --server.port "$PORT" \
  --server.headless true --browser.gatherUsageStats false \
  --server.enableCORS false --server.enableXsrfProtection false
