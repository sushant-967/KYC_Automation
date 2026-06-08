#!/usr/bin/env bash
# Launch the FastAPI orchestrator on the box. vLLM must already be up on
# localhost:8000/8001/8002 (see rocm/vllm-launch-*.sh).
set -euo pipefail
cd "$(dirname "$0")"

HOST="${KYC_HOST:-0.0.0.0}"
PORT="${KYC_PORT:-7860}"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -U pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/uvicorn app:app --host "$HOST" --port "$PORT" "$@"
