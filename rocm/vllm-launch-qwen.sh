#!/usr/bin/env bash
# Qwen2.5-VL-72B (FP8) on the MI300X — doc OCR + structured extraction (:8000).
# OpenAI-compatible API at http://localhost:8000/v1. See docs/architecture.md §3.1.
set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
PORT="${QWEN_PORT:-8000}"
SERVED_NAME="${QWEN_SERVED_NAME:-qwen2.5-vl-72b}"

# ROCm/MI300X: single 192 GB GPU, FP8 to fit alongside Llama + BGE (~75 GB each).
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len "${QWEN_MAX_LEN:-16384}" \
  --gpu-memory-utilization "${QWEN_GPU_UTIL:-0.42}" \
  --limit-mm-per-prompt image=3 \
  --trust-remote-code \
  --disable-log-requests
