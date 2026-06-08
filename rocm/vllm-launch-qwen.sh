#!/usr/bin/env bash
# Qwen2.5-VL-72B (FP8) on the MI300X — doc OCR + structured extraction (:8000).
# OpenAI-compatible API at http://localhost:8000/v1. See docs/architecture.md §3.1.
set -euo pipefail

# Pre-quantized FP8 (compressed-tensors) — vLLM auto-detects the quant from the
# model config, so no --quantization flag. ~76 GB, ungated. ROCm/MI300X gfx942.
MODEL="${QWEN_MODEL:-RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic}"
PORT="${QWEN_PORT:-8000}"
SERVED_NAME="${QWEN_SERVED_NAME:-qwen2.5-vl-72b}"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len "${QWEN_MAX_LEN:-16384}" \
  --gpu-memory-utilization "${QWEN_GPU_UTIL:-0.42}" \
  --limit-mm-per-prompt image=3 \
  --trust-remote-code \
  --disable-log-requests
