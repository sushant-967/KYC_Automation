#!/usr/bin/env bash
# Llama-3.3-70B-Instruct (FP8) on the MI300X — text reasoning / adjudication /
# explanation (:8001). OpenAI-compatible API at http://localhost:8001/v1.
set -euo pipefail

# Pre-quantized FP8 (compressed-tensors) — ~73 GB, ungated. vLLM auto-detects
# the quant from the model config, so no --quantization flag.
MODEL="${LLAMA_MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
PORT="${LLAMA_PORT:-8001}"
SERVED_NAME="${LLAMA_SERVED_NAME:-llama-3.3-70b}"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len "${LLAMA_MAX_LEN:-16384}" \
  --gpu-memory-utilization "${LLAMA_GPU_UTIL:-0.42}" \
  --disable-log-requests
