#!/usr/bin/env bash
# BGE-large-en-v1.5 (FP16) on the MI300X — 1024-dim entity-name embeddings for
# sanctions vector search (:8002). OpenAI-compatible /v1/embeddings.
set -euo pipefail

MODEL="${BGE_MODEL:-BAAI/bge-large-en-v1.5}"
PORT="${BGE_PORT:-8002}"
SERVED_NAME="${BGE_SERVED_NAME:-bge-large-en-v1.5}"

# ~2 GB; leave it FP16. Small slice of the GPU.
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --task embed \
  --port "$PORT" \
  --gpu-memory-utilization "${BGE_GPU_UTIL:-0.08}" \
  --disable-log-requests
