#!/usr/bin/env bash
# Download the three models we need into the HF cache. Idempotent — already-present
# files are skipped, so this doubles as the per-session re-pull (weights live on the
# ephemeral disk and are lost on container reset; the persistent disk is too small).
#
#   rocm/pull-models.sh            # all three
#   ONLY=bge rocm/pull-models.sh   # just the embedder
#
# No HF token required — all three checkpoints are ungated. FP8 (compressed-tensors)
# variants of the two large models keep the download to ~150 GB.
set -euo pipefail

# Keep the cache on the big ephemeral disk (/ has ~600 GB; /workspace/shared is 28 GB).
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

QWEN="${QWEN_MODEL:-RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic}"
LLAMA="${LLAMA_MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
BGE="${BGE_MODEL:-BAAI/bge-large-en-v1.5}"

pull() {
  echo "[pull] $1"
  hf download "$1" --exclude "*.pth" "original/*" "*.gguf"
}

case "${ONLY:-all}" in
  bge)   pull "$BGE" ;;
  llama) pull "$LLAMA" ;;
  qwen)  pull "$QWEN" ;;
  *)     pull "$BGE"; pull "$LLAMA"; pull "$QWEN" ;;
esac

echo "[pull] done. cache: $HF_HOME"
du -sh "$HF_HOME/hub" 2>/dev/null || true
