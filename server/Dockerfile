# server/Dockerfile — FastAPI orchestrator image.
#
# Targets laptop dev (Path B `KYC_DEMO=1` or Path C `KYC_BACKEND=groq`). Does
# NOT include vLLM — the AMD-box production path runs natively on ROCm, not in
# a container, since GPU passthrough for ROCm on a Mac is non-trivial.
#
# Build:  docker build -f server/Dockerfile -t kyc-api .  (run from repo root)
# Run:    docker compose up api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# curl is handy for healthchecks; the rest is base.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Recreate the repo layout inside /repo so the agents' Path(__file__).parents[N]
# math (e.g. rocm/prompts/extraction.md) keeps working.
WORKDIR /repo/server

# 1. Base deps — cache layer if requirements don't change.
COPY server/requirements.txt /repo/server/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2. fastembed — needed for KYC_BACKEND=groq (~130 MB model on first use).
#    Adds ~5 MB of code now; the model file lands in the HF cache at runtime.
RUN pip install fastembed

# 3. PaddleOCR — pre-OCR pass for the extraction agent. ~270 MB of paddlepaddle
#    plus paddleocr's own model weights download on first use. Heavy, but
#    fragile to install natively on Apple Silicon, so baking it into the
#    Linux image is the safest way to give all backends real OCR text.
RUN pip install paddleocr paddlepaddle pillow

# 4. Server source + the rocm/prompts the deep agents load.
COPY server/ /repo/server/
COPY rocm/prompts/ /repo/rocm/prompts/

# Directories the runtime expects to write to.
RUN mkdir -p /repo/server/uploads

EXPOSE 7860

# uvicorn directly so logs go straight to stdout (no run.sh venv dance).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
