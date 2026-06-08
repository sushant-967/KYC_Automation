# CLAUDE.md — project context for Claude Code

Agentic KYC Intelligence Platform. **TCS & AMD AI Hackathon, Track 1 (Agents);
submission June 12, 2026.** Multi-agent Customer Due Diligence (CDD) pipeline for
Indian banking KYC (RBI Master Direction / PMLA), with a legible, auditable
rationale for every decision.

## ⚠️ Architecture — read this first

**Everything runs on ONE AMD MI300X box.** vLLM serves the models locally and a
Python **FastAPI** app orchestrates the agents over `localhost`. There is **no
Cloudflare** anywhere — the original Workers/Durable-Objects/D1/Vectorize/Tunnel
plan was dropped on day 1. See `docs/adr-001-single-box-fastapi.md`. The planning
doc `docs/architecture.md` is still the source of truth for *agent behavior* (§4),
risk scoring (§4.7), metrics (§6), and personas (§8) — but its §3 topology and §10
layout are superseded by the ADR.

If a request implies Cloudflare/Workers/Durable Objects/Vectorize, stop — that's
the old design. Use the local equivalents below.

## Layout

```
server/                FastAPI orchestrator (runs on the box, next to vLLM)
  app.py               routes + SSE + HITL; background pipeline execution
  pipeline.py          9-agent orchestrator (intake→…→decision)
  schemas.py           pydantic contracts — the source of truth for agent I/O
  vllm_client.py       async httpx → localhost:8000 Qwen / 8001 Llama / 8002 BGE
  store.py             sqlite case state + append-only audit log (was the DO)
  screening_index.py   in-process numpy cosine recall over OpenSanctions vectors
  metrics.py           per-agent + per-gpu-call timing
  ingest.py            OpenSanctions JSONL → sqlite + BGE embeddings
  agents/*.py          the 9 agents (intake, extraction, entity_resolution,
                       screening, id_verify, financial_profile, risk,
                       explainability, decision)
  smoke_test.py        full pipeline with a STUBBED vLLM (no GPU) — keep it green
  requirements.txt · run.sh
rocm/                  vLLM on the AMD box
  vllm-launch-{qwen,llama,bge}.sh   FP8 (compressed-tensors) launch scripts
  pull-models.sh                    idempotent model downloader / per-session re-pull
  prompts/*.md                      loaded by the deep agents
  bench/*.py                        slide-4 latency/throughput baselines
apps/ui/               React + Vite frontend — NOT BUILT YET (deferred, see below)
data/ · personas/ · docs/
```

## Models (vLLM, localhost)

| Port | Served name | Checkpoint (ungated, FP8) | Role |
|------|-------------|---------------------------|------|
| 8000 | `qwen2.5-vl-72b` | `RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic` (~76 GB) | doc OCR + extraction (vision) |
| 8001 | `llama-3.3-70b` | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` (~73 GB) | screening adjudication + explainability |
| 8002 | `bge-large-en-v1.5` | `BAAI/bge-large-en-v1.5` (~4 GB) | entity-name embeddings |

No HF token needed (all ungated). All three at FP8 fit the single 206 GB GPU.

## Run

```bash
rocm/pull-models.sh                              # download weights (~153 GB)
rocm/vllm-launch-qwen.sh                          # :8000  (each in its own pane)
rocm/vllm-launch-llama.sh                         # :8001
rocm/vllm-launch-bge.sh                           # :8002
cd server && python ingest.py --input ../data/opensanctions/snapshot.jsonl --limit 100000
cd server && ./run.sh                              # FastAPI → http://localhost:7860
cd server && python smoke_test.py                  # verify wiring WITHOUT a GPU
```

## Conventions (don't break these)

- **No external inference or data APIs at runtime.** Everything is local: models on
  this box, OpenSanctions bundled into sqlite. (architecture.md §5.5)
- **Risk scoring is deterministic** (`agents/risk.py`, explicit weight constants).
  The score must be reproducible from inputs — that's what makes explainability
  honest. Don't let an LLM compute the score.
- **Aadhaar is masked** in the extraction agent (`XXXX-XXXX-1234`); the raw 12-digit
  value never propagates downstream. (§4.2)
- **Validate at every boundary** with the pydantic models in `schemas.py`. Deep-agent
  vLLM outputs are parsed defensively (`vllm_client._safe_json`) with fallbacks.
- **Deep vs light agents:** deep = extraction, screening, explainability (call vLLM);
  light = the rest (pure Python). Keep light agents model-free.
- Decision thresholds: score <30 approve · 30–69 review (human) · ≥70 escalate.

## Build sequence (current)

Agentic side (`server/` + `rocm/`) is **done and smoke-tested**. The **frontend
(`apps/ui/`) is intentionally deferred** until the agentic side is validated on real
GPU. Do not start the UI unless asked.

## Environment gotchas

- The container is **ephemeral** (resets ~every few hours). Only `/workspace/shared`
  (28 GB, where this repo lives) persists. **Model weights are too big for it** — they
  live on the ephemeral `/` (612 GB) and must be re-pulled each session via
  `rocm/pull-models.sh`.
- The repo **auto-commits**. Push to `origin/main` when asked.
- Tooling already on the box: `vllm` 0.11 (ROCm), `huggingface-cli`/`hf`, Python 3.12.
  Node is at `/workspace/shared/.persist/node` (for the future frontend).
