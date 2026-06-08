# Agentic KYC Intelligence Platform

Six+ specialized agents collaborate in a deterministic pipeline to perform
end-to-end Customer Due Diligence (CDD) for Indian banking KYC (RBI / PMLA), with
a human-readable audit trail of *why* every decision was made.

**TCS & AMD AI Hackathon — Track 1 (Agents).** Everything runs on a single AMD
(MI300X) box: vLLM serves the models locally and a Python FastAPI app orchestrates
the agents. No external inference or data APIs at runtime.

> Architecture: [`docs/architecture.md`](docs/architecture.md) ·
> single-box pivot: [`docs/adr-001-single-box-fastapi.md`](docs/adr-001-single-box-fastapi.md)

## Pipeline

```
intake → extraction ★ → entity-resolution
       → { sanctions ‖ PEP ‖ adverse-media ★  ‖  id-verify  ‖  financial-profile }
       → risk (deterministic) → explainability ★ → decision
                                          approve <30 · review 30–69 · escalate ≥70
```
★ = deep agent (GPU). Others are pure Python.

## Models (vLLM on the box)

| Port | Model | Role |
|------|-------|------|
| 8000 | Qwen2.5-VL-72B (FP8) | doc OCR + structured extraction (vision) |
| 8001 | Llama-3.3-70B (FP8) | screening adjudication + explainability |
| 8002 | BGE-large-en-v1.5 | entity-name embeddings for sanctions recall |

## Run

```bash
# 1. Start the models (each in its own shell/tmux pane)
rocm/vllm-launch-qwen.sh
rocm/vllm-launch-llama.sh
rocm/vllm-launch-bge.sh

# 2. Ingest the bundled OpenSanctions snapshot → sqlite + embeddings (once)
cd server && python ingest.py --input ../data/opensanctions/snapshot.jsonl --limit 100000

# 3. Start the orchestrator (creates a venv on first run)
./run.sh                      # → http://localhost:7860

# Verify the pipeline wiring without a GPU:
python smoke_test.py
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/cases` | create a case; runs the pipeline in the background |
| GET  | `/api/cases/:id` | full case state |
| GET  | `/api/cases/:id/stream` | SSE stream of pipeline events |
| POST | `/api/cases/:id/decide` | human verdict (approve / review / escalate) |
| GET  | `/healthz` | liveness + entity count |

## Layout

```
server/   FastAPI orchestrator + 9 agents + sqlite + vector recall + ingest
rocm/     vLLM launch scripts, deep-agent prompts, benchmarks
data/     bundled OpenSanctions snapshot + synthetic adverse-media (declared)
personas/ synthetic demo customers (Priya / Rajesh / Viktor)
apps/ui/  React + Vite frontend (built after the agentic side)
docs/     architecture, ADRs, metrics, demo script, slides
```

## Data & attribution

Sanctions/PEP/adverse-media data is bundled from **OpenSanctions** (CC-BY-NC 4.0,
non-commercial — see [`data/opensanctions/ATTRIBUTION.md`](data/opensanctions/ATTRIBUTION.md)).
Demo ID documents and adverse-media articles are **synthetic** and clearly labeled.
