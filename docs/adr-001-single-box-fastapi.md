# ADR-001 — Single-box FastAPI orchestrator (supersedes the Cloudflare topology)

**Status:** Accepted · June 8, 2026 (kick-off day)
**Supersedes:** the Cloudflare Workers / Durable Objects / D1 / Vectorize / Tunnel
topology in `architecture.md` §3 and the §10 repo layout.

## Context

The original plan assumed two boxes: a Cloudflare Worker for orchestration and a
separate MI300X for inference, bridged by a Cloudflare Tunnel. In practice we have
**one** AMD instance, and that instance is where vLLM and the models live. There is
no separate Cloudflare environment, and we don't want to depend on one.

## Decision

Run the entire orchestrator **locally on the same box as vLLM**, in Python.

| Original (Cloudflare) | Now (local on the box) |
|---|---|
| Worker + routes | **FastAPI** app (`server/app.py`) |
| Durable Object (per-case state, SSE) | in-process `Hub` + SQLite (`store.py`) |
| D1 (OpenSanctions entities) | SQLite file (`opensanctions.db`) |
| Vectorize (embeddings) | in-process numpy brute-force / FAISS (`screening_index.py`) |
| Cloudflare Workflows | plain async orchestrator (`pipeline.py`) |
| Tunnel → MI300X | direct `localhost:8000/8001/8002` |

Everything else from `architecture.md` stands: the 9-agent pipeline, the agent
contracts (§4), deterministic risk scoring (§4.7), the deep/light split, the
metrics taxonomy (§6), personas (§8), and the "no external APIs at runtime" rule
(§5.5) — which is now even easier to honor since nothing leaves the box.

## New repo layout

```
server/                FastAPI orchestrator (runs on the box)
  app.py               routes + SSE + HITL
  pipeline.py          9-agent orchestrator (intake→…→decision)
  schemas.py           pydantic contracts (was worker/schemas.ts)
  vllm_client.py       httpx → localhost vLLM (was rocm-client.ts)
  store.py             sqlite case state + audit log (was the Durable Object)
  metrics.py           per-agent + per-gpu-call instrumentation
  screening_index.py   numpy/FAISS recall over OpenSanctions vectors
  ingest.py            OpenSanctions JSONL → sqlite + embeddings
  agents/*.py          the 9 agents
  smoke_test.py        full pipeline with a stubbed vLLM (no GPU)
  requirements.txt, run.sh
rocm/                  vLLM on the AMD box
  vllm-launch-{qwen,llama,bge}.sh
  prompts/*.md         loaded by the deep agents
  bench/*.py           slide-4 baselines
apps/ui/               React + Vite frontend (built after the agentic side)
data/ · personas/ · docs/
```

## Consequences

- **One runtime on the box** (Python venv) shared with the vLLM ecosystem — no
  second language/toolchain, no tunnel to keep alive.
- **No managed persistence**: SQLite is single-node. Fine for a demo; if we ever
  needed multi-node we'd revisit. Audit log still survives process restarts.
- **Vector search is in-process**: ~100K×1024 float32 ≈ 0.4 GB RAM, query is one
  matmul. FAISS is a drop-in upgrade (commented in `requirements.txt`).
- The slide-3 architecture diagram must be redrawn to the single-box topology.
