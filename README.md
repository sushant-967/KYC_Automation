# Agentic KYC Intelligence Platform

A multi-agent system that does end-to-end **Customer Due Diligence** for Indian
banking KYC (RBI / PMLA). Upload a customer's documents and the platform extracts
their data, screens them against sanctions / PEP / adverse-media lists, scores
their risk on a deterministic 0–100 scale, and writes a human-readable
explanation of *why* the score is what it is — all auditable, all reproducible.

**TCS & AMD AI Hackathon — Track 1 (Agents).** Nine specialized agents
orchestrated by **LangGraph** on a Python **FastAPI** app, with all model
inference served locally by **vLLM** on a single **AMD MI300X** box. No
external inference or data APIs at runtime in the production path.

> Deeper context: [`docs/architecture.md`](docs/architecture.md) ·
> hackathon brief: [`docs/tcs-amd-ai-hackathon.md`](docs/tcs-amd-ai-hackathon.md)

---

## New here? Pick a path

There are four ways to run this. Start at the top and move down only when you
need more fidelity.

| Path | What you get | Time | Needs |
|------|--------------|------|-------|
| **A. Smoke test** | Prove the pipeline wires up correctly | ~30 sec | `python smoke_test.py` |
| **B. Interactive demo** (`KYC_DEMO=1`) | Real pipeline, planted entities, stubbed GPU | ~5 min | Python only |
| **C. Real LLMs on a laptop** (Groq) | Real LLM behavior on your Mac/PC | ~15 min | Groq API key |
| **D. Full production** (AMD box) | The actual hackathon demo target | varies | MI300X + vLLM |

### Path A — Smoke test (read this if you've never touched the repo)

The cheapest sanity check. Runs the full nine-agent LangGraph pipeline with a
fake vLLM and an empty screening index. If this fails, something is wrong with
the wiring before you spend time on anything else.

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
```

You should see all nine agents emit `running → done` in order, ending with
`PASS — full pipeline ran end-to-end with stubbed vLLM.`

### Path B — Interactive demo, no GPU, no API keys

`KYC_DEMO=1` swaps in deterministic stand-ins for the three GPU calls and seeds
two planted entities (Viktor as a sanctions hit, Rajesh as a PEP). The rest of
the pipeline — embedding-based recall, deterministic risk scoring, decision
thresholds — runs for real, so you see actual `approve` / `review` / `escalate`
outcomes for the three demo personas.

```bash
cd server
KYC_DEMO=1 ./run.sh                                # API on http://localhost:7860

# in another terminal — Streamlit dashboard talks to the API over HTTP
cd apps/ui
pip install -r requirements.txt
API_BASE=http://localhost:7860 streamlit run dashboard.py   # → http://localhost:8501
```

The dashboard lets you pick a persona, submit it, and watch the agent timeline
fill in over SSE.

**The three personas:**
- **Priya** (clean) → low score → `approve`
- **Rajesh** (PEP) → mid score → `review` (pauses for human verdict)
- **Viktor** (sanctions) → high score → `escalate`

### Path C — Real LLMs on a laptop (Groq + fastembed)

When you want to see what real LLMs produce without the AMD box. Chat goes
to **Groq's** OpenAI-compatible API (Llama 3.3 70B for reasoning, Llama 4 Scout
for vision), embeddings run locally via **`fastembed`** (ONNX BGE-small, ~130
MB, no torch).

```bash
cd server
pip install fastembed                              # one-time, ~5 MB lib
export KYC_BACKEND=groq
export GROQ_API_KEY=gsk_…                          # from console.groq.com/keys

# Optional: build a small OpenSanctions index in the same embedding space
python ingest.py --input ../data/opensanctions/snapshot.jsonl \
                 --limit 1000 --backend local

./run.sh                                           # http://localhost:7860
```

Tunables (`GROQ_VISION_MODEL`, `GROQ_REASON_MODEL`, `KYC_LOCAL_EMBEDDER`,
`GROQ_BASE_URL`) are documented at the top of
[`server/vllm_client.py`](server/vllm_client.py).

### Path D — Full production on the AMD box

The actual hackathon target. vLLM serves all three models locally; FastAPI talks
to them over `localhost`. Each launch script needs its own shell or tmux pane.

```bash
rocm/pull-models.sh                                # one-time per session (~153 GB)
rocm/vllm-launch-qwen.sh                           # :8000
rocm/vllm-launch-llama.sh                          # :8001
rocm/vllm-launch-bge.sh                            # :8002

cd server
python ingest.py --input ../data/opensanctions/snapshot.jsonl --limit 100000
./run.sh                                           # http://localhost:7860
```

---

## How it works

```
intake → extraction ★ → entity-resolution
       → { sanctions ‖ PEP ‖ adverse-media ★  ‖  id-verify  ‖  financial-profile }
       → risk (deterministic) → explainability ★ → decision
                                          approve <30 · review 30–69 · escalate ≥70
```

★ = deep agent (GPU). Others are pure Python.

| # | Agent | Type | What it does |
|---|-------|------|--------------|
| 1 | intake | light | normalize customer + document submission |
| 2 | extraction ★ | Qwen-VL | OCR each document into typed fields; **masks Aadhaar** |
| 3 | entity-resolution | light | canonical name + alias merge + prior cases |
| 4 | screening ★ | BGE + Llama | sanctions / PEP / adverse-media — recall → precision → adjudicate |
| 5 | id-verify | light | pass/fail from extraction validations (MRZ, regex, expiry) |
| 6 | financial-profile | light | income plausibility, geography risk, employment risk |
| 7 | risk | light | **deterministic** weighted sum of all signals → 0–100 score |
| 8 | explanation ★ | Llama | summary + evidence cards from the risk breakdown |
| 9 | decision | light | threshold the score into approve / review / escalate |

The deterministic risk score is the project's whole pitch — anyone can compute a
score, but making the *why* reproducible and legible is what makes the audit
trail honest. The LLM explains; it does not compute.

## Models

Production (vLLM on the AMD box):

| Port | Model | Role |
|------|-------|------|
| 8000 | Qwen2.5-VL-72B (FP8) | doc OCR + structured extraction (vision) |
| 8001 | Llama-3.3-70B-Instruct (FP8) | screening adjudication + explainability |
| 8002 | BGE-large-en-v1.5 | entity-name embeddings for sanctions recall |

Laptop dev (Groq + fastembed):

| Role | Backend | Default model |
|------|---------|---------------|
| Vision | Groq | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Reasoning | Groq | `llama-3.3-70b-versatile` |
| Embeddings | local fastembed | `BAAI/bge-small-en-v1.5` (ONNX) |

## Where to look in the code

Start at `server/app.py` and follow one request through. Everything else hangs
off that thread.

```
server/
  app.py               ← start here. routes + SSE + background pipeline runner
  pipeline.py          ← LangGraph StateGraph wiring the 9 agents
  schemas.py           ← pydantic contracts — the truth about agent inputs/outputs
  vllm_client.py       ← LLM backend (vLLM | Groq), factory at the bottom
  store.py             ← sqlite case state + append-only audit log
  screening_index.py   ← numpy cosine recall over OpenSanctions vectors
  agents/              ← one file per agent — each is a single async function
  ingest.py            ← OpenSanctions JSONL → sqlite + embeddings (one-time)
  demo.py              ← KYC_DEMO=1 stand-ins for vLLM + planted entities
  smoke_test.py        ← full pipeline against FakeVllm (zero GPU)
rocm/
  vllm-launch-*.sh     ← FP8 launch scripts (Qwen-VL / Llama / BGE)
  pull-models.sh       ← idempotent HF downloader (~153 GB)
  prompts/*.md         ← prompts loaded by deep agents
  bench/*.py           ← latency/throughput benchmarks for slide 4
apps/ui/               ← Streamlit dashboard (thin client of the API)
data/                  ← bundled OpenSanctions snapshot + synthetic adverse media
personas/              ← three demo customers (clean / PEP / sanctions)
docs/                  ← architecture, ADRs, hackathon brief
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/cases` | create a case; runs the pipeline in the background |
| GET  | `/api/cases/:id` | full case state |
| GET  | `/api/cases/:id/stream` | SSE stream of pipeline events (one per agent step) |
| POST | `/api/cases/:id/decide` | human verdict for review/escalate cases |
| GET  | `/healthz` | liveness + entity count |

Submission shape lives in
[`server/schemas.py`](server/schemas.py) — `Submission` → `CaseState`.

## Sanctions / PEP / adverse-media dataset

### Why we need one at all

The screening agent (§4.4 of the architecture doc) has to answer three
questions per customer: are they on a sanctions list, a PEP list, or in
adverse media? Each "yes" feeds the deterministic risk score (sanctions
+50, PEP +30, adverse +20×severity). With no list to query, every customer
scores low → every customer is approved → there's nothing for the
explainability agent to explain. The bundled list is what makes the demo
land for an Indian banking audience.

### What we use — OpenSanctions

Source: **[OpenSanctions](https://www.opensanctions.org/)**, the consolidated
open feed of every major global sanctions + PEP source (UN Security Council,
OFAC SDN, EU consolidated, UK HMT, Interpol, ~150 national PEP datasets
including several Indian ones). One bulk download covers what would otherwise
be 20+ separate API integrations.

- Format: FollowTheMoney JSON-lines (entity-per-line, well-structured, with
  aliases / DOB / country / dataset provenance per entity).
- Acquisition: `https://data.opensanctions.org/datasets/`
- License: **CC-BY-NC 4.0** — non-commercial. The hackathon is non-commercial,
  attribution is in slides + README + [`data/opensanctions/ATTRIBUTION.md`](data/opensanctions/ATTRIBUTION.md).
- If the NC clause is a problem, the OFAC SDN list (public domain) is
  schema-compatible — `ingest.py` accepts it as a drop-in.

### India relevance

OpenSanctions' default collection skews global (4.9 M entities, most of them
irrelevant to an Indian bank), so an unfiltered ingest matches Indian customers
against arbitrary UN designees instead of meaningful Indian PEPs / wanted
persons / sanctioned entities. The datasets that actually matter for an
Indian-bank demo — all verified to exist in the OpenSanctions catalog:

| Dataset id | Entities | Role |
|---|---|---|
| `in_sansad` | 19,079 | **PEPs** — Lok Sabha + Rajya Sabha members |
| `in_nse_debarred` | 31,391 | **adverse media** — NSE-debarred entities (fraud / insider trading) |
| `in_mha_banned` | 260 | **sanctions** — MHA-banned organizations |
| `sanctions` (global) | 283,621 | UN / OFAC / EU / UK consolidated — includes India-relevant designees |

Total: ~334 K entities, ~310 MB raw download. That's the recommended demo
dataset.

### Disk footprint

Per entity (measured, not estimated): ~500–1500 bytes of structured JSON
depending on dataset density, plus the embedding. The embedding dominates
larger entities; JSON dominates the smallest ones.

| Backend | Embedding dim | Embedding bytes |
|---|---|---|
| vLLM BGE-large (AMD box) | 1024 × float32 | ~4 KB |
| fastembed BGE-small (laptop) | 384 × float32 | ~1.5 KB |

Total `server/opensanctions.db` size:

| Entities | bge-large (AMD) | bge-small (laptop) |
|---|---|---|
| 50 K (just `in_sansad` + `in_nse_debarred`) | ~280 MB | ~135 MB |
| **334 K (India subset, recommended)** | **~1.9 GB** | **~900 MB** |
| 4.9 M (full unfiltered default) | ~28 GB | ~13 GB |

The raw JSONL is ~310 MB (India subset) or ~4.5 GB (full default). You can
delete it after ingest — only the SQLite DB is queried at runtime.

### When you actually need it

Tied to the four onboarding paths at the top of this README:

| Path | Dataset? | Why |
|---|---|---|
| A. Smoke test | **no** | `EmptyIndex` — verifying wiring only |
| B. `KYC_DEMO=1` | **no** | `demo.py` plants Viktor + Rajesh as fake entities, so screening still produces real `escalate` / `review` outcomes |
| C. Groq laptop | **yes, India subset** | run `data/opensanctions/fetch.sh` (~310 MB raw) |
| D. AMD box (production demo) | **yes, India subset** | same fetch script, then ingest with `--backend vllm` |

### Acquire & ingest

One script downloads the India-focused subset and (optionally) runs ingest
against either backend. No auth required.

```bash
# Just download — prints the ingest commands at the end
data/opensanctions/fetch.sh

# Download + ingest locally (fastembed, no GPU)
data/opensanctions/fetch.sh --ingest local

# Download + ingest via vLLM BGE on :8002 (AMD box)
data/opensanctions/fetch.sh --ingest vllm

# Pin to a specific snapshot date for a reproducible demo
data/opensanctions/fetch.sh --pin 20260610

# Or, for the full 4.9 M-entity default collection (not recommended)
data/opensanctions/fetch.sh --full
```

The script is idempotent — it skips files that are already downloaded unless
you pass `--force`. Pin the snapshot date in
[`data/opensanctions/ATTRIBUTION.md`](data/opensanctions/ATTRIBUTION.md) once
you've fetched a known-good version.

Demo ID documents and adverse-media articles in `data/adverse-media/` and
`personas/*/` are **synthetic** and clearly labeled.

## Conventions

A handful of rules the code holds itself to — useful to know before editing:

- **No external inference or data APIs at runtime in the production path.**
  Models are local (vLLM on this box). OpenSanctions is bundled into sqlite.
- **Risk scoring is deterministic.** The score must be reproducible from inputs.
  Don't let an LLM compute it.
- **Aadhaar is masked at extraction** to `XXXX-XXXX-1234`. The raw value never
  propagates downstream.
- **Validate at every boundary** using the pydantic models in `schemas.py`. LLM
  output is parsed defensively with safe-JSON + per-agent fallbacks.
- **Deep vs light agents.** Deep = extraction, screening, explainability (call
  the LLM). Light = the rest (pure Python). Keep light agents model-free.
