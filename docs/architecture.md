# Agentic KYC Intelligence Platform — Architecture

The behavioral spec for the pipeline: what each agent does, what it inputs and
outputs, how risk is scored, what we measure, and which personas drive the demo.

The platform runs on a single AMD MI300X box: vLLM serves the three models
locally and a Python FastAPI app orchestrates the nine agents through a
LangGraph `StateGraph`. This doc focuses on agent contracts and pipeline
behavior, which the topology choice does not affect.

> **Hackathon:** TCS & AMD AI Hackathon, Track 1 (Agents). Submission June 12.

## 1. Problem statement

Indian banks, NBFCs, and fintechs onboard customers under the RBI's *Master
Direction — Know Your Customer (KYC) Direction, 2016* (and its amendments), the
Prevention of Money Laundering Act 2002 (PMLA), and FIU-IND reporting
obligations. Customer Due Diligence (CDD) today is mostly manual, slow (days),
inconsistent across reviewers, and produces approve/reject decisions without an
audit trail of *why* — a problem when the regulator asks.

**Our solution:** an agentic platform where nine specialized agents collaborate
in a deterministic pipeline to perform end-to-end CDD. Each agent emits
structured evidence; the explainability agent assembles a human-readable
rationale; the decision agent renders an outcome that a compliance officer can
trust because every signal — Aadhaar verification, PAN format validation,
sanctions match, PEP flag, adverse-media hit, financial-profile risk — is
exposed end-to-end.

Although the framing is India-primary (RBI/PMLA, Aadhaar/PAN/passport), the
architecture is jurisdiction-agnostic. The same pipeline works for FATCA-driven
cross-border customers, EU AML6, or US BSA-CIP with only the document parsers
and risk weights changed.

## 2. Pipeline (high level)

```
START
  │
  ▼
[ Intake ]                            ← validate submission shape, normalize fields
  │
  ▼
[ Extraction ]    ★ DEEP              ← Qwen2.5-VL-72B — OCR + structured fields from docs
  │
  ▼
[ Entity Resolution ]                 ← dedupe against prior cases, canonical name + DOB
  │
  ├───────────────────┬───────────────────┐
  ▼                   ▼                   ▼
[ Screening ]    [ ID Verify ]    [ Financial Profile ]
   ★ DEEP        (light)          (light)
(BGE recall + Llama 3.3 70B adjudication for sanctions / PEP / adverse-media)
  │
  ▼
[ Risk Aggregation ]                  ← deterministic weighted scoring across signals
  │
  ▼
[ Explainability ]   ★ DEEP           ← Llama 3.3 70B — turn evidence into human rationale
  │
  ▼
[ Decision ]
  │
  ├── score < 30:        APPROVE
  ├── 30 ≤ score < 70:   HUMAN REVIEW   ← UI surfaces with HITL buttons
  └── score ≥ 70:        ESCALATE       ← high-severity, paged to compliance
END
```

★ = "deep" agent (calls a GPU). The other steps are real but lighter (smaller
prompts, simpler logic, no GPU).

## 3. Models — three on one GPU

| Port | Model | Role | Approx VRAM |
|------|-------|------|------|
| 8000 | Qwen2.5-VL-72B (FP8) | doc OCR + structured extraction (vision) | ~75 GB |
| 8001 | Llama-3.3-70B-Instruct (FP8) | screening adjudication + explainability | ~75 GB |
| 8002 | BGE-large-en-v1.5 | entity-name embeddings for sanctions recall | ~2 GB |

Total ~152 GB on the 192 GB MI300X — comfortable headroom.

### 3.1 FP8 routing decision

Qwen2.5-VL-72B (~145 GB FP16) + Llama 3.3 70B (~140 GB FP16) + BGE-large
(~2 GB) exceeds 192 GB if loaded in FP16. We quantize the two large LLMs to FP8:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. FP8 both LLMs + FP16 embeddings** | All three fit concurrently. Lower latency from reduced memory bandwidth. ~40 GB headroom. | Quality drop on quantization. | **Chosen.** vLLM supports FP8 on MI300X via the ROCm fork. |
| B. Hot-swap between the two LLMs | Full FP16 quality. | 30-60 s swap per case is demo-killer. | Rejected. |
| C. Qwen-VL for both vision and text, no Llama | Frees ~75 GB. | Qwen-VL weaker on long-form text reasoning + structured output. | Fallback if FP8 has issues. |

## 4. Agents — contracts & responsibilities

Every agent reads from and writes to the case's structured state, defined in
`server/schemas.py` as pydantic models. The agent's output schema is the
contract downstream agents depend on; validate at every boundary so a
malformed vLLM response fails loud and early.

### 4.1 Intake (light)

- **Input:** raw submission (form fields + document file references).
- **Output:** `IntakeOutput { case_id, customer, documents, normalized_at }`.
- **Model:** none. Pure validation + normalization.
- **Failure modes:** missing required field → reject before pipeline starts.

### 4.2 Extraction (DEEP) ★

- **Input:** document file references from intake.
- **Output:** `ExtractionOutput { documents: [ExtractedDocument {kind, fields, confidence, raw_text, validations, masked_fields}] }`.
- **Model:** Qwen2.5-VL-72B on vLLM :8000.
- **Supported document kinds (India-primary set):**
  - **Aadhaar** — 12-digit number (we **mask the first 8** per UIDAI display rules), name, DOB/YOB, gender, address.
  - **PAN** — 10-char PAN (regex `^[A-Z]{5}[0-9]{4}[A-Z]$`), name, father's name, DOB.
  - **Voter ID (EPIC)** — EPIC number, name, age, address, assembly constituency.
  - **Passport** — name, DOB, nationality, MRZ lines (parsed + checksum), expiry, place of issue.
  - **Driving license** — DL number, name, DOB, address, validity, RTO.
  - **Address proof** — utility bill / rental agreement / bank statement (name, address, date, provider).
- **Prompt strategy:** structured-output prompt with a per-doc-kind JSON schema.
  Vision model returns fields + per-field confidence; the agent validates the
  shape and runs format checks (PAN regex, MRZ checksum, Aadhaar Verhoeff).
- **Why deep:** highest-value GPU work. A 72B vision model extracting Aadhaar +
  PAN fields with confidence and producing a structurally validated record is
  the demo headline.

**Aadhaar masking:** raw 12-digit Aadhaar never leaves the extraction agent.
The downstream pipeline, audit log, and UI all see only `XXXX-XXXX-1234`. A
small compliance touch with disproportionate signal to RBI-savvy judges.

### 4.3 Entity Resolution (light)

- **Input:** customer record from intake + extraction.
- **Output:** `EntityResolutionOutput { canonical_name, dob_confirmed, alias_matches, prior_cases }`.
- **Model:** none. Deterministic name canonicalization + DOB cross-check
  between submitted form and extracted ID. Looks up prior cases in the case
  store.

### 4.4 Screening — Sanctions + PEP + Adverse Media (DEEP) ★

Internally three sub-agents but one logical agent in the pipeline. The three
sub-agents run concurrently (`asyncio.gather`). **No external APIs at runtime**
— all data is bundled (see §6). "AI-powered fuzzy matching against a local
sanctions database" is also a stronger AMD/agentic story than calling someone
else's API.

**Three-stage funnel (per sub-agent):**

1. **Recall stage — vector search.** Customer's canonical name is embedded via
   BGE-large on vLLM :8002. The in-process `ScreeningIndex` does brute-force
   cosine over the loaded OpenSanctions matrix (~100 K × 1024 ≈ 400 MB, single
   matmul → a few ms) → top-K candidate entities, K≈20.
2. **Precision stage — name normalization + DOB filter.** Deterministic checks
   (Levenshtein on canonicalized names, DOB tolerance ±2 years for handwritten
   docs) drop obvious non-matches.
3. **Adjudication stage — LLM disambiguation.** Llama 3.3 70B receives the
   remaining 1-5 candidates with full structured context (aliases, DOB,
   country, dataset source) and outputs
   `{ verdict: 'match'|'no-match'|'uncertain', confidence, rationale, evidence }`.

This broad-recall → cheap-precision → expensive-LLM funnel is the standard
pattern and keeps per-case cost low.

- **Sanctions sub-agent:** filters entities with `topic ∈ {sanction, sanction.linked}`.
- **PEP sub-agent:** filters entities with `topic ∈ {role.pep, gov.*}` plus a country match heuristic.
- **Adverse media sub-agent:** filters entities flagged `topic.crime.*` or
  with adverse-media source URLs. For hits, Llama 3.3 70B summarizes the
  available `notes` / `summary` fields into a 2-sentence risk narrative with
  `severity ∈ {low, medium, high}`.

- **Output:** `ScreeningOutput { sanctions, pep, adverse_media }`.

### 4.5 ID Verification (light)

- **Input:** extracted ID doc.
- **Output:** `IDVerificationOutput { doc_authenticity, mrz_valid, expiry_ok, face_match_score? }`.
- **Implementation:** MRZ checksum + expiry check. No new model needed; the
  extraction agent already returned MRZ. Face match deferred unless time
  permits.

### 4.6 Financial Profile (light)

- **Input:** declared income, declared employment, country, address risk.
- **Output:** `FinancialProfileOutput { income_plausibility_score, geography_risk, employment_risk }`.
- **Implementation:** rules + lookup tables (country risk index, etc.).
  Documented heuristics, not ML.

### 4.7 Risk Aggregation (light) — **deterministic**

- **Input:** outputs from §4.2–4.6.
- **Output:** `RiskOutput { score: 0-100, contributors: [{ signal, weight, value, contribution }] }`.
- **Implementation:** deterministic weighted scoring. Weights are explicit
  constants in `server/agents/risk.py`:
  - `sanctions_hit = +50`
  - `pep_hit = +30`
  - `adverse_media = +20 × severity` (`low 0.5 / medium 0.75 / high 1.0`)
  - `id_fail = +30`
  - `geography_risk = up to +10`
  - `income_implausibility = up to +10`

**Deterministic is critical** — the score must be reproducible from the
inputs, which is what makes the explainability honest. Don't let an LLM
compute the score.

### 4.8 Explainability (DEEP) ★

- **Input:** entity, screening, risk score + contributors.
- **Output:** `ExplanationOutput { summary, evidence_cards: [{ title, finding, source, severity }], recommended_action }`.
- **Model:** Llama 3.3 70B with a structured-output prompt that takes the
  deterministic scoring breakdown and turns it into prose a compliance officer
  would actually read. Falls back to a deterministic summary if the model
  output fails to parse.
- **Why deep:** explainability is the differentiator. Anyone can compute a
  score; what wins this hackathon is making the *why* legible.

### 4.9 Decision (light)

- **Input:** risk score.
- **Output:** `DecisionOutput { decision: 'approve'|'review'|'escalate', requires_human }`.
- **Implementation:** threshold rules (see pipeline diagram §2). On `review`
  or `escalate`, the orchestrator parks the case in `awaiting_human` status
  until a human posts a verdict via `POST /api/cases/:id/decide`.

## 5. Case state & audit log

One `CaseState` pydantic model per case (`server/schemas.py`) persisted to
SQLite (`server/store.py`):

```python
class CaseState(BaseModel):
    case_id: str
    status: CaseStatus  # intake | running | awaiting_human | approved | rejected | escalated
    customer: CustomerInput
    documents: list[DocumentRef]
    agent_outputs: AgentOutputs  # one optional field per agent
    audit_log: list[AuditEvent]  # append-only, one row per emit
    metrics: CaseMetrics         # per-agent latency + per-GPU-call detail
```

A separate append-only `audit` table holds the same events keyed by `case_id`
so the trail survives a process restart even if in-memory state is lost.

## 6. Datasets (bundled, not API-called)

**Hard rule:** zero external inference or data APIs at runtime. Sanctions /
PEP / adverse-media data must be downloaded once during prep and bundled into
the project.

**Primary dataset: OpenSanctions bulk export.**
- Source: `https://data.opensanctions.org/datasets/` (snapshot taken June 6, version pinned in the submission).
- Format: FollowTheMoney JSON-lines (entity-per-line, well-structured, alias arrays, dataset provenance per entity).
- Scope: `default` collection (sanctions + PEP + crime + adverse-media-flagged entities). ~500 K entities, ~1.5 GB uncompressed. We prune to the subset matching the 3 personas + a few thousand decoys for realistic vector-search behavior, getting it down to <100 MB.
- License: CC-BY-NC 4.0 — non-commercial use only. The hackathon is non-commercial (educational/competition). We attribute OpenSanctions in the submission, slides, and README.

**Storage (on the box):**
- **SQLite** (`server/opensanctions.db`) for structured entity records:
  `entities(id, name, aliases JSON, dob, countries JSON, topics JSON, datasets JSON, summary, source_url, embedding BLOB)`.
- **In-process float32 matrix** loaded from those rows at startup — brute-force cosine recall via `screening_index.ScreeningIndex`. FAISS is a drop-in upgrade if the corpus grows.
- BGE-large embeddings (1024-d) are computed once during ingest and stored as `embedding BLOB`.

**Ingestion script:** `server/ingest.py` (run once during prep):
1. Read the pruned FtM JSON-lines export.
2. For each entity, embed `name + aliases[:5]` via vLLM BGE-large.
3. Insert row into SQLite with the embedding as a float32 byte blob.

**Adverse-media supplemental corpus:** hand-crafted 5-10 short fake "news articles" tied to demo personas ("Viktor Nazarov"). Stored as a JSON file in `data/adverse-media/`. Clearly labeled synthetic.

**Declaration in submission:**
- Slide 4 lists: OpenSanctions snapshot YYYY-MM-DD (CC-BY-NC, attributed), synthetic adverse-media corpus (authored June 6, in repo).
- README has full attribution + license notice.

## 7. Metrics & instrumentation (slide 4 ammunition)

The submission rubric **demands** GPU usage, memory, latency, and token counts
on slide 4. We produce these from the start, not bolted on.

**What we log per agent run** (`server/metrics.py` + the `CaseMetrics` model):
- `agent_name`, `case_id`, `latency_ms`
- `model` (if applicable), `input_tokens`, `output_tokens`
- For GPU calls: `vram_used_gb`, `gpu_util_pct`, `batch_size` (from vLLM's `/metrics` endpoint scraped per call).

**Where it lives:** in `CaseState.metrics`, persisted to SQLite with the rest
of the case state.

**Slide 4 will show:**
- Bar chart: tokens consumed per agent per case (averaged across 3 personas).
- Bar chart: latency per agent (p50, p95).
- One-liner: peak VRAM usage on MI300X (proving the 72B model is doing the work).
- Sustained tokens/sec for both Qwen-VL-72B and Llama-3.3-70B.
- End-to-end case latency: p50, p95.

## 8. Demo personas

| # | Name (synthetic) | Profile | Expected outcome |
|---|---|---|---|
| 1 | **"Priya Sharma"** | Bengaluru-based software engineer at a Tier-1 IT firm. Aadhaar + PAN + utility bill all consistent. Declared income ₹18 LPA — plausible for role. No screening hits. | **APPROVE** |
| 2 | **"Rajesh Kumar Singh"** | Mid-level state government official from a fictional Indian state. Name + DOB match a domestic PEP entry in the bundled OpenSanctions PEP dataset. Aadhaar and PAN clean; income declared as ₹12 LPA but disclosed property holdings inconsistent. | **HUMAN REVIEW** |
| 3 | **"Viktor Nazarov"** | Cross-border customer opening an NRI account from Cyprus. Name + DOB match a UN-sanctioned individual in OpenSanctions; passport extracted, MRZ valid; adverse-media articles describe alleged sanctions evasion. | **ESCALATE** |

**Why this mix:** persona 1 + 2 are domestic Indian (demonstrates Aadhaar/PAN
parsing + domestic PEP screening — the regulator-facing demo). Persona 3 is
cross-border (demonstrates passport parsing + UN/OFAC screening — the
FATCA-relevant flow that any global Indian bank also needs).

Documents for each are fake images generated with templates (NOT real ID docs
of real people). Aadhaar, PAN, and passport templates are public/replicated
layouts; faces are synthetic (`thispersondoesnotexist` images pre-downloaded).
All clearly watermarked **SYNTHETIC** in the corner so judges immediately
understand these are test artifacts.
