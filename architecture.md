# Agentic KYC Intelligence Platform — Architecture & Build Plan

> **Status:** Pre-build planning artifact (June 5, 2026). No code yet — build phase opens June 8.
> **Hackathon:** TCS & AMD AI Hackathon, Track 1 (Agents). Submission June 12.
> **Team:** 2 people. Split = MI300X/model owner ("A") + Cloudflare/UI owner ("B").

## 1. Problem statement

Indian banks, NBFCs, and fintechs onboard customers under the RBI's *Master Direction — Know Your Customer (KYC) Direction, 2016* (and its amendments), the Prevention of Money Laundering Act 2002 (PMLA), and FIU-IND reporting obligations. Customer Due Diligence (CDD) today is mostly manual, slow (days), inconsistent across reviewers, and produces approve/reject decisions without an audit trail of *why* — a problem when the regulator asks.

**Our solution:** an agentic platform where six specialized agents collaborate in a deterministic pipeline to perform end-to-end CDD. Each agent emits structured evidence; the explainability agent assembles a human-readable rationale; the decision agent renders an outcome that a compliance officer can trust because every signal — Aadhaar verification, PAN format validation, sanctions match, PEP flag, adverse-media hit, financial-profile risk — is exposed end-to-end.

Although the framing is India-primary (RBI/PMLA, Aadhaar/PAN/passport), the architecture is jurisdiction-agnostic. Same pipeline works for FATCA-driven cross-border customers, EU AML6, or US BSA-CIP with only the document parsers and risk weights changed.

## 2. Pipeline (high level)

```
START
  │
  ▼
[ Intake Agent ]                       ← validate submission shape, normalize fields
  │
  ▼
[ Extraction Agent ]   ★ DEEP          ← Qwen2.5-VL-72B on MI300X — OCR + structured fields from docs
  │
  ▼
[ Entity Resolution ]                  ← dedupe against prior cases, canonical name + DOB
  │
  ├──────────────────┬─────────────────┬────────────────────┬─────────────────────┐
  ▼                  ▼                 ▼                    ▼                     ▼
[ ID Verify ]   [ Sanctions ]    [ PEP Screen ]   [ Adverse Media ]    [ Financial Profile ]
                       ╰────────── ★ DEEP ──────────╯                          (light)
                  (bundled OpenSanctions data in D1 + embeddings + Llama 3.3 70B
                   for fuzzy match disambiguation and adverse-media summary)
  │
  ▼
[ Risk Aggregation ]                   ← deterministic weighted scoring across signals
  │
  ▼
[ Explainability Agent ]  ★ DEEP       ← Llama 3.3 70B — turn evidence into human rationale
  │
  ▼
[ Decision Agent ]
  │
  ├── score < 30:    APPROVE
  ├── 30 ≤ score < 70:  HUMAN REVIEW   ← UI surfaces with HITL buttons
  └── score ≥ 70:    ESCALATE          ← high-severity, paged to compliance
END
```

★ = "deep" agent we invest in. The other steps are real but lighter (smaller prompts, simpler logic).

## 3. Stack & topology

```
┌────────────────────────────────────────────────────────────────────┐
│  Cloudflare Workers (owned by Teammate B)                          │
│                                                                    │
│   ┌──────────────────┐                                             │
│   │ /api/cases       │ POST → start case                           │
│   │ /api/cases/:id   │ GET  → state                                │
│   │ /api/cases/:id/  │ GET  → SSE stream of pipeline events        │
│   │     stream       │                                             │
│   │ /api/cases/:id/  │ POST → human decision (approve/review/...)  │
│   │     decide       │                                             │
│   └────────┬─────────┘                                             │
│            │                                                       │
│            ▼                                                       │
│   ┌──────────────────────────────────────┐                         │
│   │ KYCCase Durable Object (per case)    │                         │
│   │   - state machine                    │                         │
│   │   - SQLite-backed audit log          │                         │
│   │   - SSE broadcaster                  │                         │
│   │   - kicks off Workflow               │                         │
│   └──────────────────┬───────────────────┘                         │
│                      │                                             │
│            ┌─────────┴──────────────┐                              │
│            ▼                        ▼                              │
│   ┌────────────────┐      ┌──────────────────┐                     │
│   │ KYC Workflow   │      │ HITL queue       │                     │
│   │ (orchestrator) │      │ (review/escalate)│                     │
│   └────┬───────────┘      └──────────────────┘                     │
│        │                                                           │
│        │ steps call out via HTTP                                   │
│        ▼                                                           │
└────────┼───────────────────────────────────────────────────────────┘
         │
         │ HTTPS (Cloudflare Tunnel or public ingress)
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│  AMD MI300X box (owned by Teammate A)                              │
│  ROCm + vLLM, 192 GB HBM3                                          │
│                                                                    │
│   ┌─────────────────────────┐   ┌─────────────────────────────┐    │
│   │ vLLM :8000              │   │ vLLM :8001                  │    │
│   │ Qwen2.5-VL-72B (FP8)    │   │ Llama-3.3-70B-Instruct (FP8)│    │
│   │ ~75 GB VRAM             │   │ ~75 GB VRAM                 │    │
│   │ doc OCR + extraction    │   │ text reasoning              │    │
│   │ adverse-media vision    │   │ risk explanations           │    │
│   └─────────────────────────┘   └─────────────────────────────┘    │
│                                                                    │
│   ┌─────────────────────────┐                                      │
│   │ vLLM :8002              │                                      │
│   │ BGE-large-en-v1.5       │                                      │
│   │ ~2 GB VRAM              │                                      │
│   │ entity-name embeddings  │                                      │
│   │ for sanctions vector    │                                      │
│   │ search (see §4.4)       │                                      │
│   └─────────────────────────┘                                      │
│                                                                    │
│   Total ~152 GB used / 192 GB available. Comfortable headroom.     │
│   See §3.1 for the FP8 quantization decision.                      │
└────────────────────────────────────────────────────────────────────┘

         NO external inference or data APIs at runtime.
         All reasoning happens on the MI300X.
         All sanctions/PEP/adverse-media data is bundled locally
         (see §5.5 — Datasets).
```

### 3.1 Three models, one GPU — routing decision

Qwen2.5-VL-72B (~145 GB FP16) + Llama 3.3 70B (~140 GB FP16) + BGE-large embeddings (~2 GB) exceeds 192 GB if loaded in FP16. Quantize the two large models to FP8:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. FP8 quantization of both LLMs + FP16 embeddings** | All three fit concurrently (~75 + ~75 + ~2 GB). Lower latency from reduced memory bandwidth. ~40 GB headroom. | Quality drop on quantization, possible prompt-engineering retries. | **Recommended.** vLLM supports FP8 on MI300X via the ROCm fork. |
| B. Hot-swap between the two LLMs | Full FP16 quality. | 30-60s swap per case is demo-killer. | Reject. |
| C. Qwen-VL for both vision and text, no Llama | Frees ~75 GB. | Qwen-VL is weaker on long-form text reasoning + structured output. | Fallback if FP8 has issues. |

**Action for Teammate A (June 5-6):** stand up all three models on the MI300X (Qwen-VL FP8, Llama FP8, BGE FP16); benchmark tokens/sec and embedding throughput. If FP8 quality drops too far on a sample reasoning prompt, fall back to option C.

## 4. Agents — contracts & responsibilities

Every agent reads from and writes to the case's structured state. State lives in the KYCCase Durable Object as a single JSON document, plus an append-only audit log of events. Each agent's output schema is contractual — downstream agents depend on it.

### 4.1 Intake Agent (light)

- **Input:** raw submission (form fields + document file references).
- **Output:** `{ caseId, customer: { fullName, dob, address, nationality, declaredIncome, ... }, documents: [{ kind, fileId }] }`.
- **Model:** none. Pure validation + normalization in TypeScript.
- **Failure modes:** missing required field → reject before pipeline starts.

### 4.2 Extraction Agent (DEEP) ★

- **Input:** document file references from intake.
- **Output:** per-document `{ kind, fields: {...}, confidence: 0-1, rawText, boundingBoxes? }`.
- **Model:** Qwen2.5-VL-72B via vLLM on MI300X.
- **Supported document kinds (India-primary set):**
  - **Aadhaar card** — 12-digit Aadhaar number (we **mask the first 8 digits** per UIDAI rules on storage/display), name, DOB/YOB, gender, address.
  - **PAN card** — 10-char PAN (validated against regex `^[A-Z]{5}[0-9]{4}[A-Z]$`), name, father's name, DOB.
  - **Voter ID (EPIC)** — EPIC number, name, age, address, assembly constituency.
  - **Passport** — name, DOB, nationality, MRZ lines (parsed + checksum validated), expiry, place of issue.
  - **Driving license** — DL number, name, DOB, address, validity, RTO.
  - **Address proof** — electricity bill / rental agreement / bank statement (extract: name, address, date, provider).
- **Prompt strategy:** structured-output prompt with a per-doc-kind JSON schema. Vision model returns fields + per-field confidence; Worker validates against the schema and runs format checks (PAN regex, MRZ checksum, Aadhaar Verhoeff checksum).
- **Why deep:** highest-value GPU work. A 72B vision model extracting Aadhaar + PAN fields with confidence scores and producing a structurally validated record is the demo headline.

**Aadhaar masking:** raw 12-digit Aadhaar never leaves the extraction agent. The downstream pipeline, audit log, and UI all see only `XXXX-XXXX-1234`. This is a small compliance touch with disproportionate signal to RBI-savvy judges.

### 4.3 Entity Resolution (light)

- **Input:** customer record from intake + extraction.
- **Output:** `{ canonicalName, dobConfirmed, aliasMatches: [], priorCases: [] }`.
- **Model:** none. Deterministic name canonicalization + DOB cross-check between submitted form and extracted ID. Looks up prior cases in DO storage.

### 4.4 Screening Agent — Sanctions + PEP + Adverse Media (DEEP) ★

Internally three sub-agents but one logical agent in the pipeline. Run in parallel. **No external APIs at runtime** — all data is bundled (see §5.5). The "AI-powered fuzzy matching against a local sanctions database" framing is also a stronger AMD/agentic story than calling someone else's API.

**Matching pipeline (per sub-agent):**

1. **Recall stage — vector search.** Customer's canonical name is embedded via BGE-large on the MI300X (port :8002). Query D1's stored embeddings via Cloudflare Vectorize (or in-Worker brute-force if Vectorize complicates things) → top-K candidate entities, K≈20.
2. **Precision stage — name normalization + DOB filter.** Deterministic checks (Levenshtein on canonicalized names, DOB tolerance ±2 years for handwritten docs) drop obvious non-matches.
3. **Adjudication stage — LLM disambiguation.** Llama 3.3 70B receives the remaining 1-5 candidates with full structured context (aliases, DOB, country, dataset source) and outputs `{ verdict: 'match'|'no-match'|'uncertain', confidence: 0-1, rationale: string, evidence: [list of facts cited] }`.

This three-stage funnel (broad recall → cheap precision → expensive LLM) is the standard pattern and keeps per-case cost low.

- **Sanctions sub-agent:** filters D1 to entities with `topic ∈ {sanction, sanction.linked}`.
- **PEP sub-agent:** filters D1 to entities with `topic ∈ {role.pep, gov.*}` and a country match heuristic.
- **Adverse media sub-agent:** filters D1 to entities flagged `topic.crime.*` or with adverse-media source URLs. For hits, Llama 3.3 70B summarizes the available `notes` / `summary` fields from OpenSanctions into a 2-sentence risk narrative with `severity ∈ {low, medium, high}`.

- **Output:** `{ sanctions: { hit, matches: [...], rationale }, pep: {...}, adverseMedia: { hit, summary, severity } }`.

### 4.5 ID Verification (light)

- **Input:** extracted ID doc + selfie (skipped for demo unless time permits).
- **Output:** `{ docAuthenticity: 'pass'|'fail'|'unknown', mrzValid, expiryOk, faceMatchScore? }`.
- **Implementation:** MRZ checksum validation + expiry check + optional face match. No new model needed; the extraction agent already returned MRZ.

### 4.6 Financial Profile (light)

- **Input:** declared income, declared employment, country, address risk.
- **Output:** `{ incomePlausibilityScore, geographyRisk, employmentRisk }`.
- **Implementation:** rules + lookup tables (country risk index, etc.). Documented heuristics, not ML.

### 4.7 Risk Aggregation (light)

- **Input:** all outputs from §4.2–4.6.
- **Output:** `{ score: 0-100, contributors: [{ signal, weight, value, contribution }] }`.
- **Implementation:** deterministic weighted scoring. Weights are explicit constants (sanctions hit = +50, PEP hit = +30, adverse media hit = +20 × severity, ID fail = +30, geo risk = up to +10, income implausibility = up to +10). **Deterministic** is critical — the score must be reproducible from the inputs, which is what makes the explainability honest.

### 4.8 Explainability Agent (DEEP) ★

- **Input:** full case state + risk score + contributors.
- **Output:** `{ summary: string, evidenceCards: [{ title, finding, source, severity }], recommendedAction }`.
- **Model:** Llama 3.3 70B with a structured-output prompt that takes the deterministic scoring breakdown and turns it into prose a compliance officer would actually read.
- **Why deep:** explainability is the differentiator. Anyone can compute a score; what wins this hackathon is making the *why* legible. Slide 3 ("Solution Overview") will lean on this.

### 4.9 Decision Agent (light)

- **Input:** risk score + explainability output.
- **Output:** `{ decision: 'approve'|'review'|'escalate', requiresHuman: bool }`.
- **Implementation:** threshold rules (see pipeline diagram §2). On `review` or `escalate`, the DO holds the case in a paused state until a human posts a verdict via the UI.

## 5. Data flow & state

KYCCase Durable Object holds one JSON document per case:

```ts
type CaseState = {
  caseId: string;
  status: 'intake' | 'running' | 'awaiting_human' | 'approved' | 'rejected' | 'escalated';
  customer: { ... };
  documents: [...];
  agentOutputs: {
    intake?: IntakeOutput;
    extraction?: ExtractionOutput;
    entityResolution?: EntityResOutput;
    screening?: ScreeningOutput;
    idVerification?: IDVOutput;
    financialProfile?: FPOutput;
    risk?: RiskOutput;
    explanation?: ExplanationOutput;
    decision?: DecisionOutput;
  };
  auditLog: Array<{ ts, agent, event, payload }>;
  metrics: {
    perAgent: Record<AgentName, { latencyMs, inputTokens, outputTokens, model }>;
    perGpuCall: Array<{ ts, model, latencyMs, vramUsedGb, gpuUtilPct, batchSize }>;
    endToEndMs?: number;
  };
};
```

DO SQLite stores the full event stream so the audit log is persistent even if the DO restarts.

### 5.5 Datasets (bundled, not API-called)

**Hard rule (per [[feedback_no_external_apis]]):** zero external inference or data APIs at runtime. Sanctions / PEP / adverse-media data must be downloaded once during prep and bundled into the project.

**Primary dataset: OpenSanctions bulk export.**
- Source: `https://data.opensanctions.org/datasets/` (snapshot taken June 6, version pinned in the submission).
- Format: FollowTheMoney JSON (entity-per-line, well-structured, alias arrays, dataset provenance per entity).
- Scope: `default` collection (sanctions + PEP + crime + adverse-media-flagged entities). ~500 K entities, ~1.5 GB uncompressed JSON. We can prune to the subset matching our 3 personas + a few thousand decoys for realistic vector-search behavior, getting it down to <100 MB.
- License: CC-BY-NC 4.0 — non-commercial use only. The hackathon is non-commercial (educational/competition). We attribute OpenSanctions in the submission, slides, and README.

**Storage:**
- Cloudflare D1 (SQLite) for the structured entity records: `entities(id, type, name, aliases JSON, dob, countries JSON, topics JSON, datasets JSON, summary, source_url)`.
- Cloudflare Vectorize for embeddings: 1024-dim vectors from BGE-large, one per entity name + aliases.

**Ingestion script:** `rocm/ingest.py` (run once, locally during prep):
1. Download bulk JSON from data.opensanctions.org.
2. For each entity, write a D1 row.
3. For each entity, call MI300X BGE-large endpoint to embed `name + aliases`, write to Vectorize.

**Adverse-media supplemental corpus:** For demo polish, hand-craft 5-10 short fake "news articles" tied to persona "Marcus Chen" and "Viktor Nazarov" so the adverse-media sub-agent has rich prose to summarize. Stored as a JSON file in `personas/adverse-media/`. Clearly labeled synthetic.

**Declaration in submission:**
- Slide 4 lists: OpenSanctions snapshot YYYY-MM-DD (CC-BY-NC, attributed), synthetic adverse-media corpus (authored June 6, in repo).
- README has full attribution + license notice.

## 6. Metrics & instrumentation (slide 4 ammunition)

The submission rubric **demands** GPU usage, memory, latency, and token counts on slide 4. We will produce these from the start, not bolted on.

**What we log per agent run:**
- `agentName`, `caseId`, `startedAt`, `endedAt`, `latencyMs`
- `model` (if applicable), `inputTokens`, `outputTokens`
- For GPU calls: `vramUsedGb`, `gpuUtilPct`, `batchSize`, served by vLLM's `/metrics` endpoint scraped per call

**Where it lives:** in the DO state under `metrics`, plus mirrored to a Cloudflare Analytics Engine dataset for cross-case aggregation.

**Slide 4 will show:**
- Bar chart: tokens consumed per agent per case (averaged across 3 personas)
- Bar chart: latency per agent (p50, p95)
- One-liner: peak VRAM usage on MI300X (proving the 72B model is doing the work)
- Sustained tokens/sec for both Qwen-VL-72B and Llama-3.3-70B
- End-to-end case latency: p50, p95

**Action:** Teammate A captures baseline numbers for both models on June 5-6 from a smoke test — these go into the slide deck skeleton on June 6 even before the full pipeline is built.

## 7. UI (Teammate B, June 9-11)

React + Vite + Tailwind. Deployed as a static Worker site.

Three screens:
1. **Submit case** — form + file upload. Submit → POST `/api/cases`.
2. **Pipeline view** — live SSE stream from the DO. Each agent renders as a card with status (pending/running/done/flagged), a latency badge, and a click-to-expand evidence panel showing the agent's structured output. Risk score widget fills in when risk aggregation completes.
3. **Decision view** — explainability summary, evidence cards laid out as a timeline, HITL buttons (Approve / Send back / Escalate). Posting a decision unblocks the DO.

**Demo motion:** open the submit screen, paste one of the three personas with one click each (preset buttons), watch the pipeline animate, point at the explainability output, click the appropriate HITL button. Reset, run the next persona.

## 8. Demo personas

Designed June 6 in detail. Sketch now:

| # | Name (synthetic) | Profile | Expected outcome |
|---|---|---|---|
| 1 | **"Priya Sharma"** | Bengaluru-based software engineer at a Tier-1 IT firm. Aadhaar + PAN + utility bill all consistent. Declared income ₹18 LPA — plausible for role. No screening hits. | **APPROVE** |
| 2 | **"Rajesh Kumar Singh"** | Mid-level state government official from a fictional Indian state. Name + DOB match a domestic PEP entry in the bundled OpenSanctions PEP dataset. Aadhaar and PAN clean; income declared as ₹12 LPA but disclosed property holdings inconsistent. | **HUMAN REVIEW** |
| 3 | **"Viktor Nazarov"** | Cross-border customer opening an NRI account from Cyprus. Name + DOB match a UN-sanctioned individual in OpenSanctions; passport extracted, MRZ valid; adverse-media articles describe alleged sanctions evasion. | **ESCALATE** |

**Why this mix:** persona 1 + 2 are domestic Indian (demonstrates Aadhaar/PAN parsing + domestic PEP screening — the regulator-facing demo). Persona 3 is cross-border (demonstrates passport parsing + UN/OFAC screening — the FATCA-relevant flow that any global Indian bank also needs).

Documents for each will be fake images we generate with templates (NOT real ID docs of real people). Aadhaar, PAN, and passport templates are public/replicated layouts; faces are synthetic (thispersondoesnotexist images pre-downloaded). All clearly watermarked SYNTHETIC in the corner so judges immediately understand these are test artifacts.

## 9. Team-of-2 split

| | Teammate A (MI300X owner) | Teammate B (CF / UI owner) |
|---|---|---|
| **Jun 5** | Bench Qwen-VL-72B + Llama-3.3-70B on MI300X. Capture baseline GPU/latency. | Read this doc end-to-end. Sketch wireframes. Download OpenSanctions bulk snapshot. |
| **Jun 6** | Quantize both LLMs to FP8 + stand up BGE-large on port :8002. Re-bench. Persona docs (synthetic ID images). | Prune OpenSanctions JSON to demo-relevant subset. Design D1 schema. Persona profiles (text fields). Slide skeleton. |
| **Jun 7** | Prompt engineering for extraction + adverse-media + adjudication prompts (against synthetic data). | Finalize wireframes. Metrics taxonomy doc. Hand-craft adverse-media corpus for 2 of 3 personas. Repo structure draft. |
| **Jun 8** | (Kick-off AM.) Wire up `/extract`, `/reason`, `/embed` HTTP endpoints to vLLM. | (Kick-off AM.) `git init`. Scaffold Worker + DO + Workflow. Run ingestion script → populate D1 + Vectorize. Hello-world pipeline (intake → echo extraction → fake decision). |
| **Jun 9** | Extraction prompt locked. Sanctions adjudication prompt locked. | Wire screening (D1 + Vectorize + Llama). Wire decision rules. Static UI shell. |
| **Jun 10** | Explainability prompt + structured output validation. Latency tuning. | UI live SSE rendering. HITL flow end-to-end. |
| **Jun 11** | Persona stress-tests; capture final metrics for slides. | Bug fixes. Slide 3 architecture diagram. Demo rehearsals. |
| **Jun 12** | Final benchmark run. | Demo recording. Final slide polish. Submit. |

Daily 15-min sync at 9am and 6pm.

## 10. Repo layout (to create June 8)

```
kyc-platform/
├── apps/
│   ├── worker/                # Cloudflare Worker — orchestration
│   │   ├── src/
│   │   │   ├── index.ts       # routes
│   │   │   ├── case-do.ts     # KYCCase Durable Object
│   │   │   ├── workflow.ts    # KYC pipeline orchestrator
│   │   │   ├── agents/
│   │   │   │   ├── intake.ts
│   │   │   │   ├── extraction.ts      # → ROCm
│   │   │   │   ├── entity-resolution.ts
│   │   │   │   ├── screening.ts       # → OpenSanctions + ROCm
│   │   │   │   ├── id-verify.ts
│   │   │   │   ├── financial-profile.ts
│   │   │   │   ├── risk.ts
│   │   │   │   ├── explainability.ts  # → ROCm
│   │   │   │   └── decision.ts
│   │   │   ├── rocm-client.ts # HTTP client to vLLM
│   │   │   ├── metrics.ts
│   │   │   └── schemas.ts     # zod schemas for all agent contracts
│   │   ├── wrangler.jsonc
│   │   └── package.json
│   └── ui/                    # React + Vite + Tailwind
│       ├── src/
│       │   ├── App.tsx
│       │   ├── routes/
│       │   │   ├── submit.tsx
│       │   │   ├── pipeline.tsx
│       │   │   └── decision.tsx
│       │   └── lib/
│       │       ├── api.ts
│       │       └── sse.ts
│       └── vite.config.ts
├── rocm/                      # Teammate A's territory
│   ├── vllm-launch-qwen.sh        # port 8000 — Qwen2.5-VL-72B FP8
│   ├── vllm-launch-llama.sh       # port 8001 — Llama-3.3-70B FP8
│   ├── vllm-launch-bge.sh         # port 8002 — BGE-large-en-v1.5
│   ├── prompts/
│   │   ├── extraction.md
│   │   ├── adverse-media.md
│   │   ├── sanctions-adjudication.md
│   │   └── explainability.md
│   ├── bench/
│   │   ├── bench-qwen-vl.py
│   │   ├── bench-llama.py
│   │   └── bench-bge.py
│   └── ingest.py                  # one-shot: OpenSanctions JSON → D1 + Vectorize
├── data/                          # bundled, declared in submission
│   ├── opensanctions/
│   │   ├── snapshot-YYYY-MM-DD.jsonl   # pruned bulk export (~100MB)
│   │   └── ATTRIBUTION.md              # CC-BY-NC notice
│   └── adverse-media/
│       └── synthetic-articles.json     # hand-crafted, persona-tied
├── personas/                  # synthetic test data
│   ├── priya-clean/
│   ├── marcus-pep/
│   └── viktor-sanctions/
├── docs/
│   ├── architecture.md        # this file (moved here on June 8)
│   ├── metrics.md
│   ├── demo-script.md
│   └── slides/
└── README.md
```

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| All three models won't fit on MI300X even at FP8 | Low | We're at ~152/192 GB. Plenty of headroom. If something goes wrong, drop Llama and use Qwen-VL for text too (option C, §3.1). |
| OpenSanctions bulk data too large / wrong shape | Low | Prune to ~100K relevant entities at ingestion. Have a smaller OFAC SDN backup ready. |
| vLLM endpoint unreachable from Cloudflare | Medium | Cloudflare Tunnel from MI300X box → CF Worker. Set up on June 7. Test before kick-off. |
| Slide 4 metrics missing | High if bolted on | §6 plan — instrument from line 1 of the Worker code. |
| Document extraction quality poor on synthetic IDs | Medium | Tune the extraction prompt on June 7 against fake-doc templates. Have a deterministic regex fallback for MRZ if Qwen-VL underperforms. |
| Vectorize binding feels overkill / slow | Low | Fallback: brute-force cosine similarity over an in-memory array of ~100K embeddings inside the Worker. ~50ms with Float32Array. Trade-off acceptable. |
| 4-day build slips | Low | Light agents (§4.1, 4.3, 4.5–4.7, 4.9) are pure code — each <2 hours. Deep agents are the only real risk surface. |
| OpenSanctions CC-BY-NC conflicts with submission terms | Low | Hackathon is non-commercial. Attribute in slides + README. If TCS flags it post-hoc, swap to OFAC SDN (public domain) — schema is similar. |

## 12. Open questions (to answer before June 8)

- [ ] Exact vLLM ROCm fork version on the MI300X box? Confirms FP8 quantization support for both Qwen-VL-72B and Llama-3.3-70B.
- [ ] Network path from CF Worker to MI300X — Cloudflare Tunnel or public IP with auth? **Need to set up on June 7 and smoke-test before kick-off.**
- [ ] D1 vs in-memory: do we use Cloudflare Vectorize for embeddings, or load ~100K embeddings into a Worker Durable Object on first hit? (Decide June 7 after sizing the pruned dataset.)
- [ ] How do we generate fake ID images? Template-based PIL/Pillow script with synthetic faces from thispersondoesnotexist (no API — pre-downloaded). Confirm acceptable approach with teammate.
- [ ] Demo video: screen recording with voiceover (recommended — repeatable), or live narration?
- [ ] Final dataset version pin: which OpenSanctions snapshot date will we ship with? (Pick on June 6 ingest day.)

---

*This document supersedes any prior conversation. If anything here is wrong or unclear, raise it before June 8.*
