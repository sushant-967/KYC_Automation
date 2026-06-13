"""
schemas.py — the contractual surface of the whole pipeline.

Every agent reads from and writes to the case's structured state. Each agent's
OUTPUT model is a contract downstream agents depend on. Validate at every boundary
(HTTP in, agent out) so a malformed vLLM response fails loud, early, and in one
place rather than corrupting case state.

See docs/architecture.md §4 (agent contracts) and §5 (case state).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Primitives ──────────────────────────────────────────────────────────────

class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DocumentKind(str, Enum):
    aadhaar = "aadhaar"
    pan = "pan"
    voter_id = "voter_id"
    passport = "passport"
    driving_license = "driving_license"
    address_proof = "address_proof"
    dual_name_affidavit = "dual_name_affidavit"  # notarized doc bridging two name variants


AgentName = Literal[
    "intake", "extraction", "entityResolution", "screening", "idVerification",
    "financialProfile", "risk", "explanation", "decision", "eval",
]

NodeKind = Literal["raw_value", "signal", "decision"]


# ── Submission / Intake (§4.1) ──────────────────────────────────────────────

class DocumentRef(BaseModel):
    kind: DocumentKind
    file_id: str  # reference into a blob store / inline base64 for the demo


class CustomerInput(BaseModel):
    full_name: str
    dob: str  # ISO-8601 date as submitted on the form
    address: Optional[str] = None
    nationality: Optional[str] = None
    declared_income: Optional[float] = Field(default=None, ge=0)  # annual INR
    declared_employment: Optional[str] = None


class Submission(BaseModel):
    customer: CustomerInput
    documents: list[DocumentRef] = Field(min_length=1)


class IntakeOutput(BaseModel):
    case_id: str
    customer: CustomerInput
    documents: list[DocumentRef]
    normalized_at: str


# ── Extraction (§4.2) — Qwen2.5-VL-72B ──────────────────────────────────────

class Validations(BaseModel):
    pan_regex_ok: Optional[bool] = None
    mrz_checksum_ok: Optional[bool] = None
    aadhaar_verhoeff_ok: Optional[bool] = None


class ExtractedDocument(BaseModel):
    kind: DocumentKind
    fields: dict[str, Any]  # format-checked downstream (PAN regex, MRZ, Verhoeff)
    confidence: float = Field(ge=0, le=1)
    raw_text: Optional[str] = None
    validations: Optional[Validations] = None
    masked_fields: list[str] = Field(default_factory=list)  # e.g. ["aadhaarNumber"]


class ExtractionOutput(BaseModel):
    documents: list[ExtractedDocument]


# ── Entity Resolution (§4.3) ────────────────────────────────────────────────

class NameMatch(BaseModel):
    """Per-document fuzzy match of the submitted full_name against the name
    extracted from that document. `score` is a 0-1 token-set ratio after
    canonicalization; `ok` is the boolean verdict against the agent threshold."""
    doc_kind: DocumentKind
    extracted_name: str
    score: float = Field(ge=0, le=1)
    ok: bool


class EntityResolutionOutput(BaseModel):
    canonical_name: str
    dob_confirmed: bool
    name_matches: list[NameMatch] = Field(default_factory=list)
    name_consistent: bool = True  # all available per-doc name matches passed
    address_confirmed: Optional[bool] = None  # None = no address proof submitted
    address_match_score: Optional[float] = Field(default=None, ge=0, le=1)
    alias_matches: list[str] = Field(default_factory=list)
    prior_cases: list[str] = Field(default_factory=list)
    # Remediation tracking
    name_affidavit_submitted: bool = False          # dual_name_affidavit doc was present
    name_affidavit_covers_discrepancy: Optional[bool] = None  # affidavit bridges the mismatch
    affidavit_attempts: int = 0                     # how many affidavits have been submitted
    affidavit_retries_exhausted: bool = False        # True when max retries reached and affidavit still fails
    address_additional_proof_submitted: bool = False
    address_additional_proof_confirmed: Optional[bool] = None
    documents_required: list[str] = Field(default_factory=list)  # what the customer still owes


# ── Screening (§4.4) — Sanctions + PEP + Adverse Media ──────────────────────

ScreeningVerdict = Literal["match", "no-match", "uncertain"]


class CandidateMatch(BaseModel):
    entity_id: str
    name: str
    datasets: list[str] = Field(default_factory=list)
    verdict: ScreeningVerdict
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class SubScreening(BaseModel):
    hit: bool
    matches: list[CandidateMatch] = Field(default_factory=list)
    rationale: Optional[str] = None


class AdverseMedia(BaseModel):
    hit: bool
    summary: Optional[str] = None
    severity: Optional[Severity] = None


class ScreeningOutput(BaseModel):
    sanctions: SubScreening
    pep: SubScreening
    adverse_media: AdverseMedia


# ── ID Verification (§4.5) & Financial Profile (§4.6) ───────────────────────

class IDVerificationOutput(BaseModel):
    doc_authenticity: Literal["pass", "fail", "unknown"]
    mrz_valid: Optional[bool] = None
    expiry_ok: Optional[bool] = None
    face_match_score: Optional[float] = Field(default=None, ge=0, le=1)
    pan_format_valid: Optional[bool] = None       # PAN regex ^[A-Z]{5}[0-9]{4}[A-Z]$
    aadhaar_format_valid: Optional[bool] = None   # Verhoeff checksum over 12 digits


class FinancialProfileOutput(BaseModel):
    income_plausibility_score: float = Field(ge=0, le=1)
    geography_risk: float = Field(ge=0, le=1)
    employment_risk: float = Field(ge=0, le=1)


# ── Risk Aggregation (§4.7) — deterministic ─────────────────────────────────

class RiskContributor(BaseModel):
    signal: str
    weight: float
    value: Any
    contribution: float  # points added to the 0-100 score


class RiskOutput(BaseModel):
    score: float = Field(ge=0, le=100)
    contributors: list[RiskContributor]


# ── Explainability (§4.8) — Llama 3.3 70B ───────────────────────────────────

class EvidenceCard(BaseModel):
    title: str
    finding: str
    source: str
    severity: Severity


class EvidenceNode(BaseModel):
    """One node in the causal audit DAG — links a risk point to its origin."""
    node_id: str
    kind: NodeKind
    label: str          # short label shown in the graph (newlines allowed)
    agent: AgentName    # which agent produced this value
    raw_value: Any      # the exact value read from that agent's output
    rule: str           # the rule or threshold that was checked / violated
    contribution: float = 0.0  # risk points (signal nodes only; 0 for raw/decision)


class CausalEdge(BaseModel):
    source: str   # node_id
    target: str   # node_id


class ExplanationOutput(BaseModel):
    summary: str
    evidence_cards: list[EvidenceCard]
    recommended_action: str
    dag_nodes: list[EvidenceNode] = Field(default_factory=list)
    dag_edges: list[CausalEdge] = Field(default_factory=list)


# ── Eval (§4.10) — LLM-as-judge faithfulness + coverage ────────────────────

EvalVerdict = Literal["pass", "warn", "fail"]


class EvalOutput(BaseModel):
    """Structured quality score produced by the LLM-as-judge eval agent."""
    faithfulness: float = Field(ge=0, le=1)     # 1.0 = zero hallucinated signals
    coverage: float = Field(ge=0, le=1)          # fraction of high-weight signals in narrative
    missing_signals: list[str] = Field(default_factory=list)       # high-weight but absent
    hallucinated_signals: list[str] = Field(default_factory=list)  # in narrative, not in contributors
    verdict: EvalVerdict
    rationale: str                               # one-sentence LLM justification


# ── Decision (§4.9) ─────────────────────────────────────────────────────────

Decision = Literal["approve", "review", "escalate", "reject"]


class DecisionOutput(BaseModel):
    decision: Decision
    requires_human: bool


class HumanDecision(BaseModel):
    decision: Decision
    reviewer: Optional[str] = None
    note: Optional[str] = None


# ── Metrics (§6) ────────────────────────────────────────────────────────────

class AgentMetric(BaseModel):
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: Optional[str] = None


class GpuCallMetric(BaseModel):
    ts: str
    model: str
    latency_ms: float
    vram_used_gb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    batch_size: Optional[int] = None


# ── Case state (§5) — the single document persisted per case ────────────────

CaseStatus = Literal[
    "intake", "running", "awaiting_human", "awaiting_documents",
    "awaiting_id_review", "approved", "rejected", "escalated",
]


class AuditEvent(BaseModel):
    ts: str
    agent: str
    event: str
    payload: Optional[Any] = None


class AgentOutputs(BaseModel):
    intake: Optional[IntakeOutput] = None
    extraction: Optional[ExtractionOutput] = None
    entity_resolution: Optional[EntityResolutionOutput] = None
    screening: Optional[ScreeningOutput] = None
    id_verification: Optional[IDVerificationOutput] = None
    financial_profile: Optional[FinancialProfileOutput] = None
    risk: Optional[RiskOutput] = None
    explanation: Optional[ExplanationOutput] = None
    decision: Optional[DecisionOutput] = None
    eval: Optional[EvalOutput] = None


class CaseMetrics(BaseModel):
    per_agent: dict[str, AgentMetric] = Field(default_factory=dict)
    per_gpu_call: list[GpuCallMetric] = Field(default_factory=list)
    end_to_end_ms: Optional[float] = None


class CaseState(BaseModel):
    case_id: str
    status: CaseStatus
    customer: CustomerInput
    documents: list[DocumentRef]
    agent_outputs: AgentOutputs = Field(default_factory=AgentOutputs)
    audit_log: list[AuditEvent] = Field(default_factory=list)
    metrics: CaseMetrics = Field(default_factory=CaseMetrics)
