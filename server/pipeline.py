"""
pipeline.py — KYC pipeline orchestrator on LangGraph (§2, §3).

A `StateGraph` wires the 9 agents into the deterministic CDD flow. Deep agents
(extraction, screening, explainability) call vLLM on localhost; light agents are
pure Python. The graph is compiled once at import time.

    intake → extraction ★ → entity-resolution
           → { screening ★ ‖ id-verify ‖ financial-profile }
           → risk → explainability ★ → decision

Resilience: every node that calls vLLM is wrapped in a try/except. If a node
fails (circuit open, all retries exhausted, unexpected error) the pipeline
continues with a degraded stub output and emits a "degraded" SSE event instead
of hanging or returning a 500. This ensures the case always reaches a terminal
status even when a model endpoint is unavailable.

Degradation strategy per agent:
  extraction       → empty ExtractionOutput (no OCR; downstream uses form data)
  screening        → all-clear stubs (no hits) + degraded flag in audit log
  financial_profile→ mid-range defaults (0.5/0.1/0.1) — conservative, not zero
  explainability   → deterministic template built from risk.contributors (no LLM)
  eval             → silent skip (already had try/except; observability only)
"""
from __future__ import annotations

import asyncio
import operator
import os
import time
import traceback
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from datetime import datetime, timezone

from schemas import (
    AdverseMedia, AgentMetric, CaseState, EvidenceCard, ExplanationOutput,
    ExtractionOutput, FinancialProfileOutput, GpuCallMetric, GuardrailViolation,
    IDVerificationOutput, RiskOutput, ScreeningOutput, Severity, SubScreening,
)
from guardrails import (
    guard_customer_input, guard_prompt_injection, guard_llm_output, is_adversarial,
)
from screening_index import ScreeningIndex
from tavily_client import TavilyClient
from vllm_client import VllmClient

from agents.intake import run_intake
from agents.extraction import run_extraction
from agents.entity_resolution import run_entity_resolution
from agents.screening import run_screening
from agents.id_verify import run_id_verification
from agents.financial_profile import run_financial_profile
from agents.risk import run_risk
from agents.explainability import run_explainability
from agents.decision import run_decision
from agents.eval import run_eval

# Emit callback: (agent, status, payload) -> None. Used for audit log + SSE.
Emit = Callable[[str, str, object], None]


# ── Degraded fallback outputs ─────────────────────────────────────────────────

def _degraded_extraction() -> ExtractionOutput:
    """No OCR available — downstream entity-resolution uses customer form data only."""
    return ExtractionOutput(documents=[])


def _degraded_screening() -> ScreeningOutput:
    """Cannot screen — return all-clear stubs. Audit log records the failure."""
    return ScreeningOutput(
        sanctions=SubScreening(hit=False,
                               rationale="Screening unavailable — model endpoint down"),
        pep=SubScreening(hit=False,
                         rationale="Screening unavailable — model endpoint down"),
        adverse_media=AdverseMedia(hit=False),
    )


def _degraded_financial() -> FinancialProfileOutput:
    """Mid-range defaults: conservative but not zero — avoids artificially clean scores."""
    return FinancialProfileOutput(
        income_plausibility_score=0.5,
        geography_risk=0.1,
        employment_risk=0.25,
        financial_risk_rationale="Financial profile unavailable — model endpoint down",
    )


def _degraded_explanation(risk: Optional[RiskOutput]) -> ExplanationOutput:
    """Deterministic template from risk.contributors — no LLM required."""
    if risk and risk.contributors:
        score = risk.score
        signals = "; ".join(
            f"{c.signal} (+{c.contribution:.0f}pts)"
            for c in sorted(risk.contributors, key=lambda x: -x.contribution)
        )
        summary = (
            f"Risk score {score:.0f}/100. "
            f"Key contributors: {signals}. "
            "Full narrative unavailable — explanation model temporarily down."
        )
        cards = [
            EvidenceCard(
                title=c.signal.replace("_", " ").title(),
                finding=f"Contributed {c.contribution:.0f} risk points (weight {c.weight:.0f})",
                source=c.signal,
                severity=Severity.high if c.contribution >= 30 else
                         Severity.medium if c.contribution >= 15 else Severity.low,
            )
            for c in risk.contributors if c.contribution > 0
        ]
        action = ("Escalate for manual review." if score >= 70 else
                  "Compliance officer review recommended." if score >= 30 else
                  "Approved — low risk score.")
    else:
        summary = "Risk assessment unavailable — explanation model temporarily down."
        cards   = []
        action  = "Manual review required."

    return ExplanationOutput(
        summary=summary,
        evidence_cards=cards,
        recommended_action=action,
    )


@dataclass
class PipelineIO:
    load_image: Callable[[str], Awaitable[str]]  # file_id -> data: URL
    entity_index: ScreeningIndex
    prior_case_lookup: Callable[[str], list[str]]
    emit: Emit
    tavily: Optional[TavilyClient] = None        # None → DB-only screening fallback


# ── LangGraph state ─────────────────────────────────────────────────────────
# `case` carries the mutable pydantic CaseState — agent outputs and per-agent
# metrics are written directly onto it (each agent writes a distinct sub-field
# so parallel branches don't collide). `gpu` is the only key reduced by
# LangGraph: parallel deep agents each return their own list which is
# concatenated by `operator.add`.

class GraphState(TypedDict, total=False):
    case: CaseState
    vllm: VllmClient
    io: PipelineIO
    gpu: Annotated[list[GpuCallMetric], operator.add]


def _emit_degraded(state: GraphState, agent: str, exc: Exception) -> None:
    """Log a degraded-mode event to the audit log + SSE stream."""
    brief = f"{type(exc).__name__}: {str(exc)[:120]}"
    state["io"].emit(agent, "degraded", {"error": brief, "fallback": "stub output used"})


def _record_guardrails(state: GraphState, agent: str,
                       results: list) -> None:
    """Persist guardrail results to case.guardrail_flags + emit SSE events."""
    from guardrails import GuardrailResult  # local import avoids circular
    ts = datetime.now(timezone.utc).isoformat()
    for r in results:
        if not r.passed:
            violation = GuardrailViolation(
                ts=ts, agent=agent, check=r.check,
                level=r.level, violations=r.violations)
            state["case"].guardrail_flags.append(violation)
            state["io"].emit(agent, "guardrail_violation", {
                "check": r.check,
                "level": r.level,
                "violations": r.violations,
            })


async def _timed(name: str, state: GraphState, fn):
    """Emit running/done, time the call, accept sync or async agents.

    per_agent[name] is written in a finally block so it is ALWAYS set —
    even when the call raises. This ensures the agent shows as "done" (not
    "pending") on the dashboard even when degraded fallback takes over.
    """
    io = state["io"]
    io.emit(name, "running", None)
    step_delay = float(os.environ.get("KYC_STEP_DELAY_MS", "0")) / 1000
    if step_delay:
        await asyncio.sleep(step_delay)
    started = time.perf_counter()
    try:
        value = fn()
        result = await value if asyncio.iscoroutine(value) else value
        io.emit(name, "done", _jsonable(result))
        return result
    except Exception:
        # Emit "done" even on failure so the UI never sees a permanent "running"
        # spinner. The caller's except block will emit "degraded" with details.
        io.emit(name, "done", None)
        raise
    finally:
        # Always record latency regardless of success/failure.
        state["case"].metrics.per_agent[name] = AgentMetric(
            latency_ms=(time.perf_counter() - started) * 1000)


# ── Nodes ───────────────────────────────────────────────────────────────────

async def _intake_node(state: GraphState):
    case = state["case"]
    # INPUT GUARDRAIL — validate customer form data before anything else runs.
    _record_guardrails(state, "intake", guard_customer_input(case.customer))
    out = await _timed("intake", state, lambda: run_intake(
        case.case_id,
        {"customer": case.customer.model_dump(),
         "documents": [d.model_dump() for d in case.documents]}))
    case.agent_outputs.intake = out
    return {}


async def _extraction_node(state: GraphState):
    case = state["case"]
    gpu_local: list[GpuCallMetric] = []

    async def _do():
        o, g = await run_extraction(case.documents, state["vllm"], state["io"].load_image)
        gpu_local.extend(g)
        return o

    try:
        case.agent_outputs.extraction = await _timed("extraction", state, _do)
    except Exception as exc:
        _emit_degraded(state, "extraction", exc)
        case.agent_outputs.extraction = _degraded_extraction()

    # INJECTION GUARDRAIL — scan every OCR'd document page for adversarial text
    # that could hijack the Llama reasoning model downstream.
    # Also scans doc.fields JSON values, catching jailbroken field values.
    if case.agent_outputs.extraction:
        _record_guardrails(state, "extraction",
                           guard_prompt_injection(case.agent_outputs.extraction))

    return {"gpu": gpu_local}


async def _entity_node(state: GraphState):
    case = state["case"]
    case.agent_outputs.entity_resolution = await _timed(
        "entityResolution", state,
        lambda: run_entity_resolution(case.customer, case.agent_outputs.extraction,
                                      state["io"].prior_case_lookup))
    return {}


async def _screening_node(state: GraphState):
    case = state["case"]
    gpu_local: list[GpuCallMetric] = []

    async def _do():
        o, g = await run_screening(case.agent_outputs.entity_resolution,
                                   case.customer, state["vllm"],
                                   state["io"].entity_index,
                                   state["io"].tavily)
        gpu_local.extend(g)
        return o

    try:
        case.agent_outputs.screening = await _timed("screening", state, _do)
    except Exception as exc:
        _emit_degraded(state, "screening", exc)
        case.agent_outputs.screening = _degraded_screening()
    return {"gpu": gpu_local}


async def _id_verify_node(state: GraphState):
    case = state["case"]
    case.agent_outputs.id_verification = await _timed(
        "idVerification", state,
        lambda: run_id_verification(case.agent_outputs.extraction))
    return {}


async def _financial_node(state: GraphState):
    case = state["case"]
    gpu_local: list[GpuCallMetric] = []

    async def _do():
        o, g = await run_financial_profile(case.customer, state["vllm"])
        gpu_local.extend(g)
        return o

    try:
        case.agent_outputs.financial_profile = await _timed("financialProfile", state, _do)
    except Exception as exc:
        _emit_degraded(state, "financialProfile", exc)
        case.agent_outputs.financial_profile = _degraded_financial()
    return {"gpu": gpu_local}


async def _risk_node(state: GraphState):
    case = state["case"]
    o = case.agent_outputs
    flags = case.guardrail_flags  # pass adversarial-document flags to risk scorer
    case.agent_outputs.risk = await _timed(
        "risk", state,
        lambda: run_risk(o.entity_resolution, o.screening,
                         o.id_verification, o.financial_profile,
                         guardrail_flags=flags))
    return {}


async def _explanation_node(state: GraphState):
    case = state["case"]
    gpu_local: list[GpuCallMetric] = []
    o = case.agent_outputs

    async def _do():
        result, g = await run_explainability(
            o.entity_resolution, o.screening, o.risk,
            o.id_verification, o.financial_profile, o.decision,
            state["vllm"])
        gpu_local.extend(g)
        return result

    try:
        case.agent_outputs.explanation = await _timed("explanation", state, _do)
    except Exception as exc:
        _emit_degraded(state, "explanation", exc)
        case.agent_outputs.explanation = _degraded_explanation(o.risk)
    return {"gpu": gpu_local}


async def _eval_node(state: GraphState):
    case = state["case"]
    gpu_local: list[GpuCallMetric] = []
    o = case.agent_outputs

    async def _do():
        result, g = await run_eval(o.explanation, o.risk, state["vllm"])
        gpu_local.extend(g)
        return result

    try:
        # Hard 45-second cap: eval is observability-only and must never block the pipeline.
        case.agent_outputs.eval = await asyncio.wait_for(
            _timed("eval", state, _do), timeout=45.0)
    except Exception as exc:
        # Emit degraded so the UI shows ⚠️ rather than spinning forever.
        _emit_degraded(state, "eval", exc)
    return {"gpu": gpu_local}


async def _decision_node(state: GraphState):
    case = state["case"]
    case.agent_outputs.decision = await _timed(
        "decision", state, lambda: run_decision(
            case.agent_outputs.risk,
            case.agent_outputs.entity_resolution))
    return {}


# ── Graph wiring ────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("intake", _intake_node)
    g.add_node("extraction", _extraction_node)
    g.add_node("entityResolution", _entity_node)
    g.add_node("screening", _screening_node)
    g.add_node("idVerification", _id_verify_node)
    g.add_node("financialProfile", _financial_node)
    g.add_node("risk", _risk_node)
    g.add_node("explanation", _explanation_node)
    g.add_node("decision", _decision_node)
    g.add_node("eval", _eval_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "extraction")
    g.add_edge("extraction", "entityResolution")
    # Fan-out: LangGraph schedules these three concurrently because they share
    # the same parent and have no edges between them.
    g.add_edge("entityResolution", "screening")
    g.add_edge("entityResolution", "idVerification")
    g.add_edge("entityResolution", "financialProfile")
    # Join: risk waits for all three parallel branches to complete.
    g.add_edge("screening", "risk")
    g.add_edge("idVerification", "risk")
    g.add_edge("financialProfile", "risk")
    g.add_edge("risk", "decision")      # decision first so explanation can cite the verdict
    g.add_edge("decision", "explanation")
    g.add_edge("explanation", "eval")   # eval judges the explanation's own output
    g.add_edge("eval", END)
    return g.compile()


_GRAPH = _build_graph()


async def run_pipeline(state: CaseState, vllm: VllmClient, io: PipelineIO) -> CaseState:
    """Run the LangGraph pipeline. Signature preserved for `app.py` / smoke test."""
    t0 = time.perf_counter()
    state.status = "running"

    initial: GraphState = {"case": state, "vllm": vllm, "io": io, "gpu": []}
    final = await _GRAPH.ainvoke(initial)

    state.metrics.per_gpu_call.extend(final.get("gpu", []))
    state.metrics.end_to_end_ms = (time.perf_counter() - t0) * 1000

    # Pause for missing documents — skip the decision step entirely.
    er = state.agent_outputs.entity_resolution
    if er and er.documents_required:
        state.status = "awaiting_documents"
        io.emit("pipeline", state.status, {"documents_required": er.documents_required})
        return state

    # Pause for invalid PAN / Aadhaar — compliance officer must decide.
    idv = state.agent_outputs.id_verification
    id_issues = _id_issues(idv)
    if id_issues:
        state.status = "awaiting_id_review"
        io.emit("pipeline", state.status, {"id_issues": id_issues})
        return state

    # Terminal status — review/escalate pause for a human verdict.
    decision = state.agent_outputs.decision
    if decision.requires_human:
        state.status = "awaiting_human"
    elif decision.decision == "approve":
        state.status = "approved"
    else:
        state.status = "escalated"

    io.emit("pipeline", state.status, None)
    return state


def _id_issues(idv) -> list[str]:
    """Return hard-stop ID issues that require HITL.

    Aadhaar Verhoeff failure is intentionally excluded — vision models often
    misread digits from low-quality scans, producing false positives.  It is
    scored as a risk signal in risk.py instead of blocking the pipeline.
    Only structurally unambiguous failures (PAN regex, passport MRZ/expiry)
    warrant a hard stop.
    """
    if not idv:
        return []
    issues = []
    if idv.pan_format_valid is False:
        issues.append("PAN number format is invalid (expected ABCDE1234F)")
    if idv.mrz_valid is False:
        issues.append("Passport MRZ checksum failed — document may be tampered")
    if idv.expiry_ok is False:
        issues.append("Passport is expired")
    return issues


def _jsonable(v: object):
    return v.model_dump(mode="json") if hasattr(v, "model_dump") else v
