"""
pipeline.py — KYC pipeline orchestrator on LangGraph (§2, §3).

A `StateGraph` wires the 9 agents into the deterministic CDD flow. Deep agents
(extraction, screening, explainability) call vLLM on localhost; light agents are
pure Python. The graph is compiled once at import time.

    intake → extraction ★ → entity-resolution
           → { screening ★ ‖ id-verify ‖ financial-profile }
           → risk → explainability ★ → decision

The agents, schemas, audit log, SSE emit, and metrics layers are unchanged —
LangGraph only replaces the hand-written DAG that used to live here.
"""
from __future__ import annotations

import asyncio
import operator
import os
import time
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from schemas import AgentMetric, CaseState, GpuCallMetric
from screening_index import ScreeningIndex
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

# Emit callback: (agent, status, payload) -> None. Used for audit log + SSE.
Emit = Callable[[str, str, object], None]


@dataclass
class PipelineIO:
    load_image: Callable[[str], Awaitable[str]]  # file_id -> data: URL
    entity_index: ScreeningIndex
    prior_case_lookup: Callable[[str], list[str]]
    emit: Emit


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


async def _timed(name: str, state: GraphState, fn):
    """Emit running/done, time the call, accept sync or async agents."""
    io = state["io"]
    io.emit(name, "running", None)
    step_delay = float(os.environ.get("KYC_STEP_DELAY_MS", "0")) / 1000
    if step_delay:
        await asyncio.sleep(step_delay)
    started = time.perf_counter()
    value = fn()
    result = await value if asyncio.iscoroutine(value) else value
    state["case"].metrics.per_agent[name] = AgentMetric(
        latency_ms=(time.perf_counter() - started) * 1000)
    io.emit(name, "done", _jsonable(result))
    return result


# ── Nodes ───────────────────────────────────────────────────────────────────

async def _intake_node(state: GraphState):
    case = state["case"]
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

    case.agent_outputs.extraction = await _timed("extraction", state, _do)
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
                                   case.customer.dob, state["vllm"],
                                   state["io"].entity_index)
        gpu_local.extend(g)
        return o

    case.agent_outputs.screening = await _timed("screening", state, _do)
    return {"gpu": gpu_local}


async def _id_verify_node(state: GraphState):
    case = state["case"]
    case.agent_outputs.id_verification = await _timed(
        "idVerification", state,
        lambda: run_id_verification(case.agent_outputs.extraction))
    return {}


async def _financial_node(state: GraphState):
    case = state["case"]
    case.agent_outputs.financial_profile = await _timed(
        "financialProfile", state,
        lambda: run_financial_profile(case.customer))
    return {}


async def _risk_node(state: GraphState):
    case = state["case"]
    o = case.agent_outputs
    case.agent_outputs.risk = await _timed(
        "risk", state,
        lambda: run_risk(o.entity_resolution, o.screening,
                         o.id_verification, o.financial_profile))
    return {}


async def _explanation_node(state: GraphState):
    case = state["case"]
    gpu_local: list[GpuCallMetric] = []

    async def _do():
        o, g = await run_explainability(case.agent_outputs.entity_resolution,
                                        case.agent_outputs.screening,
                                        case.agent_outputs.risk, state["vllm"])
        gpu_local.extend(g)
        return o

    case.agent_outputs.explanation = await _timed("explanation", state, _do)
    return {"gpu": gpu_local}


async def _decision_node(state: GraphState):
    case = state["case"]
    case.agent_outputs.decision = await _timed(
        "decision", state, lambda: run_decision(case.agent_outputs.risk))
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
    g.add_edge("risk", "explanation")
    g.add_edge("explanation", "decision")
    g.add_edge("decision", END)
    return g.compile()


_GRAPH = _build_graph()


async def run_pipeline(state: CaseState, vllm: VllmClient, io: PipelineIO) -> CaseState:
    """Run the LangGraph pipeline. Signature preserved for `app.py` / smoke test."""
    t0 = time.perf_counter()
    state.status = "running"

    initial: GraphState = {"case": state, "vllm": vllm, "io": io, "gpu": []}
    final = await _GRAPH.ainvoke(initial)

    # Terminal status — review/escalate pause for a human verdict.
    decision = state.agent_outputs.decision
    if decision.requires_human:
        state.status = "awaiting_human"
    elif decision.decision == "approve":
        state.status = "approved"
    else:
        state.status = "escalated"

    state.metrics.per_gpu_call.extend(final.get("gpu", []))
    state.metrics.end_to_end_ms = (time.perf_counter() - t0) * 1000
    io.emit("pipeline", state.status, None)
    return state


def _jsonable(v: object):
    return v.model_dump(mode="json") if hasattr(v, "model_dump") else v
