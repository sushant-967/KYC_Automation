"""
pipeline.py — KYC pipeline orchestrator (§2, §3).

Runs the deterministic agent pipeline over a case, emitting an event after each
step so the API layer can append to the audit log and broadcast SSE. Deep agents
(extraction, screening, explainability) call vLLM on localhost; light agents are
pure Python.

    intake → extraction ★ → entity-resolution
           → { screening ★ ‖ id-verify ‖ financial-profile }
           → risk → explainability ★ → decision
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

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


async def run_pipeline(state: CaseState, vllm: VllmClient, io: PipelineIO) -> CaseState:
    t0 = time.perf_counter()
    gpu: list[GpuCallMetric] = []
    # Optional pacing so the live SSE pipeline is watchable in a dashboard.
    # Zero in production; set e.g. KYC_STEP_DELAY_MS=500 for demos.
    step_delay = float(os.environ.get("KYC_STEP_DELAY_MS", "0")) / 1000

    async def step(name: str, coro_or_value):
        io.emit(name, "running", None)
        if step_delay:
            await asyncio.sleep(step_delay)
        started = time.perf_counter()
        result = await coro_or_value if asyncio.iscoroutine(coro_or_value) else coro_or_value
        state.metrics.per_agent[name] = AgentMetric(latency_ms=(time.perf_counter() - started) * 1000)
        io.emit(name, "done", _jsonable(result))
        return result

    state.status = "running"
    out = state.agent_outputs

    # 1. Intake (re-run for the audit trail; already validated on POST).
    out.intake = await step("intake", run_intake(
        state.case_id, {"customer": state.customer.model_dump(), "documents":
                        [d.model_dump() for d in state.documents]}))

    # 2. Extraction ★
    async def _extract():
        o, g = await run_extraction(state.documents, vllm, io.load_image)
        gpu.extend(g)
        return o
    out.extraction = await step("extraction", _extract())

    # 3. Entity resolution
    out.entity_resolution = await step("entityResolution", run_entity_resolution(
        state.customer, out.extraction, io.prior_case_lookup))

    # 4. Parallel fan-out: screening ★ ‖ id-verify ‖ financial-profile
    async def _screen():
        o, g = await run_screening(out.entity_resolution, state.customer.dob, vllm, io.entity_index)
        gpu.extend(g)
        return o
    screening, idv, fin = await asyncio.gather(
        step("screening", _screen()),
        step("idVerification", run_id_verification(out.extraction)),
        step("financialProfile", run_financial_profile(state.customer)),
    )
    out.screening, out.id_verification, out.financial_profile = screening, idv, fin

    # 5. Risk aggregation (deterministic)
    out.risk = await step("risk", run_risk(screening, idv, fin))

    # 6. Explainability ★
    async def _explain():
        o, g = await run_explainability(out.entity_resolution, screening, out.risk, vllm)
        gpu.extend(g)
        return o
    out.explanation = await step("explanation", _explain())

    # 7. Decision
    out.decision = await step("decision", run_decision(out.risk))

    # Terminal status — review/escalate pause for a human verdict.
    if out.decision.requires_human:
        state.status = "awaiting_human"
    elif out.decision.decision == "approve":
        state.status = "approved"
    else:
        state.status = "escalated"

    state.metrics.per_gpu_call.extend(gpu)
    state.metrics.end_to_end_ms = (time.perf_counter() - t0) * 1000
    io.emit("pipeline", state.status, None)
    return state


def _jsonable(v: object):
    return v.model_dump(mode="json") if hasattr(v, "model_dump") else v


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
