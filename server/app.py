"""
app.py — FastAPI orchestrator entrypoint (§3, single-box edition).

Routes:
  POST /api/cases             → create a case, kick off the pipeline, return {case_id}
  GET  /api/cases/:id         → current CaseState
  GET  /api/cases/:id/stream  → SSE stream of pipeline events
  POST /api/cases/:id/decide  → human verdict (approve | review | escalate)
  GET  /api/cases             → list case ids
  GET  /healthz               → liveness

Everything runs on this box. vLLM is on localhost; state is local SQLite; vector
recall is in-process. No external services.
"""
from __future__ import annotations

import asyncio
import base64
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from schemas import (AuditEvent, CaseMetrics, CaseState, HumanDecision, Submission)
from store import CaseStore
from screening_index import ScreeningIndex
from vllm_client import VllmClient
from pipeline import PipelineIO, run_pipeline
from agents.intake import run_intake

UPLOAD_DIR = Path(os.environ.get("KYC_UPLOAD_DIR", Path(__file__).parent / "uploads"))

app = FastAPI(title="Agentic KYC Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class Hub:
    """In-process per-case SSE fan-out + shared singletons (replaces the DO)."""

    def __init__(self) -> None:
        self.store = CaseStore()
        self.index = ScreeningIndex()
        self.vllm = VllmClient()
        self._subs: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, case_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(case_id, set()).add(q)
        return q

    def unsubscribe(self, case_id: str, q: asyncio.Queue) -> None:
        self._subs.get(case_id, set()).discard(q)

    def publish(self, case_id: str, event: dict) -> None:
        for q in list(self._subs.get(case_id, set())):
            q.put_nowait(event)


hub: Hub


@app.on_event("startup")
async def _startup() -> None:
    global hub
    hub = Hub()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await hub.vllm.aclose()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "entities_loaded": len(hub.index.rows)}


@app.get("/api/cases")
async def list_cases() -> dict:
    return {"cases": hub.store.list_ids()}


@app.post("/api/cases")
async def create_case(request: Request) -> JSONResponse:
    raw = await request.json()
    try:
        submission = Submission.model_validate(raw.get("submission", raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid submission: {e}")

    case_id = uuid.uuid4().hex
    intake = run_intake(case_id, submission)
    state = CaseState(
        case_id=case_id, status="intake", customer=intake.customer,
        documents=intake.documents, metrics=CaseMetrics(),
    )
    state.agent_outputs.intake = intake
    hub.store.save(state)
    _audit(state, "do", "case_created")

    # Run the pipeline in the background; events stream as they happen.
    asyncio.create_task(_execute(case_id))
    return JSONResponse({"case_id": case_id, "status": state.status})


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str) -> Any:
    state = hub.store.get(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="no such case")
    return state.model_dump(mode="json")


@app.get("/api/cases/{case_id}/stream")
async def stream_case(case_id: str) -> StreamingResponse:
    state = hub.store.get(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="no such case")
    q = hub.subscribe(case_id)
    # Replay current status so a late subscriber isn't blank.
    q.put_nowait({"agent": "pipeline", "status": state.status})

    async def gen():
        try:
            while True:
                event = await q.get()
                yield f"data: {_dumps(event)}\n\n"
                if event.get("agent") == "pipeline" and event.get("status") in _TERMINAL:
                    break
        finally:
            hub.unsubscribe(case_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache"})


@app.post("/api/cases/{case_id}/decide")
async def decide_case(case_id: str, verdict: HumanDecision) -> dict:
    state = hub.store.get(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="no such case")
    state.status = ("approved" if verdict.decision == "approve"
                    else "escalated" if verdict.decision == "escalate" else "rejected")
    hub.store.save(state)
    _audit(state, "human", "human_decision", verdict.model_dump())
    hub.publish(case_id, {"agent": "human", "status": state.status, "payload": verdict.model_dump()})
    return {"status": state.status}


# ── Pipeline execution ──────────────────────────────────────────────────────

_TERMINAL = {"approved", "rejected", "escalated", "awaiting_human"}


async def _execute(case_id: str) -> None:
    state = hub.store.get(case_id)
    if not state:
        return

    def emit(agent: str, status: str, payload: object) -> None:
        event = {"agent": agent, "status": status, "payload": payload}
        state.audit_log.append(AuditEvent(ts=_now(), agent=agent, event=status, payload=payload))
        hub.store.append_audit(case_id, state.audit_log[-1])
        hub.publish(case_id, event)

    io = PipelineIO(
        load_image=_load_image,
        entity_index=hub.index,
        prior_case_lookup=lambda _name: [],  # TODO: query prior cases from the store
        emit=emit,
    )
    try:
        state = await run_pipeline(state, hub.vllm, io)
    except Exception as e:  # surface failures to the UI instead of hanging
        state.status = "rejected"
        emit("pipeline", "error", {"error": str(e)})
    finally:
        hub.store.save(state)


async def _load_image(file_id: str) -> str:
    """Resolve a document file_id to a data: URL. Demo: read from UPLOAD_DIR."""
    path = UPLOAD_DIR / file_id
    if path.exists():
        b = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{b}"
    return f"data:image/png;base64,"  # placeholder; extraction will low-confidence


# ── helpers ─────────────────────────────────────────────────────────────────

def _audit(state: CaseState, agent: str, event: str, payload: object = None) -> None:
    ev = AuditEvent(ts=_now(), agent=agent, event=event, payload=payload)
    state.audit_log.append(ev)
    hub.store.append_audit(state.case_id, ev)


def _dumps(obj: object) -> str:
    import json
    return json.dumps(obj, default=str)
