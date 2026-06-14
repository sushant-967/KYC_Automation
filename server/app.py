"""
app.py — FastAPI orchestrator entrypoint (§3, single-box edition).

Routes:
  POST /api/upload            → upload a document file, returns {file_id}
  POST /api/cases             → create a case, kick off the pipeline, return {case_id}
  GET  /api/cases/:id         → current CaseState
  GET  /api/cases/:id/stream  → SSE stream of pipeline events
  POST /api/cases/:id/documents → add remediation docs (awaiting_documents state)
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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from schemas import (AgentOutputs, AuditEvent, CaseMetrics, CaseState, DocumentRef, HumanDecision, Submission)
from store import CaseStore
from screening_index import ScreeningIndex
from vllm_client import VllmClient, make_vllm_client
from tavily_client import TavilyClient
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
        self.demo = os.environ.get("KYC_DEMO") == "1"
        if self.demo:
            # Deterministic stand-ins for the GPU + planted entities (see demo.py).
            from demo import DemoVllm, DemoIndex
            self.index = DemoIndex()
            self.vllm = DemoVllm()
        else:
            self.index = ScreeningIndex()
            self.vllm = make_vllm_client()
        self.tavily = TavilyClient.from_env()  # None if TAVILY_API_KEY not set
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
    # Pre-warm the local fastembed model in a background task so the first
    # screening call doesn't pay the ~130 MB download latency.
    if hasattr(hub.vllm, "cfg") and hub.vllm.cfg.local_embedder_model:
        asyncio.create_task(_prewarm_embedder())


async def _prewarm_embedder() -> None:
    try:
        await hub.vllm.embed("warmup")
    except Exception:
        pass  # non-fatal; first real call will still work


@app.on_event("shutdown")
async def _shutdown() -> None:
    await hub.vllm.aclose()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/metrics/gpu")
async def gpu_metrics() -> dict:
    """Live GPU metrics — tries rocm-smi (AMD MI300X), then nvidia-smi, then returns nulls."""
    import subprocess, json as _json

    def _rocm() -> dict | None:
        try:
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return None
            data = _json.loads(r.stdout)
            # rocm-smi JSON key is "card0" or the first GPU entry
            card = next(iter(data.values()))
            vram_total = int(card.get("VRAM Total Memory (B)", 0))
            vram_used  = int(card.get("VRAM Total Used Memory (B)", 0))
            gpu_util   = float(card.get("GPU use (%)", 0) or 0)
            temp       = float(card.get("Temperature (Sensor edge) (C)", 0) or 0)
            return {
                "vram_used_gb":  round(vram_used  / 1e9, 1),
                "vram_total_gb": round(vram_total / 1e9, 1),
                "vram_pct":      round(vram_used / vram_total * 100, 1) if vram_total else 0,
                "gpu_util_pct":  gpu_util,
                "temperature_c": temp,
                "source": "rocm-smi",
            }
        except Exception:
            return None

    def _nvidia() -> dict | None:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return None
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) < 4:
                return None
            used_mb, total_mb, util, temp = parts[:4]
            used, total = float(used_mb), float(total_mb)
            return {
                "vram_used_gb":  round(used  / 1024, 1),
                "vram_total_gb": round(total / 1024, 1),
                "vram_pct":      round(used / total * 100, 1) if total else 0,
                "gpu_util_pct":  float(util),
                "temperature_c": float(temp),
                "source": "nvidia-smi",
            }
        except Exception:
            return None

    result = await asyncio.to_thread(_rocm) or await asyncio.to_thread(_nvidia)
    if result:
        return result
    return {
        "vram_used_gb": None, "vram_total_gb": None,
        "vram_pct": None, "gpu_util_pct": None,
        "temperature_c": None, "source": "unavailable",
    }


@app.get("/healthz")
async def healthz() -> dict:
    circuit = (hub.vllm.circuit_states()
               if hasattr(hub.vllm, "circuit_states") else {})
    degraded = any(v != "closed" for k, v in circuit.items()
                   if isinstance(v, str))
    return {
        "ok": True,
        "demo": hub.demo,
        "entities_loaded": len(hub.index.rows),
        "tavily": hub.tavily is not None,
        "circuit_breakers": circuit,
        "backend_degraded": degraded,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload a single KYC document image. Returns a file_id to reference in submission."""
    ext = (file.filename or "doc").rsplit(".", 1)[-1].lower()
    allowed = {"png", "jpg", "jpeg", "pdf"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"file type .{ext} not allowed")
    file_id = f"{uuid.uuid4().hex[:12]}.{ext}"
    (UPLOAD_DIR / file_id).write_bytes(await file.read())
    return {"file_id": file_id}


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
    # Replay per-agent done/degraded events from the AUDIT TABLE (not
    # state.audit_log, which only reflects the last full hub.store.save() and
    # is empty during live pipeline runs). The audit table is written on every
    # emit() call so it always has the real-time history. This ensures a late
    # SSE subscriber (Streamlit opens the stream ~200 ms after create_case, by
    # which time intake and sometimes extraction have already completed) sees
    # all finished agents in the correct order before receiving live events.
    for ev in hub.store.get_audit(case_id):
        if ev.event in ("done", "degraded"):
            q.put_nowait({"agent": ev.agent, "status": ev.event})
    # Always send the current pipeline status last so the UI knows whether to stop.
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


@app.post("/api/cases/{case_id}/documents")
async def add_documents(case_id: str, request: Request) -> JSONResponse:
    """Accept additional documents (e.g. dual_name_affidavit, fresh address_proof)
    for a case that is paused in awaiting_documents status, then re-run the pipeline."""
    state = hub.store.get(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="no such case")
    if state.status != "awaiting_documents":
        raise HTTPException(status_code=409,
                            detail=f"case is '{state.status}', not awaiting_documents")

    raw = await request.json()
    new_docs: list[dict] = raw.get("documents", [])
    if not new_docs:
        raise HTTPException(status_code=400, detail="documents list required")

    # Save uploaded files and append DocumentRefs to the case.
    for doc in new_docs:
        file_id: str = doc.get("file_id", "")
        b64: str = doc.get("data", "")
        if file_id and b64:
            import base64 as _b64
            (UPLOAD_DIR / file_id).write_bytes(_b64.b64decode(b64))
        try:
            ref = DocumentRef.model_validate({"kind": doc["kind"], "file_id": doc["file_id"]})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid document ref: {e}")
        state.documents.append(ref)

    # Reset agent outputs so the pipeline re-runs cleanly.
    state.agent_outputs = AgentOutputs(intake=state.agent_outputs.intake)
    state.metrics = CaseMetrics()
    state.status = "running"   # move out of awaiting_documents before SSE replay
    hub.store.save(state)
    _audit(state, "do", "documents_added", {"added": [d["kind"] for d in new_docs]})

    asyncio.create_task(_execute(case_id))
    return JSONResponse({"case_id": case_id, "status": "running"})


@app.post("/api/cases/{case_id}/decide")
async def decide_case(case_id: str, verdict: HumanDecision) -> dict:
    state = hub.store.get(case_id)
    if not state:
        raise HTTPException(status_code=404, detail="no such case")

    if state.status == "awaiting_id_review":
        if verdict.decision == "approve":
            # Officer accepts the invalid format — apply the computed decision.
            dec = state.agent_outputs.decision
            if dec:
                if dec.decision == "approve":
                    state.status = "approved"
                elif dec.requires_human:
                    # Medium-risk case: officer still needs to review the risk
                    state.status = "awaiting_human"
                else:
                    state.status = "escalated"
            else:
                state.status = "approved"
        elif verdict.decision == "reject":
            state.status = "rejected"
        else:
            state.status = "escalated"
    else:
        state.status = ("approved" if verdict.decision == "approve"
                        else "escalated" if verdict.decision == "escalate" else "rejected")

    hub.store.save(state)
    _audit(state, "human", "human_decision", verdict.model_dump())
    hub.publish(case_id, {"agent": "human", "status": state.status, "payload": verdict.model_dump()})
    return {"status": state.status}


# ── Pipeline execution ──────────────────────────────────────────────────────

_TERMINAL = {"approved", "rejected", "escalated", "awaiting_human", "awaiting_documents", "awaiting_id_review"}


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
        prior_case_lookup=lambda _name: [],
        emit=emit,
        tavily=hub.tavily,
    )
    try:
        state = await run_pipeline(state, hub.vllm, io)
    except Exception as e:  # surface failures to the UI instead of hanging
        state.status = "escalated"   # "rejected" implies a KYC decision; this is a system error
        emit("pipeline", "error", {"error": str(e)})
    finally:
        hub.store.save(state)


_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
         "gif": "image/gif", "webp": "image/webp"}

async def _load_image(file_id: str) -> str:
    """Resolve a document file_id to a data: URL (PNG) for the vision model.
    PDFs are converted to a PNG image by rendering their extracted text onto a
    white canvas with PIL — no native PDF renderer needed."""
    path = UPLOAD_DIR / file_id
    if not path.exists():
        return "data:image/png;base64,"  # placeholder

    ext = path.suffix.lstrip(".").lower()

    if ext == "pdf":
        return _pdf_to_image_url(path)

    mime = _MIME.get(ext, "image/png")
    b = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b}"


def _pdf_to_image_url(path) -> str:
    """Extract text from a PDF and render it as a PNG data URL using PIL."""
    import io as _io
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip() or "(no text extracted from PDF)"
    except Exception:
        text = "(could not read PDF)"

    from PIL import Image, ImageDraw, ImageFont
    W, H = 900, max(600, 30 * (text.count("\n") + 5))
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 20), text, fill="#111111", font=font)

    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ── helpers ─────────────────────────────────────────────────────────────────

def _audit(state: CaseState, agent: str, event: str, payload: object = None) -> None:
    ev = AuditEvent(ts=_now(), agent=agent, event=event, payload=payload)
    state.audit_log.append(ev)
    hub.store.append_audit(state.case_id, ev)


def _dumps(obj: object) -> str:
    import json
    return json.dumps(obj, default=str)
