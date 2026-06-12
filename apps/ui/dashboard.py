"""
dashboard.py — Streamlit visualization client for the Agentic KYC API.

This is a THIN CLIENT: it talks to the FastAPI backend over HTTP (POST a case,
consume the SSE pipeline stream, GET the final state, POST a human decision). The
API remains the product/source-of-truth — Streamlit is one of potentially several
front-ends. Run the API separately (server/run.sh); point this at it via API_BASE.

    API_BASE=http://localhost:7860 streamlit run apps/ui/dashboard.py
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_BASE", "http://localhost:7860")
PERSONA_DIR = Path(__file__).resolve().parents[2] / "personas"

AGENTS = [
    ("intake", "Intake"),
    ("extraction", "Extraction ★"),
    ("entityResolution", "Entity Resolution"),
    ("screening", "Screening ★"),
    ("idVerification", "ID Verify"),
    ("financialProfile", "Financial Profile"),
    ("risk", "Risk Aggregation"),
    ("explanation", "Explainability ★"),
    ("decision", "Decision"),
]
TERMINAL = {"approved", "rejected", "escalated", "awaiting_human", "awaiting_documents", "awaiting_id_review", "error"}
ICON = {"pending": "⚪", "running": "🟡", "done": "✅"}
DECISION_STYLE = {
    "approve": ("APPROVE", "#16a34a"), "approved": ("APPROVED", "#16a34a"),
    "review": ("HUMAN REVIEW", "#d97706"), "awaiting_human": ("AWAITING HUMAN", "#d97706"),
    "escalate": ("ESCALATE", "#dc2626"), "escalated": ("ESCALATED", "#dc2626"),
    "rejected": ("REJECTED", "#dc2626"),
    "awaiting_documents": ("DOCUMENTS REQUIRED", "#7c3aed"),
    "awaiting_id_review": ("ID REVIEW REQUIRED", "#dc6803"),
}

DOC_LABELS = {
    "dual_name_affidavit": "Dual Name Affidavit (notarized)",
    "address_proof": "Current Address Proof (utility bill / bank statement < 3 months)",
}

st.set_page_config(page_title="Agentic KYC", page_icon="🛡️", layout="wide")


# ── API helpers ─────────────────────────────────────────────────────────────

def api_health() -> dict | None:
    try:
        return httpx.get(f"{API}/healthz", timeout=3).json()
    except Exception:
        return None


def upload_file(data: bytes, filename: str) -> str:
    """Upload a document to the API and return its file_id."""
    r = httpx.post(f"{API}/api/upload",
                   files={"file": (filename, io.BytesIO(data), "application/octet-stream")},
                   timeout=30)
    r.raise_for_status()
    return r.json()["file_id"]


def create_case(submission: dict) -> str:
    r = httpx.post(f"{API}/api/cases", json=submission, timeout=15)
    r.raise_for_status()
    return r.json()["case_id"]


def get_case(cid: str) -> dict:
    return httpx.get(f"{API}/api/cases/{cid}", timeout=10).json()


def decide_case(cid: str, decision: str, note: str = "") -> dict:
    r = httpx.post(f"{API}/api/cases/{cid}/decide",
                   json={"decision": decision, "reviewer": "dashboard", "note": note}, timeout=10)
    r.raise_for_status()
    return r.json()


def submit_documents(cid: str, docs: list[dict]) -> dict:
    r = httpx.post(f"{API}/api/cases/{cid}/documents", json={"documents": docs}, timeout=15)
    r.raise_for_status()
    return r.json()


# ── rendering ───────────────────────────────────────────────────────────────

def render_pipeline(container, statuses: dict) -> None:
    with container.container():
        for row in range(0, len(AGENTS), 3):
            cols = st.columns(3)
            for col, (key, label) in zip(cols, AGENTS[row:row + 3]):
                s = statuses.get(key, "pending")
                col.markdown(
                    f"<div style='padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;"
                    f"background:{'#f0fdf4' if s=='done' else '#fffbeb' if s=='running' else '#fafafa'}'>"
                    f"<span style='font-size:18px'>{ICON.get(s,'⚪')}</span>&nbsp;"
                    f"<b>{label}</b><br><span style='color:#6b7280;font-size:12px'>{s}</span></div>",
                    unsafe_allow_html=True)


def render_results(case: dict) -> None:
    ao = case.get("agent_outputs") or {}
    risk = ao.get("risk") or {}
    decision = ao.get("decision") or {}
    explanation = ao.get("explanation") or {}
    screening = ao.get("screening") or {}
    metrics = case.get("metrics", {})

    # Decision banner
    label, color = DECISION_STYLE.get(case.get("status", ""),
                                      DECISION_STYLE.get(decision.get("decision", ""), ("—", "#6b7280")))
    st.markdown(
        f"<div style='padding:18px 22px;border-radius:12px;background:{color}18;border:1px solid {color}55'>"
        f"<span style='font-size:13px;color:#6b7280;letter-spacing:.05em'>DECISION</span><br>"
        f"<span style='font-size:30px;font-weight:700;color:{color}'>{label}</span>"
        f"&nbsp;&nbsp;<span style='color:#6b7280'>risk score "
        f"<b style='font-size:22px;color:{color}'>{round(risk.get('score',0))}</b>/100</span></div>",
        unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Risk contributors")
        contribs = risk.get("contributors", [])
        if contribs:
            df = pd.DataFrame([{"signal": c["signal"], "points": round(c["contribution"], 1)} for c in contribs])
            st.bar_chart(df.set_index("signal"), horizontal=True, color="#dc2626")
        else:
            st.caption("No risk signals fired.")

        st.subheader("Screening")
        for k in ("sanctions", "pep"):
            sub = screening.get(k, {})
            hit = sub.get("hit")
            st.markdown(f"{'🔴' if hit else '🟢'} **{k.upper()}** — {'MATCH' if hit else 'clear'}")
            for m in sub.get("matches", [])[:3]:
                st.caption(f"· {m.get('name')} ({m.get('verdict')}, conf {m.get('confidence')}) — {m.get('rationale','')}")
        am = screening.get("adverse_media", {})
        st.markdown(f"{'🔴' if am.get('hit') else '🟢'} **ADVERSE MEDIA** — "
                    f"{am.get('severity','none') if am.get('hit') else 'clear'}")
        if am.get("summary"):
            st.caption(am["summary"])

    with right:
        st.subheader("Explainability")
        if explanation.get("summary"):
            st.info(explanation["summary"])
        for card in explanation.get("evidence_cards", []):
            sev = card.get("severity", "low")
            dot = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")
            with st.expander(f"{dot} {card.get('title','')}  ·  {card.get('source','')}"):
                st.write(card.get("finding", ""))

    # Metrics
    st.divider()
    st.subheader("Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("End-to-end", f"{round(metrics.get('end_to_end_ms') or 0)} ms")
    m2.metric("GPU calls", len(metrics.get("per_gpu_call", [])))
    m3.metric("Agents run", len(metrics.get("per_agent", {})))
    per_agent = metrics.get("per_agent", {})
    if per_agent:
        dfm = pd.DataFrame([{"agent": k, "latency_ms": round(v.get("latency_ms", 0), 1)}
                            for k, v in per_agent.items()])
        st.bar_chart(dfm.set_index("agent"), color="#2563eb")

    # Awaiting ID review — invalid PAN / Aadhaar / Passport
    idv = ao.get("id_verification") or {}
    if case.get("status") == "awaiting_id_review":
        st.divider()
        st.subheader("ID Document Review Required")
        st.error("One or more identity documents failed format validation. "
                 "A compliance officer must decide how to proceed.")

        # Collect which checks failed
        if idv.get("pan_format_valid") is False:
            st.warning("**PAN** — format invalid (expected 5 letters + 4 digits + 1 letter, e.g. ABCDE1234F)")
        if idv.get("aadhaar_format_valid") is False:
            st.warning("**Aadhaar** — Verhoeff checksum failed (12-digit number is incorrect or tampered)")
        if idv.get("mrz_valid") is False:
            st.warning("**Passport MRZ** — checksum failed (machine-readable zone may be tampered)")
        if idv.get("expiry_ok") is False:
            st.warning("**Passport** — document is expired")

        st.caption("Risk score and full pipeline analysis are still shown below for reference.")
        st.divider()
        c1, c2, c3 = st.columns(3)
        if c1.button("Accept & Proceed", use_container_width=True, type="primary",
                     help="Override the format issue — apply the computed risk decision"):
            _apply_hitl(case["case_id"], "approve")
        if c2.button("Reject Case", use_container_width=True,
                     help="Document is invalid/fake — reject the KYC application"):
            _apply_hitl(case["case_id"], "reject")
        if c3.button("Escalate", use_container_width=True,
                     help="Flag for senior compliance review"):
            _apply_hitl(case["case_id"], "escalate")

    # Awaiting documents — show what's needed + upload UI
    er = ao.get("entity_resolution") or {}
    docs_required = er.get("documents_required", [])
    if case.get("status") == "awaiting_documents" and docs_required:
        st.divider()
        st.subheader("Action Required — Additional Documents")
        for doc_kind in docs_required:
            label = DOC_LABELS.get(doc_kind, doc_kind.replace("_", " ").title())
            st.warning(f"**{label}** is required to proceed.")
            uploaded = st.file_uploader(f"Upload {label}", type=["png", "jpg", "jpeg", "pdf"],
                                        key=f"upload_{doc_kind}_{case['case_id']}")
            if uploaded and st.button(f"Submit {label}", key=f"submit_{doc_kind}_{case['case_id']}"):
                import base64 as _b64
                b64 = _b64.b64encode(uploaded.read()).decode()
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                file_id = f"{case['case_id'][:8]}-{doc_kind}.{ext}"
                submit_documents(case["case_id"], [{"kind": doc_kind, "file_id": file_id, "data": b64}])
                st.session_state.case = get_case(case["case_id"])
                st.rerun()
        # Show what was already verified
        if er.get("name_affidavit_submitted"):
            covers = er.get("name_affidavit_covers_discrepancy")
            st.info(f"Dual Name Affidavit received — {'covers discrepancy' if covers else 'does NOT cover all name variants'}.")
        if er.get("address_additional_proof_submitted"):
            confirmed = er.get("address_additional_proof_confirmed")
            st.info(f"Additional address proof received — {'address confirmed' if confirmed else 'address still does not match'}.")

    # HITL
    if decision.get("requires_human") and case.get("status") == "awaiting_human":
        st.divider()
        st.subheader("Human-in-the-loop")
        st.caption("This case is paused for a compliance officer's verdict.")
        h1, h2, h3 = st.columns(3)
        if h1.button("✅ Approve", use_container_width=True):
            _apply_hitl(case["case_id"], "approve")
        if h2.button("↩️ Send back (review)", use_container_width=True):
            _apply_hitl(case["case_id"], "review")
        if h3.button("⛔ Escalate", use_container_width=True):
            _apply_hitl(case["case_id"], "escalate")
    elif case.get("status") in ("approved", "rejected", "escalated") and decision.get("requires_human"):
        st.success(f"Final human verdict recorded: **{case['status'].upper()}**")

    with st.expander("Audit log"):
        for e in case.get("audit_log", []):
            st.text(f"{e['ts']}  [{e['agent']}] {e['event']}")


def _apply_hitl(cid: str, decision: str) -> None:
    decide_case(cid, decision)
    st.session_state.case = get_case(cid)
    st.rerun()


# ── app ─────────────────────────────────────────────────────────────────────

st.title("🛡️ Agentic KYC — Intelligence Dashboard")
st.caption(f"Multi-agent Customer Due Diligence · backend API: `{API}`")

health = api_health()
with st.sidebar:
    st.header("New KYC Case")
    if health:
        st.success(f"API up · {'DEMO' if health.get('demo') else 'LIVE'} mode · "
                   f"{health.get('entities_loaded',0)} entities")
    else:
        st.error(f"API unreachable at {API}")

    tab_demo, tab_manual = st.tabs(["Demo Personas", "Manual KYC"])
    submission = None

    # ── Tab 1: pre-built personas ────────────────────────────────────────────
    with tab_demo:
        personas = sorted(p.name for p in PERSONA_DIR.iterdir() if p.is_dir()) if PERSONA_DIR.exists() else []
        choice = st.selectbox("Persona", personas, index=0 if personas else None)
        if choice:
            data = json.load(open(PERSONA_DIR / choice / "persona.json"))
            submission = data.get("submission", data)
            with st.expander("Payload"):
                st.json(submission)
        run = st.button("▶ Run Demo Case", type="primary", use_container_width=True,
                        disabled=not (health and submission), key="run_demo")

    # ── Tab 2: manual upload ─────────────────────────────────────────────────
    with tab_manual:
        st.caption("Fill in customer details and upload identity documents.")

        m_name    = st.text_input("Full name", placeholder="As on Aadhaar / PAN")
        m_dob     = st.text_input("Date of birth", placeholder="YYYY-MM-DD")
        m_address = st.text_area("Current address", height=80)
        m_nat     = st.selectbox("Nationality", ["india", "usa", "uk", "uae", "cyprus", "other"])
        m_income  = st.number_input("Declared annual income (INR)", min_value=0, step=10000)
        m_emp     = st.text_input("Employment", placeholder="e.g. Software Engineer, TCS")

        st.markdown("**Upload documents** (at least one required)")
        DOC_TYPES = [
            ("aadhaar",           "Aadhaar Card"),
            ("pan",               "PAN Card"),
            ("passport",          "Passport"),
            ("voter_id",          "Voter ID"),
            ("driving_license",   "Driving Licence"),
            ("address_proof",     "Address Proof (utility bill etc.)"),
            ("dual_name_affidavit", "Dual Name Affidavit"),
        ]
        uploaded_docs: list[tuple[str, bytes, str]] = []  # (kind, bytes, filename)
        for kind, label in DOC_TYPES:
            f = st.file_uploader(label, type=["png", "jpg", "jpeg", "pdf"], key=f"doc_{kind}")
            if f:
                uploaded_docs.append((kind, f.read(), f.name))

        run_manual = st.button("▶ Submit KYC Case", type="primary", use_container_width=True,
                               disabled=not (health and m_name and m_dob and uploaded_docs),
                               key="run_manual")

        if run_manual and m_name and m_dob and uploaded_docs:
            with st.spinner("Uploading documents…"):
                doc_refs = []
                for kind, data_bytes, fname in uploaded_docs:
                    try:
                        fid = upload_file(data_bytes, fname)
                        doc_refs.append({"kind": kind, "file_id": fid})
                    except Exception as e:
                        st.error(f"Upload failed for {kind}: {e}")
                        doc_refs = []
                        break

            if doc_refs:
                submission = {
                    "customer": {
                        "full_name": m_name,
                        "dob": m_dob,
                        "address": m_address or None,
                        "nationality": m_nat,
                        "declared_income": float(m_income) if m_income else None,
                        "declared_employment": m_emp or None,
                    },
                    "documents": doc_refs,
                }
                run = True   # fall through to the shared pipeline runner below
            else:
                run = False
    # run is set by either tab above; if neither tab triggered it, default False
    if "run" not in dir():
        run = False

pipeline_box = st.empty()

if run and submission:
    statuses = {k: "pending" for k, _ in AGENTS}
    render_pipeline(pipeline_box, statuses)
    cid = create_case(submission)
    st.session_state.case_id = cid
    # Consume the SSE stream and animate agents as they fire.
    try:
        with httpx.stream("GET", f"{API}/api/cases/{cid}/stream", timeout=120) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                ag, sttus = ev.get("agent"), ev.get("status")
                if ag in statuses:
                    statuses[ag] = sttus
                    render_pipeline(pipeline_box, statuses)
                if ag == "pipeline" and sttus in TERMINAL:
                    break
    except Exception as e:
        st.warning(f"stream ended: {e}")
    st.session_state.case = get_case(cid)

# Render whichever case is current (persists across HITL reruns).
case = st.session_state.get("case")
if case:
    render_pipeline(pipeline_box, {k: ("done" if k in case.get("metrics", {}).get("per_agent", {}) else "pending")
                                   for k, _ in AGENTS})
    st.divider()
    render_results(case)
else:
    st.info("Pick a persona in the sidebar and hit **Run KYC case** to watch the pipeline.")
