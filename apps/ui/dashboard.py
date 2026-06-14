"""
dashboard.py — Streamlit visualization client for the Agentic KYC API.

Thin client: talks to the FastAPI backend over HTTP. Run the API separately
(server/run.sh) and point this at it via API_BASE env var.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.environ.get("API_BASE", "http://localhost:7860")
PERSONA_DIR = Path(__file__).resolve().parents[2] / "personas"

AGENTS = [
    ("intake",           "Intake"),
    ("extraction",       "Extraction ★"),
    ("entityResolution", "Entity Resolution"),
    ("screening",        "Screening ★"),
    ("idVerification",   "ID Verify"),
    ("financialProfile", "Financial Profile"),
    ("risk",             "Risk"),
    ("explanation",      "Explainability ★"),
    ("decision",         "Decision"),
    ("eval",             "LLM Eval ★"),
]
TERMINAL = {
    "approved", "rejected", "escalated", "awaiting_human",
    "awaiting_documents", "awaiting_id_review", "error",
}

# Pipeline DAG edges for the flow diagram
_PIPELINE_EDGES = [
    ("intake", "extraction"),
    ("extraction", "entityResolution"),
    ("entityResolution", "screening"),
    ("entityResolution", "idVerification"),
    ("entityResolution", "financialProfile"),
    ("screening", "risk"),
    ("idVerification", "risk"),
    ("financialProfile", "risk"),
    ("risk", "decision"),
    ("decision", "explanation"),
    ("explanation", "eval"),
]

DECISION_STYLE = {
    "approve":            ("APPROVE",           "#10b981", "#d1fae5"),
    "approved":           ("APPROVED",          "#10b981", "#d1fae5"),
    "review":             ("HUMAN REVIEW",      "#f59e0b", "#fef3c7"),
    "awaiting_human":     ("AWAITING REVIEW",   "#f59e0b", "#fef3c7"),
    "escalate":           ("ESCALATE",          "#ef4444", "#fee2e2"),
    "escalated":          ("ESCALATED",         "#ef4444", "#fee2e2"),
    "rejected":           ("REJECTED",          "#ef4444", "#fee2e2"),
    "awaiting_documents": ("DOCS REQUIRED",     "#8b5cf6", "#ede9fe"),
    "awaiting_id_review": ("ID REVIEW",         "#f97316", "#ffedd5"),
}

DOC_LABELS = {
    "dual_name_affidavit": "Dual Name Affidavit (notarized)",
    "address_proof":        "Address Proof (utility bill / bank statement < 3 months)",
}

_MODEL_INFO: dict[str, tuple[str, list[str], float]] = {
    "qwen2.5-vl-72b":    ("Qwen2.5-VL-72B",  ["extraction"],                                76.0),
    "llama-3.3-70b":     ("Llama-3.3-70B",   ["screening", "financialProfile",
                                                "explanation", "eval"],                      73.0),
    "bge-large-en-v1.5": ("BGE-large-en-v1.5", ["screening (embed)"],                        4.0),
}
_TOTAL_MODEL_VRAM = sum(v[2] for v in _MODEL_INFO.values())


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic KYC — AMD Hackathon",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Force dark text in main content (counteract OS/Streamlit dark-mode white text) ── */
.stApp { color: #1e293b; }
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: #1e293b; }
[data-testid="stCaptionContainer"] { color: #64748b !important; }
[data-testid="stCaptionContainer"] p { color: #64748b !important; }
/* Native Streamlit text elements in the main area */
[data-testid="stVerticalBlock"] > div > .stMarkdown p { color: #1e293b; }
/* Tabs: text in tab panels */
[data-testid="stTabPanel"] { color: #1e293b; }
[data-testid="stTabPanel"] p, [data-testid="stTabPanel"] span { color: #1e293b; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    color: #0f172a !important;
}
[data-testid="metric-container"] * { color: #0f172a !important; }
[data-testid="metric-container"] > div:first-child {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .03em !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #ef4444, #b91c1c) !important;
    box-shadow: 0 4px 12px rgba(220,38,38,.4) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
    font-size: 13px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    color: #1e293b !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] div { color: #1e293b !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* ── Running pulse animation ── */
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(251,191,36,.5); }
    70%  { box-shadow: 0 0 0 8px rgba(251,191,36,0); }
    100% { box-shadow: 0 0 0 0 rgba(251,191,36,0); }
}
.agent-running { animation: pulse-ring 1.2s ease infinite; }

/* ── Section headers ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #7f1d1d 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 24px rgba(0,0,0,.18);
">
  <div>
    <div style="font-size:11px;letter-spacing:.2em;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:4px">
      TCS &amp; AMD AI Hackathon · Track 1 — Agents
    </div>
    <div style="font-size:28px;font-weight:800;color:#f1f5f9;letter-spacing:-.02em">
      🛡️ Agentic KYC Intelligence Platform
    </div>
    <div style="font-size:13px;color:#94a3b8;margin-top:4px">
      Multi-agent Customer Due Diligence · RBI / PMLA compliant · AMD MI300X
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:11px;color:#64748b;margin-bottom:6px">POWERED BY</div>
    <div style="font-size:14px;font-weight:700;color:#e2e8f0">Qwen2.5-VL-72B</div>
    <div style="font-size:14px;font-weight:700;color:#e2e8f0">Llama-3.3-70B</div>
    <div style="font-size:14px;font-weight:700;color:#e2e8f0">BGE-large-en-v1.5</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _card(html: str, padding: str = "18px 22px", radius: str = "12px",
          border: str = "#e2e8f0", bg: str = "#ffffff", shadow: bool = True,
          color: str = "#1e293b") -> None:
    sh = "box-shadow:0 1px 4px rgba(0,0,0,.07);" if shadow else ""
    st.markdown(
        f"<div style='padding:{padding};border-radius:{radius};border:1px solid {border};"
        f"background:{bg};color:{color};{sh}margin-bottom:12px'>{html}</div>",
        unsafe_allow_html=True)


def api_gpu_metrics() -> dict:
    try:
        return httpx.get(f"{API}/api/metrics/gpu", timeout=3).json()
    except Exception:
        return {}


def api_health() -> dict | None:
    try:
        return httpx.get(f"{API}/healthz", timeout=3).json()
    except Exception:
        return None


def upload_file(data: bytes, filename: str) -> str:
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
                   json={"decision": decision, "reviewer": "dashboard", "note": note},
                   timeout=10)
    r.raise_for_status()
    return r.json()


def submit_documents(cid: str, docs: list[dict]) -> dict:
    r = httpx.post(f"{API}/api/cases/{cid}/documents", json={"documents": docs}, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Pipeline visualization ───────────────────────────────────────────────────

_STATUS_STYLE = {
    "pending":  ("⚪", "#f8fafc", "#cbd5e1", "#64748b"),
    "running":  ("⏳", "#fefce8", "#fbbf24", "#92400e"),
    "done":     ("✅", "#f0fdf4", "#86efac", "#166534"),
    "degraded": ("⚠️",  "#fff7ed", "#fed7aa", "#9a3412"),
}

# Horizontal pipeline layout — two rows matching the actual DAG
_PIPELINE_LAYOUT = [
    ["intake", "extraction", "entityResolution", None,          None,             "risk", "decision", "explanation", "eval"],
    [None,     None,         None,               "screening",   "idVerification", None,  None,       None,          None],
    [None,     None,         None,               "financialProfile", None,        None,  None,       None,          None],
]

_AGENT_LABEL = {k: v for k, v in AGENTS}


def render_pipeline(container, statuses: dict) -> None:
    """Render animated pipeline cards in a grid reflecting the DAG structure."""
    with container.container():
        # Main row
        cols = st.columns(len(_PIPELINE_LAYOUT[0]))
        for i, key in enumerate(_PIPELINE_LAYOUT[0]):
            if key is None:
                cols[i].empty()
                continue
            s = statuses.get(key, "pending")
            icon, bg, border, txt = _STATUS_STYLE.get(s, _STATUS_STYLE["pending"])
            anim = ' class="agent-running"' if s == "running" else ""
            label = _AGENT_LABEL.get(key, key)
            cols[i].markdown(
                f"<div{anim} style='padding:10px 12px;border-radius:10px;"
                f"border:1.5px solid {border};background:{bg};text-align:center;"
                f"min-height:64px;display:flex;flex-direction:column;"
                f"align-items:center;justify-content:center'>"
                f"<div style='font-size:20px'>{icon}</div>"
                f"<div style='font-size:11px;font-weight:600;color:{txt};"
                f"margin-top:2px;line-height:1.3'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True)

        # Parallel branch row (screening / idVerification / financialProfile)
        st.markdown(
            "<div style='margin:-4px 0 4px;font-size:11px;color:#94a3b8;padding-left:4px'>"
            "↳ parallel branches</div>", unsafe_allow_html=True)
        branch_keys = ["screening", "idVerification", "financialProfile"]
        b_cols = st.columns([1, 1, 1, 1, 1, 1])
        # Position them under entityResolution (index 2) → columns 3,4,5
        for i in range(3):
            b_cols[i].empty()
        for j, key in enumerate(branch_keys):
            s = statuses.get(key, "pending")
            icon, bg, border, txt = _STATUS_STYLE.get(s, _STATUS_STYLE["pending"])
            anim = ' class="agent-running"' if s == "running" else ""
            label = _AGENT_LABEL.get(key, key)
            b_cols[3 + j].markdown(
                f"<div{anim} style='padding:8px 10px;border-radius:8px;"
                f"border:1.5px solid {border};background:{bg};text-align:center'>"
                f"<div style='font-size:16px'>{icon}</div>"
                f"<div style='font-size:10px;font-weight:600;color:{txt}'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True)


# ── Causal DAG ───────────────────────────────────────────────────────────────

def _dag_to_dot(dag_nodes: list[dict], dag_edges: list[dict], decision_str: str) -> str:
    dec_color = {
        "approve": "#10b981", "approved": "#10b981",
        "review":  "#f59e0b",
        "escalate": "#ef4444", "escalated": "#ef4444",
        "reject": "#8b5cf6",
    }.get(decision_str, "#64748b")

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "digraph causal_chain {",
        '    rankdir=LR;',
        '    graph [bgcolor="#0f172a", pad="0.5"];',
        '    node [fontname="Helvetica", fontsize=10, margin="0.3,0.15", fontcolor="white"];',
        '    edge [color="#475569", arrowsize=0.8, penwidth=1.5];',
    ]
    for n in dag_nodes:
        nid   = f'"{n["node_id"]}"'
        label = _esc(n.get("label", n["node_id"]))
        kind  = n.get("kind", "raw_value")
        contrib = n.get("contribution", 0)

        if kind == "decision":
            attrs = (f'label="{label}", shape=diamond, style=filled, '
                     f'fillcolor="{dec_color}", fontcolor="white", penwidth=2.5')
        elif kind == "signal":
            if contrib >= 30:
                fill, border = "#7f1d1d", "#ef4444"
            elif contrib >= 15:
                fill, border = "#78350f", "#f97316"
            else:
                fill, border = "#713f12", "#eab308"
            attrs = (f'label="{label}", shape=ellipse, style=filled, '
                     f'fillcolor="{fill}", color="{border}", penwidth=2')
        else:
            attrs = (f'label="{label}", shape=box, style="filled,rounded", '
                     f'fillcolor="#1e3a5f", color="#3b82f6"')
        lines.append(f"    {nid} [{attrs}];")

    for e in dag_edges:
        lines.append(f'    "{e["source"]}" -> "{e["target"]}";')

    lines.append("}")
    return "\n".join(lines)


# ── Plotly charts ────────────────────────────────────────────────────────────

def _risk_gauge(score: float) -> go.Figure:
    color = "#10b981" if score < 30 else "#f59e0b" if score < 70 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": color, "family": "Inter"},
                "suffix": ""},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569",
                     "tickfont": {"color": "#64748b", "size": 11}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#064e3b"},
                {"range": [30, 70], "color": "#451a03"},
                {"range": [70,100], "color": "#450a0a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": score,
            },
        },
        title={"text": "Risk Score / 100",
               "font": {"size": 13, "color": "#94a3b8", "family": "Inter"}},
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(t=30, b=10, l=20, r=20),
        height=220,
        font={"family": "Inter"},
    )
    return fig


def _contrib_bar(contribs: list[dict]) -> go.Figure:
    sorted_c = sorted(contribs, key=lambda x: x.get("contribution", 0))
    labels = [c["signal"].replace("_", " ").title() for c in sorted_c]
    values = [round(c.get("contribution", 0), 1) for c in sorted_c]
    colors = ["#ef4444" if v >= 30 else "#f97316" if v >= 15 else "#eab308" if v >= 5
              else "#94a3b8" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"+{v}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="%{y}: %{x} pts<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(color="#64748b"),
                   title=dict(text="Risk points", font=dict(color="#64748b"))),
        yaxis=dict(tickfont=dict(color="#e2e8f0"), automargin=True),
        margin=dict(t=10, b=10, l=10, r=40),
        height=max(160, len(contribs) * 34 + 40),
        font={"family": "Inter"},
    )
    return fig


def _latency_bar(per_agent: dict) -> go.Figure:
    items = sorted(per_agent.items(), key=lambda x: x[1].get("latency_ms", 0), reverse=True)
    labels = [k for k, _ in items]
    values = [round(v.get("latency_ms", 0)) for _, v in items]
    colors = ["#dc2626" if v > 2000 else "#f97316" if v > 500 else "#3b82f6" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v} ms" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=10),
        hovertemplate="%{y}: %{x} ms<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(color="#64748b"),
                   title=dict(text="Latency (ms)", font=dict(color="#64748b"))),
        yaxis=dict(tickfont=dict(color="#e2e8f0"), automargin=True),
        margin=dict(t=10, b=10, l=10, r=50),
        height=max(160, len(per_agent) * 30 + 40),
        font={"family": "Inter"},
    )
    return fig


def _accuracy_radar(faith: float, cov: float, align: float) -> go.Figure:
    cats = ["Faithfulness", "Coverage", "Score Alignment"]
    vals = [faith, cov, align]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,.25)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6, color="#60a5fa"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e293b",
            radialaxis=dict(range=[0, 1], tickfont=dict(color="#64748b", size=9),
                            gridcolor="#334155"),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=11),
                             gridcolor="#334155"),
        ),
        paper_bgcolor="#0f172a",
        margin=dict(t=20, b=20, l=30, r=30),
        height=220,
        font={"family": "Inter"},
    )
    return fig


# ── Main results renderer ────────────────────────────────────────────────────

def render_results(case: dict) -> None:
    ao         = case.get("agent_outputs") or {}
    risk       = ao.get("risk") or {}
    decision   = ao.get("decision") or {}
    explanation = ao.get("explanation") or {}
    screening  = ao.get("screening") or {}
    metrics    = case.get("metrics", {})

    # ── Guardrail banners ────────────────────────────────────────────────────
    flags    = case.get("guardrail_flags") or []
    critical = [f for f in flags if f.get("level") == "critical"]
    if critical:
        injection  = [f for f in critical
                      if "injection" in f.get("check","") or "jailbreak" in f.get("check","")]
        input_errs = [f for f in critical if f not in injection]

        if injection:
            st.markdown("""
<div style='background:linear-gradient(135deg,#7f1d1d,#450a0a);border:1px solid #ef4444;
border-radius:12px;padding:16px 20px;margin-bottom:12px'>
<div style='font-size:18px;font-weight:700;color:#fca5a5;margin-bottom:4px'>
🚨 ADVERSARIAL DOCUMENT DETECTED</div>
<div style='color:#fca5a5;font-size:13px'>
Prompt injection / jailbreak attempt found in submitted documents.
+50 risk points applied. Case auto-escalated.</div></div>
""", unsafe_allow_html=True)
            for f in injection:
                with st.expander(f"🔴 {f.get('check')}  [{f.get('agent')}]"):
                    for v in f.get("violations", []):
                        st.code(v)

        if input_errs:
            for f in input_errs:
                for v in f.get("violations", []):
                    st.warning(f"⚠️ {v}")

    warn_flags = [f for f in flags if f.get("level") == "warn" and f not in critical]
    if warn_flags:
        with st.expander(f"⚠️ {len(warn_flags)} guardrail warning(s)"):
            for f in warn_flags:
                st.caption(f"[{f.get('agent')}] {f.get('check')}: " +
                           "; ".join(f.get("violations", [])))

    # ── Decision banner ──────────────────────────────────────────────────────
    status   = case.get("status", "")
    dec_key  = status if status in DECISION_STYLE else decision.get("decision", "")
    label, color, bg = DECISION_STYLE.get(dec_key, ("—", "#64748b", "#f8fafc"))
    score    = round(risk.get("score", 0))

    st.markdown(
        f"<div style='padding:20px 28px;border-radius:14px;background:{bg};"
        f"border:1.5px solid {color}55;margin-bottom:20px;"
        f"display:flex;align-items:center;justify-content:space-between;color:#1e293b'>"
        f"<div>"
        f"<div style='font-size:11px;font-weight:700;letter-spacing:.15em;"
        f"color:{color};text-transform:uppercase;margin-bottom:4px'>DECISION</div>"
        f"<div style='font-size:32px;font-weight:800;color:{color}'>{label}</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='font-size:11px;color:#475569;font-weight:600;letter-spacing:.08em;margin-bottom:4px'>RISK SCORE</div>"
        f"<div style='font-size:48px;font-weight:800;color:{color};line-height:1'>{score}</div>"
        f"<div style='font-size:13px;color:#64748b'>/ 100</div>"
        f"</div></div>",
        unsafe_allow_html=True)

    # ── Risk + Screening ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Risk gauge + contributors
        contribs = risk.get("contributors", [])
        st.markdown('<div class="section-label">Risk Analysis</div>', unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(_risk_gauge(score), use_container_width=True, config={"displayModeBar": False})
        with g2:
            if contribs:
                st.plotly_chart(_contrib_bar(contribs), use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("No risk signals fired.")

        # Screening
        st.markdown('<div class="section-label" style="margin-top:12px">Screening</div>',
                    unsafe_allow_html=True)
        scr_cols = st.columns(3)
        for i, (k, title) in enumerate([("sanctions","Sanctions"),("pep","PEP"),("adverse_media","Adverse Media")]):
            sub = screening.get(k, {})
            hit = sub.get("hit")
            bg_c = "#450a0a" if hit else "#052e16"
            bd_c = "#ef4444" if hit else "#16a34a"
            ic   = "🔴" if hit else "🟢"
            txt  = "MATCH" if hit else "CLEAR"
            scr_cols[i].markdown(
                f"<div style='padding:12px;border-radius:10px;background:{bg_c};"
                f"border:1px solid {bd_c};text-align:center'>"
                f"<div style='font-size:22px'>{ic}</div>"
                f"<div style='font-size:11px;font-weight:700;color:{'#fca5a5' if hit else '#86efac'}"
                f";letter-spacing:.06em'>{title}</div>"
                f"<div style='font-size:10px;color:{'#fca5a5' if hit else '#6ee7b7'}"
                f";margin-top:2px'>{txt}</div>"
                f"</div>",
                unsafe_allow_html=True)

        am = screening.get("adverse_media", {})
        if am.get("summary"):
            st.caption(f"📰 {am['summary']}")
        for url in (am.get("sources") or [])[:2]:
            st.caption(f"🔗 [{url[:70]}]({url})")

    with col_right:
        # Explainability
        st.markdown('<div class="section-label">Causal Audit Trail</div>', unsafe_allow_html=True)
        if explanation.get("summary"):
            st.markdown(
                f"<div style='padding:12px 16px;border-radius:10px;background:#1e293b;"
                f"border:1px solid #334155;color:#e2e8f0;font-size:13px;"
                f"line-height:1.6;margin-bottom:12px'>"
                f"💡 {explanation['summary']}</div>",
                unsafe_allow_html=True)

        dag_nodes = explanation.get("dag_nodes", [])
        dag_edges = explanation.get("dag_edges", [])
        if dag_nodes:
            dot_src = _dag_to_dot(dag_nodes, dag_edges, decision.get("decision", ""))
            try:
                st.graphviz_chart(dot_src, use_container_width=True)
            except Exception:
                pass

        if not dag_nodes and explanation.get("evidence_cards"):
            for card in explanation.get("evidence_cards", []):
                sev   = card.get("severity", "low")
                sev_c = {"high": "#ef4444", "medium": "#f97316", "low": "#eab308"}.get(sev, "#94a3b8")
                with st.expander(f"● {card.get('title','')}"):
                    st.write(card.get("finding", ""))

        if explanation.get("recommended_action"):
            st.markdown(
                f"<div style='margin-top:8px;padding:10px 14px;border-radius:8px;"
                f"background:#1e3a5f;border-left:3px solid #3b82f6;"
                f"color:#bfdbfe;font-size:12px'>"
                f"📋 {explanation['recommended_action']}</div>",
                unsafe_allow_html=True)

        # LLM Eval badge
        eval_out = ao.get("eval") or {}
        if eval_out:
            st.markdown('<div class="section-label" style="margin-top:16px">LLM-as-Judge Accuracy</div>',
                        unsafe_allow_html=True)
            verdict    = eval_out.get("verdict", "")
            faith      = eval_out.get("faithfulness", 0)
            cov        = eval_out.get("coverage", 0)
            align      = eval_out.get("score_alignment", 1.0)
            v_color    = {"pass": "#10b981", "warn": "#f59e0b", "fail": "#ef4444"}.get(verdict, "#64748b")
            v_icon     = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(verdict, "❓")
            e1, e2 = st.columns([1, 1])
            with e1:
                st.plotly_chart(_accuracy_radar(faith, cov, align),
                                use_container_width=True, config={"displayModeBar": False})
            with e2:
                st.markdown(
                    f"<div style='padding:12px;border-radius:10px;background:#0f172a;"
                    f"border:1px solid {v_color}55;margin-top:8px'>"
                    f"<div style='font-size:20px;margin-bottom:6px'>{v_icon} "
                    f"<span style='font-weight:700;color:{v_color}'>{verdict.upper()}</span></div>"
                    f"<div style='font-size:12px;color:#94a3b8;margin:3px 0'>"
                    f"Faithfulness&nbsp;&nbsp;<b style='color:#e2e8f0'>{faith:.0%}</b></div>"
                    f"<div style='font-size:12px;color:#94a3b8;margin:3px 0'>"
                    f"Coverage&nbsp;&nbsp;<b style='color:#e2e8f0'>{cov:.0%}</b></div>"
                    f"<div style='font-size:12px;color:#94a3b8;margin:3px 0'>"
                    f"Score Alignment&nbsp;&nbsp;<b style='color:#e2e8f0'>{align:.0%}</b></div>"
                    f"</div>",
                    unsafe_allow_html=True)
                if eval_out.get("rationale"):
                    st.caption(eval_out["rationale"])

        # ID Verification
        idv = ao.get("id_verification") or {}
        if idv:
            st.markdown('<div class="section-label" style="margin-top:14px">ID Verification</div>',
                        unsafe_allow_html=True)
            _IDV_CHECKS = [
                ("pan_format_valid", "PAN Format",
                 "Valid (ABCDE1234F)", "Invalid format"),
                ("mrz_valid",        "Passport MRZ",
                 "Checksum valid",   "Checksum failed — possible tampering"),
                ("expiry_ok",        "Passport Expiry",
                 "Valid (not expired)", "Expired"),
            ]
            for field, lbl, ok_msg, fail_msg in _IDV_CHECKS:
                val = idv.get(field)
                if val is None:
                    continue
                ok_c = "#10b981" if val else "#ef4444"
                ok_i = "✅" if val else "❌"
                msg  = ok_msg if val else fail_msg
                st.markdown(
                    f"<div style='font-size:12px;color:#94a3b8;padding:3px 0'>"
                    f"{ok_i}&nbsp;<span style='color:#e2e8f0;font-weight:600'>{lbl}</span>"
                    f"&nbsp;—&nbsp;{msg}</div>",
                    unsafe_allow_html=True)

    # ── System Performance ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">⚡ System Performance · AMD MI300X</div>',
                unsafe_allow_html=True)

    gpu_calls = metrics.get("per_gpu_call", [])
    per_agent = metrics.get("per_agent", {})
    e2e_ms    = metrics.get("end_to_end_ms") or 0
    total_in  = sum(c.get("input_tokens") or 0 for c in gpu_calls)
    total_out = sum(c.get("output_tokens") or 0 for c in gpu_calls)
    total_tok = total_in + total_out
    avg_tps   = (total_tok / (e2e_ms / 1000)) if e2e_ms > 0 and total_tok else None

    # Headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("End-to-end Latency", f"{round(e2e_ms):,} ms")
    m2.metric("LLM Calls",          len([c for c in gpu_calls
                                         if not c.get("model","").startswith("cache:")]))
    m3.metric("Total Tokens",       f"{total_tok:,}" if total_tok else "—")
    m4.metric("Avg Tok / sec",      f"{avg_tps:,.0f}" if avg_tps else "—")

    perf_left, perf_right = st.columns([1, 1])

    with perf_left:
        # Agent latency chart
        if per_agent:
            st.markdown('<div class="section-label" style="margin-top:12px">Agent Latency Breakdown</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(_latency_bar(per_agent), use_container_width=True,
                            config={"displayModeBar": False})

        # Live GPU monitor
        gpu = api_gpu_metrics()
        gpu_available = gpu.get("source") != "unavailable" and gpu.get("vram_total_gb")
        if gpu_available:
            st.markdown(
                f'<div class="section-label" style="margin-top:4px">Live GPU · {gpu.get("source","")}</div>',
                unsafe_allow_html=True)
            vram_pct = gpu.get("vram_pct") or 0
            util_pct = gpu.get("gpu_util_pct") or 0
            vg1, vg2, vg3, vg4 = st.columns(4)
            vg1.metric("VRAM Used",
                       f"{gpu.get('vram_used_gb','?')} GB")
            vg2.metric("VRAM %",      f"{vram_pct:.1f}%")
            vg3.metric("GPU Util",    f"{util_pct:.1f}%")
            vg4.metric("Temp",        f"{gpu.get('temperature_c','?')} °C"
                       if gpu.get("temperature_c") else "—")

            vram_color = "#ef4444" if vram_pct > 90 else "#f97316" if vram_pct > 75 else "#10b981"
            for label, pct, bar_c in [
                (f"VRAM {vram_pct:.1f}%", vram_pct, vram_color),
                (f"GPU util {util_pct:.1f}%", util_pct, "#3b82f6"),
            ]:
                st.markdown(
                    f"<div style='margin:4px 0 2px;font-size:11px;color:#94a3b8'>{label}</div>"
                    f"<div style='height:8px;border-radius:4px;background:#1e293b'>"
                    f"<div style='height:8px;border-radius:4px;background:{bar_c};"
                    f"width:{pct:.1f}%;transition:width .5s ease'></div></div>",
                    unsafe_allow_html=True)

            # Model VRAM breakdown
            st.markdown('<div class="section-label" style="margin-top:12px">Model VRAM Allocation</div>',
                        unsafe_allow_html=True)
            vram_total_gpu = gpu.get("vram_total_gb") or _TOTAL_MODEL_VRAM
            bar_colors_vram = {"qwen": "#8b5cf6", "llama": "#3b82f6", "bge": "#10b981"}
            for mk, (display, agents, mvram) in _MODEL_INFO.items():
                pct  = round(mvram / vram_total_gpu * 100, 1)
                key  = next((k for k in bar_colors_vram if k in mk), "llama")
                bc   = bar_colors_vram[key]
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0'>"
                    f"<div style='width:130px;font-size:11px;color:#e2e8f0;font-weight:600'>{display}</div>"
                    f"<div style='flex:1;background:#1e293b;border-radius:4px;height:14px'>"
                    f"<div style='width:{pct}%;background:{bc};border-radius:4px;height:14px'></div></div>"
                    f"<div style='width:65px;font-size:11px;color:#94a3b8;text-align:right'>"
                    f"{mvram} GB</div>"
                    f"<div style='width:190px;font-size:10px;color:#64748b'>"
                    f"→ {', '.join(agents)}</div></div>",
                    unsafe_allow_html=True)
            st.caption(f"Model footprint: {_TOTAL_MODEL_VRAM:.0f} GB / {vram_total_gpu} GB total")

    with perf_right:
        # LLM call breakdown table
        real_calls = [c for c in gpu_calls if not c.get("model", "").startswith("cache:")]
        if real_calls:
            st.markdown('<div class="section-label" style="margin-top:12px">LLM Call Breakdown</div>',
                        unsafe_allow_html=True)
            has_vram = any(c.get("vram_used_gb") is not None for c in real_calls)
            rows = []
            for c in real_calls:
                in_t  = c.get("input_tokens")
                out_t = c.get("output_tokens")
                tps   = c.get("tokens_per_second")
                model = c.get("model", "")
                mvram = next((info[2] for k, info in _MODEL_INFO.items()
                              if k in model or model in k), None)
                row = {
                    "agent":        c.get("agent") or "—",
                    "model":        model[-26:] or "—",
                    "duration ms":  round(c.get("latency_ms", 0)),
                    "in tok":       in_t  if in_t  is not None else "—",
                    "out tok":      out_t if out_t is not None else "—",
                    "tok/s":        f"{tps:,.0f}" if tps else "—",
                }
                if has_vram:
                    lv = c.get("vram_used_gb")
                    row["VRAM GB"] = f"{lv:.1f}" if lv else "—"
                row["model VRAM"] = f"{mvram} GB" if mvram else "—"
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={
                             "duration ms": st.column_config.NumberColumn(format="%d ms"),
                             "in tok":      st.column_config.NumberColumn(),
                             "out tok":     st.column_config.NumberColumn(),
                         })

            # Screening parallelism note
            screening_calls = [c for c in real_calls
                               if (c.get("agent") or "").startswith("screening")]
            if len(screening_calls) > 1:
                llm_c   = [c for c in screening_calls if c.get("model") != "tavily-web-search"]
                tav     = next((c for c in screening_calls
                                if c.get("model") == "tavily-web-search"), None)
                scr_w   = (per_agent.get("screening") or {}).get("latency_ms")
                parts   = [f"{len(llm_c)} LLM calls run in parallel (asyncio.gather)"]
                if tav:
                    parts.append(f"Tavily = {round(tav['latency_ms'])} ms (bottleneck)")
                if scr_w:
                    parts.append(f"wall-clock = {round(scr_w)} ms")
                st.caption("ℹ️ Screening: " + " · ".join(parts))

    # ── HITL / ID review / awaiting docs ────────────────────────────────────
    idv = ao.get("id_verification") or {}
    if case.get("status") == "awaiting_id_review":
        st.markdown("---")
        st.markdown("""
<div style='padding:16px 20px;border-radius:12px;background:#450a0a;border:1px solid #ef4444;margin-bottom:12px'>
<div style='font-size:16px;font-weight:700;color:#fca5a5;margin-bottom:4px'>
🔴 ID Document Review Required</div>
<div style='color:#fca5a5;font-size:13px'>One or more documents failed validation. Compliance officer must decide.</div>
</div>""", unsafe_allow_html=True)
        if idv.get("pan_format_valid") is False:
            st.warning("**PAN** — format invalid (expected ABCDE1234F)")
        if idv.get("mrz_valid") is False:
            st.warning("**Passport MRZ** — checksum failed")
        if idv.get("expiry_ok") is False:
            st.warning("**Passport** — expired")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Accept & Proceed", use_container_width=True, type="primary"):
            _apply_hitl(case["case_id"], "approve")
        if c2.button("❌ Reject Case", use_container_width=True):
            _apply_hitl(case["case_id"], "reject")
        if c3.button("⛔ Escalate", use_container_width=True):
            _apply_hitl(case["case_id"], "escalate")

    er = ao.get("entity_resolution") or {}
    docs_required = er.get("documents_required", [])
    if case.get("status") == "awaiting_documents" and docs_required:
        st.markdown("---")
        st.markdown("""
<div style='padding:16px 20px;border-radius:12px;background:#2e1065;border:1px solid #8b5cf6;margin-bottom:12px'>
<div style='font-size:16px;font-weight:700;color:#c4b5fd;margin-bottom:4px'>
📄 Additional Documents Required</div></div>""", unsafe_allow_html=True)
        for doc_kind in docs_required:
            label = DOC_LABELS.get(doc_kind, doc_kind.replace("_", " ").title())
            st.warning(f"**{label}** is required.")
            uploaded = st.file_uploader(f"Upload {label}", type=["png","jpg","jpeg","pdf"],
                                        key=f"upload_{doc_kind}_{case['case_id']}")
            if uploaded and st.button(f"Submit {label}", key=f"submit_{doc_kind}_{case['case_id']}"):
                import base64 as _b64
                b64    = _b64.b64encode(uploaded.read()).decode()
                ext    = uploaded.name.rsplit(".", 1)[-1].lower()
                fid    = f"{case['case_id'][:8]}-{doc_kind}.{ext}"
                cid    = case["case_id"]
                with st.spinner("Submitting — re-running pipeline…"):
                    submit_documents(cid, [{"kind": doc_kind, "file_id": fid, "data": b64}])
                    try:
                        with httpx.stream("GET", f"{API}/api/cases/{cid}/stream",
                                          timeout=120) as sse:
                            for line in sse.iter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                ev = json.loads(line[6:])
                                if ev.get("agent") == "pipeline" and ev.get("status") in TERMINAL:
                                    break
                    except Exception:
                        pass
                st.session_state.case = get_case(cid)
                st.rerun()

    if decision.get("requires_human") and case.get("status") == "awaiting_human":
        st.markdown("---")
        st.markdown("""
<div style='padding:16px 20px;border-radius:12px;background:#1c1917;border:1px solid #f59e0b;margin-bottom:12px'>
<div style='font-size:16px;font-weight:700;color:#fde68a;margin-bottom:4px'>
👤 Human-in-the-Loop Review</div>
<div style='color:#fde68a;font-size:13px'>Pipeline paused — compliance officer decision required.</div>
</div>""", unsafe_allow_html=True)
        reasons = decision.get("reasons", [])
        for reason in reasons:
            st.warning(f"⚠️ {reason}")
        h1, h2, h3 = st.columns(3)
        if h1.button("✅ Approve", use_container_width=True, type="primary"):
            _apply_hitl(case["case_id"], "approve")
        if h2.button("↩️ Send Back", use_container_width=True):
            _apply_hitl(case["case_id"], "review")
        if h3.button("⛔ Escalate", use_container_width=True):
            _apply_hitl(case["case_id"], "escalate")

    elif case.get("status") in ("approved", "rejected", "escalated") and decision.get("requires_human"):
        st.success(f"✅ Final human verdict recorded: **{case['status'].upper()}**")

    with st.expander("🗂 Audit Log"):
        for e in case.get("audit_log", []):
            ts = e["ts"][:19].replace("T", " ")
            st.markdown(
                f"<span style='color:#64748b;font-size:11px'>{ts}</span>&nbsp;"
                f"<span style='color:#94a3b8;font-size:11px'>[{e['agent']}]</span>&nbsp;"
                f"<span style='color:#e2e8f0;font-size:11px'>{e['event']}</span>",
                unsafe_allow_html=True)


def _apply_hitl(cid: str, decision: str) -> None:
    decide_case(cid, decision)
    st.session_state.case = get_case(cid)
    st.rerun()


# ── Sidebar (Demo personas + API status) ─────────────────────────────────────

health = api_health()

# Read pipeline trigger saved by buttons lower in the page (session_state survives rerun)
run        = st.session_state.pop("_run", False)
submission = st.session_state.pop("_submission", None)

DOC_TYPES = [
    ("aadhaar", "Aadhaar Card"), ("pan", "PAN Card"), ("passport", "Passport"),
    ("voter_id", "Voter ID"), ("driving_license", "Driving Licence"),
    ("address_proof", "Address Proof"), ("dual_name_affidavit", "Name Affidavit"),
]

with st.sidebar:
    st.markdown("""
<div style='padding:14px 16px;background:#dc2626;border-radius:10px;margin-bottom:14px;text-align:center'>
<div style='font-size:10px;letter-spacing:.15em;color:#fecaca;font-weight:700'>AMD HACKATHON</div>
<div style='font-size:16px;font-weight:800;color:white;margin-top:2px'>KYC Pipeline</div>
</div>""", unsafe_allow_html=True)

    if health:
        mode     = "🔬 DEMO" if health.get("demo") else "🟢 LIVE"
        entities = health.get("entities_loaded", 0)
        tavily   = "🌐 Tavily" if health.get("tavily") else "📁 DB-only"
        cb       = health.get("circuit_breakers", {})
        cb_str   = " · ".join(f"{'🟢' if v=='closed' else '🔴'}{k}" for k, v in cb.items())
        st.markdown(
            f"<div style='padding:10px 12px;background:#0f172a;border:1px solid #22c55e55;"
            f"border-radius:8px;margin-bottom:14px'>"
            f"<div style='font-size:11px;color:#22c55e;font-weight:600'>● API HEALTHY</div>"
            f"<div style='font-size:11px;color:#94a3b8;margin-top:4px'>"
            f"{mode} · {entities:,} entities · {tavily}</div>"
            f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{cb_str}</div>"
            f"</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='padding:10px 12px;background:#450a0a;border:1px solid #ef444455;"
            f"border-radius:8px;margin-bottom:14px'>"
            f"<div style='font-size:11px;color:#ef4444;font-weight:600'>● API UNREACHABLE</div>"
            f"<div style='font-size:10px;color:#94a3b8'>{API}</div></div>",
            unsafe_allow_html=True)

    st.markdown('<div class="section-label">🎭 Demo Personas</div>', unsafe_allow_html=True)
    personas = sorted(p.name for p in PERSONA_DIR.iterdir() if p.is_dir()) \
               if PERSONA_DIR.exists() else []
    choice = st.selectbox("Demo persona", personas,
                          index=0 if personas else None, label_visibility="collapsed")
    # Use a LOCAL variable — never overwrite the global `submission` which may
    # already hold a manual-form or landing-page payload from session state.
    _demo_sub = None
    if choice:
        _d    = json.load(open(PERSONA_DIR / choice / "persona.json"))
        _demo_sub = _d.get("submission", _d)
        cust  = _demo_sub.get("customer", {})
        docs  = _demo_sub.get("documents", [])
        st.markdown(
            f"<div style='padding:10px 12px;background:#0f172a;border-radius:8px;"
            f"border:1px solid #334155;margin:8px 0'>"
            f"<div style='font-size:12px;font-weight:600;color:#e2e8f0'>"
            f"{cust.get('full_name','—')}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:2px'>"
            f"{cust.get('nationality','—').title()} · {cust.get('declared_employment','—')[:30]}</div>"
            f"<div style='font-size:10px;color:#475569;margin-top:2px'>"
            f"{len(docs)} doc(s): {', '.join(d['kind'] for d in docs[:3])}</div>"
            f"</div>",
            unsafe_allow_html=True)
        with st.expander("JSON payload"):
            st.json(_demo_sub)
    if st.button("▶ Run Demo Pipeline", type="primary", use_container_width=True,
                 disabled=not (health and _demo_sub), key="run_demo"):
        st.session_state._submission = _demo_sub
        st.session_state._run = True
        st.rerun()

    st.divider()

    st.divider()
    if st.session_state.get("case_id"):
        st.markdown('<div class="section-label">Start New Case</div>', unsafe_allow_html=True)
    if st.button("🎭 New Demo KYC", type="primary", use_container_width=True, key="new_demo_sb"):
        st.session_state.pop("case", None)
        st.session_state.pop("case_id", None)
        st.session_state.show_manual = False
        st.rerun()
    if st.button("✏️ New Manual KYC", use_container_width=True, key="new_manual_sb"):
        st.session_state.pop("case", None)
        st.session_state.pop("case_id", None)
        st.session_state.show_manual = True
        st.rerun()


# ── Pipeline live view ───────────────────────────────────────────────────────

pipeline_box = st.empty()

if run and submission:
    import time as _time
    statuses = {k: "pending" for k, _ in AGENTS}
    render_pipeline(pipeline_box, statuses)
    cid = create_case(submission)
    st.session_state.case_id = cid
    _stream_start = _time.monotonic()

    try:
        with httpx.stream("GET", f"{API}/api/cases/{cid}/stream", timeout=180) as r:
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
    except Exception:
        pass  # timeout or connection drop — fall through to fetch results

    st.session_state.case = get_case(cid)


# ── Render current case ──────────────────────────────────────────────────────

case = st.session_state.get("case")
if case:
    _degraded = {ev["agent"] for ev in case.get("audit_log", [])
                 if ev.get("event") == "degraded"}
    _ran      = set(case.get("metrics", {}).get("per_agent", {}).keys())
    statuses  = {}
    for key, _ in AGENTS:
        if key in _degraded:
            statuses[key] = "degraded"
        elif key in _ran:
            statuses[key] = "done"
        else:
            statuses[key] = "pending"
    render_pipeline(pipeline_box, statuses)

    # ── Action bar: always visible at top of results ─────────────────────────
    ab1, ab2, ab3 = st.columns([2, 1, 1])
    with ab1:
        cust_name = case.get("customer", {}).get("full_name", "Unknown")
        st.markdown(
            f"<div style='padding:8px 0;font-size:13px;color:#64748b'>"
            f"Case <code style='color:#94a3b8'>{case.get('case_id','')[:8]}…</code>"
            f" · <b style='color:#e2e8f0'>{cust_name}</b></div>",
            unsafe_allow_html=True)
    with ab2:
        if st.button("🎭 New Demo KYC", type="primary", use_container_width=True, key="rerun_demo_top"):
            st.session_state.pop("case", None)
            st.session_state.pop("case_id", None)
            st.session_state.show_manual = False
            st.rerun()
    with ab3:
        if st.button("✏️ New Manual KYC", use_container_width=True, key="rerun_manual_top"):
            st.session_state.pop("case", None)
            st.session_state.pop("case_id", None)
            st.session_state.show_manual = True
            st.rerun()

    render_results(case)
elif st.session_state.get("show_manual"):
    # ── Manual KYC entry form (full main-area width) ──────────────────────────
    st.markdown("""
<div style='padding:16px 24px;border-radius:12px;
background:linear-gradient(135deg,#1e3a5f,#0f172a);
border:1px solid #3b82f655;margin-bottom:20px'>
<div style='font-size:18px;font-weight:700;color:#e2e8f0'>✏️ Manual KYC Entry</div>
<div style='font-size:13px;color:#64748b;margin-top:2px'>
Fill in customer details and upload at least one identity document to run the pipeline.</div>
</div>""", unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown('<div class="section-label">Customer Details</div>', unsafe_allow_html=True)
        m_name    = st.text_input("Full name *", placeholder="As on Aadhaar / PAN")
        m_dob     = st.text_input("Date of birth *", placeholder="YYYY-MM-DD")
        m_nat     = st.selectbox("Nationality", ["india","usa","uk","uae","cyprus","other"])
        m_emp     = st.text_input("Employment", placeholder="e.g. Software Engineer, TCS")
    with fc2:
        m_address = st.text_area("Address", height=100, placeholder="Full residential address")
        m_income  = st.number_input("Annual income (INR)", min_value=0, step=10000)

    st.markdown('<div class="section-label" style="margin-top:8px">Documents — drag &amp; drop or browse (≥ 1 required)</div>',
                unsafe_allow_html=True)

    # 3-column grid for file uploaders so they're visible and spacious
    uploaded_docs: list[tuple[str, bytes, str]] = []
    doc_cols = st.columns(3)
    for idx, (kind, lbl) in enumerate(DOC_TYPES):
        with doc_cols[idx % 3]:
            uf = st.file_uploader(lbl, type=["png","jpg","jpeg","pdf"], key=f"doc_{kind}")
            if uf:
                uploaded_docs.append((kind, uf.read(), uf.name))

    st.markdown("<br>", unsafe_allow_html=True)
    run_manual = st.button("▶ Submit Case", type="primary",
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
            st.session_state._submission = {
                "customer": {
                    "full_name":           m_name,
                    "dob":                 m_dob,
                    "address":             m_address or None,
                    "nationality":         m_nat,
                    "declared_income":     float(m_income) if m_income else None,
                    "declared_employment": m_emp or None,
                },
                "documents": doc_refs,
            }
            st.session_state._run = True
            st.session_state.show_manual = False
            st.rerun()

else:
    # ── Landing state with real clickable actions ─────────────────────────────
    st.markdown("""
<div style='text-align:center;padding:40px 20px 24px'>
<div style='font-size:52px;margin-bottom:14px'>🛡️</div>
<div style='font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:6px'>
Agentic KYC Intelligence Platform</div>
<div style='font-size:14px;color:#64748b;margin-bottom:28px'>
Multi-agent Customer Due Diligence · RBI / PMLA compliant · AMD MI300X
</div>
</div>""", unsafe_allow_html=True)

    land_l, land_r = st.columns(2, gap="large")

    with land_l:
        st.markdown("""
<div style='padding:20px 24px;background:#0f172a;border:1.5px solid #334155;
border-radius:14px;margin-bottom:16px;min-height:120px'>
<div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:6px'>🎭 Demo Personas</div>
<div style='font-size:13px;color:#64748b'>Pre-loaded synthetic customers with documents —
no file uploads needed. Select a persona and run the full 9-agent pipeline instantly.</div>
</div>""", unsafe_allow_html=True)

        personas_l = sorted(p.name for p in PERSONA_DIR.iterdir() if p.is_dir()) \
                     if PERSONA_DIR.exists() else []
        choice_l = st.selectbox("Choose a persona", personas_l,
                                index=0 if personas_l else None, key="landing_persona")
        if choice_l:
            pd_l   = json.load(open(PERSONA_DIR / choice_l / "persona.json"))
            sub_l  = pd_l.get("submission", pd_l)
            cust_l = sub_l.get("customer", {})
            docs_l = sub_l.get("documents", [])
            st.markdown(
                f"<div style='padding:10px 12px;background:#1e293b;border-radius:8px;"
                f"border:1px solid #334155;margin:8px 0;color:#e2e8f0'>"
                f"<b style='color:#f1f5f9'>{cust_l.get('full_name','—')}</b><br>"
                f"<span style='font-size:12px;color:#94a3b8'>"
                f"{cust_l.get('nationality','—').title()} · "
                f"{cust_l.get('declared_employment','—')[:40]}</span><br>"
                f"<span style='font-size:11px;color:#64748b'>"
                f"{len(docs_l)} document(s): {', '.join(d['kind'] for d in docs_l[:3])}"
                f"</span></div>",
                unsafe_allow_html=True)

        if st.button("▶ Run Demo Pipeline", type="primary", use_container_width=True,
                     disabled=not (health and choice_l), key="run_demo_landing"):
            pd_l  = json.load(open(PERSONA_DIR / choice_l / "persona.json"))
            st.session_state._submission = pd_l.get("submission", pd_l)
            st.session_state._run = True
            st.rerun()

    with land_r:
        st.markdown("""
<div style='padding:20px 24px;background:#0f172a;border:1.5px solid #334155;
border-radius:14px;margin-bottom:16px;min-height:120px'>
<div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:6px'>✏️ Manual KYC Entry</div>
<div style='font-size:13px;color:#64748b'>Upload your own Aadhaar, PAN, Passport or other
documents and enter customer details to run a real KYC check.</div>
</div>""", unsafe_allow_html=True)

        if st.button("Open Manual Entry Form →", use_container_width=True,
                     key="open_manual_landing"):
            st.session_state.show_manual = True
            st.rerun()
