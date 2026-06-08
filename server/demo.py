"""
demo.py — deterministic stand-ins for the GPU so the pipeline runs end-to-end
WITHOUT vLLM or ingested data. Enabled with KYC_DEMO=1.

It is NOT a mock of the whole pipeline — only the three GPU touchpoints (embed,
reason, extract) and the entity index are replaced. The real screening funnel
(embed → vector recall → precision filter → adjudication), the real deterministic
risk scoring, and the real decision thresholds all execute. Embeddings are a
deterministic function of the name, and a couple of planted sanctioned/PEP entities
are seeded so recall genuinely retrieves them — so you see real APPROVE / REVIEW /
ESCALATE outcomes for the three demo personas.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from schemas import GpuCallMetric
from screening_index import EntityRow

DIM = 1024


def name_vec(name: str) -> np.ndarray:
    """Deterministic unit vector seeded by the canonicalized name."""
    h = hashlib.sha256(name.strip().lower().encode()).digest()
    seed = int.from_bytes(h[:8], "big") % (2**32)
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# Planted entities (synthetic — mirror personas 2 & 3). Priya matches nothing.
_DEMO_ROWS = [
    EntityRow(id="UN-DEMO-001", name="Viktor Nazarov", aliases=["Viktor A. Nazarov"],
              dob="1979", countries=["cy", "ru"], topics=["sanction", "crime.fin"],
              datasets=["un_sc_sanctions", "eu_sanctions"],
              summary="Alleged sanctions-evasion network (synthetic demo entity).",
              source_url="https://example/synthetic"),
    EntityRow(id="IN-PEP-DEMO-002", name="Rajesh Kumar Singh", aliases=[],
              dob="1971", countries=["in"], topics=["role.pep"],
              datasets=["in_pep"], summary="State government official (synthetic demo PEP).",
              source_url="https://example/synthetic"),
]


class DemoIndex:
    """Same interface as ScreeningIndex, backed by the planted rows."""

    def __init__(self) -> None:
        self.rows = _DEMO_ROWS
        self._matrix = np.vstack([name_vec(r.name) for r in self.rows])
        self._topics = [set(r.topics) for r in self.rows]

    def recall(self, query_vector, topic_prefixes, k: int = 20) -> list[EntityRow]:
        q = np.asarray(query_vector, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        out = []
        sims = self._matrix @ q
        for i, sim in enumerate(sims):
            if sim > 0.95 and _topic_match(self._topics[i], topic_prefixes):
                out.append(self.rows[i])
        return out[:k]


def _topic_match(topics, prefixes) -> bool:
    return any(t == p or t.startswith(p + ".") for t in topics for p in prefixes)


class _Res:
    def __init__(self, j, model):
        self.json, self.text = j, ""
        self.metric = GpuCallMetric(ts="demo", model=model, latency_ms=1.0)


class DemoVllm:
    """Canned stand-ins for the three model endpoints."""

    async def aclose(self):
        pass

    async def embed(self, text):
        items = text if isinstance(text, list) else [text]
        return [name_vec(t).tolist() for t in items], GpuCallMetric(ts="demo", model="bge-demo", latency_ms=1.0)

    async def extract(self, messages, **kw):
        kind = _kind_from(messages)
        return _Res(_canned_doc(kind), "qwen-demo")

    async def reason(self, messages, **kw):
        user = _user_text(messages)
        if '"candidates"' in user:  # screening adjudication
            data = json.loads(user)
            matches = [{
                "entity_id": c["entity_id"], "name": c["name"],
                "datasets": c.get("datasets", []), "verdict": "match", "confidence": 0.92,
                "rationale": f"Name and DOB align with {c['name']} in {','.join(c.get('datasets', []) or ['demo'])}.",
                "evidence": [f"name≈{c['name']}", f"dob≈{c.get('dob')}"],
            } for c in data.get("candidates", [])]
            return _Res({"matches": matches}, "llama-demo")
        # explainability
        return _Res(_canned_explanation(user), "llama-demo")


def _user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m["content"]
            return c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _kind_from(messages) -> str:
    txt = (messages[0].get("content", "") if messages else "") + _user_text(messages)
    for k in ("passport", "aadhaar", "pan", "voter_id", "driving_license", "address_proof"):
        if k in txt:
            return k
    return "aadhaar"


def _canned_doc(kind: str) -> dict:
    base = {
        "passport": {"name": "Viktor Nazarov", "dob": "1979-04-02", "passportNumber": "C1234567",
                     "nationality": "Cyprus", "expiry": "2030-01-01", "_confidence": 0.93},
        "aadhaar": {"aadhaarNumber": "123456781234", "name": "Demo Customer", "dob": "1992-07-14",
                    "gender": "F", "address": "Bengaluru", "_confidence": 0.95},
        "pan": {"pan": "ABCDE1234F", "name": "Demo Customer", "dob": "1992-07-14", "_confidence": 0.94},
        "address_proof": {"name": "Demo Customer", "address": "Bengaluru", "date": "2026-05-01",
                          "provider": "BESCOM", "_confidence": 0.9},
    }
    return base.get(kind, base["aadhaar"])


def _canned_explanation(user: str) -> dict:
    try:
        data = json.loads(user)
    except Exception:
        data = {}
    score = data.get("score", 0)
    contributors = data.get("contributors", [])
    cards = [{
        "title": c["signal"].replace("_", " ").title(),
        "finding": f"{c['signal']} added {round(c['contribution'])} points (value={c['value']}).",
        "source": "risk-aggregation",
        "severity": "high" if c["contribution"] >= 30 else "medium" if c["contribution"] >= 15 else "low",
    } for c in contributors]
    return {
        "summary": f"Aggregate KYC risk score is {round(score)}/100 for {data.get('subject','the customer')}, "
                   f"driven by {len(contributors)} signal(s): {', '.join(c['signal'] for c in contributors) or 'none'}.",
        "evidence_cards": cards,
        "recommended_action": "escalate" if score >= 70 else "review" if score >= 30 else "approve",
    }
