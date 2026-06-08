"""
Decision Agent (light, §4.9) — threshold rules over the deterministic score.
    score < 30        → approve
    30 ≤ score < 70   → review   (requires human)
    score ≥ 70        → escalate (requires human)
"""
from __future__ import annotations

from schemas import DecisionOutput, RiskOutput

THRESHOLDS = {"approve": 30, "escalate": 70}


def run_decision(risk: RiskOutput) -> DecisionOutput:
    s = risk.score
    if s < THRESHOLDS["approve"]:
        return DecisionOutput(decision="approve", requires_human=False)
    if s < THRESHOLDS["escalate"]:
        return DecisionOutput(decision="review", requires_human=True)
    return DecisionOutput(decision="escalate", requires_human=True)
