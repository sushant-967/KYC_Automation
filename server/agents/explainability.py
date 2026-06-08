"""
Explainability Agent (DEEP ★, §4.8) — Llama 3.3 70B (:8001).

Turns the deterministic scoring breakdown into prose a compliance officer would
actually read. This is the differentiator — anyone can compute a score; making the
*why* legible is what wins. Falls back to a deterministic summary if the model
output fails to parse.
"""
from __future__ import annotations

import json

from schemas import (EntityResolutionOutput, EvidenceCard, ExplanationOutput,
                     GpuCallMetric, RiskOutput, ScreeningOutput, Severity)
from vllm_client import VllmClient


async def run_explainability(
    entity: EntityResolutionOutput, screening: ScreeningOutput, risk: RiskOutput,
    vllm: VllmClient,
) -> tuple[ExplanationOutput, list[GpuCallMetric]]:
    result = await vllm.reason(
        [
            {"role": "system",
             "content": "You are a KYC compliance analyst. Given a deterministic risk "
                        "breakdown, write a clear, defensible rationale. Output ONLY JSON "
                        '{"summary","evidence_cards":[{"title","finding","source","severity"}],'
                        '"recommended_action"}. Cite the contributors; do not invent signals.'},
            {"role": "user", "content": json.dumps({
                "subject": entity.canonical_name,
                "score": risk.score,
                "contributors": [c.model_dump() for c in risk.contributors],
                "screening": screening.model_dump(mode="json"),
            })},
        ],
        json_mode=True, max_tokens=1536,
    )

    try:
        output = ExplanationOutput.model_validate(result.json)
    except Exception:
        output = _fallback(entity, risk)
    return output, [result.metric]


def _fallback(entity: EntityResolutionOutput, risk: RiskOutput) -> ExplanationOutput:
    cards = [
        EvidenceCard(
            title=c.signal,
            finding=f"{c.signal} contributed {c.contribution:.0f} points.",
            source="risk-aggregation",
            severity=Severity.high if c.contribution >= 30
            else Severity.medium if c.contribution >= 15 else Severity.low,
        )
        for c in risk.contributors
    ]
    return ExplanationOutput(
        summary=f"Risk score {risk.score:.0f}/100 for {entity.canonical_name}.",
        evidence_cards=cards,
        recommended_action="See decision agent.",
    )
