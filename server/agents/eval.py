"""
Eval Agent (DEEP ★, §4.10) — LLM-as-judge faithfulness + coverage check.

Evaluates the explainability agent's own output. Answers:
  - Coverage:     did the narrative mention every high-weight risk signal?
  - Faithfulness: did the narrative invent signals not in the contributors list?

High-weight threshold: contribution >= 15 pts (medium severity cutoff).

Returns a structured EvalOutput with scores, lists of missing / hallucinated
signals, a pass/warn/fail verdict, and a one-sentence rationale. The verdict
is also enforced deterministically from the scores so the judge cannot
contradict itself.
"""
from __future__ import annotations

import json

from schemas import EvalOutput, EvalVerdict, ExplanationOutput, GpuCallMetric, RiskOutput
from vllm_client import VllmClient

HIGH_WEIGHT_THRESHOLD = 15.0   # contribution pts — aligns with Severity.medium cutoff


async def run_eval(
    explanation: ExplanationOutput,
    risk: RiskOutput,
    vllm: VllmClient,
) -> tuple[EvalOutput, list[GpuCallMetric]]:
    high_weight = [c for c in risk.contributors if c.contribution >= HIGH_WEIGHT_THRESHOLD]

    # No high-weight signals → coverage is vacuously complete; nothing to evaluate.
    if not high_weight:
        return EvalOutput(
            faithfulness=1.0, coverage=1.0,
            missing_signals=[], hallucinated_signals=[],
            verdict="pass",
            rationale="No high-weight signals (all contributions < 15 pts) — coverage is vacuously complete.",
        ), []

    all_contribs = [{"signal": c.signal, "contribution": round(c.contribution, 1)}
                    for c in risk.contributors]
    hw_contribs = [{"signal": c.signal, "contribution": round(c.contribution, 1)}
                   for c in high_weight]

    result = await vllm.reason(
        [
            {"role": "system",
             "content": (
                 "You are a KYC audit quality evaluator. Given the ground-truth risk "
                 "contributors and an AI-generated narrative summary, evaluate two things:\n\n"
                 "1. COVERAGE — what fraction of HIGH-WEIGHT signals (contribution >= 15 pts, "
                 "listed in high_weight_contributors) are reflected in the summary? "
                 "Paraphrases count — 'ID document issues' covers 'id_fail', etc.\n\n"
                 "2. FAITHFULNESS — does the summary mention risk topics that have NO "
                 "corresponding entry anywhere in all_contributors? "
                 "Be semantic and generous: 'geographical risks' = geography_risk, "
                 "'income concerns' = income_implausibility, 'name discrepancy' = name_mismatch*, "
                 "'address issues' = address_*, 'ID problems' = id_fail, etc. "
                 "Only flag a genuine hallucination if the topic has absolutely no basis "
                 "in any contributor. Do NOT flag natural-language paraphrases of real signals.\n\n"
                 "Return ONLY valid JSON: "
                 '{"missing_signals": [...], "hallucinated_signals": [...], '
                 '"coverage": 0.0-1.0, "faithfulness": 0.0-1.0, '
                 '"verdict": "pass"|"warn"|"fail", "rationale": "one sentence"}. '
                 "verdict rules: pass = coverage>=0.8 AND faithfulness>=0.9; "
                 "fail = coverage<0.5 OR faithfulness<0.7; warn = otherwise."
             )},
            {"role": "user", "content": json.dumps({
                "all_contributors": all_contribs,
                "high_weight_contributors": hw_contribs,
                "summary": explanation.summary,
            })},
        ],
        json_mode=True, max_tokens=512,
    )

    try:
        raw = result.json or {}
        output = EvalOutput(
            faithfulness=float(raw.get("faithfulness", 1.0)),
            coverage=float(raw.get("coverage", 1.0)),
            missing_signals=[str(s) for s in raw.get("missing_signals", [])],
            hallucinated_signals=[str(s) for s in raw.get("hallucinated_signals", [])],
            verdict=_enforce_verdict(
                float(raw.get("coverage", 1.0)),
                float(raw.get("faithfulness", 1.0)),
                str(raw.get("verdict", "pass")),
            ),
            rationale=str(raw.get("rationale", "")),
        )
    except Exception:
        output = _deterministic_fallback(explanation, risk, high_weight)

    return output, [result.metric]


def _enforce_verdict(coverage: float, faithfulness: float, llm_verdict: str) -> EvalVerdict:
    """Override LLM verdict if scores contradict it."""
    if coverage < 0.5 or faithfulness < 0.7:
        return "fail"
    if coverage >= 0.8 and faithfulness >= 0.9:
        return "pass"
    return "warn"


def _deterministic_fallback(explanation, risk, high_weight) -> EvalOutput:
    """Score coverage by checking whether each high-weight signal name appears
    in the summary text. Used when the LLM call fails entirely."""
    summary_lower = explanation.summary.lower()
    missing = [
        c.signal for c in high_weight
        if not _signal_mentioned(c.signal, summary_lower)
    ]
    n = len(high_weight)
    coverage = (n - len(missing)) / n if n else 1.0
    faithfulness = 1.0  # can't detect hallucinations without LLM

    verdict: EvalVerdict = (
        "pass" if coverage >= 0.8 else
        "warn" if coverage >= 0.5 else
        "fail"
    )
    return EvalOutput(
        faithfulness=faithfulness,
        coverage=round(coverage, 2),
        missing_signals=missing,
        hallucinated_signals=[],
        verdict=verdict,
        rationale="Deterministic fallback — LLM call unavailable.",
    )


def _signal_mentioned(signal: str, text: str) -> bool:
    """Heuristic: check if the signal name or its key words appear in the text."""
    keywords = signal.replace("_", " ").split()
    return any(kw in text for kw in keywords if len(kw) > 3)
