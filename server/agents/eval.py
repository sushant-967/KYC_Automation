"""
Eval Agent (DEEP ★, §4.10) — LLM-as-judge quality evaluation.

Evaluates the explainability agent's output on four axes:
  1. Coverage       — did the narrative mention every high-weight signal?
  2. Faithfulness   — did the narrative stay grounded in the contributors list?
  3. Hallucinations — list of topics mentioned that have NO basis in contributors.
  4. Score alignment — does the narrative tone/recommendation match the risk score tier?
                       (score <30 → approve tone · 30–69 → review · ≥70 → escalate/reject)

High-weight threshold: contribution >= 15 pts (Severity.medium cutoff).

Returns EvalOutput with all four scores, pass/warn/fail verdict (deterministically
enforced so the judge cannot contradict itself), and a one-sentence rationale.
"""
from __future__ import annotations

import json

from schemas import EvalOutput, EvalVerdict, ExplanationOutput, GpuCallMetric, RiskOutput
from vllm_client import VllmClient

HIGH_WEIGHT_THRESHOLD = 15.0   # contribution pts — aligns with Severity.medium cutoff

# Score tier boundaries (must match decision agent thresholds)
TIER_APPROVE  = 30
TIER_ESCALATE = 70


async def run_eval(
    explanation: ExplanationOutput,
    risk: RiskOutput,
    vllm: VllmClient,
) -> tuple[EvalOutput, list[GpuCallMetric]]:
    high_weight = [c for c in risk.contributors if c.contribution >= HIGH_WEIGHT_THRESHOLD]

    # No high-weight signals → coverage/faithfulness/score_alignment are vacuously perfect.
    if not high_weight and risk.score < TIER_APPROVE:
        return EvalOutput(
            faithfulness=1.0, coverage=1.0, score_alignment=1.0,
            hallucinations=[], missing_signals=[], hallucinated_signals=[],
            verdict="pass",
            rationale="No high-weight signals and low score — explanation quality is vacuously perfect.",
        ), []

    all_contribs = [{"signal": c.signal, "contribution": round(c.contribution, 1)}
                    for c in risk.contributors]
    hw_contribs  = [{"signal": c.signal, "contribution": round(c.contribution, 1)}
                    for c in high_weight]
    score_tier = ("low-risk (approve)"   if risk.score < TIER_APPROVE  else
                  "medium-risk (review)" if risk.score < TIER_ESCALATE else
                  "high-risk (escalate/reject)")

    result = await vllm.reason(
        [
            {"role": "system",
             "content": (
                 "You are a KYC audit quality evaluator. Given the ground-truth risk "
                 "contributors and an AI-generated explanation, evaluate FOUR things:\n\n"
                 "1. COVERAGE (0–1) — what fraction of HIGH-WEIGHT signals "
                 "(contribution >= 15 pts, listed in high_weight_contributors) are "
                 "reflected in the summary? Paraphrases count — 'ID document issues' "
                 "covers 'id_fail', 'geographical risks' = geography_risk, etc.\n\n"
                 "2. FAITHFULNESS (0–1) — does the summary stay grounded in the "
                 "contributors? 1.0 = no invented risks. Be semantic and generous: "
                 "'geographical risks' = geography_risk, 'income concerns' = "
                 "income_implausibility, 'name discrepancy' = name_mismatch*, "
                 "'address issues' = address_*, 'ID problems' = id_fail. "
                 "Only deduct if a topic has absolutely NO basis in any contributor.\n\n"
                 "3. HALLUCINATIONS — list only the topics that are in the summary but "
                 "have NO basis whatsoever in any contributor. Natural-language "
                 "paraphrases of real signals are NOT hallucinations.\n\n"
                 "4. SCORE ALIGNMENT (0–1) — does the narrative tone and recommended "
                 "action match the score tier? The score tier is provided in the input. "
                 "low-risk → should recommend approval / no concern; "
                 "medium-risk → should recommend review / caution; "
                 "high-risk → should recommend escalation or rejection. "
                 "1.0 = tone perfectly matches tier, 0.0 = tone directly contradicts tier.\n\n"
                 "Return ONLY valid JSON: "
                 '{"coverage": 0.0-1.0, "faithfulness": 0.0-1.0, "score_alignment": 0.0-1.0, '
                 '"hallucinations": [...], "missing_signals": [...], '
                 '"verdict": "pass"|"warn"|"fail", "rationale": "one sentence"}. '
                 "verdict rules: pass = all three scores >=0.8; "
                 "fail = coverage<0.5 OR faithfulness<0.7 OR score_alignment<0.5; "
                 "warn = otherwise."
             )},
            {"role": "user", "content": json.dumps({
                "risk_score": risk.score,
                "score_tier": score_tier,
                "all_contributors": all_contribs,
                "high_weight_contributors": hw_contribs,
                "summary": explanation.summary,
                "recommended_action": explanation.recommended_action,
            })},
        ],
        json_mode=True, max_tokens=512, agent="eval",
    )

    try:
        raw = result.json or {}
        coverage      = float(raw.get("coverage", 1.0))
        faithfulness  = float(raw.get("faithfulness", 1.0))
        score_align   = float(raw.get("score_alignment", 1.0))
        hallucinations = [str(s) for s in raw.get("hallucinations", [])]
        # backwards compat — LLM might use either key
        if not hallucinations:
            hallucinations = [str(s) for s in raw.get("hallucinated_signals", [])]
        output = EvalOutput(
            faithfulness=faithfulness,
            coverage=coverage,
            score_alignment=score_align,
            hallucinations=hallucinations,
            missing_signals=[str(s) for s in raw.get("missing_signals", [])],
            hallucinated_signals=hallucinations,
            verdict=_enforce_verdict(coverage, faithfulness, score_align,
                                     str(raw.get("verdict", "pass"))),
            rationale=str(raw.get("rationale", "")),
        )
    except Exception:
        output = _deterministic_fallback(explanation, risk, high_weight)

    return output, [result.metric]


def _enforce_verdict(coverage: float, faithfulness: float,
                     score_alignment: float, llm_verdict: str) -> EvalVerdict:
    """Deterministically override LLM verdict when scores contradict it."""
    if coverage < 0.5 or faithfulness < 0.7 or score_alignment < 0.5:
        return "fail"
    if coverage >= 0.8 and faithfulness >= 0.9 and score_alignment >= 0.8:
        return "pass"
    return "warn"


def _score_alignment_deterministic(explanation: ExplanationOutput, score: float) -> float:
    """Heuristic score_alignment: does recommended_action match the score tier?"""
    action = (explanation.recommended_action or "").lower()
    summary = (explanation.summary or "").lower()
    text = action + " " + summary

    if score < TIER_APPROVE:
        # Expect approval language; penalise escalation / rejection language
        if any(w in text for w in ("escalat", "reject", "senior compliance")):
            return 0.3
        if any(w in text for w in ("proceed", "onboard", "approv", "no concern", "low risk")):
            return 1.0
        return 0.7  # neutral text — partial credit

    if score < TIER_ESCALATE:
        # Expect review / caution language
        if any(w in text for w in ("compliance officer", "manual review", "review", "caution")):
            return 1.0
        if any(w in text for w in ("proceed", "onboard")) and "review" not in text:
            return 0.4
        return 0.7

    # High risk — expect escalation or rejection language
    if any(w in text for w in ("escalat", "senior compliance", "reject", "immediately")):
        return 1.0
    if any(w in text for w in ("proceed", "onboard", "approv")):
        return 0.2
    return 0.6


def _deterministic_fallback(explanation, risk, high_weight) -> EvalOutput:
    """Coverage by keyword matching + deterministic score alignment. Used when LLM fails."""
    summary_lower = explanation.summary.lower()
    missing = [
        c.signal for c in high_weight
        if not _signal_mentioned(c.signal, summary_lower)
    ]
    n = len(high_weight)
    coverage     = (n - len(missing)) / n if n else 1.0
    faithfulness = 1.0   # can't detect hallucinations without LLM
    score_align  = _score_alignment_deterministic(explanation, risk.score)

    verdict: EvalVerdict = _enforce_verdict(coverage, faithfulness, score_align, "warn")
    return EvalOutput(
        faithfulness=faithfulness,
        coverage=round(coverage, 2),
        score_alignment=round(score_align, 2),
        hallucinations=[],
        missing_signals=missing,
        hallucinated_signals=[],
        verdict=verdict,
        rationale="Deterministic fallback — LLM call unavailable.",
    )


def _signal_mentioned(signal: str, text: str) -> bool:
    keywords = signal.replace("_", " ").split()
    return any(kw in text for kw in keywords if len(kw) > 3)
