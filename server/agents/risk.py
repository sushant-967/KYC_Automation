"""
Risk Aggregation (light, §4.7) — DETERMINISTIC weighted scoring.

The score must be reproducible from the inputs; that reproducibility is what makes
the explainability honest. Weights are explicit constants (§4.7).
"""
from __future__ import annotations

from typing import Optional
from schemas import (EntityResolutionOutput, FinancialProfileOutput,
                     GuardrailViolation, IDVerificationOutput, RiskContributor,
                     RiskOutput, ScreeningOutput)

W = {
    # ── Baseline (always present — residual KYC uncertainty) ─────────────────
    "baseline":                        5,  # no customer is zero-risk; screening DBs are not omniscient
    # ── Adversarial document (guardrail-detected injection / jailbreak) ───────
    "adversarial_document":           50,  # someone tried to manipulate the KYC system → auto-escalate
    # ── Catastrophic (must escalate alone) ──────────────────────────────────
    "sanctions_hit":                  75,  # confirmed watchlist match → always escalate
    # ── High (review alone; two together escalate) ───────────────────────────
    "pep_hit":                        40,  # politically exposed person
    "id_fail":                        35,  # document authenticity failure
    "name_mismatch":                  30,  # name inconsistency — aligns with hard-block threshold
    "employment_income_contradiction": 25, # logically impossible income for job type
    # ── Medium (contribute meaningfully in combination) ──────────────────────
    "adverse_media_base":             20,  # × severity multiplier (low 0.5 · med 0.75 · high 1.0)
    "employment_category_risk_max":   20,  # × employment_risk score (fires if ≥ 0.30)
    "address_unverified":             15,  # address could not be confirmed
    # ── Low (fine-tuning signals) ────────────────────────────────────────────
    "geo_risk_max":                   10,  # × country risk (0–1, FATF-aligned)
    "income_implausibility_max":      10,  # × (1 − plausibility_score)
}
# Design invariants (enforce these if weights change):
#   sanctions alone (75)           → ESCALATE  (≥ 70)
#   pep alone (40)                 → REVIEW    (30–69)
#   id_fail alone (35)             → REVIEW
#   name_mismatch alone (30)       → REVIEW    (aligns with hard block)
#   pep + id_fail (75)             → ESCALATE
#   all financial at max (~57)     → REVIEW at worst
SEVERITY_MULT   = {"low": 0.5, "medium": 0.75, "high": 1.0}
EMP_RISK_THRESHOLD = 0.30   # corporate (0.15) and public_sector (0.10) do not fire


def run_risk(entity: EntityResolutionOutput, screening: ScreeningOutput,
             idv: IDVerificationOutput, fin: FinancialProfileOutput,
             guardrail_flags: Optional[list[GuardrailViolation]] = None) -> RiskOutput:
    # Every customer carries residual KYC risk — screening databases are not
    # omniscient, documents have OCR error rates, and circumstances change.
    # RBI / FATF risk-based approach requires Low / Medium / High tiers, not zero.
    c: list[RiskContributor] = [
        RiskContributor(signal="baseline_kyc_risk", weight=W["baseline"],
                        value="inherent", contribution=W["baseline"]),
    ]

    if screening.sanctions.hit:
        c.append(RiskContributor(signal="sanctions_hit", weight=W["sanctions_hit"],
                                 value=True, contribution=W["sanctions_hit"]))
    if screening.pep.hit:
        c.append(RiskContributor(signal="pep_hit", weight=W["pep_hit"],
                                 value=True, contribution=W["pep_hit"]))
    if screening.adverse_media.hit:
        sev = screening.adverse_media.severity.value if screening.adverse_media.severity else "low"
        c.append(RiskContributor(signal="adverse_media", weight=W["adverse_media_base"],
                                 value=sev, contribution=W["adverse_media_base"] * SEVERITY_MULT[sev]))
    if idv.doc_authenticity == "fail":
        failed_checks = []
        if idv.pan_format_valid is False:
            failed_checks.append("pan_format")
        if idv.mrz_valid is False:
            failed_checks.append("passport_mrz")
        if idv.expiry_ok is False:
            failed_checks.append("passport_expired")
        c.append(RiskContributor(signal="id_fail", weight=W["id_fail"],
                                 value=failed_checks or ["unknown"],
                                 contribution=W["id_fail"]))
    if not entity.name_consistent:
        failing = [f"{m.doc_kind.value}:{m.score:.2f}" for m in entity.name_matches if not m.ok]
        if entity.name_affidavit_covers_discrepancy:
            # Affidavit submitted and verified — residual 5-pt flag for audit trail.
            c.append(RiskContributor(signal="name_mismatch_affidavit_resolved",
                                     weight=W["name_mismatch"], value=failing,
                                     contribution=5))
        elif entity.affidavit_retries_exhausted:
            # Customer submitted affidavit multiple times but it never covered the mismatch.
            c.append(RiskContributor(signal="name_mismatch_affidavit_exhausted",
                                     weight=W["name_mismatch"], value=failing,
                                     contribution=W["name_mismatch"]))
        elif entity.name_affidavit_submitted:
            # Affidavit submitted but doesn't cover all failing names — partial credit.
            c.append(RiskContributor(signal="name_mismatch_affidavit_insufficient",
                                     weight=W["name_mismatch"], value=failing,
                                     contribution=W["name_mismatch"] * 0.6))
        else:
            # No affidavit at all — full penalty.
            c.append(RiskContributor(signal="name_mismatch", weight=W["name_mismatch"],
                                     value=failing, contribution=W["name_mismatch"]))

    # Address present on form but no submitted proof confirmed it.
    if entity.address_confirmed is False:
        if entity.address_additional_proof_confirmed:
            # Additional proof submitted and matched — residual 3-pt audit flag.
            c.append(RiskContributor(signal="address_resolved_by_additional_proof",
                                     weight=W["address_unverified"],
                                     value=entity.address_match_score, contribution=3))
        elif entity.address_additional_proof_submitted:
            # Additional proof submitted but still doesn't match — partial.
            c.append(RiskContributor(signal="address_additional_proof_mismatch",
                                     weight=W["address_unverified"],
                                     value=entity.address_match_score,
                                     contribution=W["address_unverified"] * 0.7))
        else:
            # No additional proof provided — full penalty.
            c.append(RiskContributor(signal="address_unverified",
                                     weight=W["address_unverified"],
                                     value=entity.address_match_score,
                                     contribution=W["address_unverified"]))

    # Geography risk: only fire for medium-risk countries (> 0.15).
    # Standard low-risk jurisdictions (India, USA, UK … = 0.10) are clean baseline
    # and produce 1-pt noise that would appear on every Indian KYC case.
    geo = fin.geography_risk * W["geo_risk_max"]
    if fin.geography_risk > 0.15:
        c.append(RiskContributor(signal="geography_risk", weight=W["geo_risk_max"],
                                 value=fin.geography_risk, contribution=round(geo, 1)))

    # Income implausibility: only fire when plausibility is below 0.80.
    # A perfectly within-band income (plausibility = 0.90) yields penalty = 1.0 pt —
    # noise that would appear on every clean case.
    income_penalty = (1 - fin.income_plausibility_score) * W["income_implausibility_max"]
    if income_penalty >= 2.0:
        c.append(RiskContributor(signal="income_implausibility", weight=W["income_implausibility_max"],
                                 value=fin.income_plausibility_score, contribution=round(income_penalty, 1)))

    # Employment category risk — fires for inherently high-risk occupation types.
    if fin.employment_risk >= EMP_RISK_THRESHOLD:
        emp_contribution = round(fin.employment_risk * W["employment_category_risk_max"], 1)
        c.append(RiskContributor(
            signal="employment_category_risk",
            weight=W["employment_category_risk_max"],
            value={"category": fin.employment_category, "risk_score": fin.employment_risk},
            contribution=emp_contribution,
        ))

    # Employment-income contradiction — logically impossible combination.
    if fin.employment_contradiction:
        c.append(RiskContributor(
            signal="employment_income_contradiction",
            weight=W["employment_income_contradiction"],
            value={"reason": fin.contradiction_reason, "category": fin.employment_category},
            contribution=W["employment_income_contradiction"],
        ))

    # Adversarial document: injection or jailbreak attempt detected by guardrails.
    # +50 pts auto-pushes any case into REVIEW or ESCALATE — someone tampered.
    if guardrail_flags:
        adversarial = [
            f for f in guardrail_flags
            if f.level == "critical" and
               ("injection" in f.check or "jailbreak" in f.check)
        ]
        if adversarial:
            checks = ", ".join(sorted({f.check for f in adversarial}))
            c.append(RiskContributor(
                signal="adversarial_document",
                weight=W["adversarial_document"],
                value=checks,
                contribution=W["adversarial_document"],
            ))

    score = min(100, round(sum(x.contribution for x in c)))
    return RiskOutput(score=score, contributors=c)
