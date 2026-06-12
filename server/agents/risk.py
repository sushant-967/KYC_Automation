"""
Risk Aggregation (light, §4.7) — DETERMINISTIC weighted scoring.

The score must be reproducible from the inputs; that reproducibility is what makes
the explainability honest. Weights are explicit constants (§4.7).
"""
from __future__ import annotations

from schemas import (EntityResolutionOutput, FinancialProfileOutput,
                     IDVerificationOutput, RiskContributor, RiskOutput,
                     ScreeningOutput)

W = {
    "sanctions_hit": 50,
    "pep_hit": 30,
    "adverse_media_base": 20,  # × severity multiplier
    "id_fail": 30,
    "name_mismatch": 25,
    "address_unverified": 15,
    "geo_risk_max": 10,
    "income_implausibility_max": 10,
}
SEVERITY_MULT = {"low": 0.5, "medium": 0.75, "high": 1.0}


def run_risk(entity: EntityResolutionOutput, screening: ScreeningOutput,
             idv: IDVerificationOutput, fin: FinancialProfileOutput) -> RiskOutput:
    c: list[RiskContributor] = []

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
        if idv.aadhaar_format_valid is False:
            failed_checks.append("aadhaar_verhoeff")
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

    geo = fin.geography_risk * W["geo_risk_max"]
    if geo > 0:
        c.append(RiskContributor(signal="geography_risk", weight=W["geo_risk_max"],
                                 value=fin.geography_risk, contribution=geo))

    income_penalty = (1 - fin.income_plausibility_score) * W["income_implausibility_max"]
    if income_penalty > 0:
        c.append(RiskContributor(signal="income_implausibility", weight=W["income_implausibility_max"],
                                 value=fin.income_plausibility_score, contribution=income_penalty))

    score = min(100, round(sum(x.contribution for x in c)))
    return RiskOutput(score=score, contributors=c)
