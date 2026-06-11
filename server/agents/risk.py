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
        c.append(RiskContributor(signal="id_fail", weight=W["id_fail"],
                                 value="fail", contribution=W["id_fail"]))
    if not entity.name_consistent:
        failing = [f"{m.doc_kind.value}:{m.score:.2f}" for m in entity.name_matches if not m.ok]
        c.append(RiskContributor(signal="name_mismatch", weight=W["name_mismatch"],
                                 value=failing, contribution=W["name_mismatch"]))
    # Address present on form but no submitted proof confirmed it.
    if entity.address_confirmed is False:
        c.append(RiskContributor(signal="address_unverified", weight=W["address_unverified"],
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
