"""
Financial Profile (light, §4.6) — rules + lookup tables, no ML.
Documented heuristics so the score stays explainable.
"""
from __future__ import annotations

from schemas import CustomerInput, FinancialProfileOutput

# Country risk index (0 = low, 1 = high). TODO: load a fuller FATF-aligned table.
COUNTRY_RISK: dict[str, float] = {
    "india": 0.1,
    "cyprus": 0.6,
}


def run_financial_profile(customer: CustomerInput) -> FinancialProfileOutput:
    country = (customer.nationality or "").lower()
    geography_risk = COUNTRY_RISK.get(country, 0.4)

    # Income plausibility — placeholder band. 0 = implausible, 1 = plausible.
    income = customer.declared_income or 0
    income_plausibility = 0.8 if 0 < income < 50_000_000 else 0.4

    employment_risk = 0.2 if customer.declared_employment else 0.5

    return FinancialProfileOutput(
        income_plausibility_score=income_plausibility,
        geography_risk=geography_risk,
        employment_risk=employment_risk,
    )
