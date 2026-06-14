"""
Financial Profile (DEEP ★, §4.6) — Llama 3.3 70B + FATF-aligned rules.

Employment is classified into one of 8 AML risk categories by Llama, with a
deterministic keyword fallback for when the model is unavailable. Income
plausibility is scored against the category's expected INR band — the formula
maps each outcome (within-band, moderately-above, suspiciously-above, below-min)
to a calibrated 0–1 score. Geography risk uses a ~40-country FATF-aligned table.
"""
from __future__ import annotations

import json
from typing import Optional

from schemas import CustomerInput, FinancialProfileOutput, GpuCallMetric
from vllm_client import VllmClient

_LAKH  = 100_000
_CRORE = 10_000_000

# ── Geography: FATF-aligned country risk ────────────────────────────────────

COUNTRY_RISK: dict[str, float] = {
    # FATF blacklist (call for action) ----------------------------------------
    "myanmar":                    0.95,
    "iran":                       0.95,
    "north korea":                0.95,
    "dprk":                       0.95,

    # FATF grey list (increased monitoring, FATF Feb-2024 plenary) ------------
    "algeria":                    0.75,
    "angola":                     0.75,
    "bulgaria":                   0.75,
    "burkina faso":               0.75,
    "cameroon":                   0.75,
    "croatia":                    0.75,
    "democratic republic of congo": 0.75,
    "haiti":                      0.75,
    "kenya":                      0.75,
    "laos":                       0.75,
    "lao pdr":                    0.75,
    "mali":                       0.75,
    "monaco":                     0.75,
    "mozambique":                 0.75,
    "namibia":                    0.75,
    "nigeria":                    0.75,
    "south africa":               0.75,
    "south sudan":                0.75,
    "syria":                      0.75,
    "tanzania":                   0.75,
    "vietnam":                    0.75,
    "venezuela":                  0.75,
    "yemen":                      0.75,

    # OFAC bilateral high-risk -------------------------------------------------
    "russia":                     0.80,
    "belarus":                    0.75,
    "cuba":                       0.80,
    "sudan":                      0.80,
    "eritrea":                    0.75,

    # Tax havens / secrecy jurisdictions (OECD/FSF list) ----------------------
    "cayman islands":             0.65,
    "panama":                     0.65,
    "british virgin islands":     0.65,
    "bvi":                        0.65,
    "samoa":                      0.65,
    "vanuatu":                    0.65,
    "marshall islands":           0.65,
    "seychelles":                 0.65,
    "liechtenstein":              0.65,
    "cyprus":                     0.60,
    "malta":                      0.55,

    # Standard low-risk (FATF founding members, strong AML) -------------------
    "india":                      0.10,
    "united states":              0.10,
    "usa":                        0.10,
    "united kingdom":             0.10,
    "uk":                         0.10,
    "germany":                    0.10,
    "france":                     0.10,
    "japan":                      0.10,
    "australia":                  0.10,
    "canada":                     0.10,
    "singapore":                  0.10,
    "new zealand":                0.10,
    "switzerland":                0.20,

    # Standard medium-risk -----------------------------------------------------
    "brazil":                     0.20,
    "mexico":                     0.25,
    "indonesia":                  0.20,
    "turkey":                     0.25,
    "egypt":                      0.25,
    "pakistan":                   0.30,
    "bangladesh":                 0.25,
    "sri lanka":                  0.25,
    "nepal":                      0.25,
    "uae":                        0.30,
    "united arab emirates":       0.30,
    "qatar":                      0.25,
    "saudi arabia":               0.25,
    "china":                      0.20,
    "hong kong":                  0.20,
}

_DEFAULT_COUNTRY_RISK = 0.35  # unknown nationality — elevated but not suspicious

# ── Employment: 8 AML risk categories ───────────────────────────────────────
# (risk_score, band_min_inr, band_max_inr or None for no cap)

_EMP_CATEGORIES: dict[str, tuple[float, float, float | None]] = {
    "pep_adjacent":           (0.85,  0,              25 * _LAKH),
    "cash_intensive":         (0.75,  1 * _LAKH,      None),
    "regulated_professional": (0.40,  5 * _LAKH,      5 * _CRORE),
    "senior_corporate":       (0.25, 20 * _LAKH,      5 * _CRORE),
    "corporate":              (0.15,  2.4 * _LAKH,    1 * _CRORE),
    "public_sector":          (0.10,  2.4 * _LAKH,   30 * _LAKH),
    "self_employed":          (0.40,  1 * _LAKH,      2 * _CRORE),
    "unemployed":             (0.50,  0,               5 * _LAKH),
}

_DEFAULT_CATEGORY = "corporate"
VALID_CATEGORIES  = set(_EMP_CATEGORIES)

# Contradiction thresholds: income above this level is logically impossible for the category.
# Distinct from the ≥3× implausibility ladder — this is a hard logical impossibility.
_CONTRADICTION_THRESHOLDS: dict[str, tuple[float, str]] = {
    "public_sector": (
        50 * _LAKH,
        "Government pay-commission scales cap at ~₹30L — declared income is logically "
        "inconsistent with public sector employment.",
    ),
    "unemployed": (
        15 * _LAKH,
        "Significant declared income from a person who is unemployed/retired — "
        "source of funds requires explicit explanation (undisclosed business / asset income?).",
    ),
}

# More-specific patterns must come before generic ones within each group.
_KEYWORD_MAP: list[tuple[str, list[str]]] = [
    ("pep_adjacent", [
        "minister", "politician", "parliamentarian", "member of parliament",
        " mp ", " mla ", "senator", "governor", "chief minister",
        "judge", "magistrate", "justice", "army general", "brigadier",
        "major general", "lieutenant general", "air marshal", "admiral",
        "senior government", "senior govt", " ias ", " ips ", " ifs ",
        "bureaucrat", "collector", "secretary to govt",
    ]),
    ("cash_intensive", [
        "jeweller", "jewelry", "jewellery", "bullion", "gold dealer",
        "real estate", "property dealer", "realty", "pawn shop", "pawn broker",
        "money lender", "moneylender", "chit fund", "casino", "gaming",
        "liquor", "wine shop", "bar owner", "scrap dealer",
    ]),
    ("regulated_professional", [
        "chartered accountant", " ca ", "company secretary", " cs ",
        "lawyer", "advocate", "attorney", "solicitor", "barrister",
        "legal ", "law firm",
    ]),
    ("senior_corporate", [
        "chief executive", " ceo", "chief financial", " cfo",
        "chief operating", " coo", "managing director", " md ",
        "vice president", " vp ", " svp", " evp",
        "board member", "board director",
    ]),
    ("public_sector", [
        "government employee", "govt employee", "central government",
        "state government", " psu ", "public sector",
        "railway", "post office", "municipality", "panchayat",
        "teacher", "professor", "lecturer",
        "police constable", "constable", "sub inspector",
        "armed forces", "paramilitary", "defence ", "defense ",
    ]),
    ("unemployed", [
        "student", "homemaker", "housewife", "house wife", "retired",
        "pensioner", "unemployed", "not employed", "no employment",
        "looking for work", "job seeker",
    ]),
    ("self_employed", [
        "consultant", "freelancer", "freelance", "contractor",
        "self employed", "self-employed", "business owner", "entrepreneur",
        "proprietor", "trader", "merchant", "shopkeeper", "vendor",
        "small business",
    ]),
    ("corporate", [
        "engineer", "software", "developer", "programmer", "analyst",
        "manager", "executive", "officer", "associate", "specialist",
        "doctor", "physician", "dentist", "nurse", "pharmacist",
        "accountant", "auditor", "architect", "designer",
        "marketing", "sales", " hr ", "finance ", "operations",
        "employee", "salaried", "employed",
    ]),
]

_SYSTEM_PROMPT = (
    "You are an AML financial profiling analyst for a KYC pipeline. "
    "Classify the customer's employment into exactly one of these 8 categories "
    "and flag any AML concerns. "
    "You MUST respond with valid json only — no markdown, no prose outside json.\n\n"
    "Categories:\n"
    "  pep_adjacent           — politician, minister, judge, army general, senior govt official\n"
    "  cash_intensive         — jeweller, real estate dealer, pawn shop, money lender, casino\n"
    "  regulated_professional — lawyer, CA/chartered accountant, company secretary\n"
    "  senior_corporate       — CEO, CFO, MD, Director, VP (C-suite / board level)\n"
    "  corporate              — software engineer, manager, doctor, salaried employee\n"
    "  public_sector          — govt employee, PSU, teacher, non-PEP armed forces\n"
    "  self_employed          — consultant, freelancer, trader, small business owner\n"
    "  unemployed             — student, homemaker, retired, not employed\n\n"
    'Output ONLY this json shape: '
    '{"category": "<one of the 8 above>", '
    '"red_flags": ["<flag>"], '
    '"rationale": "<one sentence explaining the category and any AML concern>"}'
)


# ── Public API ───────────────────────────────────────────────────────────────

async def run_financial_profile(
    customer: CustomerInput,
    vllm: VllmClient,
) -> tuple[FinancialProfileOutput, list[GpuCallMetric]]:
    # 1. Geography risk — deterministic lookup
    country = (customer.nationality or "").lower().strip()
    geography_risk = COUNTRY_RISK.get(country, _DEFAULT_COUNTRY_RISK)

    # 2. LLM employment classification
    gpu: list[GpuCallMetric] = []
    llm_category: Optional[str] = None
    llm_rationale: Optional[str] = None

    try:
        result = await vllm.reason(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps({
                    "declared_employment": customer.declared_employment or "",
                    "declared_income_inr": customer.declared_income,
                    "nationality":         customer.nationality or "",
                })},
            ],
            json_mode=True,
            max_tokens=256, agent="financialProfile",
        )
        gpu.append(result.metric)
        raw = result.json if isinstance(result.json, dict) else {}
        candidate = str(raw.get("category", "")).strip().lower().replace("-", "_")
        if candidate in VALID_CATEGORIES:
            llm_category = candidate
        llm_rationale = str(raw.get("rationale", "")).strip() or None
    except Exception:
        pass  # deterministic keyword fallback below

    # 3. Keyword fallback when LLM is unavailable or returned an invalid category
    if llm_category is None:
        llm_category = (_keyword_category(customer.declared_employment)
                        if customer.declared_employment else _DEFAULT_CATEGORY)

    # 4. Deterministic scoring from category + income band
    emp_risk, band_min, band_max = _EMP_CATEGORIES[llm_category]
    plausibility, band_ok = _income_plausibility(customer.declared_income, llm_category)

    # 5. Contradiction check — is the income logically impossible for this employment type?
    contradiction, contradiction_reason = _check_contradiction(
        customer.declared_income, llm_category, band_max)

    return FinancialProfileOutput(
        income_plausibility_score=plausibility,
        geography_risk=geography_risk,
        employment_risk=emp_risk,
        employment_category=llm_category,
        income_band_min=band_min,
        income_band_max=band_max,
        income_band_ok=band_ok,
        financial_risk_rationale=llm_rationale,
        employment_contradiction=contradiction,
        contradiction_reason=contradiction_reason,
    ), gpu


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_contradiction(
    income: Optional[float],
    category: str,
    band_max: Optional[float],
) -> tuple[bool, Optional[str]]:
    """Return (is_contradiction, reason).

    Two triggers:
    1. Category-specific hard cap (public_sector, unemployed) — income above threshold
       is a logical impossibility, not just implausibility.
    2. Generic: income ≥ 5× band_max for categories with a finite cap — at this level
       the mismatch crosses from 'high earner' into 'structurally impossible'.
    """
    if income is None or income == 0:
        return False, None

    # Category-specific hard thresholds
    if category in _CONTRADICTION_THRESHOLDS:
        threshold, reason = _CONTRADICTION_THRESHOLDS[category]
        if income >= threshold:
            return True, reason

    # Generic 2× band ceiling check — income more than 2× the ceiling needs explanation
    if band_max is not None and income >= band_max * 2:
        cap_str = (f"₹{band_max/_CRORE:.1f}Cr" if band_max >= _CRORE
                   else f"₹{band_max/_LAKH:.0f}L")
        return True, (
            f"Declared income is {income/band_max:.1f}× above the {cap_str} ceiling "
            f"for {category.replace('_', ' ')} employment — "
            "source of additional income requires explicit documentation."
        )

    return False, None


def _keyword_category(employment: str) -> str:
    text = employment.lower()
    for category, keywords in _KEYWORD_MAP:
        if any(kw in text for kw in keywords):
            return category
    return _DEFAULT_CATEGORY


def _income_plausibility(
    income: Optional[float],
    category: str,
) -> tuple[float, Optional[bool]]:
    """
    Plausibility ladder (AML rationale):
      no income declared   → 0.50  (neutral; cannot assess)
      within band          → 0.90  (plausible)
      above band < 3× max  → 0.70  (high earner; EDD warranted)
      above band ≥ 3× max  → 0.35  (suspicious over-declaration)
      below band min       → 0.65  (under-reporting or new joiner)
      cash_intensive       → 0.90  (no cap — open-ended business income)
      pep_adjacent > cap   → 0.70  (assets expected beyond official salary)
    """
    _, band_min, band_max = _EMP_CATEGORIES[category]

    if income is None or income == 0:
        return 0.50, None

    if band_min > 0 and income < band_min:
        return 0.65, False

    if band_max is None:          # cash_intensive — no upper bound
        return 0.90, True

    if income <= band_max:
        return 0.90, True

    # Income exceeds band ceiling
    if category == "pep_adjacent":
        return 0.70, False        # PEPs legitimately hold assets above official salary

    multiplier = income / band_max
    if multiplier < 3.0:
        return 0.70, False
    return 0.35, False
