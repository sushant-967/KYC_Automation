"""
Entity Resolution (light, §4.3) — deterministic name canonicalization + DOB
cross-check between the submitted form and the extracted ID. No model.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Callable

from schemas import CustomerInput, EntityResolutionOutput, ExtractionOutput


def run_entity_resolution(
    customer: CustomerInput,
    extraction: ExtractionOutput,
    prior_case_lookup: Callable[[str], list[str]],
) -> EntityResolutionOutput:
    canonical = canonicalize(customer.full_name)

    extracted_dobs = [str(d.fields.get("dob", "")) for d in extraction.documents]
    dob_confirmed = any(_same_date(d, customer.dob) for d in extracted_dobs if d)

    return EntityResolutionOutput(
        canonical_name=canonical,
        dob_confirmed=dob_confirmed,
        alias_matches=[],
        prior_cases=prior_case_lookup(canonical),
    )


def canonicalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _same_date(a: str, b: str) -> bool:
    try:
        return date.fromisoformat(a[:10]) == date.fromisoformat(b[:10])
    except ValueError:
        return a.strip() == b.strip()
