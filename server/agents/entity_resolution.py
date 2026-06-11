"""
Entity Resolution (light, §4.3) — answers "is this the same person across all
data points the customer gave us?" No model. Deterministic checks only.

Three signals:
  - DOB cross-check: submitted form vs each extracted ID.
  - Per-document name match: submitted full_name vs each doc's extracted name,
    via a transliteration-tolerant token-set match (Dice coefficient over
    canonicalized tokens, with initials matched by prefix). Tuned for Indian
    names where "Rajesh Kr." legitimately equals "Rajesh Kumar".
  - Address match: submitted address vs Aadhaar / driving_license / voter_id /
    address_proof. Stopwords stripped before scoring.

Risk consumes the booleans; the raw scores are surfaced so HITL can override.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Callable, Optional

from schemas import (CustomerInput, DocumentKind, EntityResolutionOutput,
                     ExtractionOutput, NameMatch)

NAME_MATCH_THRESHOLD = 0.70
ADDRESS_MATCH_THRESHOLD = 0.60

# Address-noise tokens that carry no identity signal — strip before scoring.
_ADDRESS_STOPWORDS = {
    "road", "rd", "street", "st", "lane", "ln", "ave", "avenue", "marg",
    "near", "opp", "opposite", "behind", "next", "to", "of",
    "house", "flat", "block", "sector", "apt", "apartment", "no", "number",
    "the", "and", "po", "dist", "district", "taluk", "tehsil", "via", "pin",
}

# Doc kinds known to carry a usable address field (matches rocm/prompts/extraction.md).
_ADDRESS_DOCS = {
    DocumentKind.aadhaar, DocumentKind.voter_id,
    DocumentKind.driving_license, DocumentKind.address_proof,
}


def run_entity_resolution(
    customer: CustomerInput,
    extraction: ExtractionOutput,
    prior_case_lookup: Callable[[str], list[str]],
) -> EntityResolutionOutput:
    canonical = canonicalize(customer.full_name)

    extracted_dobs = [str(d.fields.get("dob", "")) for d in extraction.documents]
    dob_confirmed = any(_same_date(d, customer.dob) for d in extracted_dobs if d)

    name_matches: list[NameMatch] = []
    for d in extraction.documents:
        extracted = str(d.fields.get("name", "")).strip()
        if not extracted:
            continue
        score = _name_score(customer.full_name, extracted)
        name_matches.append(NameMatch(
            doc_kind=d.kind, extracted_name=extracted,
            score=round(score, 3), ok=score >= NAME_MATCH_THRESHOLD,
        ))
    name_consistent = all(m.ok for m in name_matches) if name_matches else True

    address_confirmed, address_score = _address_match(customer.address, extraction)

    return EntityResolutionOutput(
        canonical_name=canonical,
        dob_confirmed=dob_confirmed,
        name_matches=name_matches,
        name_consistent=name_consistent,
        address_confirmed=address_confirmed,
        address_match_score=address_score,
        alias_matches=[],
        prior_cases=prior_case_lookup(canonical),
    )


# ── Canonicalization ────────────────────────────────────────────────────────

def canonicalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in canonicalize(s).split(" ") if t]


# ── Name match (Dice over tokens, initials by prefix) ───────────────────────

def _name_score(a: str, b: str) -> float:
    """Symmetric token-set match. Tokens ≤ 2 chars are treated as initials
    and match any longer token starting with the same letter — handles the
    "Rajesh Kr." ≡ "Rajesh Kumar" case common on Indian docs."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0

    # Greedy match: each token consumed at most once.
    remaining = list(tb)
    matched = 0
    for tok in ta:
        for j, other in enumerate(remaining):
            if _token_eq(tok, other):
                matched += 1
                remaining.pop(j)
                break

    # Dice: 2|A∩B| / (|A|+|B|). Penalizes both missing and extra tokens.
    return (2 * matched) / (len(ta) + len(tb))


def _token_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    # 1-2 char tokens are initials/abbreviations — match by first letter
    # ("Kr." → "Kumar", "R." → "Rajesh"). Prefix would miss the "Kr/Kumar" case.
    if len(short) <= 2 and short[0] == long[0]:
        return True
    return False


# ── Address match (Dice over tokens, with stopwords) ────────────────────────

def _address_match(
    submitted: Optional[str], extraction: ExtractionOutput,
) -> tuple[Optional[bool], Optional[float]]:
    if not submitted or not submitted.strip():
        return None, None

    candidates: list[str] = []
    for d in extraction.documents:
        if d.kind not in _ADDRESS_DOCS:
            continue
        addr = str(d.fields.get("address", "")).strip()
        if addr:
            candidates.append(addr)
    if not candidates:
        return None, None

    best = max(_address_score(submitted, c) for c in candidates)
    return best >= ADDRESS_MATCH_THRESHOLD, round(best, 3)


def _address_score(a: str, b: str) -> float:
    ta = [t for t in _tokens(a) if t not in _ADDRESS_STOPWORDS]
    tb = [t for t in _tokens(b) if t not in _ADDRESS_STOPWORDS]
    if not ta or not tb:
        return 0.0
    inter = len(set(ta) & set(tb))
    return (2 * inter) / (len(set(ta)) + len(set(tb)))


# ── DOB ─────────────────────────────────────────────────────────────────────

def _same_date(a: str, b: str) -> bool:
    try:
        return date.fromisoformat(a[:10]) == date.fromisoformat(b[:10])
    except ValueError:
        return a.strip() == b.strip()
