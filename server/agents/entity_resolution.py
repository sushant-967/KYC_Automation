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
MAX_AFFIDAVIT_RETRIES = 2   # customer gets 2 attempts to submit a valid affidavit

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
    dob_confirmed = bool(customer.dob and
                         any(_same_date(d, customer.dob) for d in extracted_dobs if d))

    # Exclude the affidavit itself from per-document name matching.
    id_docs = [d for d in extraction.documents
               if d.kind.value != "dual_name_affidavit"]

    name_matches: list[NameMatch] = []
    for d in id_docs:
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

    # ── Remediation: dual name affidavit ────────────────────────────────────
    affidavit_submitted = False
    affidavit_covers = None
    affidavit_attempts = 0
    affidavit_retries_exhausted = False
    docs_required: list[str] = []

    if not name_consistent:
        affidavit_docs = [d for d in extraction.documents
                          if d.kind.value == "dual_name_affidavit"]
        affidavit_attempts = len(affidavit_docs)
        if affidavit_docs:
            affidavit_submitted = True
            # Use the most recently submitted affidavit for the check.
            affidavit_covers = _affidavit_covers_discrepancy(
                customer.full_name, name_matches, affidavit_docs[-1])
            if not affidavit_covers:
                if affidavit_attempts < MAX_AFFIDAVIT_RETRIES:
                    # Still has retries — pause and ask for a better affidavit.
                    docs_required.append("dual_name_affidavit")
                else:
                    # Retries exhausted — proceed with full penalty.
                    affidavit_retries_exhausted = True
        else:
            docs_required.append("dual_name_affidavit")

    # ── Remediation: additional address proof ────────────────────────────────
    addr_additional_submitted = False
    addr_additional_confirmed = None

    if address_confirmed is False:
        # Check if more than one address-bearing doc was submitted; the extras
        # are treated as the re-submitted current-address proof.
        extra_addr = [d for d in extraction.documents
                      if d.kind in _ADDRESS_DOCS
                      and d.kind.value != "aadhaar"]   # Aadhaar is the original
        if extra_addr:
            addr_additional_submitted = True
            best_extra = max(
                _address_score(customer.address or "", str(d.fields.get("address", "")))
                for d in extra_addr
            )
            addr_additional_confirmed = best_extra >= ADDRESS_MATCH_THRESHOLD
            # Promote overall address_confirmed when the fresh proof matches.
            if addr_additional_confirmed:
                address_confirmed = True
                address_score = round(best_extra, 3)
        else:
            docs_required.append("address_proof")

    return EntityResolutionOutput(
        canonical_name=canonical,
        dob_confirmed=dob_confirmed,
        name_matches=name_matches,
        name_consistent=name_consistent,
        address_confirmed=address_confirmed,
        address_match_score=address_score,
        alias_matches=[],
        prior_cases=prior_case_lookup(canonical),
        name_affidavit_submitted=affidavit_submitted,
        name_affidavit_covers_discrepancy=affidavit_covers,
        affidavit_attempts=affidavit_attempts,
        affidavit_retries_exhausted=affidavit_retries_exhausted,
        address_additional_proof_submitted=addr_additional_submitted,
        address_additional_proof_confirmed=addr_additional_confirmed,
        documents_required=docs_required,
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

_DATE_FMTS = [
    "%Y-%m-%d",   # ISO — 1990-06-14
    "%d/%m/%Y",   # Aadhaar / PAN / DL printed format — 14/06/1990
    "%d-%m-%Y",   # dash variant — 14-06-1990
    "%Y/%m/%d",   # rare — 1990/06/14
    "%d %b %Y",   # 14 Jun 1990
    "%d %B %Y",   # 14 June 1990
    "%B %d, %Y",  # June 14, 1990
    "%b %d, %Y",  # Jun 14, 1990
]


def _parse_date(s: str) -> Optional[date]:
    s = s.strip()
    for fmt in _DATE_FMTS:
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # last resort: ISO fromisoformat on first 10 chars
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _same_date(a: str, b: Optional[str]) -> bool:
    if not a or not b:
        return False
    da, db = _parse_date(a), _parse_date(b)
    if da and db:
        return da == db
    return a.strip() == b.strip()


# ── Dual name affidavit verification ────────────────────────────────────────

def _affidavit_covers_discrepancy(
    submitted_name: str,
    name_matches: list,
    affidavit_doc,
) -> bool:
    """Return True if the affidavit explicitly links the submitted name to every
    failing document name.  The extraction agent pulls `name_1` and `name_2`
    (or a `names` list) from the affidavit; we verify both ends are covered."""
    fields = affidavit_doc.fields or {}

    # Collect names the affidavit declares as equivalent.
    affidavit_names: list[str] = []
    if "names" in fields and isinstance(fields["names"], list):
        affidavit_names = [str(n) for n in fields["names"]]
    else:
        for key in ("name_1", "name_2", "name1", "name2", "name"):
            val = fields.get(key)
            if val:
                affidavit_names.append(str(val))

    if not affidavit_names:
        return False

    # Submitted name must appear in the affidavit.
    submitted_covered = any(
        _name_score(submitted_name, n) >= NAME_MATCH_THRESHOLD
        for n in affidavit_names
    )
    if not submitted_covered:
        return False

    # Every failing document name must also appear in the affidavit.
    failing_names = [m.extracted_name for m in name_matches if not m.ok]
    for fname in failing_names:
        if not any(_name_score(fname, n) >= NAME_MATCH_THRESHOLD
                   for n in affidavit_names):
            return False

    return True
