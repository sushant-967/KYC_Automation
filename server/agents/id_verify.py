"""
ID Verification (light, §4.5) — MRZ checksum + expiry + optional face match.
No new model; consumes fields the extraction agent already returned.
"""
from __future__ import annotations

from datetime import date

from schemas import ExtractionOutput, IDVerificationOutput


def run_id_verification(extraction: ExtractionOutput) -> IDVerificationOutput:
    passport = next((d for d in extraction.documents if d.kind.value == "passport"), None)
    mrz_valid = passport.validations.mrz_checksum_ok if passport and passport.validations else None

    expiry_ok = None
    if passport:
        expiry = str(passport.fields.get("expiry", ""))
        try:
            expiry_ok = date.fromisoformat(expiry[:10]) > date.today()
        except ValueError:
            expiry_ok = None

    checks = [c for c in (mrz_valid, expiry_ok) if c is not None]
    if not checks:
        authenticity = "unknown"
    elif all(checks):
        authenticity = "pass"
    else:
        authenticity = "fail"

    return IDVerificationOutput(
        doc_authenticity=authenticity, mrz_valid=mrz_valid, expiry_ok=expiry_ok
    )
