"""
ID Verification (light, §4.5) — format checks on all submitted ID documents.
No new model; consumes Validations the extraction agent already computed.

Checks by document type:
  passport         — MRZ checksum (TODO) + expiry date
  pan              — regex ^[A-Z]{5}[0-9]{4}[A-Z]$
  aadhaar          — Verhoeff checksum over 12 digits
  voter_id /
  driving_license  — presence + name field non-empty (basic completeness)
"""
from __future__ import annotations

from datetime import date

from schemas import ExtractionOutput, IDVerificationOutput


def run_id_verification(extraction: ExtractionOutput) -> IDVerificationOutput:
    checks: list[bool] = []

    mrz_valid = None
    expiry_ok = None
    pan_format_valid = None
    aadhaar_format_valid = None

    for doc in extraction.documents:
        kind = doc.kind.value
        v = doc.validations

        if kind == "passport":
            mrz_valid = v.mrz_checksum_ok if v else None
            expiry = str(doc.fields.get("expiry", ""))
            try:
                expiry_ok = date.fromisoformat(expiry[:10]) > date.today()
                checks.append(expiry_ok)
            except ValueError:
                pass
            if mrz_valid is not None:
                checks.append(mrz_valid)

        elif kind == "pan":
            pan_format_valid = v.pan_regex_ok if v else None
            if pan_format_valid is not None:
                checks.append(pan_format_valid)

        elif kind == "aadhaar":
            aadhaar_format_valid = v.aadhaar_verhoeff_ok if v else None
            if aadhaar_format_valid is not None:
                checks.append(aadhaar_format_valid)

        elif kind in ("voter_id", "driving_license"):
            # Basic completeness — name must be extractable
            name_present = bool(str(doc.fields.get("name", "")).strip())
            checks.append(name_present)

    if not checks:
        authenticity = "unknown"
    elif all(checks):
        authenticity = "pass"
    else:
        authenticity = "fail"

    return IDVerificationOutput(
        doc_authenticity=authenticity,
        mrz_valid=mrz_valid,
        expiry_ok=expiry_ok,
        pan_format_valid=pan_format_valid,
        aadhaar_format_valid=aadhaar_format_valid,
    )
