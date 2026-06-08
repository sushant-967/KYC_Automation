"""Intake Agent (light, §4.1) — validate + normalize. No model."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from schemas import Submission, IntakeOutput


def run_intake(case_id: str, raw: dict | Submission) -> IntakeOutput:
    sub = raw if isinstance(raw, Submission) else Submission.model_validate(raw)
    customer = sub.customer.model_copy(
        update={"full_name": re.sub(r"\s+", " ", sub.customer.full_name).strip()}
    )
    return IntakeOutput(
        case_id=case_id,
        customer=customer,
        documents=sub.documents,
        normalized_at=datetime.now(timezone.utc).isoformat(),
    )
