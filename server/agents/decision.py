"""
Decision Agent (light, §4.9) — threshold rules over the deterministic score.
    score < 30        → approve
    30 ≤ score < 70   → review   (requires human)
    score ≥ 70        → escalate (requires human)

Hard blocks (override score — these three checks must pass before approval):
    - DOB not confirmed                              → escalate
    - Name unresolved (inconsistent + no valid affidavit) → at minimum review
    - Address unconfirmed (submitted but no proof)   → at minimum review
"""
from __future__ import annotations

from typing import Optional

from schemas import DecisionOutput, EntityResolutionOutput, RiskOutput

THRESHOLDS = {"approve": 30, "escalate": 70}


def run_decision(risk: RiskOutput,
                 entity: Optional[EntityResolutionOutput] = None) -> DecisionOutput:
    # ── Hard blocks — critical identity checks that must pass ────────────────
    if entity is not None:
        # 1. DOB must be confirmed by at least one document.
        if not entity.dob_confirmed:
            return DecisionOutput(
                decision="escalate", requires_human=True,
                reasons=["Date of birth could not be confirmed from any submitted document — manual verification required."],
            )

        # 2. Name must be either consistent across docs or resolved by a valid
        #    affidavit. Unresolved name cannot be approved.
        name_resolved = entity.name_consistent or entity.name_affidavit_covers_discrepancy
        if not name_resolved:
            if entity.affidavit_retries_exhausted:
                name_reason = (
                    f"Name inconsistency across documents — dual-name affidavit was submitted "
                    f"{entity.affidavit_attempts} time(s) but did not cover all failing name variants."
                )
            elif entity.name_affidavit_submitted:
                name_reason = (
                    "Name inconsistency across documents — the affidavit submitted does not cover all "
                    "name variants; please resubmit a notarized dual-name affidavit listing all name forms."
                )
            else:
                name_reason = (
                    "Name inconsistency across documents — a notarized dual-name affidavit is required "
                    "to bridge the discrepancy between the application name and the ID documents."
                )
            if risk.score >= THRESHOLDS["escalate"]:
                return DecisionOutput(decision="escalate", requires_human=True, reasons=[name_reason])
            return DecisionOutput(decision="review", requires_human=True, reasons=[name_reason])

        # 3. Address — if the customer supplied an address on the form it must
        #    be confirmed by at least one document or additional proof.
        address_ok = (
            entity.address_confirmed is None          # no address on form — not required
            or entity.address_confirmed is True
            or entity.address_additional_proof_confirmed is True
        )
        if not address_ok:
            if entity.address_additional_proof_submitted:
                addr_reason = (
                    "Address on the application does not match the additional proof submitted — "
                    "please provide a utility bill or bank statement (< 3 months) matching the declared address."
                )
            else:
                addr_reason = (
                    "Address on the application could not be confirmed from any submitted document — "
                    "please provide a utility bill or bank statement dated within 3 months."
                )
            if risk.score >= THRESHOLDS["escalate"]:
                return DecisionOutput(decision="escalate", requires_human=True, reasons=[addr_reason])
            return DecisionOutput(decision="review", requires_human=True, reasons=[addr_reason])

    # ── Risk-based decision ──────────────────────────────────────────────────
    s = risk.score
    if s < THRESHOLDS["approve"]:
        return DecisionOutput(decision="approve", requires_human=False)
    score_reason = f"Risk score {s}/100 exceeds the approval threshold — compliance officer review required."
    if s < THRESHOLDS["escalate"]:
        return DecisionOutput(decision="review", requires_human=True, reasons=[score_reason])
    return DecisionOutput(decision="escalate", requires_human=True, reasons=[score_reason])
