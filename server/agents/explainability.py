"""
Explainability Agent (DEEP ★, §4.8) — Llama 3.3 70B (:8001).

Builds a deterministic causal DAG of evidence nodes (raw value → risk signal →
decision) from the upstream agent outputs, then asks Llama for a 1-sentence
summary and recommended action. The DAG is the primary audit artifact — every
risk point is linked to the agent that produced it, the raw value it read, and
the rule it triggered. Llama writes the prose; the DAG is the ground truth.
"""
from __future__ import annotations

import json
from typing import Optional

from schemas import (
    CausalEdge, DecisionOutput, EvidenceCard, EvidenceNode,
    EntityResolutionOutput, ExplanationOutput, FinancialProfileOutput,
    GpuCallMetric, IDVerificationOutput, RiskOutput, ScreeningOutput, Severity,
)
from vllm_client import VllmClient

NAME_MATCH_THRESHOLD = 0.70
ADDRESS_MATCH_THRESHOLD = 0.60

_SIGNAL_AGENT = {
    "sanctions_hit":                        "screening",
    "pep_hit":                              "screening",
    "adverse_media":                        "screening",
    "id_fail":                              "idVerification",
    "name_mismatch":                        "entityResolution",
    "name_mismatch_affidavit_resolved":     "entityResolution",
    "name_mismatch_affidavit_insufficient": "entityResolution",
    "name_mismatch_affidavit_exhausted":    "entityResolution",
    "address_unverified":                   "entityResolution",
    "address_resolved_by_additional_proof": "entityResolution",
    "address_additional_proof_mismatch":    "entityResolution",
    "geography_risk":                       "financialProfile",
    "income_implausibility":                "financialProfile",
}

_SIGNAL_RULE = {
    "sanctions_hit":
        "weight=50 applied when sanctions.hit=True",
    "pep_hit":
        "weight=30 applied when pep.hit=True",
    "adverse_media":
        "weight=20 × severity_mult (low:0.5 · med:0.75 · high:1.0)",
    "id_fail":
        "weight=30 applied when doc_authenticity=fail",
    "name_mismatch":
        "weight=25 when name_consistent=False and no affidavit submitted",
    "name_mismatch_affidavit_resolved":
        "5-pt residual audit flag — affidavit covers the discrepancy",
    "name_mismatch_affidavit_insufficient":
        "weight=25 × 0.6 — affidavit submitted but does not cover all name variants",
    "name_mismatch_affidavit_exhausted":
        "weight=25 — max retries exhausted, affidavit never covered the mismatch",
    "address_unverified":
        "weight=15 when address_confirmed=False and no additional proof provided",
    "address_resolved_by_additional_proof":
        "3-pt residual flag — additional proof confirmed the address",
    "address_additional_proof_mismatch":
        "weight=15 × 0.7 — additional proof submitted but address still does not match",
    "geography_risk":
        "weight=10 × geography_risk (0–1 from financialProfile lookup table)",
    "income_implausibility":
        "weight=10 × (1 − income_plausibility_score)",
}


async def run_explainability(
    entity: EntityResolutionOutput,
    screening: ScreeningOutput,
    risk: RiskOutput,
    idv: IDVerificationOutput,
    fin: FinancialProfileOutput,
    decision: DecisionOutput,
    vllm: VllmClient,
) -> tuple[ExplanationOutput, list[GpuCallMetric]]:
    dag_nodes, dag_edges = _build_dag(entity, screening, risk, idv, fin, decision)
    evidence_cards = _cards_from_dag(dag_nodes)

    result = await vllm.reason(
        [
            {"role": "system",
             "content": "You are a KYC compliance analyst. Write ONE sentence for "
                        '"summary" (key finding) and ONE sentence for "recommended_action". '
                        'Output ONLY valid JSON: {"summary": "...", "recommended_action": "..."}.',},
            {"role": "user", "content": json.dumps({
                "subject": entity.canonical_name,
                "score": risk.score,
                "decision": decision.decision,
                "signals": [c.signal for c in risk.contributors],
            })},
        ],
        json_mode=True, max_tokens=256,
    )

    try:
        raw = result.json or {}
        summary = str(raw.get("summary") or _default_summary(entity, risk, decision))
        recommended_action = str(raw.get("recommended_action") or _default_action(decision))
    except Exception:
        summary = _default_summary(entity, risk, decision)
        recommended_action = _default_action(decision)

    return ExplanationOutput(
        summary=summary,
        evidence_cards=evidence_cards,
        recommended_action=recommended_action,
        dag_nodes=dag_nodes,
        dag_edges=dag_edges,
    ), [result.metric]


# ── Deterministic DAG builder ────────────────────────────────────────────────

def _build_dag(
    entity: EntityResolutionOutput,
    screening: ScreeningOutput,
    risk: RiskOutput,
    idv: IDVerificationOutput,
    fin: FinancialProfileOutput,
    decision: DecisionOutput,
) -> tuple[list[EvidenceNode], list[CausalEdge]]:
    nodes: list[EvidenceNode] = []
    edges: list[CausalEdge] = []

    dec_id = "node_decision"
    nodes.append(EvidenceNode(
        node_id=dec_id,
        kind="decision",
        label=f"{decision.decision.upper()}\nscore {risk.score:.0f}/100",
        agent="decision",
        raw_value={"decision": decision.decision, "score": risk.score,
                   "requires_human": decision.requires_human},
        rule="score<30→approve · 30–69→review · ≥70→escalate; hard blocks on DOB/name/address",
        contribution=0.0,
    ))

    for contrib in risk.contributors:
        sig_id = f"node_sig_{contrib.signal}"
        ev_id = f"node_raw_{contrib.signal}"

        nodes.append(EvidenceNode(
            node_id=sig_id,
            kind="signal",
            label=f"{contrib.signal}\n+{contrib.contribution:.0f} pts",
            agent="risk",
            raw_value=contrib.value,
            rule=_SIGNAL_RULE.get(contrib.signal, f"weight={contrib.weight}"),
            contribution=contrib.contribution,
        ))
        edges.append(CausalEdge(source=sig_id, target=dec_id))

        ev_node = _raw_node(ev_id, contrib.signal, entity, screening, idv, fin)
        if ev_node:
            nodes.append(ev_node)
            edges.append(CausalEdge(source=ev_id, target=sig_id))

    return nodes, edges


def _raw_node(
    node_id: str,
    signal: str,
    entity: EntityResolutionOutput,
    screening: ScreeningOutput,
    idv: IDVerificationOutput,
    fin: FinancialProfileOutput,
) -> Optional[EvidenceNode]:
    agent = _SIGNAL_AGENT.get(signal, "risk")

    if signal == "sanctions_hit":
        m = screening.sanctions.matches[0] if screening.sanctions.matches else None
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=(f"Sanctions\n{m.name}\nconf {m.confidence:.2f}" if m
                   else "Sanctions\nmatch"),
            raw_value=m.model_dump(mode="json") if m else {"hit": True},
            rule="BGE cosine recall → Llama adjudication → verdict=match",
            contribution=0,
        )

    if signal == "pep_hit":
        m = screening.pep.matches[0] if screening.pep.matches else None
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=(f"PEP\n{m.name}\nconf {m.confidence:.2f}" if m
                   else "PEP\nmatch"),
            raw_value=m.model_dump(mode="json") if m else {"hit": True},
            rule="BGE cosine recall → Llama adjudication → verdict=match",
            contribution=0,
        )

    if signal == "adverse_media":
        sev = (screening.adverse_media.severity.value
               if screening.adverse_media.severity else "unknown")
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=f"Adverse Media\n{sev}",
            raw_value={"severity": sev, "summary": screening.adverse_media.summary},
            rule="Llama adjudication on bundled adverse-media articles",
            contribution=0,
        )

    if signal == "id_fail":
        failed = []
        if idv.pan_format_valid is False:
            failed.append("PAN format")
        if idv.mrz_valid is False:
            failed.append("Passport MRZ")
        if idv.expiry_ok is False:
            failed.append("Passport expiry")
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label="ID Checks\n" + "\n".join(failed or ["unknown"]),
            raw_value={"failed_checks": failed, "doc_authenticity": idv.doc_authenticity},
            rule="PAN regex · Aadhaar Verhoeff · MRZ checksum · expiry date",
            contribution=0,
        )

    if "name_mismatch" in signal:
        failing = [m for m in entity.name_matches if not m.ok]
        pairs = ", ".join(
            f"{m.doc_kind.value}:{m.score:.2f}" for m in failing
        ) or "all docs"
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=f"Name Match\n{pairs}",
            raw_value=[{"doc": m.doc_kind.value, "extracted": m.extracted_name,
                        "score": m.score} for m in failing],
            rule=f"token-set ratio ≥ {NAME_MATCH_THRESHOLD} required per document",
            contribution=0,
        )

    if "address" in signal:
        score = entity.address_match_score
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=(f"Address Match\nscore {score:.2f}" if score is not None
                   else "Address Match\nno score"),
            raw_value={"match_score": score, "confirmed": entity.address_confirmed,
                       "additional_proof": entity.address_additional_proof_submitted},
            rule=f"fuzzy address match ≥ {ADDRESS_MATCH_THRESHOLD} required",
            contribution=0,
        )

    if signal == "geography_risk":
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=f"Geography\nrisk {fin.geography_risk:.2f}",
            raw_value={"geography_risk": fin.geography_risk},
            rule="nationality × declared address → lookup table risk score",
            contribution=0,
        )

    if signal == "income_implausibility":
        return EvidenceNode(
            node_id=node_id, kind="raw_value", agent=agent,
            label=f"Income\nplausibility {fin.income_plausibility_score:.2f}",
            raw_value={"income_plausibility_score": fin.income_plausibility_score},
            rule="declared_income vs employment_band → plausibility ratio",
            contribution=0,
        )

    return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cards_from_dag(nodes: list[EvidenceNode]) -> list[EvidenceCard]:
    """Derive evidence cards from signal nodes for backward compatibility."""
    cards = []
    for n in nodes:
        if n.kind != "signal":
            continue
        signal = n.node_id.removeprefix("node_sig_")
        contrib = n.contribution
        sev = (Severity.high if contrib >= 30 else
               Severity.medium if contrib >= 15 else Severity.low)
        cards.append(EvidenceCard(
            title=signal,
            finding=f"{n.rule} — contributed {contrib:.0f} pts.",
            source=_SIGNAL_AGENT.get(signal, "risk"),
            severity=sev,
        ))
    return cards


def _default_summary(entity: EntityResolutionOutput, risk: RiskOutput,
                     decision: DecisionOutput) -> str:
    n = len(risk.contributors)
    signals = f" ({n} risk signal{'s' if n != 1 else ''} fired)" if n else ""
    return (f"Risk score {risk.score:.0f}/100 for {entity.canonical_name}"
            f"{signals} → {decision.decision.upper()}.")


def _default_action(decision: DecisionOutput) -> str:
    return {
        "approve":  "Proceed with onboarding.",
        "review":   "Route to a compliance officer for manual review.",
        "escalate": "Escalate to the senior compliance team immediately.",
        "reject":   "Reject the application and notify the customer.",
    }.get(decision.decision, "Refer to the decision agent for next steps.")
