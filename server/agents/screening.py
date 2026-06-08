"""
Screening Agent (DEEP ★, §4.4) — Sanctions + PEP + Adverse Media.

Three sub-agents over the bundled OpenSanctions data, run concurrently. Per
sub-agent, a three-stage funnel:
  1. Recall    — embed canonical name (BGE :8002) → top-K via ScreeningIndex
  2. Precision — Levenshtein on canonical names + DOB tolerance (±2y) filter
  3. Adjudicate — Llama 3.3 70B (:8001) emits match/no-match/uncertain + rationale

NO external APIs at runtime — all data is local (§5.5).
"""
from __future__ import annotations

import asyncio
import json
from difflib import SequenceMatcher

from schemas import (AdverseMedia, CandidateMatch, EntityResolutionOutput,
                     GpuCallMetric, ScreeningOutput, Severity, SubScreening)
from screening_index import EntityRow, ScreeningIndex
from vllm_client import VllmClient


async def run_screening(
    entity: EntityResolutionOutput, customer_dob: str,
    vllm: VllmClient, index: ScreeningIndex,
) -> tuple[ScreeningOutput, list[GpuCallMetric]]:
    gpu: list[GpuCallMetric] = []

    # Recall stage — one embedding shared by all three sub-agents.
    vectors, embed_metric = await vllm.embed(entity.canonical_name)
    gpu.append(embed_metric)
    qv = vectors[0]

    sanctions, pep, adverse = await asyncio.gather(
        _adjudicate(qv, ["sanction"], entity, customer_dob, vllm, index, gpu),
        _adjudicate(qv, ["role.pep", "gov"], entity, customer_dob, vllm, index, gpu),
        _adjudicate(qv, ["crime"], entity, customer_dob, vllm, index, gpu),
    )

    adverse_hit = next((m for m in adverse if m.verdict == "match"), None)
    out = ScreeningOutput(
        sanctions=SubScreening(hit=any(m.verdict == "match" for m in sanctions), matches=sanctions),
        pep=SubScreening(hit=any(m.verdict == "match" for m in pep), matches=pep),
        adverse_media=AdverseMedia(
            hit=adverse_hit is not None,
            summary=adverse_hit.rationale if adverse_hit else None,  # TODO: Llama summary of notes
            severity=Severity.medium if adverse_hit else None,
        ),
    )
    return out, gpu


async def _adjudicate(query_vector, topic_prefixes, entity, customer_dob,
                      vllm: VllmClient, index: ScreeningIndex,
                      gpu: list[GpuCallMetric]) -> list[CandidateMatch]:
    candidates = index.recall(query_vector, topic_prefixes, k=20)
    filtered = [c for c in candidates if _precision_filter(c, entity.canonical_name, customer_dob)]
    if not filtered:
        return []

    result = await vllm.reason(
        [
            {"role": "system",
             "content": "You are a sanctions/PEP screening adjudicator. Output ONLY JSON "
                        'of shape {"matches":[{"entity_id","name","datasets","verdict",'
                        '"confidence","rationale","evidence"}]}.'},
            {"role": "user", "content": json.dumps({
                "subject": {"name": entity.canonical_name, "dob": customer_dob},
                "candidates": [_cand(c) for c in filtered[:5]],
            })},
        ],
        json_mode=True, max_tokens=1024,
    )
    gpu.append(result.metric)
    raw = (result.json or {}).get("matches", []) if isinstance(result.json, dict) else []
    out: list[CandidateMatch] = []
    for m in raw:
        try:
            out.append(CandidateMatch.model_validate(m))
        except Exception:
            continue  # skip malformed candidate, keep the rest
    return out


def _cand(c: EntityRow) -> dict:
    return {"entity_id": c.id, "name": c.name, "aliases": c.aliases,
            "dob": c.dob, "countries": c.countries, "datasets": c.datasets}


def _precision_filter(c: EntityRow, canonical_name: str, dob: str) -> bool:
    """Cheap precision gate before the expensive LLM: name similarity + DOB tolerance."""
    names = [c.name] + c.aliases
    sim = max(SequenceMatcher(None, canonical_name, n.lower()).ratio() for n in names)
    if sim < 0.6:
        return False
    if c.dob and dob and not _dob_within(c.dob, dob, years=2):
        return False
    return True


def _dob_within(a: str, b: str, years: int) -> bool:
    try:
        ya, yb = int(a[:4]), int(b[:4])
        return abs(ya - yb) <= years
    except (ValueError, IndexError):
        return True  # can't compare → don't drop on DOB alone
