"""
Screening Agent (DEEP ★, §4.4) — Sanctions + PEP + Adverse Media.

Three sub-checks, run concurrently:

  Sanctions / PEP  — three-stage funnel (unchanged):
    1. Recall      — embed canonical name (BGE) → top-K via ScreeningIndex
    2. Precision   — Levenshtein + DOB tolerance (±2y) filter
    3. Adjudicate  — Llama emits match/no-match/uncertain + rationale

  Adverse Media + PEP (Tavily, when TAVILY_API_KEY set):
    - Queries include nationality + employment + DOB year for disambiguation
    - Common placeholder names (John Doe, Jane Doe) trigger extra disambiguation
    - Llama receives full customer profile and is instructed to reject namesake hits
    - Severity properly assessed from actual news content (not hardcoded)

Tavily is optional: None → DB-only behaviour, pipeline always completes.
"""
from __future__ import annotations

import asyncio
import json
from difflib import SequenceMatcher
from typing import Optional

from schemas import (AdverseMedia, CandidateMatch, CustomerInput,
                     EntityResolutionOutput, GpuCallMetric, ScreeningOutput,
                     Severity, SubScreening)
from screening_index import EntityRow, ScreeningIndex
from tavily_client import TavilyClient
from vllm_client import VllmClient

# Names that are legally used as anonymous placeholders — searches for these
# return court documents about unknown defendants, not the actual individual.
_PLACEHOLDER_NAMES = {
    "john doe", "jane doe", "john smith", "jane smith",
    "richard roe", "mary major", "unknown person",
}


async def run_screening(
    entity: EntityResolutionOutput,
    customer: CustomerInput,
    vllm: VllmClient,
    index: ScreeningIndex,
    tavily: Optional[TavilyClient] = None,
) -> tuple[ScreeningOutput, list[GpuCallMetric]]:
    gpu: list[GpuCallMetric] = []

    # Recall stage — one embedding shared by all DB sub-agents.
    vectors, embed_metric = await vllm.embed(entity.canonical_name, agent="screening")
    gpu.append(embed_metric)
    qv = vectors[0]

    # Run all lookups concurrently.
    # Tavily makes ONE combined search — Llama splits results into adverse vs PEP signals.
    sanctions_task  = _adjudicate(qv, ["sanction"],        entity, customer.dob, vllm, index, gpu)
    pep_task        = _adjudicate(qv, ["role.pep", "gov"], entity, customer.dob, vllm, index, gpu)
    db_adverse_task = _adjudicate(qv, ["crime"],           entity, customer.dob, vllm, index, gpu)
    web_task        = _web_news_combined(entity.canonical_name, customer, tavily)

    sanctions_matches, pep_db_matches, db_adverse_matches, (web_articles, tavily_ms) = \
        await asyncio.gather(sanctions_task, pep_task, db_adverse_task, web_task)

    # Record Tavily latency as a synthetic metric so the dashboard shows
    # the full accounting (Tavily runs concurrently but its time dominates the gather phase).
    if tavily_ms > 0:
        from datetime import datetime, timezone as _tz
        from schemas import GpuCallMetric as _M
        gpu.append(_M(
            ts=datetime.now(_tz.utc).isoformat(),
            model="tavily-web-search",
            latency_ms=round(tavily_ms, 1),
            agent="screening:web_search",
        ))

    adverse_media = await _assess_adverse_media(
        entity.canonical_name, customer, db_adverse_matches, web_articles, vllm, gpu)

    pep_matches = await _enrich_pep(
        entity.canonical_name, customer, pep_db_matches, web_articles, vllm, gpu)

    return ScreeningOutput(
        sanctions=SubScreening(
            hit=any(m.verdict == "match" for m in sanctions_matches),
            matches=sanctions_matches,
        ),
        pep=SubScreening(
            hit=any(m.verdict == "match" for m in pep_matches),
            matches=pep_matches,
        ),
        adverse_media=adverse_media,
    ), gpu


# ── Tavily query builder ──────────────────────────────────────────────────────

async def _web_news_combined(
    name: str,
    customer: CustomerInput,
    tavily: Optional[TavilyClient],
) -> tuple[list[dict], float]:
    """Single Tavily call covering both adverse media and PEP signals.
    Returns (articles, latency_ms) so the caller can surface Tavily time in metrics.
    """
    if tavily is None:
        return [], 0.0

    import time as _time
    year        = customer.dob[:4] if customer.dob and len(customer.dob) >= 4 else ""
    nationality = (customer.nationality or "").strip()
    employment  = (customer.declared_employment or "").strip()
    short_emp   = " ".join(employment.split()[:3]) if employment else ""

    context = " ".join(filter(None, [nationality, short_emp, year]))

    query = (
        f'"{name}" '
        f'(fraud OR corruption OR crime OR "money laundering" OR scandal OR conviction) '
        f'OR (politician OR minister OR "government official" OR parliament OR senator OR judge) '
        f'{context}'
    ).strip()

    t0 = _time.perf_counter()
    articles = await tavily.search_news(query, max_results=7)
    latency_ms = (_time.perf_counter() - t0) * 1000
    return articles, latency_ms


# ── Adverse media assessment ──────────────────────────────────────────────────

def _is_placeholder_name(name: str) -> bool:
    return name.lower().strip() in _PLACEHOLDER_NAMES


async def _assess_adverse_media(
    name: str,
    customer: CustomerInput,
    db_matches: list[CandidateMatch],
    web_articles: list[dict],
    vllm: VllmClient,
    gpu: list[GpuCallMetric],
) -> AdverseMedia:
    has_db_hit = any(m.verdict == "match" for m in db_matches)
    has_web    = bool(web_articles)

    if not has_db_hit and not has_web:
        return AdverseMedia(hit=False)

    db_evidence  = [{"source": "OpenSanctions", "name": m.name,
                     "datasets": m.datasets, "rationale": m.rationale}
                    for m in db_matches if m.verdict == "match"]
    web_evidence = [{"title":   a.get("title", ""),
                     "url":     a.get("url", ""),
                     "snippet": (a.get("content") or "")[:400]}
                    for a in web_articles]
    sources = [a.get("url", "") for a in web_articles if a.get("url")]

    # Build a disambiguation note for the prompt when the name is a common placeholder.
    disambiguation_note = ""
    if _is_placeholder_name(name):
        disambiguation_note = (
            f'\n\nIMPORTANT: "{name}" is frequently used as a legal placeholder for '
            "anonymous/unknown defendants in court filings. Such generic mentions are "
            "NOT adverse media about this specific individual. Only flag if the article "
            "explicitly identifies this person by additional details that match the "
            "subject profile (nationality, employment, DOB year)."
        )

    result = await vllm.reason(
        [
            {"role": "system",
             "content": (
                 "You are a KYC adverse media analyst with high precision standards. "
                 "Given evidence, determine if there is credible adverse media about "
                 "THIS SPECIFIC INDIVIDUAL — not a namesake, not a placeholder.\n\n"
                 "IDENTITY RULE: You must confirm the article is about the same person "
                 "using AT LEAST TWO independent corroborating details from this list:\n"
                 "  1. nationality or country of origin\n"
                 "  2. occupation or employer\n"
                 "  3. age or date of birth (within 3 years)\n"
                 "  4. city or region of residence\n"
                 "If fewer than two details match, set hit=false — do not flag.\n\n"
                 "Severity:\n"
                 "  high   — active criminal conviction, terrorism, confirmed fraud, "
                 "current sanctions violation\n"
                 "  medium — ongoing investigation, fraud/corruption allegation, "
                 "regulatory fine, money-laundering suspicion\n"
                 "  low    — minor controversy, historical resolved issue, "
                 "civil dispute unrelated to financial crime\n\n"
                 "If you cannot confirm the article is about THIS person specifically "
                 "with at least two details, set hit=false. "
                 "False positives on common names cause wrongful KYC rejection."
                 f"{disambiguation_note}\n\n"
                 "Output ONLY valid json: "
                 '{"hit": true|false, "severity": "high"|"medium"|"low"|null, '
                 '"summary": "<one sentence or null>", '
                 '"confirmed_details": ["<detail 1>", "<detail 2>"]}'
             )},
            {"role": "user", "content": json.dumps({
                "subject": {
                    "name":        name,
                    "dob":         customer.dob,
                    "nationality": customer.nationality,
                    "employment":  customer.declared_employment,
                },
                "db_evidence":  db_evidence,
                "web_evidence": web_evidence,
            })},
        ],
        json_mode=True,
        max_tokens=300, agent="screening:adverse_media",
    )
    gpu.append(result.metric)

    raw      = result.json if isinstance(result.json, dict) else {}
    hit      = bool(raw.get("hit", has_db_hit))
    sev_raw  = raw.get("severity")
    severity = (Severity(sev_raw) if sev_raw in ("low", "medium", "high") else
                Severity.medium if hit else None)
    summary  = str(raw.get("summary") or "") or None

    return AdverseMedia(hit=hit, severity=severity, summary=summary, sources=sources)


# ── PEP web enrichment ────────────────────────────────────────────────────────

async def _enrich_pep(
    name: str,
    customer: CustomerInput,
    db_matches: list[CandidateMatch],
    web_articles: list[dict],
    vllm: VllmClient,
    gpu: list[GpuCallMetric],
) -> list[CandidateMatch]:
    if any(m.verdict == "match" for m in db_matches):
        return db_matches
    if not web_articles:
        return db_matches

    disambiguation_note = ""
    if _is_placeholder_name(name):
        disambiguation_note = (
            f' Note: "{name}" is a common placeholder name — require strong '
            "corroborating evidence before confirming PEP status."
        )

    result = await vllm.reason(
        [
            {"role": "system",
             "content": (
                 "You are a PEP screening analyst. Determine if web results confirm "
                 "this person holds or recently held a senior political/government/judicial role. "
                 "IDENTITY RULE: confirm with AT LEAST TWO independent details from: "
                 "nationality/country, age/DOB, city/region, employer/party. "
                 "Name alone or one detail is insufficient — set is_pep=false."
                 f"{disambiguation_note} "
                 "Output ONLY valid json: "
                 '{"is_pep": true|false, "role": "<role or null>", '
                 '"confirmed_details": ["<detail 1>", "<detail 2>"], '
                 '"confidence": 0.0-1.0, "rationale": "<one sentence>"}'
             )},
            {"role": "user", "content": json.dumps({
                "subject":      {"name": name, "dob": customer.dob,
                                 "nationality": customer.nationality},
                "web_evidence": [{"title":   a.get("title", ""),
                                  "snippet": (a.get("content") or "")[:300]}
                                 for a in web_articles[:3]],
            })},
        ],
        json_mode=True, max_tokens=200, agent="screening:pep",
    )
    gpu.append(result.metric)

    raw = result.json if isinstance(result.json, dict) else {}
    if raw.get("is_pep") and float(raw.get("confidence", 0)) >= 0.7:
        db_matches = list(db_matches) + [CandidateMatch(
            entity_id="web_pep_" + name[:8].replace(" ", "_").lower(),
            name=name,
            datasets=["web_search"],
            verdict="match",
            confidence=float(raw.get("confidence", 0.7)),
            rationale=str(raw.get("rationale", "")),
            evidence=[a.get("url", "") for a in web_articles[:3] if a.get("url")],
        )]
    return db_matches


# ── DB adjudication (unchanged) ───────────────────────────────────────────────

async def _adjudicate(
    query_vector, topic_prefixes, entity, customer_dob,
    vllm: VllmClient, index: ScreeningIndex,
    gpu: list[GpuCallMetric],
) -> list[CandidateMatch]:
    candidates = index.recall(query_vector, topic_prefixes, k=20)
    filtered   = [c for c in candidates if _precision_filter(c, entity.canonical_name, customer_dob)]
    if not filtered:
        return []

    result = await vllm.reason(
        [
            {"role": "system",
             "content": "You are a sanctions/PEP screening adjudicator. Output ONLY JSON "
                        'of shape {"matches":[{"entity_id","name","datasets","verdict",'
                        '"confidence","rationale","evidence"}]}.'},
            {"role": "user", "content": json.dumps({
                "subject":    {"name": entity.canonical_name, "dob": customer_dob},
                "candidates": [_cand(c) for c in filtered[:5]],
            })},
        ],
        json_mode=True, max_tokens=1024, agent="screening:sanctions",
    )
    gpu.append(result.metric)
    raw = (result.json or {}).get("matches", []) if isinstance(result.json, dict) else []
    out: list[CandidateMatch] = []
    for m in raw:
        try:
            out.append(CandidateMatch.model_validate(m))
        except Exception:
            continue
    return out


def _cand(c: EntityRow) -> dict:
    return {"entity_id": c.id, "name": c.name, "aliases": c.aliases,
            "dob": c.dob, "countries": c.countries, "datasets": c.datasets}


def _precision_filter(c: EntityRow, canonical_name: str, dob: str) -> bool:
    names = [c.name] + c.aliases
    sim   = max(SequenceMatcher(None, canonical_name, n.lower()).ratio() for n in names)
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
        return True
