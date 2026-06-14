"""
guardrails.py — Safety, input-validation, prompt-injection, and jailbreak guards.

Three layers:

  INPUT        — validate customer-submitted data before the pipeline starts.
                 Hard blocks: under-18, future DOB, empty name, negative income.
                 Warn blocks: suspiciously long fields (possible injection payload).

  INJECTION    — scan every OCR'd document page for adversarial text patterns
                 that could hijack the reasoning LLM downstream.  Covers:
                   • role-override phrases ("ignore previous instructions", "you are now")
                   • KYC-specific bypass attempts ("approve this case", "zero risk")
                   • LLM special tokens (<|im_start|>, [INST], <<SYS>>, etc.)
                   • Template injection ({{ }})
                   • Suspicious URLs embedded in documents (exfiltration vector)

  JAILBREAK    — scan each raw LLM response for signs the model broke character:
                   • Model refuses its KYC role ("I cannot", "as an AI …")
                   • Model output contains compliance-bypass language
                   • JSON expected but model returned prose (went off-script)

A GuardrailResult with level="critical" that contains "injection" or "jailbreak"
in its check name triggers an `adversarial_document` risk contributor (+50 pts)
that pushes the case into REVIEW or ESCALATE regardless of other signals.

Results flow three ways:
  1. audit log — emitted as event="guardrail_violation" for the timeline
  2. case.guardrail_flags — a typed list on CaseState for downstream agents
  3. SSE stream — Streamlit shows an alert banner for any critical violation
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from schemas import CustomerInput, ExtractionOutput


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    level: str        # "info" | "warn" | "critical"
    check: str        # short slug used for grouping / risk contribution
    violations: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def guard_customer_input(customer: CustomerInput) -> list[GuardrailResult]:
    """Validate customer-submitted form data before the pipeline starts."""
    results: list[GuardrailResult] = []

    # ── Date of birth ─────────────────────────────────────────────────────────
    r = _check_dob(customer.dob)
    if r:
        results.append(r)

    # ── Full name ─────────────────────────────────────────────────────────────
    name = (customer.full_name or "").strip()
    if len(name) < 2:
        results.append(GuardrailResult(
            passed=False, level="critical", check="name_empty",
            violations=["Customer full name is empty or too short — cannot proceed"]))
    elif len(name) > 200:
        # Legitimate names are never this long; a very long string is a sign of
        # an injection payload embedded in the name field.
        results.append(GuardrailResult(
            passed=False, level="warn", check="name_too_long",
            violations=[f"Full name is {len(name)} chars (max 200) — possible injection payload"]))

    # ── Income ────────────────────────────────────────────────────────────────
    if customer.declared_income is not None and customer.declared_income < 0:
        results.append(GuardrailResult(
            passed=False, level="critical", check="negative_income",
            violations=["Declared annual income is negative — invalid value"]))

    # ── Free-text fields: suspicious length ───────────────────────────────────
    emp = customer.declared_employment or ""
    if len(emp) > 500:
        results.append(GuardrailResult(
            passed=False, level="warn", check="employment_too_long",
            violations=[
                f"Employment field is {len(emp)} chars (max 500) — "
                "may contain injection payload; field will be truncated"]))

    addr = customer.address or ""
    if len(addr) > 1000:
        results.append(GuardrailResult(
            passed=False, level="warn", check="address_too_long",
            violations=[f"Address field is {len(addr)} chars (max 1000)"]))

    return results


def _check_dob(dob_str: Optional[str]) -> Optional[GuardrailResult]:
    if not dob_str:
        return GuardrailResult(
            passed=False, level="critical", check="dob_missing",
            violations=["Date of birth is required for KYC"])

    try:
        dob = date.fromisoformat(dob_str)
    except ValueError:
        return GuardrailResult(
            passed=False, level="critical", check="dob_invalid",
            violations=[f"DOB '{dob_str}' is not a valid ISO-8601 date (expected YYYY-MM-DD)"])

    today = date.today()
    age   = (today - dob).days / 365.25

    if dob > today:
        return GuardrailResult(
            passed=False, level="critical", check="dob_future",
            violations=[f"DOB {dob_str} is in the future — invalid"])

    if age > 120:
        return GuardrailResult(
            passed=False, level="critical", check="dob_implausible",
            violations=[f"DOB {dob_str} implies age {age:.0f} — not biologically plausible"])

    if age < 18:
        return GuardrailResult(
            passed=False, level="critical", check="under_18",
            violations=[
                f"Customer age {age:.1f} is below 18 — "
                "RBI Master Direction §3 requires customers to be adults"])

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT INJECTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # ── Role override / instruction hijack ───────────────────────────────────
    (r"ignore\s+(previous|prior|all|above)\s+instruction", "role_override"),
    (r"disregard\s+(previous|prior|all|the\s+above)", "role_override"),
    (r"forget\s+(everything|all|prior\s+context|previous)", "role_override"),
    (r"you\s+are\s+now\s+(a|an|the)\s+\w+", "role_override"),
    (r"act\s+as\s+(a|an|the)\s+\w+", "role_override"),
    (r"(switch|change)\s+(to|your)\s+(mode|role|persona)", "role_override"),
    (r"new\s+(role|persona|instruction|system\s+(prompt|message))", "role_override"),
    (r"pretend\s+(you\s+are|to\s+be)", "role_override"),
    (r"roleplay\s+as", "role_override"),
    # ── KYC-specific bypass ───────────────────────────────────────────────────
    (r"approve\s+(this|the)\s+(case|application|customer|kyc)", "kyc_bypass"),
    (r"(mark|set|flag|classify)\s+(as|this\s+as)\s+(approved|safe|clean|clear)", "kyc_bypass"),
    (r"(zero|0|no)\s+(risk|flags|alert|sanctions?)", "kyc_bypass"),
    (r"not\s+a\s+(pep|politically\s+exposed\s+person)", "kyc_bypass"),
    (r"(clear|remove|delete|ignore)\s+(all\s+)?(risk\s+)?(flag|signal|alert|warning)", "kyc_bypass"),
    (r"(override|bypass|skip|circumvent)\s+(risk|decision|compliance|screening)", "kyc_bypass"),
    (r"this\s+(customer|person|applicant)\s+(is\s+)?(safe|trusted|verified|approved)", "kyc_bypass"),
    (r"no\s+(further\s+)?(review|check|investigation)\s+(is\s+)?(needed|required)", "kyc_bypass"),
    # ── LLM special tokens (model-specific control sequences) ─────────────────
    (r"<\|.{1,40}\|>",              "special_token"),   # <|im_start|>, <|endoftext|>
    (r"\[INST\]|\[/?INST\]",        "special_token"),   # Llama2 instruction token
    (r"<<SYS>>|<</SYS>>",           "special_token"),   # Llama2 system block
    (r"<\|system\|>|<\|user\|>",    "special_token"),   # Phi-3 chat template
    (r"### (Instruction|Response|System):", "special_token"),  # Alpaca template
    (r"Human:|Assistant:|HUMAN:|ASSISTANT:", "special_token"),  # ChatML alt
    # ── Template injection ────────────────────────────────────────────────────
    (r"\{\{.*?\}\}",                "template_injection"),  # Jinja2 / Handlebars
    (r"\{%.*?%\}",                  "template_injection"),
    # ── Data exfiltration ─────────────────────────────────────────────────────
    (r"(send|forward|email|upload|post|exfiltrate)\s+(the\s+)?(data|result|output|case|report)", "exfiltration"),
    (r"https?://\S+",               "external_url"),    # URLs embedded in docs
    (r"webhook\s*=",                "exfiltration"),
]

_COMPILED_INJECTION = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), kind)
    for p, kind in _INJECTION_PATTERNS
]


def guard_prompt_injection(extraction: ExtractionOutput) -> list[GuardrailResult]:
    """
    Scan every OCR'd document for adversarial text that could manipulate the
    reasoning LLM downstream (Llama sees this text as part of its context window).
    """
    results: list[GuardrailResult] = []

    for doc in extraction.documents:
        # Scan both raw OCR text and extracted field values.
        raw   = doc.raw_text or ""
        field_text = " ".join(str(v) for v in doc.fields.values() if v)
        text  = raw + " " + field_text

        hits: dict[str, list[str]] = {}
        for pattern, kind in _COMPILED_INJECTION:
            for m in pattern.finditer(text):
                snippet = m.group(0)[:80].replace("\n", " ")
                hits.setdefault(kind, []).append(snippet)

        if hits:
            kinds   = ", ".join(sorted(hits.keys()))
            samples = [f"[{k}] «{vs[0]}»" for k, vs in list(hits.items())[:3]]
            results.append(GuardrailResult(
                passed=False,
                level="critical",
                check=f"prompt_injection:{doc.kind.value}",
                violations=[
                    f"Document '{doc.kind.value}' contains adversarial patterns: {kinds}",
                    *samples,
                ],
            ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. JAILBREAK DETECTION (LLM output analysis)
# ══════════════════════════════════════════════════════════════════════════════

# Signs the model broke its KYC analyst role.
_JB_REFUSAL: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"i (cannot|can'?t|am unable to|will not|won'?t) (help|assist|provide|do)",
        r"as (a|an) (ai|language model|llm|large language model|assistant)",
        r"i'?m (an ai|a language model|a large language model|not able)",
        r"i (don'?t|do not) have (the ability|access|permission|authority)",
        r"that'?s (not |outside )(my |the )(role|scope|purpose|function)",
        r"i (am|was) trained (to|not to)",
        r"my (guidelines|policy|training|creators|developers) (prevent|prohibit|don'?t allow)",
    ]
]

_JB_BYPASS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(approve|clear|whitelist|authorise|authorize)\s+(this|the)\s+(case|customer|application|kyc)",
        r"(ignore|disregard|skip|bypass)\s+(the\s+)?(risk|flag|sanction|screening|alert|warning)",
        r"(set|change|update|modify)\s+(risk\s+)?score\s+to\s+0",
        r"(no|zero)\s+risk\s+(detected|found|identified|present)",
        r"this\s+(individual|person|customer|applicant)\s+(should\s+be|is|must\s+be)\s+(approved|cleared|safe)",
        r"(remove|delete|clear)\s+(the\s+)?(sanctions?|pep|adverse\s+media)\s+(hit|match|flag)",
    ]
]


def guard_llm_output(agent: str, raw_text: str,
                     expected_json: bool = True) -> Optional[GuardrailResult]:
    """
    Detect if an LLM response shows signs of jailbreak success:
      - Model refused its KYC analyst role (refusal patterns)
      - Model output contains compliance-bypass instructions
      - JSON was requested but model returned prose (went off-script)
    """
    violations: list[str] = []

    # JSON expected but got prose — model was manipulated off its output format.
    if expected_json:
        s = raw_text.strip()
        if s and not s.startswith(("{", "[")):
            if len(s) > 40 and "{" not in s[:30]:
                violations.append(
                    f"Expected JSON but received prose — model may have been jailbroken: "
                    f"«{s[:100]}»")

    # Refusal: model broke its KYC analyst persona.
    for pat in _JB_REFUSAL:
        m = pat.search(raw_text)
        if m:
            violations.append(f"Model refused KYC role: «{m.group(0)}»")
            break

    # Bypass: model output contains language that would override KYC decisions.
    for pat in _JB_BYPASS:
        m = pat.search(raw_text)
        if m:
            violations.append(f"Compliance-bypass language in model output: «{m.group(0)}»")
            break

    if violations:
        return GuardrailResult(
            passed=False,
            level="critical",
            check=f"jailbreak:{agent}",
            violations=violations,
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_adversarial(results: list[GuardrailResult]) -> bool:
    """True if any result is a critical injection or jailbreak finding."""
    return any(
        r.level == "critical" and ("injection" in r.check or "jailbreak" in r.check)
        for r in results
    )
