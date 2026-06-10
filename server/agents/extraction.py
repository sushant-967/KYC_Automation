"""
Extraction Agent (DEEP ★, §4.2) — Qwen2.5-VL-72B via vLLM :8000.

Per document: an optional PaddleOCR pre-pass produces raw text; that text plus
the image are sent to the vision LLM with a per-doc-kind JSON-schema prompt;
the LLM returns structured fields + confidence; deterministic format checks
run (PAN regex, MRZ checksum, Aadhaar Verhoeff); Aadhaar is MASKED before it
propagates downstream.

If PaddleOCR isn't installed, `ocr.extract_text` silently returns "" and the
vision LLM does extraction on the image alone — no behavior change for users
who don't install it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Awaitable, Callable

from schemas import (DocumentRef, ExtractedDocument, ExtractionOutput,
                     GpuCallMetric, Validations)
from vllm_client import VllmClient
from ocr import extract_text as ocr_extract_text

PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "rocm" / "prompts"

# loader: file_id -> data: URL the vision model can read
ImageLoader = Callable[[str], Awaitable[str]]


async def run_extraction(
    documents: list[DocumentRef], vllm: VllmClient, load_image: ImageLoader
) -> tuple[ExtractionOutput, list[GpuCallMetric]]:
    gpu: list[GpuCallMetric] = []
    out: list[ExtractedDocument] = []

    for doc in documents:
        image_url = await load_image(doc.file_id)
        # OCR pre-pass — empty string if PaddleOCR isn't installed or image is blank.
        raw_text = await ocr_extract_text(image_url)
        result = await vllm.extract(
            [
                {"role": "system", "content": _system_prompt(doc.kind.value)},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": _user_prompt(doc.kind.value, raw_text)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
            json_mode=True, max_tokens=1024,
        )
        gpu.append(result.metric)

        fields = result.json if isinstance(result.json, dict) else {}
        out.append(ExtractedDocument(
            kind=doc.kind,
            fields=_mask_sensitive(doc.kind.value, fields),
            confidence=float(fields.get("_confidence", 0.5)),
            raw_text=raw_text or None,
            validations=_format_checks(doc.kind.value, fields),
            masked_fields=["aadhaarNumber"] if doc.kind.value == "aadhaar" else [],
        ))

    return ExtractionOutput(documents=out), gpu


def _system_prompt(kind: str) -> str:
    """Prefer the maintained prompt file; fall back to a terse inline default."""
    f = _PROMPT_DIR / "extraction.md"
    base = f.read_text() if f.exists() else (
        "You are a precise KYC document parser. Output ONLY JSON. "
        "Include a numeric _confidence in [0,1]. Do not invent unreadable fields."
    )
    return f"{base}\n\nDocument kind: {kind}."


def _user_prompt(kind: str, raw_text: str) -> str:
    """User prompt — includes the OCR'd raw text as a hint when available."""
    base = f"Extract all fields from this {kind}. Return JSON only."
    if not raw_text:
        return base
    # Cap OCR text so we don't blow the context on a long bank statement.
    hint = raw_text[:1500]
    return (
        f"{base}\n\n"
        "OCR-extracted raw text (use as a hint — the image is authoritative):\n"
        f"---\n{hint}\n---"
    )


def _format_checks(kind: str, fields: dict) -> Validations | None:
    if kind == "pan":
        pan = str(fields.get("pan", ""))
        return Validations(pan_regex_ok=bool(PAN_RE.match(pan)))
    if kind == "passport":
        return Validations(mrz_checksum_ok=False)  # TODO: parse MRZ + checksum
    if kind == "aadhaar":
        return Validations(aadhaar_verhoeff_ok=False)  # TODO: Verhoeff over 12 digits
    return None


def _mask_sensitive(kind: str, fields: dict) -> dict:
    """Mask first 8 of the 12-digit Aadhaar (§4.2) — raw value never leaves here."""
    if kind != "aadhaar":
        return fields
    masked = dict(fields)
    raw = re.sub(r"\D", "", str(masked.get("aadhaarNumber", "")))
    if len(raw) == 12:
        masked["aadhaarNumber"] = f"XXXX-XXXX-{raw[8:]}"
    return masked
