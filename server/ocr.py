"""
ocr.py — PaddleOCR pre-pass for the extraction agent (§4.2).

Reads raw text out of an ID document image before the vision LLM sees it. The
vision model still produces the final structured fields, but with the OCR'd
text as a hint in the user prompt — that defends against vision-only mistakes
(small fonts, smudged stamps, Devanagari/Tamil dates) and means even the
KYC_DEMO=1 / smoke-test paths get real `raw_text` populated on case state,
not just canned fields.

CPU only. Lazy-loaded once per process — the first call is slow (~3-5 s
warm-up on a laptop), subsequent calls are ~100-300 ms per image.

Soft-fail on every error: if PaddleOCR isn't installed, or the image is
empty/corrupt, or recognition fails, the function returns "" and the
extraction agent falls back to LLM-only behavior. We never raise.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
from typing import Optional

_DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,(.*)$", re.DOTALL)
_paddle_ocr = None  # lazy singleton


def _parse_data_url(image_url: str) -> Optional[bytes]:
    """Decode `data:image/...;base64,...` → bytes. Returns None on shape errors."""
    m = _DATA_URL_RE.match(image_url)
    if not m or not m.group(1):
        return None
    try:
        return base64.b64decode(m.group(1))
    except Exception:
        return None


def _ensure_paddle():
    """Load PaddleOCR once. Raises ImportError if not installed."""
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    from paddleocr import PaddleOCR  # type: ignore
    # use_angle_cls handles rotated text (common on phone-shot ID photos).
    # lang='en' covers English on Indian docs; Devanagari numbers in dates
    # are usually fine. For full Devanagari we'd flip to lang='hi'.
    _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddle_ocr


def _ocr_sync(image_bytes: bytes) -> str:
    """Run PaddleOCR synchronously on raw image bytes. Returns joined text."""
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    paddle = _ensure_paddle()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    result = paddle.ocr(arr, cls=True)
    if not result or not result[0]:
        return ""
    # Each row: [[bbox], (text, confidence)]. Keep only text with conf > 0.5.
    lines = [
        line[1][0]
        for line in result[0]
        if line and len(line) >= 2 and len(line[1]) >= 2 and line[1][1] >= 0.5
    ]
    return "\n".join(lines).strip()


async def extract_text(image_url: str) -> str:
    """Async wrapper. Returns "" on empty input, missing paddleocr, or any failure."""
    image_bytes = _parse_data_url(image_url)
    if not image_bytes:
        return ""
    try:
        return await asyncio.to_thread(_ocr_sync, image_bytes)
    except Exception:
        # Never let OCR break the pipeline — the LLM can still extract from the
        # image alone, and downstream agents tolerate raw_text == None.
        return ""
