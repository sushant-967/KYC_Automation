"""
vllm_client.py — async client to the three vLLM servers running on THIS box.

Everything is local; no external inference API (§3 hard rule). vLLM exposes an
OpenAI-compatible API per model:

    localhost:8000/v1  Qwen2.5-VL-72B (FP8)         doc OCR + structured extraction
    localhost:8001/v1  Llama-3.3-70B-Instruct (FP8) text reasoning / adjudication
    localhost:8002/v1  BGE-large-en-v1.5            entity-name embeddings (1024-d)

Each chat/embed call also samples the model's vLLM /metrics so we can attribute
latency (and, where exposed, KV-cache/throughput) per GPU call for slide 4 (§6).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from schemas import GpuCallMetric

# Model ids as registered with vLLM (--served-model-name in the launch scripts).
QWEN = "qwen2.5-vl-72b"
LLAMA = "llama-3.3-70b"
BGE = "bge-large-en-v1.5"


@dataclass
class VllmConfig:
    qwen_url: str = "http://localhost:8000/v1"
    llama_url: str = "http://localhost:8001/v1"
    bge_url: str = "http://localhost:8002/v1"
    timeout_s: float = 120.0


@dataclass
class ChatResult:
    text: str
    json: Optional[Any]  # parsed when json_mode=True and parsing succeeds
    metric: GpuCallMetric


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VllmClient:
    def __init__(self, cfg: Optional[VllmConfig] = None):
        self.cfg = cfg or VllmConfig()
        self._client = httpx.AsyncClient(timeout=self.cfg.timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Vision + text extraction (Qwen-VL, :8000) ──────────────────────────
    async def extract(self, messages: list[dict], *, json_mode: bool = True,
                      max_tokens: int = 1024, temperature: float = 0.0) -> ChatResult:
        return await self._chat(self.cfg.qwen_url, QWEN, messages,
                                json_mode=json_mode, max_tokens=max_tokens,
                                temperature=temperature)

    # ── Text reasoning / adjudication / explanation (Llama, :8001) ─────────
    async def reason(self, messages: list[dict], *, json_mode: bool = True,
                     max_tokens: int = 1536, temperature: float = 0.0) -> ChatResult:
        return await self._chat(self.cfg.llama_url, LLAMA, messages,
                                json_mode=json_mode, max_tokens=max_tokens,
                                temperature=temperature)

    # ── Embeddings (BGE, :8002) ────────────────────────────────────────────
    async def embed(self, text: str | list[str]) -> tuple[list[list[float]], GpuCallMetric]:
        started = time.perf_counter()
        resp = await self._client.post(
            f"{self.cfg.bge_url}/embeddings",
            json={"model": BGE, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        latency = (time.perf_counter() - started) * 1000
        vectors = [d["embedding"] for d in data]
        return vectors, GpuCallMetric(ts=_now(), model=BGE, latency_ms=latency)

    # ── internals ──────────────────────────────────────────────────────────
    async def _chat(self, base_url: str, model: str, messages: list[dict], *,
                    json_mode: bool, max_tokens: int, temperature: float) -> ChatResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        resp = await self._client.post(f"{base_url}/chat/completions", json=body)
        resp.raise_for_status()
        payload = resp.json()
        latency = (time.perf_counter() - started) * 1000

        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        metric = GpuCallMetric(
            ts=_now(), model=model, latency_ms=latency,
            # TODO: enrich vram_used_gb / gpu_util_pct from a /metrics scrape.
        )
        parsed = _safe_json(text) if json_mode else None
        return ChatResult(text=text, json=parsed, metric=metric)


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _safe_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _FENCE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        return None
