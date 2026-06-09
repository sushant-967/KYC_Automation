"""
vllm_client.py — async OpenAI-compatible client for the project's three model roles.

Default backend is the three vLLM servers on this box (§3 hard rule for the demo):

    localhost:8000/v1  Qwen2.5-VL-72B (FP8)         doc OCR + structured extraction
    localhost:8001/v1  Llama-3.3-70B-Instruct (FP8) text reasoning / adjudication
    localhost:8002/v1  BGE-large-en-v1.5            entity-name embeddings (1024-d)

For laptop dev (no MI300X), the client can also point at Groq for chat — set
`KYC_BACKEND=groq` and provide `GROQ_API_KEY`. Embeddings then fall back to a
local sentence-transformers BGE so the screening recall index stays in the same
vector space; vector embeddings are not a Groq-served API.

Each chat/embed call records latency into a `GpuCallMetric` for slide 4 (§6).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from schemas import GpuCallMetric

# Default model ids as registered with vLLM (--served-model-name in the launch scripts).
QWEN = "qwen2.5-vl-72b"
LLAMA = "llama-3.3-70b"
BGE = "bge-large-en-v1.5"


@dataclass
class VllmConfig:
    qwen_url: str = "http://localhost:8000/v1"
    qwen_model: str = QWEN
    llama_url: str = "http://localhost:8001/v1"
    llama_model: str = LLAMA
    bge_url: Optional[str] = "http://localhost:8002/v1"
    bge_model: str = BGE
    api_key: Optional[str] = None  # sent as Bearer on every chat call when set
    local_embedder_model: Optional[str] = None  # if set, embed() uses sentence-transformers locally
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
        self._st_model = None  # lazy sentence-transformers, only for local-embedder path

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Vision + text extraction (Qwen-VL :8000 / Groq Llama-4 vision) ─────
    async def extract(self, messages: list[dict], *, json_mode: bool = True,
                      max_tokens: int = 1024, temperature: float = 0.0) -> ChatResult:
        return await self._chat(self.cfg.qwen_url, self.cfg.qwen_model, messages,
                                json_mode=json_mode, max_tokens=max_tokens,
                                temperature=temperature)

    # ── Text reasoning / adjudication / explanation (Llama :8001 / Groq) ───
    async def reason(self, messages: list[dict], *, json_mode: bool = True,
                     max_tokens: int = 1536, temperature: float = 0.0) -> ChatResult:
        return await self._chat(self.cfg.llama_url, self.cfg.llama_model, messages,
                                json_mode=json_mode, max_tokens=max_tokens,
                                temperature=temperature)

    # ── Embeddings (BGE :8002 or local sentence-transformers) ──────────────
    async def embed(self, text: str | list[str]) -> tuple[list[list[float]], GpuCallMetric]:
        if self.cfg.local_embedder_model:
            return await self._embed_local(text)
        if not self.cfg.bge_url:
            raise RuntimeError("embed() called but neither bge_url nor local_embedder_model is set")
        started = time.perf_counter()
        resp = await self._client.post(
            f"{self.cfg.bge_url}/embeddings",
            json={"model": self.cfg.bge_model, "input": text},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        latency = (time.perf_counter() - started) * 1000
        vectors = [d["embedding"] for d in data]
        return vectors, GpuCallMetric(ts=_now(), model=self.cfg.bge_model, latency_ms=latency)

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
        resp = await self._client.post(f"{base_url}/chat/completions", json=body,
                                       headers=self._auth_headers())
        resp.raise_for_status()
        payload = resp.json()
        latency = (time.perf_counter() - started) * 1000

        text = payload["choices"][0]["message"]["content"]
        metric = GpuCallMetric(ts=_now(), model=model, latency_ms=latency)
        parsed = _safe_json(text) if json_mode else None
        return ChatResult(text=text, json=parsed, metric=metric)

    def _auth_headers(self) -> Optional[dict[str, str]]:
        return {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else None

    async def _embed_local(self, text: str | list[str]) -> tuple[list[list[float]], GpuCallMetric]:
        """Compute embeddings on-process via sentence-transformers (CPU, lazy-loaded)."""
        self._ensure_st_model()
        inputs = [text] if isinstance(text, str) else list(text)
        started = time.perf_counter()
        arr = await asyncio.to_thread(
            self._st_model.encode, inputs, normalize_embeddings=True)
        latency = (time.perf_counter() - started) * 1000
        vectors = arr.tolist() if hasattr(arr, "tolist") else [list(v) for v in arr]
        return vectors, GpuCallMetric(
            ts=_now(), model=f"local:{self.cfg.local_embedder_model}", latency_ms=latency)

    def _ensure_st_model(self) -> None:
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "KYC_BACKEND=groq needs `sentence-transformers` for the embedding path. "
                "Install it: pip install sentence-transformers"
            ) from e
        self._st_model = SentenceTransformer(self.cfg.local_embedder_model)


# ── Backend factory ─────────────────────────────────────────────────────────

def make_vllm_client() -> VllmClient:
    """Pick a backend off env vars. Default is local vLLM (the AMD-box demo path).

    Set KYC_BACKEND=groq to route chat through Groq's OpenAI-compatible API and
    embeddings through a local sentence-transformers BGE. Useful for laptop dev
    without a GPU. Requires GROQ_API_KEY.

    Other tunables (all have sensible defaults):
        GROQ_BASE_URL          (default https://api.groq.com/openai/v1)
        GROQ_VISION_MODEL      (default meta-llama/llama-4-scout-17b-16e-instruct)
        GROQ_REASON_MODEL      (default llama-3.3-70b-versatile)
        KYC_LOCAL_EMBEDDER     (default BAAI/bge-large-en-v1.5)
    """
    backend = os.environ.get("KYC_BACKEND", "vllm").lower()
    if backend in ("vllm", ""):
        return VllmClient()
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("KYC_BACKEND=groq requires GROQ_API_KEY in env")
        base = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return VllmClient(VllmConfig(
            qwen_url=base,
            qwen_model=os.environ.get(
                "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            llama_url=base,
            llama_model=os.environ.get("GROQ_REASON_MODEL", "llama-3.3-70b-versatile"),
            bge_url=None,
            api_key=api_key,
            local_embedder_model=os.environ.get(
                "KYC_LOCAL_EMBEDDER", "BAAI/bge-large-en-v1.5"),
        ))
    raise RuntimeError(f"unknown KYC_BACKEND={backend!r} (expected 'vllm' or 'groq')")


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
