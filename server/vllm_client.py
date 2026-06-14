"""
vllm_client.py — async OpenAI-compatible client for the project's three model roles.

Default backend is the three vLLM servers on this box (§3 hard rule for the demo):

    localhost:8000/v1  Qwen2.5-VL-72B (FP8)         doc OCR + structured extraction
    localhost:8001/v1  Llama-3.3-70B-Instruct (FP8) text reasoning / adjudication
    localhost:8002/v1  BGE-large-en-v1.5            entity-name embeddings (1024-d)

For laptop dev (no MI300X), the client can also point at Groq for chat — set
`KYC_BACKEND=groq` and provide `GROQ_API_KEY`. Embeddings then fall back to a
local `fastembed` BGE (ONNX, ~130 MB, no torch) so the screening recall index
stays in the BGE family — Groq doesn't serve embeddings. The local BGE model
defaults to bge-small (384-d), so ingest must use the same model: `ingest.py
--backend local`. The AMD-box path keeps using bge-large (1024-d) via vLLM.

Each chat/embed call records latency into a `GpuCallMetric` for slide 4 (§6).

Resilience layers
─────────────────
• Concurrency semaphore  — GROQ_MAX_CONCURRENCY (default 4 Groq / 16 vLLM) caps
  simultaneous in-flight requests so parallel pipeline branches don't exhaust
  Groq's per-minute token quota.

• Retry with exponential backoff + jitter — up to GROQ_MAX_RETRIES (default 3)
  retries on HTTP 429 / 5xx / network error. Respects retry-after headers.

• Circuit breaker (per endpoint) — after CB_FAILURE_THRESHOLD (default 3)
  consecutive failures the circuit opens and calls fail fast with CircuitOpenError.
  After CB_RECOVERY_S (default 30) seconds it enters HALF-OPEN and sends one
  probe request; on success it closes again. Prevents cascading timeouts.

• Automatic backend fallback — if KYC_FALLBACK_BACKEND=groq is set alongside
  KYC_BACKEND=vllm, any CircuitOpenError or network-level failure that exhausts
  retries is transparently re-routed to a Groq-backed VllmClient. Allows the
  AMD-box demo to survive a vLLM crash without manual intervention.

• Embedding LRU cache — same canonical name always produces the same vector.
  EMBED_CACHE_SIZE (default 4096) entries cached in-memory; hit returns latency=0.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from schemas import GpuCallMetric


async def _snapshot_vram() -> tuple[Optional[float], Optional[float]]:
    """Return (vram_used_gb, gpu_util_pct) from rocm-smi or nvidia-smi.

    Runs in a thread so it never blocks the event loop. Returns (None, None)
    when neither tool is available (Docker dev, CI, non-GPU machines).
    """
    def _read() -> tuple[Optional[float], Optional[float]]:
        import json as _j
        try:
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
                capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                data = _j.loads(r.stdout)
                card = next(iter(data.values()))
                total = int(card.get("VRAM Total Memory (B)", 0))
                used  = int(card.get("VRAM Total Used Memory (B)", 0))
                util  = float(card.get("GPU use (%)", 0) or 0)
                if total:
                    return round(used / 1e9, 1), util
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                if len(parts) >= 2:
                    return round(float(parts[0]) / 1024, 1), float(parts[1])
        except Exception:
            pass
        return None, None

    return await asyncio.to_thread(_read)

# Default model ids as registered with vLLM (--served-model-name in the launch scripts).
QWEN  = "qwen2.5-vl-72b"
LLAMA = "llama-3.3-70b"
BGE   = "bge-large-en-v1.5"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BASE_DELAY_S     = 1.0   # first retry wait; doubles each attempt + jitter


# ── Exceptions ───────────────────────────────────────────────────────────────

class CircuitOpenError(RuntimeError):
    """Raised immediately when a circuit breaker is OPEN — fail fast, don't block."""


# ── Circuit Breaker ──────────────────────────────────────────────────────────

class _CircuitBreaker:
    """
    Three-state machine: CLOSED → OPEN → HALF-OPEN → CLOSED.

      CLOSED    — normal operation; failures are counted.
      OPEN      — endpoint considered down; all calls rejected immediately.
      HALF-OPEN — cooldown elapsed; one probe allowed to test recovery.
    """

    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_s: float = 30.0) -> None:
        self.name = name
        self._threshold   = failure_threshold
        self._recovery_s  = recovery_s
        self._failures    = 0
        self._state       = "closed"
        self._opened_at: float = 0.0
        self._lock        = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    async def allow(self) -> bool:
        """Return True if a request should be allowed through."""
        async with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._recovery_s:
                    self._state = "half-open"
                    return True
                return False
            # half-open: allow exactly one probe
            return True

    async def on_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._state    = "closed"

    async def on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._state == "half-open" or self._failures >= self._threshold:
                self._state    = "open"
                self._opened_at = time.monotonic()


# ── Embedding LRU cache ───────────────────────────────────────────────────────

class _LruEmbedCache:
    def __init__(self, maxsize: int) -> None:
        self._cache: dict[str, list[list[float]]] = {}
        self._order: list[str] = []
        self._maxsize = maxsize
        self._lock    = asyncio.Lock()

    async def get(self, key: str) -> Optional[list[list[float]]]:
        async with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            return None

    async def put(self, key: str, value: list[list[float]]) -> None:
        async with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._maxsize:
                del self._cache[self._order.pop(0)]
            self._cache[key] = value
            self._order.append(key)


# ── Config / Result ──────────────────────────────────────────────────────────

@dataclass
class VllmConfig:
    qwen_url:  str = "http://localhost:8000/v1"
    qwen_model: str = QWEN
    llama_url: str = "http://localhost:8001/v1"
    llama_model: str = LLAMA
    bge_url:   Optional[str] = "http://localhost:8002/v1"
    bge_model: str = BGE
    api_key:   Optional[str] = None
    local_embedder_model: Optional[str] = None
    timeout_s:            float = 120.0
    max_concurrency:      int   = 4
    max_retries:          int   = 3
    embed_cache_size:     int   = 4096
    cb_failure_threshold: int   = 3    # failures before circuit opens
    cb_recovery_s:        float = 30.0 # seconds before HALF-OPEN probe


@dataclass
class ChatResult:
    text:   str
    json:   Optional[Any]
    metric: GpuCallMetric


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Client ────────────────────────────────────────────────────────────────────

class VllmClient:
    def __init__(self, cfg: Optional[VllmConfig] = None, *,
                 fallback: Optional["VllmClient"] = None) -> None:
        self.cfg      = cfg or VllmConfig()
        self._fallback = fallback

        limits = httpx.Limits(
            max_connections=self.cfg.max_concurrency * 2,
            max_keepalive_connections=self.cfg.max_concurrency,
        )
        self._client = httpx.AsyncClient(timeout=self.cfg.timeout_s, limits=limits)
        self._sem    = asyncio.Semaphore(self.cfg.max_concurrency)

        self._cb_qwen  = _CircuitBreaker("qwen",  self.cfg.cb_failure_threshold, self.cfg.cb_recovery_s)
        self._cb_llama = _CircuitBreaker("llama", self.cfg.cb_failure_threshold, self.cfg.cb_recovery_s)

        self._embed_cache     = _LruEmbedCache(self.cfg.embed_cache_size)
        self._local_embedder  = None

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._fallback:
            await self._fallback.aclose()

    # ── Public API ─────────────────────────────────────────────────────────

    async def extract(self, messages: list[dict], *, json_mode: bool = True,
                      max_tokens: int = 1024, temperature: float = 0.0,
                      agent: str = "") -> ChatResult:
        try:
            return await self._chat(
                self.cfg.qwen_url, self.cfg.qwen_model, self._cb_qwen,
                messages, json_mode=json_mode, max_tokens=max_tokens,
                temperature=temperature, agent=agent)
        except (CircuitOpenError, _InfraError):
            if self._fallback:
                return await self._fallback.extract(
                    messages, json_mode=json_mode,
                    max_tokens=max_tokens, temperature=temperature, agent=agent)
            raise

    async def reason(self, messages: list[dict], *, json_mode: bool = True,
                     max_tokens: int = 1536, temperature: float = 0.0,
                     agent: str = "") -> ChatResult:
        try:
            return await self._chat(
                self.cfg.llama_url, self.cfg.llama_model, self._cb_llama,
                messages, json_mode=json_mode, max_tokens=max_tokens,
                temperature=temperature, agent=agent)
        except (CircuitOpenError, _InfraError):
            if self._fallback:
                return await self._fallback.reason(
                    messages, json_mode=json_mode,
                    max_tokens=max_tokens, temperature=temperature, agent=agent)
            raise

    async def embed(self, text: str | list[str], *,
                    agent: str = "") -> tuple[list[list[float]], GpuCallMetric]:
        cache_key = json.dumps(text, ensure_ascii=False)
        cached = await self._embed_cache.get(cache_key)
        if cached is not None:
            model_tag = self.cfg.local_embedder_model or self.cfg.bge_model
            return cached, GpuCallMetric(ts=_now(), model=f"cache:{model_tag}",
                                         latency_ms=0.0, agent=agent or None)

        if self.cfg.local_embedder_model:
            vectors, metric = await self._embed_local(text, agent=agent)
        elif self.cfg.bge_url:
            vectors, metric = await self._embed_remote(text, agent=agent)
        else:
            raise RuntimeError("embed() called but neither bge_url nor local_embedder_model is set")

        await self._embed_cache.put(cache_key, vectors)
        return vectors, metric

    def circuit_states(self) -> dict:
        """Return circuit breaker states for observability (/healthz, dashboard)."""
        states: dict = {
            "qwen":  self._cb_qwen.state,
            "llama": self._cb_llama.state,
        }
        if self._fallback:
            fallback_states = self._fallback.circuit_states()
            states["fallback"] = fallback_states
            states["fallback_active"] = any(
                s != "closed" for k, s in states.items()
                if k not in ("fallback", "fallback_active"))
        return states

    # ── Internals ──────────────────────────────────────────────────────────

    async def _chat(self, base_url: str, model: str, cb: _CircuitBreaker,
                    messages: list[dict], *, json_mode: bool,
                    max_tokens: int, temperature: float, agent: str = "") -> ChatResult:
        if not await cb.allow():
            raise CircuitOpenError(
                f"Circuit breaker OPEN for {cb.name} — endpoint considered down")

        body: dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        attempt = 0
        delay   = _BASE_DELAY_S
        last_exc: Exception = RuntimeError("unreachable")

        async with self._sem:
            while attempt <= self.cfg.max_retries:
                try:
                    started = time.perf_counter()
                    resp = await self._client.post(
                        f"{base_url}/chat/completions",
                        json=body, headers=self._auth_headers())

                    if resp.status_code in _RETRYABLE_STATUS:
                        await cb.on_failure()
                        if attempt >= self.cfg.max_retries:
                            resp.raise_for_status()  # raises HTTPStatusError
                        retry_after = float(resp.headers.get("retry-after", delay))
                        jitter = random.uniform(0, 0.3 * retry_after)
                        await asyncio.sleep(retry_after + jitter)
                        attempt += 1
                        delay = min(delay * 2, 30.0)
                        continue

                    resp.raise_for_status()
                    await cb.on_success()

                    payload = resp.json()
                    latency = (time.perf_counter() - started) * 1000
                    text    = payload["choices"][0]["message"]["content"]

                    usage        = payload.get("usage") or {}
                    in_tok       = usage.get("prompt_tokens") or usage.get("input_tokens")
                    out_tok      = usage.get("completion_tokens") or usage.get("output_tokens")
                    total_tok    = (in_tok or 0) + (out_tok or 0)
                    tps          = round(total_tok / (latency / 1000), 1) if total_tok and latency > 0 else None

                    # Fire VRAM snapshot concurrently with JSON parsing.
                    # _safe_json is fast (μs); _snapshot_vram hits rocm-smi (~ms).
                    vram_task = asyncio.ensure_future(_snapshot_vram())
                    parsed    = _safe_json(text) if json_mode else None
                    vram_gb, gpu_util = await vram_task
                    metric = GpuCallMetric(
                        ts=_now(), model=model, latency_ms=latency,
                        agent=agent or None,
                        input_tokens=in_tok, output_tokens=out_tok,
                        tokens_per_second=tps,
                        vram_used_gb=vram_gb,
                        gpu_util_pct=gpu_util,
                    )
                    return ChatResult(text=text, json=parsed, metric=metric)

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_exc = exc
                    await cb.on_failure()
                    if attempt >= self.cfg.max_retries:
                        raise _InfraError(str(exc)) from exc
                    jitter = random.uniform(0, 0.3 * delay)
                    await asyncio.sleep(delay + jitter)
                    attempt += 1
                    delay = min(delay * 2, 30.0)

        raise _InfraError(str(last_exc))

    async def _embed_remote(self, text: str | list[str],
                            agent: str = "") -> tuple[list[list[float]], GpuCallMetric]:
        started = time.perf_counter()
        resp = await self._client.post(
            f"{self.cfg.bge_url}/embeddings",
            json={"model": self.cfg.bge_model, "input": text},
            headers=self._auth_headers())
        resp.raise_for_status()
        data    = resp.json()["data"]
        latency = (time.perf_counter() - started) * 1000
        vectors = [d["embedding"] for d in data]
        return vectors, GpuCallMetric(ts=_now(), model=self.cfg.bge_model,
                                      latency_ms=latency, agent=agent or None)

    async def _embed_local(self, text: str | list[str],
                           agent: str = "") -> tuple[list[list[float]], GpuCallMetric]:
        inputs  = [text] if isinstance(text, str) else list(text)
        started = time.perf_counter()

        def _run():
            self._ensure_local_embedder()
            return list(self._local_embedder.embed(inputs))

        arrs    = await asyncio.to_thread(_run)
        latency = (time.perf_counter() - started) * 1000
        vectors = [a.tolist() for a in arrs]
        return vectors, GpuCallMetric(
            ts=_now(), model=f"local:{self.cfg.local_embedder_model}",
            latency_ms=latency, agent=agent or None)

    def _ensure_local_embedder(self) -> None:
        if self._local_embedder is not None:
            return
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "KYC_BACKEND=groq needs `fastembed`. Install: pip install fastembed") from e
        self._local_embedder = TextEmbedding(model_name=self.cfg.local_embedder_model)

    def _auth_headers(self) -> Optional[dict[str, str]]:
        return {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else None


class _InfraError(RuntimeError):
    """Network / timeout error after all retries — signals that the backend is unreachable."""


# ── Backend factory ───────────────────────────────────────────────────────────

def make_vllm_client() -> VllmClient:
    """Pick a backend from env vars and wire up the optional fallback.

    Primary backend:
        KYC_BACKEND=vllm   (default) — local vLLM on this box
        KYC_BACKEND=groq             — Groq cloud API

    Automatic fallback:
        KYC_FALLBACK_BACKEND=groq    — if primary is vLLM and this is set, a
        GROQ_API_KEY must also be present. When the vLLM circuit opens (e.g. the
        GPU OOM crashes during the demo) every subsequent request silently routes
        to Groq until vLLM recovers (HALF-OPEN probe succeeds).

    Scaling tunables (all optional, have sensible defaults):
        GROQ_MAX_CONCURRENCY   simultaneous in-flight requests (default 4 Groq / 16 vLLM)
        GROQ_MAX_RETRIES       retry attempts on 429/5xx (default 3)
        EMBED_CACHE_SIZE       LRU embedding cache capacity (default 4096)
        CB_FAILURE_THRESHOLD   failures before circuit opens (default 3)
        CB_RECOVERY_S          seconds in OPEN before HALF-OPEN probe (default 30)

    Groq-specific:
        GROQ_BASE_URL          (default https://api.groq.com/openai/v1)
        GROQ_VISION_MODEL      (default meta-llama/llama-4-scout-17b-16e-instruct)
        GROQ_REASON_MODEL      (default llama-3.3-70b-versatile)
        KYC_LOCAL_EMBEDDER     local fastembed model for embeddings (default BAAI/bge-small-en-v1.5)
    """
    backend          = os.environ.get("KYC_BACKEND", "vllm").lower()
    fallback_backend = os.environ.get("KYC_FALLBACK_BACKEND", "").lower()
    max_retries      = int(os.environ.get("GROQ_MAX_RETRIES",      "3"))
    embed_cache      = int(os.environ.get("EMBED_CACHE_SIZE",      "4096"))
    cb_threshold     = int(os.environ.get("CB_FAILURE_THRESHOLD",  "3"))
    cb_recovery      = float(os.environ.get("CB_RECOVERY_S",       "30"))

    def _groq_cfg(concurrency: int) -> VllmConfig:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY required for Groq backend")
        base = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return VllmConfig(
            qwen_url=base,
            qwen_model=os.environ.get("GROQ_VISION_MODEL",
                                      "meta-llama/llama-4-scout-17b-16e-instruct"),
            llama_url=base,
            llama_model=os.environ.get("GROQ_REASON_MODEL", "llama-3.3-70b-versatile"),
            bge_url=None,
            api_key=api_key,
            local_embedder_model=os.environ.get("KYC_LOCAL_EMBEDDER", "BAAI/bge-small-en-v1.5"),
            max_concurrency=concurrency,
            max_retries=max_retries,
            embed_cache_size=embed_cache,
            cb_failure_threshold=cb_threshold,
            cb_recovery_s=cb_recovery,
        )

    if backend in ("vllm", ""):
        concurrency = int(os.environ.get("GROQ_MAX_CONCURRENCY", "16"))
        primary_cfg = VllmConfig(
            max_concurrency=concurrency,
            max_retries=max_retries,
            embed_cache_size=embed_cache,
            cb_failure_threshold=cb_threshold,
            cb_recovery_s=cb_recovery,
        )
        # Wire Groq as automatic fallback when KYC_FALLBACK_BACKEND=groq is set.
        fallback_client: Optional[VllmClient] = None
        if fallback_backend == "groq":
            fallback_client = VllmClient(_groq_cfg(concurrency=4))
        return VllmClient(primary_cfg, fallback=fallback_client)

    if backend == "groq":
        # Groq free tier: ~30 req/min per model — keep concurrency low.
        concurrency = int(os.environ.get("GROQ_MAX_CONCURRENCY", "4"))
        return VllmClient(_groq_cfg(concurrency))

    raise RuntimeError(f"unknown KYC_BACKEND={backend!r} (expected 'vllm' or 'groq')")


# ── JSON helpers ──────────────────────────────────────────────────────────────

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
