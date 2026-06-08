"""
metrics.py — per-agent + per-GPU-call instrumentation (§6).

Time every agent step so latency/token/model data lands in case state from day 1
(slide-4 evidence + Technical-Implementation rubric). Never bolt this on later.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from schemas import AgentMetric, CaseMetrics, GpuCallMetric


@contextmanager
def timed(metrics: CaseMetrics, agent: str, model: str | None = None) -> Iterator[list[GpuCallMetric]]:
    """Record an agent's wall-clock latency; collect any GPU-call metrics it appends.

    Usage:
        with timed(state.metrics, "extraction", model=QWEN) as gpu:
            out, calls = await run_extraction(...)
            gpu.extend(calls)
    """
    gpu: list[GpuCallMetric] = []
    started = time.perf_counter()
    try:
        yield gpu
    finally:
        metrics.per_agent[agent] = AgentMetric(
            latency_ms=(time.perf_counter() - started) * 1000, model=model
        )
        metrics.per_gpu_call.extend(gpu)
