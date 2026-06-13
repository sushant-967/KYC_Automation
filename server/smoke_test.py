"""
smoke_test.py — exercise the full pipeline with a stubbed vLLM (no GPU).

Proves the agentic wiring end-to-end: intake → extraction → entity → screening ‖
id ‖ financial → risk → explainability → decision, with deep-agent calls faked.
Run:  python smoke_test.py
"""
from __future__ import annotations

import asyncio

from schemas import CaseState, CaseMetrics, Submission
from pipeline import PipelineIO, run_pipeline
from agents.intake import run_intake
from vllm_client import GpuCallMetric


class FakeVllm:
    """Canned responses standing in for the three vLLM servers."""

    async def extract(self, messages, **kw):
        return _R({"name": "Viktor Nazarov", "dob": "1979-04-02",
                   "passportNumber": "C1234567", "expiry": "2030-01-01",
                   "_confidence": 0.93}, "qwen")

    async def reason(self, messages, **kw):
        # Screening adjudication or explanation — return a benign no-match / summary.
        return _R({"matches": []}, "llama")

    async def embed(self, text):
        return [[0.0] * 1024], GpuCallMetric(ts="t", model="bge", latency_ms=1.0)

    async def aclose(self):
        pass


class _R:
    def __init__(self, j, model):
        self.json = j
        self.text = ""
        self.metric = GpuCallMetric(ts="t", model=model, latency_ms=1.0)


class EmptyIndex:
    rows: list = []
    def recall(self, *a, **k):
        return []


async def main() -> int:
    submission = Submission.model_validate({
        "customer": {"full_name": "Viktor Nazarov", "dob": "1979-04-02",
                     "nationality": "cyprus", "declared_income": 4200000,
                     "declared_employment": "consultant"},
        "documents": [{"kind": "passport", "file_id": "viktor-passport.png"}],
    })
    intake = run_intake("case-smoke", submission)
    state = CaseState(case_id="case-smoke", status="intake",
                      customer=intake.customer, documents=intake.documents,
                      metrics=CaseMetrics())
    state.agent_outputs.intake = intake

    events: list[str] = []
    io = PipelineIO(
        load_image=lambda fid: _img(),
        entity_index=EmptyIndex(),
        prior_case_lookup=lambda n: [],
        emit=lambda agent, status, payload: events.append(f"{agent}:{status}"),
    )
    state = await run_pipeline(state, FakeVllm(), io)

    print("events:", " -> ".join(events))
    print("status:", state.status)
    print("risk score:", state.agent_outputs.risk.score)
    print("decision:", state.agent_outputs.decision.decision)
    print("contributors:", [c.signal for c in state.agent_outputs.risk.contributors])
    print("per-agent latencies:", {k: round(v.latency_ms, 1) for k, v in state.metrics.per_agent.items()})
    print("end-to-end ms:", round(state.metrics.end_to_end_ms or 0, 1))

    assert state.agent_outputs.decision is not None, "no decision produced"
    assert state.agent_outputs.explanation is not None, "no explanation produced"
    assert state.agent_outputs.explanation.dag_nodes, "DAG must have at least one node"
    assert state.status in {
        "approved", "awaiting_human", "escalated",
        "awaiting_documents", "awaiting_id_review",
    }, state.status
    print("\nPASS — full pipeline ran end-to-end with stubbed vLLM.")
    return 0


async def _img() -> str:
    return "data:image/png;base64,"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
