---
description: Run the agentic pipeline smoke test (stubbed vLLM, no GPU)
---

Verify the agent pipeline wiring end-to-end without a GPU by running the stubbed
smoke test, then report the result.

Run: `cd server && python smoke_test.py`

Report the event ordering, final status, risk score, decision, and PASS/FAIL. If it
fails, diagnose against `server/pipeline.py` and the agent contracts in
`server/schemas.py` — do not modify the test to make it pass.
