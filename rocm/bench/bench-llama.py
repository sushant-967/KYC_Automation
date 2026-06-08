"""
bench-llama.py — throughput + latency smoke test for Llama-3.3-70B (:8001).

Captures baseline tokens/sec and p50/p95 latency for slide 4 (§6). Run after the
model is up:  python bench-llama.py --n 20
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx

PROMPT = ("You are a KYC adjudicator. Decide if 'Viktor Nazarov' (DOB 1979) matches "
          "a UN-sanctioned 'Viktor A. Nazarov' (DOB 1979, Cyprus). Reply with a short "
          "JSON {verdict, confidence, rationale}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    client = httpx.Client(timeout=120)
    latencies, tok_per_s = [], []
    for i in range(args.n):
        t0 = time.perf_counter()
        r = client.post(f"{args.url}/chat/completions", json={
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": args.max_tokens, "temperature": 0,
        })
        r.raise_for_status()
        dt = time.perf_counter() - t0
        out_tokens = r.json().get("usage", {}).get("completion_tokens", 0)
        latencies.append(dt * 1000)
        if out_tokens:
            tok_per_s.append(out_tokens / dt)
        print(f"\r {i+1}/{args.n}", end="")

    print()
    _report("llama-3.3-70b", latencies, tok_per_s)
    return 0


def _report(name, latencies, tok_per_s):
    p = lambda xs, q: statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 1 else xs[0]
    print(f"[{name}]")
    print(f"  latency  p50={p(latencies,50):.0f}ms  p95={p(latencies,95):.0f}ms")
    if tok_per_s:
        print(f"  output   {statistics.mean(tok_per_s):.1f} tok/s (mean)")


if __name__ == "__main__":
    raise SystemExit(main())
