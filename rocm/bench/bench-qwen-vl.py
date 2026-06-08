"""
bench-qwen-vl.py — latency + tokens/sec for Qwen2.5-VL-72B (:8000) on a real
document-extraction prompt (vision + structured JSON out). Slide-4 baseline (§6).

    python bench-qwen-vl.py --image ../../personas/viktor-sanctions/passport.png --n 10
"""
from __future__ import annotations

import argparse
import base64
import statistics
import time
from pathlib import Path

import httpx

SYS = ("You are a precise KYC document parser. Output ONLY JSON with the passport "
       "fields and a numeric _confidence in [0,1].")


def _data_url(path: Path) -> str:
    b = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen2.5-vl-72b")
    ap.add_argument("--image", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    image_url = _data_url(Path(args.image))
    client = httpx.Client(timeout=180)
    latencies, tok_per_s = [], []
    for i in range(args.n):
        t0 = time.perf_counter()
        r = client.post(f"{args.url}/chat/completions", json={
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all fields. JSON only."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
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
    p = lambda xs, q: statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 1 else xs[0]
    print("[qwen2.5-vl-72b]")
    print(f"  latency  p50={p(latencies,50):.0f}ms  p95={p(latencies,95):.0f}ms")
    if tok_per_s:
        print(f"  output   {statistics.mean(tok_per_s):.1f} tok/s (mean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
