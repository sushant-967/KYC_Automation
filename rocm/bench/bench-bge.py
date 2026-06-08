"""
bench-bge.py — embedding throughput for BGE-large (:8002).
Reports embeddings/sec at a few batch sizes — feeds the ingest-time sizing.
    python bench-bge.py
"""
from __future__ import annotations

import argparse
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8002/v1")
    ap.add_argument("--model", default="bge-large-en-v1.5")
    args = ap.parse_args()

    client = httpx.Client(timeout=120)
    sample = "Viktor Nazarov"
    for batch in (1, 8, 32, 64):
        inputs = [f"{sample} {i}" for i in range(batch)]
        t0 = time.perf_counter()
        r = client.post(f"{args.url}/embeddings", json={"model": args.model, "input": inputs})
        r.raise_for_status()
        dt = time.perf_counter() - t0
        dim = len(r.json()["data"][0]["embedding"])
        print(f"batch={batch:>3}  {batch/dt:8.1f} emb/s  dim={dim}  ({dt*1000:.0f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
