"""
ingest.py — OpenSanctions bulk export → local SQLite + embeddings (§5.5).

Run once during prep (data is bundled, never API-called at runtime). Reads a
FollowTheMoney JSON-lines export, writes one row per entity into opensanctions.db,
and embeds `name + aliases` via the local BGE server (:8002) so screening_index.py
can do in-process vector recall.

    python ingest.py --input ../data/opensanctions/snapshot.jsonl \
                     --limit 100000 --bge http://localhost:8002/v1

Topics drive the sanctions / PEP / crime sub-agent filters (§4.4):
    sanction, sanction.linked     → sanctions sub-agent
    role.pep, gov.*               → PEP sub-agent
    crime.*                       → adverse-media sub-agent
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import httpx
import numpy as np

DEFAULT_DB = Path(__file__).parent / "opensanctions.db"
BGE_MODEL = "bge-large-en-v1.5"
BATCH = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="OpenSanctions FtM JSON-lines export")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--bge", default="http://localhost:8002/v1")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    conn = _open_db(Path(args.db))
    client = httpx.Client(timeout=120)

    batch_rows: list[dict] = []
    total = 0
    for entity in _read_entities(Path(args.input), args.limit):
        batch_rows.append(entity)
        if len(batch_rows) >= BATCH:
            total += _flush(conn, client, args.bge, batch_rows)
            batch_rows.clear()
            print(f"\r ingested {total} entities…", end="", file=sys.stderr)
    if batch_rows:
        total += _flush(conn, client, args.bge, batch_rows)

    conn.commit()
    conn.close()
    print(f"\n done. {total} entities → {args.db}", file=sys.stderr)
    return 0


def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entities ("
        "  id TEXT PRIMARY KEY, name TEXT, aliases TEXT, dob TEXT,"
        "  countries TEXT, topics TEXT, datasets TEXT, summary TEXT,"
        "  source_url TEXT, embedding BLOB)"
    )
    return conn


def _read_entities(path: Path, limit: int):
    """Yield normalized entity dicts from an FtM JSON-lines export.

    Only Person/Organization/LegalEntity schemas with a name are kept.
    """
    n = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema") not in ("Person", "Organization", "LegalEntity", "Company"):
                continue
            props = row.get("properties", {})
            names = props.get("name", [])
            if not names:
                continue
            yield {
                "id": row.get("id", ""),
                "name": names[0],
                "aliases": props.get("alias", []) + names[1:],
                "dob": (props.get("birthDate", []) or [None])[0],
                "countries": props.get("country", []),
                "topics": props.get("topics", []),
                "datasets": row.get("datasets", []),
                "summary": (props.get("notes", []) or [None])[0],
                "source_url": (props.get("sourceUrl", []) or [None])[0],
            }
            n += 1
            if limit and n >= limit:
                return


def _flush(conn: sqlite3.Connection, client: httpx.Client, bge_url: str,
           rows: list[dict]) -> int:
    texts = [f"{r['name']} {' '.join(r['aliases'][:5])}".strip() for r in rows]
    vectors = _embed(client, bge_url, texts)
    conn.executemany(
        "INSERT OR REPLACE INTO entities VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (r["id"], r["name"], json.dumps(r["aliases"]), r["dob"],
             json.dumps(r["countries"]), json.dumps(r["topics"]),
             json.dumps(r["datasets"]), r["summary"], r["source_url"],
             np.asarray(v, dtype=np.float32).tobytes())
            for r, v in zip(rows, vectors)
        ],
    )
    return len(rows)


def _embed(client: httpx.Client, bge_url: str, texts: list[str]) -> list[list[float]]:
    resp = client.post(f"{bge_url}/embeddings", json={"model": BGE_MODEL, "input": texts})
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


if __name__ == "__main__":
    raise SystemExit(main())
