"""
screening_index.py — local recall over the bundled OpenSanctions entities (§4.4).

Two backends, same interface:
  * numpy brute-force cosine over an in-memory float32 matrix (default; ~100K×1024
    is ~400 MB and a query is a single matmul — a few ms on this box).
  * FAISS index if available (drop-in upgrade for larger corpora).

Entities + their embeddings are produced by ingest.py and stored in SQLite. On
startup we load them once into memory.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DB_PATH = Path(__file__).parent / "opensanctions.db"


@dataclass
class EntityRow:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    dob: Optional[str] = None
    countries: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    source_url: Optional[str] = None


class ScreeningIndex:
    """In-memory recall over OpenSanctions vectors. Load once at startup."""

    def __init__(self, db_path: Path = DB_PATH):
        self.rows: list[EntityRow] = []
        self._matrix: Optional[np.ndarray] = None  # (N, D) L2-normalized float32
        self._topics: list[set[str]] = []
        if db_path.exists():
            self._load(db_path)

    def _load(self, db_path: Path) -> None:
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT id,name,aliases,dob,countries,topics,datasets,summary,source_url,embedding "
            "FROM entities"
        )
        vecs: list[np.ndarray] = []
        for r in cur:
            self.rows.append(EntityRow(
                id=r[0], name=r[1], aliases=json.loads(r[2] or "[]"), dob=r[3],
                countries=json.loads(r[4] or "[]"), topics=json.loads(r[5] or "[]"),
                datasets=json.loads(r[6] or "[]"), summary=r[7], source_url=r[8],
            ))
            self._topics.append(set(json.loads(r[5] or "[]")))
            vecs.append(np.frombuffer(r[9], dtype=np.float32))
        conn.close()
        if vecs:
            m = np.vstack(vecs)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            self._matrix = (m / np.clip(norms, 1e-8, None)).astype(np.float32)

    def recall(self, query_vector: list[float], topic_prefixes: list[str], k: int = 20) -> list[EntityRow]:
        """Top-k entities by cosine similarity, restricted to entities whose topics
        match any of the given prefixes (e.g. ['sanction'], ['role.pep','gov'])."""
        if self._matrix is None or not self.rows:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-8)

        mask = np.array([_topic_match(t, topic_prefixes) for t in self._topics])
        if not mask.any():
            return []
        sims = self._matrix @ q
        sims[~mask] = -np.inf
        top = np.argpartition(-sims, min(k, mask.sum() - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [self.rows[i] for i in top if np.isfinite(sims[i])]


def _topic_match(topics: set[str], prefixes: list[str]) -> bool:
    return any(t == p or t.startswith(p + ".") for t in topics for p in prefixes)
