"""
store.py — SQLite-backed case store (replaces the Cloudflare Durable Object).

One row per case holds the full CaseState JSON; a separate append-only table is
the audit log so it survives process restarts (§5: "audit log is persistent even
if the DO restarts"). SQLite with WAL is plenty for a single-box demo.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schemas import CaseState, AuditEvent

DB_PATH = Path(__file__).parent / "kyc.db"


class CaseStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cases ("
            "  case_id TEXT PRIMARY KEY,"
            "  status  TEXT NOT NULL,"
            "  state   TEXT NOT NULL,"  # CaseState as JSON
            "  updated TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS audit ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  case_id TEXT NOT NULL,"
            "  ts TEXT NOT NULL,"
            "  agent TEXT NOT NULL,"
            "  event TEXT NOT NULL,"
            "  payload TEXT)"
        )
        self._conn.commit()

    def save(self, state: CaseState) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO cases(case_id,status,state,updated) VALUES(?,?,?,?) "
                "ON CONFLICT(case_id) DO UPDATE SET status=excluded.status,"
                " state=excluded.state, updated=excluded.updated",
                (state.case_id, state.status, state.model_dump_json(),
                 datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def get(self, case_id: str) -> Optional[CaseState]:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
        return CaseState.model_validate_json(row[0]) if row else None

    def append_audit(self, case_id: str, event: AuditEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit(case_id,ts,agent,event,payload) VALUES(?,?,?,?,?)",
                (case_id, event.ts, event.agent, event.event,
                 json.dumps(event.payload) if event.payload is not None else None),
            )
            self._conn.commit()

    def list_ids(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT case_id FROM cases ORDER BY updated DESC").fetchall()]
