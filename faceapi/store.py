"""Subject store — generic (not attendance-specific) SQLite gallery.

A 'subject' is any enrolled person/identity. Each subject holds one or more
512-d embeddings (multi-prototype). Pure stdlib sqlite3, no ORM, so it stays
light on edge devices.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id   TEXT PRIMARY KEY,
    name         TEXT,
    metadata     TEXT,
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id   TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
    vector       TEXT NOT NULL,
    model        TEXT NOT NULL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_subject ON embeddings(subject_id);
"""


class SubjectStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── subjects ─────────────────────────────────────────────────────────────
    def upsert_subject(self, subject_id: str, name: str | None, metadata: dict | None) -> None:
        self._conn.execute(
            "INSERT INTO subjects(subject_id,name,metadata,created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(subject_id) DO UPDATE SET name=excluded.name, metadata=excluded.metadata",
            (subject_id, name, json.dumps(metadata or {}), time.time()))
        self._conn.commit()

    def add_embeddings(self, subject_id: str, vectors: list[np.ndarray], model: str) -> int:
        rows = [(subject_id, json.dumps([round(float(x), 6) for x in v]), model, time.time())
                for v in vectors]
        self._conn.executemany(
            "INSERT INTO embeddings(subject_id,vector,model,created_at) VALUES(?,?,?,?)", rows)
        self._conn.commit()
        return len(rows)

    def delete_subject(self, subject_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM subjects WHERE subject_id=?", (subject_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_subjects(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT s.subject_id, s.name, s.metadata, "
            "(SELECT COUNT(*) FROM embeddings e WHERE e.subject_id=s.subject_id) "
            "FROM subjects s ORDER BY s.created_at").fetchall()
        return [{"subject_id": r[0], "name": r[1], "metadata": json.loads(r[2] or "{}"),
                 "embeddings": r[3]} for r in rows]

    def count(self) -> tuple[int, int]:
        s = self._conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        e = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return s, e

    # ── gallery for matching ─────────────────────────────────────────────────
    def gallery(self) -> dict[str, tuple[str | None, np.ndarray]]:
        """Return {subject_id: (name, matrix[N,512])} for all embeddings."""
        rows = self._conn.execute(
            "SELECT e.subject_id, s.name, e.vector FROM embeddings e "
            "JOIN subjects s ON s.subject_id=e.subject_id").fetchall()
        acc: dict[str, tuple[str | None, list]] = {}
        for sid, name, vec in rows:
            acc.setdefault(sid, (name, []))[1].append(np.asarray(json.loads(vec), dtype=np.float32))
        return {sid: (name, np.stack(vs)) for sid, (name, vs) in acc.items()}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def subject_vectors(self, subject_id: str) -> np.ndarray | None:
        rows = self._conn.execute(
            "SELECT vector FROM embeddings WHERE subject_id=?", (subject_id,)).fetchall()
        if not rows:
            return None
        return np.stack([np.asarray(json.loads(r[0]), dtype=np.float32) for r in rows])
