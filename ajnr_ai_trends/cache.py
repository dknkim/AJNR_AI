"""Tiny SQLite-backed key/value JSON cache.

Used to make every network/LLM call idempotent: the same key always returns the
same payload without re-hitting the API. Keys are arbitrary strings (we use
EIDs, DOIs, or hashes of LLM prompts); values are JSON-serializable objects.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL,"
            " ts REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv(key, value, ts) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self._conn.commit()

    def has(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM kv WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def keys(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT key FROM kv").fetchall()
        return [r[0] for r in rows]

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
