"""SQLite-backed relational store for structured research records."""
import json
import sqlite3
from pathlib import Path
from typing import Any

from config import config


class RelationalStore:
    """Stores research sessions, queries, and reports in SQLite."""

    def __init__(self) -> None:
        Path(config.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(config.sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                results_json TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        self._conn.commit()

    # --- sessions ---
    def create_session(self, session_id: str, topic: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (id, topic) VALUES (?, ?)",
            (session_id, topic),
        )
        self._conn.commit()

    def update_session_status(self, session_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET status=?, updated_at=datetime('now') WHERE id=?",
            (status, session_id),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    # --- queries ---
    def save_query(self, session_id: str, query_text: str, results: list[dict[str, Any]]) -> None:
        self._conn.execute(
            "INSERT INTO queries (session_id, query_text, results_json) VALUES (?, ?, ?)",
            (session_id, query_text, json.dumps(results, ensure_ascii=False)),
        )
        self._conn.commit()

    def get_queries(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM queries WHERE session_id=? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- reports ---
    def save_report(self, session_id: str, content: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO reports (session_id, content) VALUES (?, ?)",
            (session_id, content),
        )
        self._conn.commit()

    def get_report(self, session_id: str) -> dict[str, Any] | None:
        if row := self._conn.execute(
            "SELECT * FROM reports WHERE session_id=?", (session_id,)
        ).fetchone():
            return dict(row)
        return None


db = RelationalStore()
