"""SQLite summaries for the local application."""

import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter
from uuid import uuid4

HISTORY_PATH = Path("data/runtime/runs.sqlite3")


class RunHistory:
    def __init__(self, path):
        self.path = path

    def connect(self):
        return sqlite3.connect(self.path)

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    duration_ms REAL NOT NULL,
                    citation_count INTEGER NOT NULL,
                    tool_call_count INTEGER NOT NULL
                )
            """)

    def record(self, kind, result, started, *, run_id=None):
        timestamp, clock = started
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id or uuid4().hex, kind, timestamp, result["status"], result.get("reason"),
                 round((perf_counter() - clock) * 1000, 3), len(result.get("citations", [])),
                 len(result.get("tool_calls", []))),
            )

    def recent(self, limit):
        with closing(self.connect()) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM runs ORDER BY rowid DESC LIMIT ?", (limit,))
            return [dict(row) for row in rows]
