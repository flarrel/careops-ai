"""Persist a minimal, redacted event stream for local review runs."""

import hmac
import json
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_bytes
from threading import Lock
from time import perf_counter

TRACE_PATH = Path("data/runtime/agent-traces.jsonl")
TRACE_KEY = token_bytes(32)


def fingerprint(task):
    payload = json.dumps(task, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hmac.digest(TRACE_KEY, payload, "sha256").hex()


def start_event():
    return datetime.now(UTC).isoformat(), perf_counter()


class JsonlTrace:
    def __init__(self, path=TRACE_PATH):
        self.path = path
        self.lock = Lock()

    def __call__(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")


def record(trace, run_id, task_hash, stage, outcome, start, **details):
    started_at, clock = start
    trace({
        "run_id": run_id,
        "started_at": started_at,
        "stage": stage,
        "outcome": outcome,
        "duration_ms": round((perf_counter() - clock) * 1000, 3),
        "redacted_input_hash": task_hash,
        **details,
    })
