"""Read a small run window through the CareOps REST API."""

from collections import Counter
import os

import httpx


def fetch_summary():
    root = os.environ.get("CAREOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")
    response = httpx.get(f"{root}/runs", params={"limit": 10}, timeout=5, trust_env=False)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError("Expected an array of run summaries")
    outcomes = Counter(row["status"] for row in rows)
    if not all(isinstance(key, str) for key in outcomes):
        raise ValueError("Expected a status string for each run")
    return {"runs": len(rows), "outcomes": dict(outcomes)}
