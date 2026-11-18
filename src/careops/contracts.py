"""Small dependency-free checks for the foundation contracts."""

from __future__ import annotations

from collections.abc import Mapping


def require_keys(value: Mapping[str, object], keys: tuple[str, ...]) -> None:
    """Raise a clear error when a contract object is missing required keys."""
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")


def validate_evidence_packet(value: Mapping[str, object]) -> None:
    require_keys(value, ("query", "items"))
    if not isinstance(value["query"], str) or not value["query"].strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(value["items"], list):
        raise ValueError("items must be a list")
    for item in value["items"]:
        if not isinstance(item, Mapping):
            raise ValueError("evidence items must be objects")
        require_keys(item, ("chunk_id", "source_id", "text", "score"))


def validate_agent_result(value: Mapping[str, object]) -> None:
    require_keys(value, ("status", "summary", "citations", "tool_calls"))
    if value["status"] not in ("review", "abstain", "escalate"):
        raise ValueError("invalid agent status")
    if not isinstance(value["citations"], list) or not isinstance(value["tool_calls"], list):
        raise ValueError("citations and tool_calls must be lists")
