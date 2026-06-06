"""Recall hits must expose session_id + captured_at.

Verifies that _hit_to_json / MemoryHit carry the session_id + captured_at fields.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from tests.conftest_recall import make_tmp_store


def test_recall_hit_carries_session_id_and_captured_at(tmp_path):
    """A recall hit for a record captured with a known session_id must
    expose that session_id (non-null) and a parseable captured_at timestamp.

    _hit_to_json (core.py) includes session_id and captured_at in the hit dict.
    """
    store = make_tmp_store(tmp_path)

    result = capture_turn(
        store,
        cue="known user line phase59",
        text="known user line phase59 distinctive text",
        tier="episodic",
        session_id="sess-A1-phase59",
        role="user",
    )
    assert result["status"] == "inserted", f"capture failed: {result}"

    recall = dispatch(
        store,
        "memory_recall",
        {"cue": "known user line phase59"},
    )
    hits = recall.get("hits", [])
    assert hits, "expected at least one recall hit"

    top = hits[0]

    # These assertions are RED today: both keys are absent from _hit_to_json.
    assert top.get("session_id") == "sess-A1-phase59", (
        f"hit session_id not surfaced (got {top.get('session_id')!r}); "
        "session_id must be added to _hit_to_json / MemoryHit"
    )

    captured_at = top.get("captured_at")
    assert captured_at is not None, (
        "hit captured_at is null; it must be populated from record.created_at"
    )
    # Must parse as ISO-8601 UTC.
    dt = datetime.fromisoformat(captured_at)
    assert dt.tzinfo is not None, f"captured_at must be timezone-aware, got {captured_at!r}"
