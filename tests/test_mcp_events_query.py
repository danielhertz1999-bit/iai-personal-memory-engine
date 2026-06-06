"""Tests for MCP-05 events_query dispatch (Task 1).

events_query exposes the events table to users with a STRICT whitelist of
user-visible event kinds. Non-whitelisted kinds (e.g. s5_invariant_update)
are rejected with an error to prevent identity-kernel leakage.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp.core import dispatch
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore


def test_events_query_rejects_non_whitelisted_kind(tmp_path):
    """Identity-kernel kinds MUST be rejected (threat model)."""
    store = MemoryStore(path=tmp_path)
    write_event(
        store,
        kind="s5_invariant_update",
        data={"fact": "private"},
        severity="info",
    )
    out = dispatch(store, "events_query", {"kind": "s5_invariant_update"})
    assert "error" in out


def test_events_query_filters_kind(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(store, kind="s4_contradiction", data={"a": 1}, severity="warning")
    write_event(store, kind="trajectory_metric", data={"metric": "m1", "value": 1.0}, severity="info")
    write_event(store, kind="schema_induction_run", data={"pattern": "x"}, severity="info")

    out = dispatch(store, "events_query", {"kind": "s4_contradiction"})
    assert "events" in out
    assert len(out["events"]) == 1
    assert out["events"][0]["kind"] == "s4_contradiction"


def test_events_query_filters_since(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(store, kind="llm_health", data={"component": "test"}, severity="info")
    # future since -> zero
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    out = dispatch(store, "events_query", {"kind": "llm_health", "since": future})
    assert out["events"] == []


def test_events_query_filters_severity(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(store, kind="llm_health", data={}, severity="info")
    write_event(store, kind="llm_health", data={}, severity="warning")
    write_event(store, kind="llm_health", data={}, severity="critical")
    out = dispatch(store, "events_query", {"kind": "llm_health", "severity": "warning"})
    assert all(e["severity"] == "warning" for e in out["events"])


def test_events_query_respects_limit(tmp_path):
    store = MemoryStore(path=tmp_path)
    for i in range(10):
        write_event(store, kind="llm_health", data={"i": i}, severity="info")
    out = dispatch(store, "events_query", {"kind": "llm_health", "limit": 3})
    assert len(out["events"]) == 3


def test_events_query_default_limit(tmp_path):
    store = MemoryStore(path=tmp_path)
    for i in range(150):
        write_event(store, kind="llm_health", data={"i": i}, severity="info")
    out = dispatch(store, "events_query", {"kind": "llm_health"})
    # default limit = 100
    assert len(out["events"]) == 100


def test_events_query_crypto_key_rotated_whitelisted(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(
        store,
        kind="crypto_key_rotated",
        data={"source": "test"},
        severity="info",
    )
    out = dispatch(store, "events_query", {"kind": "crypto_key_rotated"})
    assert "error" not in out
    assert len(out["events"]) == 1


def test_events_query_ts_serialised_as_iso(tmp_path):
    """Timestamps are returned as ISO-8601 strings, not pandas Timestamps."""
    store = MemoryStore(path=tmp_path)
    write_event(store, kind="llm_health", data={}, severity="info")
    out = dispatch(store, "events_query", {"kind": "llm_health"})
    assert len(out["events"]) == 1
    assert isinstance(out["events"][0]["ts"], str)


def test_events_query_ordered_newest_first(tmp_path):
    store = MemoryStore(path=tmp_path)
    for i in range(5):
        write_event(store, kind="llm_health", data={"i": i}, severity="info")
    out = dispatch(store, "events_query", {"kind": "llm_health"})
    # Newest written last -> should appear first.
    indices = [e["data"].get("i") for e in out["events"]]
    assert indices == sorted(indices, reverse=True)
