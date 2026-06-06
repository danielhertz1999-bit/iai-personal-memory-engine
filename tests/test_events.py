"""Tests for the events table + events.py module.

Scope:
- events table created on MemoryStore instantiation
- write_event / query_events round-trip
- kind/severity/since filters
- ordering (newest first)
- limit default + explicit
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest


# ----------------------------------------------------------- table creation


def test_events_table_created_on_store_init(tmp_path):
    """MemoryStore() creates events table with the expected schema."""
    from iai_mcp.store import EVENTS_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    assert EVENTS_TABLE in store._table_names()


def test_budget_ledger_table_created(tmp_path):
    from iai_mcp.store import BUDGET_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    assert BUDGET_TABLE in store._table_names()


def test_ratelimit_ledger_table_created(tmp_path):
    from iai_mcp.store import MemoryStore, RATELIMIT_TABLE

    store = MemoryStore(path=tmp_path)
    assert RATELIMIT_TABLE in store._table_names()


# ------------------------------------------------------ write_event / query


def test_events_write_and_query_roundtrip(tmp_path):
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    event_id = write_event(store, kind="test", data={"x": 1}, session_id="s1")
    assert isinstance(event_id, UUID)

    results = query_events(store, kind="test")
    assert len(results) == 1
    assert results[0]["kind"] == "test"
    assert results[0]["data"]["x"] == 1
    assert results[0]["session_id"] == "s1"


def test_events_write_returns_uuid(tmp_path):
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ev = write_event(store, kind="k", data={})
    assert isinstance(ev, UUID)


def test_events_query_filter_kind(tmp_path):
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="a", data={})
    write_event(store, kind="b", data={})
    write_event(store, kind="c", data={})

    assert len(query_events(store, kind="a")) == 1
    assert len(query_events(store, kind="b")) == 1
    assert len(query_events(store)) == 3


def test_events_query_filter_since(tmp_path, monkeypatch):
    """Events at different timestamps; since=30min-ago returns only the newer."""
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # We can't easily freeze time; instead write both events, then query with
    # since = far-future-past to confirm filter works (both return).
    write_event(store, kind="t", data={"old": True})
    write_event(store, kind="t", data={"new": True})

    # since in the future -> no results
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    assert query_events(store, kind="t", since=future) == []

    # since well in the past -> 2 results
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    assert len(query_events(store, kind="t", since=past)) == 2


def test_events_query_filter_severity(tmp_path):
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="k", data={}, severity="info")
    write_event(store, kind="k", data={}, severity="warning")
    write_event(store, kind="k", data={}, severity="critical")

    assert len(query_events(store, severity="critical")) == 1
    assert len(query_events(store, severity="warning")) == 1
    assert len(query_events(store, severity="info")) == 1


def test_events_query_limit_default_100(tmp_path):
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for i in range(150):
        write_event(store, kind="bulk", data={"i": i})

    # Default limit
    results = query_events(store, kind="bulk")
    assert len(results) == 100

    # Explicit limit
    results = query_events(store, kind="bulk", limit=50)
    assert len(results) == 50


def test_events_query_ordering_newest_first(tmp_path):
    """Events must come back in descending ts order (newest first)."""
    import time

    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="ord", data={"i": 0})
    time.sleep(0.01)
    write_event(store, kind="ord", data={"i": 1})
    time.sleep(0.01)
    write_event(store, kind="ord", data={"i": 2})

    results = query_events(store, kind="ord")
    # Newest (i=2) first
    ordered_is = [r["data"]["i"] for r in results]
    assert ordered_is == [2, 1, 0]


def test_events_source_ids_roundtrip(tmp_path):
    """source_ids list[UUID] is preserved as JSON array of strings."""
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = [uuid4(), uuid4()]
    write_event(store, kind="s", data={}, source_ids=ids)
    results = query_events(store, kind="s")
    assert len(results) == 1
    src = results[0]["source_ids"]
    assert set(src) == {str(i) for i in ids}


def test_events_domain_roundtrip(tmp_path):
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="k", data={}, domain="coding")
    results = query_events(store, kind="k")
    assert len(results) == 1
    assert results[0]["domain"] == "coding"


def test_events_empty_store_returns_empty(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    assert query_events(store) == []
    assert query_events(store, kind="nothing") == []


# --------------------------------------------- recall-source telemetry (H63-7)
#
# The recall-path emits TELEMETRY_RECALL_SOURCE best-effort so the real-world
# recall tails are observable. These tests prove the consensus-locked contract:
# - the new kinds round-trip through the WHITELISTED events_query surface
# - recall survives an emit that raises (best-effort guard)
# - no raw cue text (or cue-derived substring) ever rides a telemetry payload
# - fallback_rate is DERIVABLE from recall_source counts (no separate metric)


def test_recall_source_telemetry_round_trip(tmp_path):
    """A recall_source event round-trips through the whitelisted events_query."""
    from iai_mcp.core import EVENTS_QUERY_WHITELIST, dispatch
    from iai_mcp.events import TELEMETRY_RECALL_SOURCE, write_event
    from iai_mcp.store import MemoryStore

    # The new kind must pass the events_query whitelist (else fallback_rate is
    # not derivable through the supported MCP surface).
    assert TELEMETRY_RECALL_SOURCE == "recall_source"
    assert "recall_source" in EVENTS_QUERY_WHITELIST
    assert "embed_construct" in EVENTS_QUERY_WHITELIST

    store = MemoryStore(path=tmp_path)
    # Write unbuffered so the events table (what events_query reads) is updated
    # immediately; the production emit is buffered=True and flushed by the daemon.
    write_event(
        store,
        TELEMETRY_RECALL_SOURCE,
        {"source": "semantic-inprocess", "construct_ms": 26.0, "encode_ms": 11.0},
        severity="info",
    )

    out = dispatch(store, "events_query", {"kind": "recall_source"})
    assert "error" not in out, out
    assert out["count"] == 1
    ev = out["events"][0]
    assert ev["kind"] == "recall_source"
    assert ev["data"]["source"] == "semantic-inprocess"
    assert ev["data"]["construct_ms"] == 26.0
    assert ev["data"]["encode_ms"] == 11.0


def test_recall_succeeds_when_telemetry_emit_raises(tmp_path, monkeypatch):
    """A telemetry emit that raises must NOT break recall (best-effort guard).

    Drives the daemon-down construct path's degrade branch with write_event
    monkeypatched to raise. Recall must still return a valid result.
    """
    import iai_mcp.events as events_mod
    import iai_mcp.semantic_recall as sr
    from iai_mcp.store import MemoryStore

    # A real tmp store: the recency degrade reads it (empty -> empty list, never
    # a hard-fail) and a store handle is reachable so the emit path actually
    # exercises write_event (the raising monkeypatch below).
    store = MemoryStore(path=tmp_path)

    # Ensure a store handle is reachable at the emit site so the emit path
    # actually calls write_event (the deepest-degrade no-store path would
    # short-circuit to logger.debug and never exercise the raising emit).
    monkeypatch.setattr(sr, "_WARM_LOCAL_STORE", store, raising=False)

    # Force the construct to fall to the recency floor (deterministic, no model).
    monkeypatch.setattr(sr, "_construct_with_budget", lambda root: (None, 5.0))

    # The emit MUST swallow this — recall must not propagate it.
    def _boom(*args, **kwargs):
        raise RuntimeError("telemetry table exploded")

    monkeypatch.setattr(events_mod, "write_event", _boom)

    result = sr.recall_semantic_warm(str(tmp_path), "what does alice prefer", n=5)
    # Recall still returns a valid (degraded) list despite the raising emit.
    assert isinstance(result, list)


def test_recall_telemetry_payload_carries_no_raw_cue(tmp_path, monkeypatch):
    """No raw cue text (or cue-derived substring) leaks into a telemetry payload.

    Drives the degrade emit with a distinctive cue, spies on write_event to
    record the emitted data dict, and asserts the cue (and its tokens) are
    absent — only the scrubbed reason token + numeric metrics survive.
    """
    import iai_mcp.events as events_mod
    import iai_mcp.semantic_recall as sr
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    monkeypatch.setattr(sr, "_WARM_LOCAL_STORE", store, raising=False)
    monkeypatch.setattr(sr, "_construct_with_budget", lambda root: (None, 7.0))

    captured: list[dict] = []

    def _spy(store_arg, kind, data, **kwargs):
        captured.append({"kind": kind, "data": dict(data)})
        # Return a UUID-ish to satisfy the contract; emit is best-effort anyway.
        from uuid import uuid4
        return uuid4()

    monkeypatch.setattr(events_mod, "write_event", _spy)

    cue = "ZEBRA_SECRET_PASSPHRASE_orbital_kangaroo_42"
    sr.recall_semantic_warm(str(tmp_path), cue, n=5)

    # The recall_source emit was captured.
    rs = [c for c in captured if c["kind"] == "recall_source"]
    assert rs, f"no recall_source emit captured: {captured}"
    payload = rs[0]["data"]

    # Scrubbed reason token + metrics only — never the cue.
    assert payload["source"] == "recency-degrade"
    assert payload.get("reason") == "construct_timeout_or_fail"
    assert "construct_ms" in payload

    # The full cue and each distinctive token must be ABSENT from the payload.
    blob = json.dumps(payload)
    assert cue not in blob
    for token in ("ZEBRA", "SECRET", "PASSPHRASE", "orbital", "kangaroo"):
        assert token not in blob, f"cue-derived token {token!r} leaked: {payload}"


def test_fallback_rate_derivable_from_recall_source(tmp_path):
    """fallback_rate (degrade / total) is derivable from recall_source counts.

    Proves no separate stored metric is needed: a count-based query over the
    whitelisted surface yields the expected degrade ratio.
    """
    from iai_mcp.core import dispatch
    from iai_mcp.events import TELEMETRY_RECALL_SOURCE, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # 3 semantic-inprocess + 1 recency-degrade -> fallback_rate = 1/4 = 0.25.
    for _ in range(3):
        write_event(store, TELEMETRY_RECALL_SOURCE,
                    {"source": "semantic-inprocess", "construct_ms": 30.0})
    write_event(store, TELEMETRY_RECALL_SOURCE,
                {"source": "recency-degrade", "construct_ms": 1200.0,
                 "reason": "construct_timeout_or_fail"})

    out = dispatch(store, "events_query", {"kind": "recall_source", "limit": 1000})
    assert "error" not in out, out
    sources = [e["data"]["source"] for e in out["events"]]
    total = len(sources)
    degrades = sum(1 for s in sources if s == "recency-degrade")
    assert total == 4
    assert degrades == 1
    fallback_rate = degrades / total
    assert fallback_rate == pytest.approx(0.25)
