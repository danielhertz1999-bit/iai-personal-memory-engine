"""Tests for the events LanceDB table + events.py module (, D-STORAGE).

Covers:
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
    """MemoryStore() creates events table with the D-STORAGE schema."""
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
