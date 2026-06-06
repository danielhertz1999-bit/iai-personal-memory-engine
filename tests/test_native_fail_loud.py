"""Regression guard: native compute failures emit diagnostic events then raise.

Tests in this file:

1. topology dispatch (core.py): a native failure in graph build or sigma
   compute emits a ``topology_native_failed`` store event and re-raises.

2. topology empty-graph (core.py): the legitimate N=0 path returns the
   ``insufficient_data`` stub WITHOUT raising.

3. recall cue-encode (pipeline.py via core dispatch): a native encode failure
   emits an ``embed_native_failure`` store event and propagates as
   ``NativeError`` (not swallowed by the soft availability fallback).

4. capture encode (capture.py): a native encode failure emits an
   ``embed_native_failure`` store event and propagates.
"""
from __future__ import annotations

import pytest

from iai_mcp.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path) -> MemoryStore:
    """Return an empty MemoryStore using the test temp directory."""
    return MemoryStore(path=tmp_path)


def _seed_one_record(store: MemoryStore, text: str = "seed record") -> None:
    """Insert a minimal record so records_count > 0."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.types import EMBED_DIM, MemoryRecord

    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------


def test_topology_native_failure_emits_and_raises(tmp_path, monkeypatch):
    """topology dispatch: a native failure emits topology_native_failed then raises.

    Monkeypatches ``retrieve.build_runtime_graph`` to raise ``RuntimeError``
    so the test does not require a live native extension. Asserts:
    - the exception propagates (pytest.raises)
    - a ``topology_native_failed`` event was written to the store
    """
    from iai_mcp import core, retrieve
    from iai_mcp.events import flush_event_buffer, query_events

    store = _make_store(tmp_path)
    _seed_one_record(store)  # ensure records_count > 0

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated native build failure")

    monkeypatch.setattr(retrieve, "build_runtime_graph", _fail)

    with pytest.raises(RuntimeError, match="simulated native build failure"):
        core.dispatch(store, "topology", {})

    # Flush buffered events before querying.
    flush_event_buffer(store)

    rows = query_events(store, kind="topology_native_failed", limit=10)
    assert rows, (
        "no topology_native_failed event written; expected emit before raise"
    )
    ev = rows[0]
    data = ev.get("data") or {}
    assert "error_type" in data, f"event data missing error_type: {data}"
    assert "error" in data, f"event data missing error field: {data}"


def test_topology_empty_graph_returns_stub_without_raising(tmp_path):
    """topology dispatch: empty store returns insufficient_data stub, does not raise.

    The N=0 early-return path is a legitimate empty-graph case, NOT a native
    error. It must return the stub dict with ``regime == 'insufficient_data'``
    and ``sigma is None``, without emitting a failure event or raising.
    """
    from iai_mcp import core
    from iai_mcp.events import flush_event_buffer, query_events

    store = _make_store(tmp_path)
    # No records inserted -- count_rows() == 0.

    result = core.dispatch(store, "topology", {})

    assert result.get("regime") == "insufficient_data", (
        f"expected regime=insufficient_data on empty store, got: {result}"
    )
    assert result.get("sigma") is None, (
        f"expected sigma=None on empty store, got: {result.get('sigma')}"
    )
    assert result.get("N") == 0, (
        f"expected N=0 on empty store, got: {result.get('N')}"
    )

    # No topology_native_failed event should have been written.
    flush_event_buffer(store)
    rows = query_events(store, kind="topology_native_failed", limit=5)
    assert not rows, (
        "topology_native_failed event must NOT be emitted for the empty-graph "
        f"legitimate path; got {rows}"
    )


# ---------------------------------------------------------------------------
# Recall cue-encode Layer-2 test
# ---------------------------------------------------------------------------


def test_recall_cue_encode_failure_emits_store_event_and_raises(
    tmp_path, monkeypatch
):
    """recall dispatch: a native encode failure emits embed_native_failure and propagates.

    Monkeypatches the embedder's ``embed`` method to raise. Asserts:
    - the exception propagates past the dispatch recall path (not swallowed)
    - a store-backed ``embed_native_failure`` event was written (Layer 2)

    Drives via ``core.dispatch`` with ``memory_recall`` so the full dispatch
    path is exercised including the availability-fallback narrowing.
    """
    from iai_mcp import core
    from iai_mcp.embed import Embedder
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.exceptions import NativeError
    from iai_mcp.types import EMBED_DIM

    store = _make_store(tmp_path)
    _seed_one_record(store, "test content for recall")

    # Patch Embedder.embed to raise on every call.
    def _broken_embed(self, text: str) -> list[float]:
        raise RuntimeError("boom native encode")

    monkeypatch.setattr(Embedder, "_encode_one", lambda self, t: (_ for _ in ()).throw(RuntimeError("boom native encode")))
    monkeypatch.setattr(Embedder, "embed", _broken_embed)

    # Patch daemon_state so recall goes through the full pipeline branch.
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"current_state": "WAKE"},
    )
    monkeypatch.setattr(
        "iai_mcp.daemon_state.save_state",
        lambda s: None,
    )

    with pytest.raises((NativeError, RuntimeError)):
        core.dispatch(store, "memory_recall", {
            "cue": "test content",
            "session_id": "test-session",
        })

    # Check the store-backed embed_native_failure event was written (Layer 2).
    flush_event_buffer(store)
    rows = query_events(store, kind="embed_native_failure", limit=10)
    assert rows, (
        "no embed_native_failure store event written on recall cue-encode failure; "
        "expected Layer-2 store-backed emit before raise"
    )
    ev = rows[0]
    data = ev.get("data") or {}
    assert data.get("op_type") == "recall_cue", (
        f"expected op_type='recall_cue', got: {data}"
    )


# ---------------------------------------------------------------------------
# Capture encode Layer-2 test
# ---------------------------------------------------------------------------


def test_capture_encode_failure_emits_store_event_and_raises(
    tmp_path, monkeypatch
):
    """capture path: a native encode failure emits embed_native_failure and propagates.

    Monkeypatches the embedder's ``embed`` method to raise. Asserts:
    - the exception propagates from ``capture_turn`` (not swallowed)
    - a store-backed ``embed_native_failure`` event was written (Layer 2)
    """
    from iai_mcp.capture import capture_turn
    from iai_mcp.embed import Embedder
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.exceptions import NativeError

    store = _make_store(tmp_path)

    def _broken_embed(self, text: str) -> list[float]:
        raise RuntimeError("boom capture encode")

    monkeypatch.setattr(Embedder, "embed", _broken_embed)

    with pytest.raises((NativeError, RuntimeError)):
        capture_turn(
            store,
            cue="test cue",
            text="this is a long enough capture text to pass the minimum length guard",
            tier="episodic",
            session_id="test-session",
        )

    # Check the store-backed embed_native_failure event was written (Layer 2).
    flush_event_buffer(store)
    rows = query_events(store, kind="embed_native_failure", limit=10)
    assert rows, (
        "no embed_native_failure store event written on capture encode failure; "
        "expected Layer-2 store-backed emit before raise"
    )
    ev = rows[0]
    data = ev.get("data") or {}
    assert data.get("op_type") == "capture", (
        f"expected op_type='capture', got: {data}"
    )
