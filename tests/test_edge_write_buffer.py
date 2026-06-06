"""Tests for the edges write-buffer infrastructure.

Scope:
- boost_edges new-row inserts buffer rows in _edge_buffer, not the store directly
- flush_edge_buffer drains N rows in one batch, returns N, empties buffer
- flush_edge_buffer on empty buffer returns 0 and does not raise (idempotent)
- should_flush_edge_buffer size-threshold helper (env var IAI_MCP_EDGE_BUFFER_MAX)
- CRITICAL: merge_insert path in boost_edges REMAINS synchronous (static check)
- CRITICAL: boost_edges return value (new_weights dict) is built locally and
  includes both update + insert rows regardless of buffer state
- add_contradicts_edge appends to _edge_buffer instead of writing to the store
- lance_buffer_flush telemetry event emitted with {table: "edges", count: N}
- Static source check: _edge_buffer.setdefault used at exactly 2 call sites
- daemon.py wiring presence: periodic-tick, WAKE drain, shutdown
- REGRESSION: contradict() on a buffered-but-not-yet-flushed record does not
  raise "unknown record" (SRC durability)
- REGRESSION: new_rec created by contradict() is durable immediately after
  return (DST durability) — temporal-validity hydration can find it
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    """Buffer-internals tests assert un-flushed state — disable the
    conftest-level autoflush patch for every test in this file."""
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

import pytest


# ----------------------------------------------------------- helpers


def _clear_edge_buffer(store) -> None:
    """Pop any leftover buffer state for this store id."""
    from iai_mcp import store as store_mod

    store_mod._edge_buffer.pop(id(store), None)
    store_mod._edge_last_flush_at.pop(id(store), None)


# ----------------------------------------------------------- Test 1


def test_boost_edges_insert_buffers_rows_not_lancedb(tmp_path):
    """boost_edges with only new (src,dst,edge_type) pairs buffers rows, does not write immediately."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        tbl = store.db.open_table(EDGES_TABLE)
        n_before = len(tbl.to_pandas())

        src_id = uuid4()
        dst_id = uuid4()

        # Default threshold is 100 — a single pair will land in buffer, not the store.
        store.boost_edges([(src_id, dst_id)], delta=0.5, edge_type="hebbian")

        # Buffer must hold the new row.
        buf = store_mod._edge_buffer.get(id(store), [])
        assert len(buf) >= 1, "expected _edge_buffer to accumulate rows after boost_edges insert"

        # The store table must be unchanged.
        tbl = store.db.open_table(EDGES_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before, (
            f"boost_edges insert changed LanceDB row count before flush: {n_before} -> {n_after}"
        )


# ----------------------------------------------------------- Test 2


def test_flush_edge_buffer_writes_batch_and_clears(tmp_path):
    """flush_edge_buffer drains N buffered rows in one batch tbl.add, returns N, empties buffer."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        tbl = store.db.open_table(EDGES_TABLE)
        n_before = len(tbl.to_pandas())

        # Append 3 rows directly to the buffer (bypass boost_edges to avoid threshold auto-flush).
        for i in range(3):
            row = {
                "src": str(uuid4()),
                "dst": str(uuid4()),
                "edge_type": "hebbian",
                "weight": float(i + 1) * 0.1,
                "updated_at": datetime.now(timezone.utc),
            }
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        assert len(store_mod._edge_buffer.get(id(store), [])) == 3

        flushed = flush_edge_buffer(store)
        assert flushed == 3

        # Buffer is empty after flush.
        assert not store_mod._edge_buffer.get(id(store))

        # Rows landed in the store.
        tbl = store.db.open_table(EDGES_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before + 3


# ----------------------------------------------------------- Test 3


def test_flush_edge_buffer_empty_returns_zero(tmp_path):
    """flush_edge_buffer on empty buffer returns 0 and does not raise (idempotent)."""
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        flushed = flush_edge_buffer(store)
        assert flushed == 0

        # Second call is also 0.
        flushed2 = flush_edge_buffer(store)
        assert flushed2 == 0


# ----------------------------------------------------------- Test 4


def test_should_flush_edge_buffer_size_threshold(tmp_path, monkeypatch):
    """should_flush_edge_buffer returns True when buffer length >= max_size (env var respected)."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, should_flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        monkeypatch.setenv("IAI_MCP_EDGE_BUFFER_MAX", "5")

        # Empty -> False.
        assert should_flush_edge_buffer(id(store)) is False

        # 4 rows -> False (under threshold).
        for i in range(4):
            row = {
                "src": str(uuid4()),
                "dst": str(uuid4()),
                "edge_type": "hebbian",
                "weight": 0.1,
                "updated_at": datetime.now(timezone.utc),
            }
            store_mod._edge_buffer.setdefault(id(store), []).append(row)
        assert should_flush_edge_buffer(id(store)) is False

        # 5th row -> True (at threshold).
        row = {
            "src": str(uuid4()),
            "dst": str(uuid4()),
            "edge_type": "hebbian",
            "weight": 0.1,
            "updated_at": datetime.now(timezone.utc),
        }
        store_mod._edge_buffer.setdefault(id(store), []).append(row)
        assert should_flush_edge_buffer(id(store)) is True

        # Explicit max_size override still works.
        assert should_flush_edge_buffer(id(store), max_size=100) is False


# ----------------------------------------------------------- Test 5 (CRITICAL — merge_insert invariant)


def test_merge_insert_remains_synchronous():
    """Static source check: boost_edges update path (merge_insert) is unchanged.

    merge_insert has read-before-write conflict semantics and MUST remain
    synchronous. This test asserts the phrase is still present in store.py.
    """
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store.py"
    text = store_py.read_text(encoding="utf-8")

    # merge_insert must still exist in the source (update path preserved).
    assert "tbl.merge_insert" in text, (
        "tbl.merge_insert not found in store.py — the synchronous update path was removed or renamed"
    )

    # The full update chain must be intact: merge_insert -> when_matched_update_all -> execute.
    assert ".when_matched_update_all().execute(" in text, (
        ".when_matched_update_all().execute( chain not found in store.py — update path is broken"
    )

    # boost_edges body must contain both phrases (not just in an unrelated context).
    boost_start = text.find("def boost_edges(")
    return_idx = text.find("return new_weights", boost_start)
    assert boost_start > 0, "def boost_edges( not found in store.py"
    assert return_idx > boost_start, "return new_weights not found after def boost_edges"

    body = text[boost_start:return_idx]
    assert "tbl.merge_insert" in body, (
        "tbl.merge_insert not found in boost_edges body — synchronous update path was removed"
    )
    assert ".when_matched_update_all().execute(" in body, (
        ".when_matched_update_all().execute( not found in boost_edges body"
    )


# ----------------------------------------------------------- Test 6 (CRITICAL — return-value invariant)


def test_boost_edges_returns_weights_for_buffered_rows(tmp_path):
    """boost_edges return value includes both update + insert rows regardless of buffer state.

    Proves new_weights is built locally (from update_rows + insert_rows), not
    from a store re-read. Buffering insert_rows does NOT affect any caller.
    """
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        uuid_a = uuid4()
        uuid_b = uuid4()
        uuid_c = uuid4()

        # Seed (a, b) to the store: insert + flush so it is visible as an existing edge.
        store.boost_edges([(uuid_a, uuid_b)], delta=0.5, edge_type="hebbian")
        flush_edge_buffer(store)

        # Now call with mixed pairs: (a, b) is an existing edge (update path),
        # (a, c) is new (insert path). Result must have BOTH keys.
        result = store.boost_edges(
            [(uuid_a, uuid_b), (uuid_a, uuid_c)], delta=0.1, edge_type="hebbian"
        )

        # new_weights is built locally — both keys must appear regardless of flush state.
        assert len(result) == 2, (
            f"expected 2 keys in new_weights (one update, one insert); got {len(result)}: {result}"
        )

        # Canonical sorted keys: boost_edges sorts (src, dst) alphabetically.
        str_a = str(uuid_a)
        str_b = str(uuid_b)
        str_c = str(uuid_c)
        key_ab = tuple(sorted([str_a, str_b]))
        key_ac = tuple(sorted([str_a, str_c]))
        assert key_ab in result, f"key_ab={key_ab} not in result; keys={list(result.keys())}"
        assert key_ac in result, f"key_ac={key_ac} not in result; keys={list(result.keys())}"


# ----------------------------------------------------------- Test 7


def test_add_contradicts_edge_flushes_immediately(tmp_path):
    """add_contradicts_edge writes the contradicts edge through to the store
    immediately (no deferral to the size/time buffer threshold).

    Contradicts edges are rare and constitutionally load-bearing (MEM-05):
    the superseded-original recall path reads them from the edges TABLE, not
    the in-memory buffer. On small stores the hebbian flush threshold is never
    reached, so a deferred contradicts edge would stay invisible to recall
    until an unrelated edge write happened to trip the flush. The edge must be
    durable the instant it is recorded.
    """
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        tbl = store.db.open_table(EDGES_TABLE)
        n_before = len(tbl.to_pandas())

        uuid_orig = uuid4()
        uuid_new = uuid4()
        store.add_contradicts_edge(uuid_orig, uuid_new)

        # Buffer must be drained (immediate flush), not holding the row.
        buf = store_mod._edge_buffer.get(id(store), [])
        assert len(buf) == 0, (
            f"contradicts edge must flush immediately; found {len(buf)} "
            f"row(s) still buffered"
        )

        # The row is in the store right away, without a manual flush call.
        tbl = store.db.open_table(EDGES_TABLE)
        assert len(tbl.to_pandas()) == n_before + 1, (
            "contradicts edge row did not land in LanceDB on write"
        )


# ----------------------------------------------------------- Test 8 (telemetry)


def test_flush_edge_buffer_emits_telemetry_event(tmp_path):
    """Successful flush emits lance_buffer_flush event with {table: 'edges', count: N}."""
    from iai_mcp import store as store_mod
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        # Buffer 2 rows manually.
        for _ in range(2):
            row = {
                "src": str(uuid4()),
                "dst": str(uuid4()),
                "edge_type": "hebbian",
                "weight": 0.3,
                "updated_at": datetime.now(timezone.utc),
            }
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        flushed = flush_edge_buffer(store)
        assert flushed == 2

        # Expect a lance_buffer_flush event for the edges table.
        events = query_events(store, kind="lance_buffer_flush")
        edges_events = [e for e in events if e["data"].get("table") == "edges"]
        assert len(edges_events) >= 1, (
            f"expected lance_buffer_flush event for edges table; found: {events}"
        )
        latest = edges_events[0]
        assert latest["data"]["count"] == 2, (
            f"expected count=2 in telemetry event; got: {latest['data']}"
        )


# ----------------------------------------------------------- Test 9 (static: _edge_buffer.setdefault count)


def test_edge_buffer_setdefault_used_at_two_call_sites():
    """Static source check: _edge_buffer.setdefault appears at exactly 2 sites in store.py.

    boost_edges insert seam + add_contradicts_edge. No other direct inserts.
    """
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store.py"
    text = store_py.read_text(encoding="utf-8")

    count = text.count("_edge_buffer.setdefault")
    assert count == 2, (
        f"expected exactly 2 '_edge_buffer.setdefault' in store.py; got {count}"
    )


# ----------------------------------------------------------- Test 10 (static: flush helpers present)


def test_store_has_three_edge_flush_helpers():
    """Static source check: store.py defines all three EDGES buffer functions."""
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store.py"
    text = store_py.read_text(encoding="utf-8")

    for fn_name in (
        "def flush_edge_buffer",
        "def should_flush_edge_buffer",
        "def should_flush_edge_buffer_by_time",
    ):
        assert fn_name in text, (
            f"expected '{fn_name}' to be defined in store.py"
        )


# ----------------------------------------------------------- Test 11 (daemon periodic-tick wiring)


def test_daemon_periodic_tick_calls_flush_edge_buffer():
    """daemon.py periodic-tick body imports + calls flush_edge_buffer and should_flush_edge_buffer_by_time."""
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_edge_buffer" in text, (
        "flush_edge_buffer not found in daemon.py"
    )
    assert "should_flush_edge_buffer_by_time" in text, (
        "periodic-tick wiring uses should_flush_edge_buffer_by_time helper — missing from daemon.py"
    )

    tick_idx = text.find("should_flush_edge_buffer_by_time")
    assert tick_idx > 0, "should_flush_edge_buffer_by_time must appear in daemon.py"


# ----------------------------------------------------------- Test 12 (daemon WAKE drain wiring)


def test_daemon_wake_drain_calls_flush_edge_buffer():
    """daemon.py per-tick path wires flush_edge_buffer with a should_flush_edge_buffer_by_time gate.

    After the single-driver consolidation collapse the per-tick path is the
    sole daemon flush for edges (no dedicated wake-hook). The invariant is:
    edges buffer IS flushed by the daemon (no data loss), using the
    should_flush_edge_buffer_by_time time-threshold gate. We verify:
      (a) flush_edge_buffer appears in daemon.py
      (b) should_flush_edge_buffer_by_time appears in daemon.py (per-tick gate)
      (c) the records gate precedes the edges gate in daemon source — records
          flush before edges (ordering invariant preserved on the per-tick path)
    """
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_edge_buffer" in text, (
        "flush_edge_buffer not found in daemon.py — per-tick flush wiring missing"
    )
    assert "should_flush_edge_buffer_by_time" in text, (
        "should_flush_edge_buffer_by_time gate not found in daemon.py — per-tick time-threshold missing"
    )
    # The edges gate must appear after the records gate in daemon source
    # (ordering: records flush before edges).
    records_gate_idx = text.find("should_flush_record_buffer_by_time")
    edges_gate_idx = text.find("should_flush_edge_buffer_by_time")
    assert edges_gate_idx > records_gate_idx, (
        "should_flush_edge_buffer_by_time must appear after should_flush_record_buffer_by_time "
        "(records before edges ordering); "
        f"records_gate_idx={records_gate_idx}, edges_gate_idx={edges_gate_idx}"
    )
    # The edge flush must appear after its gate.
    edges_flush_idx = text.find("flush_edge_buffer", edges_gate_idx)
    assert edges_flush_idx > edges_gate_idx, (
        "flush_edge_buffer must appear after should_flush_edge_buffer_by_time; "
        f"edges_gate_idx={edges_gate_idx}, edges_flush_idx={edges_flush_idx}"
    )


# ----------------------------------------------------------- Test 13 (daemon shutdown wiring)


def test_daemon_shutdown_calls_flush_edge_buffer():
    """daemon.py graceful-shutdown path flushes edges buffer before daemon_stopped."""
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    shutdown_idx = text.find("edges buffer flushed on shutdown")
    assert shutdown_idx > 0, (
        "'edges buffer flushed on shutdown' marker not found in daemon.py"
    )

    daemon_stopped_idx = text.find("daemon_stopped", shutdown_idx)
    assert daemon_stopped_idx > shutdown_idx, (
        "edges buffer flush must precede 'daemon_stopped' event write in daemon.py shutdown"
    )


# ----------------------------------------------------------- Test 14 (regression: contradict SRC durability)


def test_contradict_buffered_src_no_unknown_record_error(tmp_path, monkeypatch):
    """Regression: contradict() on a record that is still in _record_buffer does not
    raise 'unknown record'.

    This file's autouse fixture sets IAI_MCP_TEST_NO_AUTOFLUSH=1, so
    store.insert() leaves the record in _record_buffer.  Before the fix,
    retrieve.contradict() called store.get(original_id) which reads SQLite, not
    the buffer, and raised ValueError("unknown record ...").

    The fix adds flush_record_buffer(store) at the top of contradict() so the
    original record is durable before the point-read.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp import store as store_mod
    from iai_mcp.retrieve import contradict
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, _record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    # Pin buffer threshold to a large value so no auto-threshold flush occurs.
    monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "9999")

    with MemoryStore(path=tmp_path) as store:
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface="alice uses monotropic focus",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=1,
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

        # The record must still be buffered (autoflush is opted out for this file).
        buf = _record_buffer.get(id(store), [])
        assert any(r["id"] == str(rec.id) for r in buf), (
            "Pre-condition failed: record should be in _record_buffer (autoflush is disabled)"
        )

        # Before the fix this raised ValueError("unknown record ...").
        receipt = contradict(
            store,
            rec.id,
            "alice uses hyper-focus on single topic",
            [0.2] * EMBED_DIM,
        )

        # The contradicts edge must exist and reference the correct records.
        assert str(receipt.original_id) == str(rec.id), (
            f"expected original_id={rec.id}, got {receipt.original_id}"
        )
        assert receipt.edge_type == "contradicts", (
            f"expected edge_type='contradicts', got {receipt.edge_type!r}"
        )

        # Both SRC and DST must be durable in SQLite (not stranded in buffer).
        assert store.get(rec.id) is not None, (
            "SRC record must be durable in SQLite after contradict()"
        )
        assert store.get(receipt.new_record_id) is not None, (
            "DST (new_rec) must be durable in SQLite after contradict() returns"
        )

        # The contradicts edge must be queryable from the edges table.
        from iai_mcp.store import EDGES_TABLE
        tbl = store.db.open_table(EDGES_TABLE)
        df = tbl.to_pandas()
        c_edges = df[
            (df["src"] == str(rec.id)) & (df["edge_type"] == "contradicts")
        ]
        assert len(c_edges) >= 1, (
            f"expected at least one contradicts edge from {rec.id} in edges table; "
            f"found {len(c_edges)} edge(s)"
        )


# ----------------------------------------------------------- Test 15 (regression: contradict chain DST-becomes-SRC)


def test_contradict_chain_second_contradict_no_unknown_record_error(tmp_path, monkeypatch):
    """Regression: contradicting the *result* of a previous contradict() works correctly.

    The new_rec created by the first contradict() call must be durable so it can
    serve as the original_id for a second contradict().  Before the fix,
    new_rec was in _record_buffer and the second call raised 'unknown record'.

    This exercises the DST-durability half of the fix (flush_record_buffer inside
    add_contradicts_edge).
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.retrieve import contradict
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "9999")

    with MemoryStore(path=tmp_path) as store:
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface="alice prefers written communication",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=1,
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

        # First contradiction — new_rec lands in _record_buffer after insert().
        receipt1 = contradict(
            store,
            rec.id,
            "alice prefers async written communication",
            [0.2] * EMBED_DIM,
        )
        new_id = receipt1.new_record_id

        # Immediately contradict the result of the first contradict() (DST-as-SRC).
        # Before the fix this raised 'unknown record' for new_id.
        receipt2 = contradict(
            store,
            new_id,
            "alice prefers async text with bullet points",
            [0.3] * EMBED_DIM,
        )

        assert str(receipt2.original_id) == str(new_id), (
            "second contradict's original_id must be the first contradict's new_record_id"
        )
        # All three generations must be durable.
        assert store.get(rec.id) is not None, "gen-0 record must be durable"
        assert store.get(new_id) is not None, "gen-1 (first new_rec) must be durable"
        assert store.get(receipt2.new_record_id) is not None, "gen-2 (second new_rec) must be durable"
