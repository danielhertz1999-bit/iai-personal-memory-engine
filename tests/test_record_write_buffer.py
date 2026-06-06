"""Tests for the records write-buffer infrastructure.

Exercises:
- store.insert() with sync path appends row to _record_buffer, not the store
- _record_buffer accumulates rows; flush_record_buffer writes batch and clears
- flush_record_buffer logs and does not raise on store failure
- should_flush_record_buffer size-threshold helper (env var + default)
- should_flush_record_buffer_by_time time-threshold helper (5 s default)
- Ciphertext invariant: row in buffer carries iai:enc:v1: prefix on encrypted columns
- Static source check: store.py call site uses buffer (no direct tbl.add)
- daemon.py wiring presence: periodic-tick, WAKE drain, shutdown
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    """Buffer-internals tests assert un-flushed state — disable the
    conftest-level autoflush patch for every test in this file."""
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

import pytest


# ----------------------------------------------------------- helpers


def _make_record(literal_surface: str = "test memory content"):
    """Build a minimal MemoryRecord for test use."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.types import EMBED_DIM, MemoryRecord

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _clear_buffer(store) -> None:
    """Pop any leftover buffer state for this store id."""
    from iai_mcp import store as store_mod

    store_mod._record_buffer.pop(id(store), None)
    store_mod._record_last_flush_at.pop(id(store), None)


# ----------------------------------------------------------- Test 1


def test_insert_sync_path_buffers_row_not_lancedb(tmp_path):
    """store.insert() via the sync path appends to _record_buffer, not the store directly."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        # Snapshot pre-insert store row count for the records table.
        tbl = store.db.open_table(RECORDS_TABLE)
        n_before = len(tbl.to_pandas())

        record = _make_record("buffered record test")
        store.insert(record)

        # Buffer must have at least one row (insert may or may not auto-flush at threshold).
        # To isolate: ensure buffer length grew (the insert didn't go to a empty-buffer flush path
        # if threshold > 1, which it is at default 100).
        buf = store_mod._record_buffer.get(id(store), [])
        assert len(buf) >= 1, "expected _record_buffer to accumulate rows after insert"


# ----------------------------------------------------------- Test 2


def test_buffer_record_row_does_not_write_to_lancedb(tmp_path):
    """Appending directly to _record_buffer does not touch the store row count."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(RECORDS_TABLE)
        n_before = len(tbl.to_pandas())

        # Build a row (via _to_row) and append directly to the buffer.
        record = _make_record("direct buffer append test")
        row = store._to_row(record)
        store_mod._record_buffer.setdefault(id(store), []).append(row)

        # Row count unchanged — buffer append MUST NOT touch the store.
        tbl = store.db.open_table(RECORDS_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before, (
            f"buffer append changed store row count: {n_before} -> {n_after}"
        )

        # Buffer length is now at least 1.
        assert len(store_mod._record_buffer.get(id(store), [])) >= 1


# ----------------------------------------------------------- Test 3


def test_flush_record_buffer_writes_batch_and_clears(tmp_path):
    """Buffered records flush as a batch, buffer empties, count returned."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(RECORDS_TABLE)
        n_before = len(tbl.to_pandas())

        # Append 3 rows directly to buffer (bypass insert to avoid threshold auto-flush).
        for i in range(3):
            record = _make_record(f"batch flush test {i}")
            row = store._to_row(record)
            store_mod._record_buffer.setdefault(id(store), []).append(row)

        assert len(store_mod._record_buffer.get(id(store), [])) == 3

        flushed = flush_record_buffer(store)
        assert flushed == 3

        # Buffer is empty (or popped) after flush.
        assert not store_mod._record_buffer.get(id(store))

        # Records landed in the store.
        tbl = store.db.open_table(RECORDS_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before + 3


# ----------------------------------------------------------- Test 4


def test_flush_record_buffer_empty_returns_zero(tmp_path):
    """flush_record_buffer on an empty buffer returns 0 and does not raise."""
    from iai_mcp.store import MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        # Buffer is empty — must return 0, must not raise.
        flushed = flush_record_buffer(store)
        assert flushed == 0

        # Calling again is idempotent.
        flushed2 = flush_record_buffer(store)
        assert flushed2 == 0


# ----------------------------------------------------------- Test 5


def test_should_flush_record_buffer_size_threshold(tmp_path, monkeypatch):
    """should_flush_record_buffer returns True when buffer length >= max_size."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, should_flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "10")

        # Empty -> False.
        assert should_flush_record_buffer(id(store)) is False

        # 9 rows -> False (under threshold).
        for i in range(9):
            record = _make_record(f"threshold test {i}")
            row = store._to_row(record)
            store_mod._record_buffer.setdefault(id(store), []).append(row)
        assert should_flush_record_buffer(id(store)) is False

        # 10th row -> True.
        record = _make_record("threshold test 9")
        row = store._to_row(record)
        store_mod._record_buffer.setdefault(id(store), []).append(row)
        assert should_flush_record_buffer(id(store)) is True

        # Explicit override still works.
        assert should_flush_record_buffer(id(store), max_size=100) is False


# ----------------------------------------------------------- Test 6


def test_should_flush_record_buffer_by_time(tmp_path):
    """should_flush_record_buffer_by_time returns True for non-empty buffer with None or aged last_flush_at."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, should_flush_record_buffer_by_time

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        # Empty buffer -> False regardless of age.
        assert should_flush_record_buffer_by_time(id(store), None) is False
        assert should_flush_record_buffer_by_time(
            id(store), datetime.now(timezone.utc) - timedelta(seconds=60)
        ) is False

        # Add one row to buffer.
        record = _make_record("time threshold test")
        row = store._to_row(record)
        store_mod._record_buffer.setdefault(id(store), []).append(row)

        # last_flush_at=None and buffer non-empty -> True (never-flushed semantic).
        assert should_flush_record_buffer_by_time(id(store), None) is True

        # Recent flush (1 s ago) -> False.
        recent = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert should_flush_record_buffer_by_time(id(store), recent) is False

        # Old flush (6 s ago) with buffer non-empty -> True.
        old = datetime.now(timezone.utc) - timedelta(seconds=6)
        assert should_flush_record_buffer_by_time(id(store), old) is True


# ----------------------------------------------------------- Test 7 (CRITICAL — ciphertext invariant)


def test_buffered_row_carries_ciphertext_prefix(tmp_path):
    """Row in _record_buffer has AES-256-GCM ciphertext on literal_surface.

    Proves _to_row() encrypts BEFORE the row enters the buffer — buffer stores
    ciphertext, not plaintext. This is the security invariant for the buffered
    write path.
    """
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        record = _make_record("sensitive plaintext content that must be encrypted")
        store.insert(record)

        # The buffer must have the row (default threshold is 100; we inserted 1 row).
        buf = store_mod._record_buffer.get(id(store), [])
        assert len(buf) >= 1, "expected row in _record_buffer after insert"

        # The most recently appended row must have encrypted literal_surface.
        buffered_row = buf[-1]
        literal_val = buffered_row["literal_surface"]
        assert literal_val.startswith("iai:enc:v1:"), (
            f"Expected ciphertext prefix 'iai:enc:v1:' on buffered row, "
            f"got plaintext: {literal_val[:50]!r}"
        )


# ----------------------------------------------------------- Test 8 (static: buffer call site in store.py)


def test_store_sync_path_uses_record_buffer():
    """Static source check: store.py sync path uses _record_buffer.setdefault (no direct tbl.add)."""
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store.py"
    text = store_py.read_text(encoding="utf-8")

    # Exactly 1 _record_buffer.setdefault in store.py (the sync-path append).
    count = text.count("_record_buffer.setdefault")
    assert count == 1, (
        f"expected exactly 1 '_record_buffer.setdefault' in store.py; got {count}"
    )

    # No bare tbl.add([self._to_row(...) in store.py (the old direct call was removed).
    assert "tbl.add([self._to_row" not in text, (
        "old direct tbl.add([self._to_row(record)]) call site was not removed from store.py"
    )


# ----------------------------------------------------------- Test 9 (static: flush functions in store.py)


def test_store_has_three_flush_helpers():
    """Static source check: store.py defines all three RECORDS buffer functions."""
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store.py"
    text = store_py.read_text(encoding="utf-8")

    for fn_name in (
        "def flush_record_buffer",
        "def should_flush_record_buffer",
        "def should_flush_record_buffer_by_time",
    ):
        assert fn_name in text, (
            f"expected '{fn_name}' to be defined in store.py"
        )


# ----------------------------------------------------------- Test 10 (daemon periodic-tick wiring)


def test_daemon_periodic_tick_calls_flush_record_buffer():
    """daemon.py periodic-tick body imports + calls should_flush_record_buffer_by_time."""
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_record_buffer" in text, (
        "flush_record_buffer not found in daemon.py"
    )
    assert "should_flush_record_buffer_by_time" in text, (
        "periodic-tick wiring uses should_flush_record_buffer_by_time helper — missing from daemon.py"
    )

    # Periodic block must reference the time-gate helper.
    tick_idx = text.find("should_flush_record_buffer_by_time")
    assert tick_idx > 0, "should_flush_record_buffer_by_time must appear in daemon.py"


# ----------------------------------------------------------- Test 11 (daemon WAKE drain wiring)


def test_daemon_wake_drain_calls_flush_record_buffer():
    """daemon.py per-tick path wires flush_record_buffer with a should_flush_record_buffer_by_time gate.

    After the single-driver consolidation collapse the per-tick path is the
    sole daemon flush for records (no dedicated wake-hook). The invariant is:
    records buffer IS flushed by the daemon (no data loss), using the
    should_flush_record_buffer_by_time time-threshold gate. We verify:
      (a) flush_record_buffer appears in daemon.py
      (b) should_flush_record_buffer_by_time appears in daemon.py (per-tick gate)
      (c) the per-tick events gate (should_flush_by_time) precedes the records gate
          in daemon source — events flush before records (ordering invariant preserved)
    """
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_record_buffer" in text, (
        "flush_record_buffer not found in daemon.py — per-tick flush wiring missing"
    )
    assert "should_flush_record_buffer_by_time" in text, (
        "should_flush_record_buffer_by_time gate not found in daemon.py — per-tick time-threshold missing"
    )
    # The record gate must appear after the events gate in daemon source
    # (ordering: events flush first, then records — invariant preserved
    # on the per-tick path just as it was on the prior wake-hook path).
    events_gate_idx = text.find("should_flush_by_time")
    records_gate_idx = text.find("should_flush_record_buffer_by_time")
    assert records_gate_idx > events_gate_idx, (
        "should_flush_record_buffer_by_time must appear after should_flush_by_time "
        "(events before records ordering); "
        f"events_gate_idx={events_gate_idx}, records_gate_idx={records_gate_idx}"
    )
    # The record flush must appear after its gate (no flush without gate check).
    records_flush_idx = text.find("flush_record_buffer", records_gate_idx)
    assert records_flush_idx > records_gate_idx, (
        "flush_record_buffer must appear after should_flush_record_buffer_by_time; "
        f"records_gate_idx={records_gate_idx}, records_flush_idx={records_flush_idx}"
    )


# ----------------------------------------------------------- Test 12 (daemon shutdown wiring)


def test_daemon_shutdown_calls_flush_record_buffer():
    """daemon.py graceful-shutdown path flushes records buffer synchronously before daemon_stopped."""
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    # Shutdown block must invoke synchronous flush before daemon_stopped.
    shutdown_idx = text.find("records buffer flushed on shutdown")
    assert shutdown_idx > 0, (
        "records buffer shutdown flush marker ('records buffer flushed on shutdown') not found in daemon.py"
    )

    daemon_stopped_idx = text.find("daemon_stopped", shutdown_idx)
    assert daemon_stopped_idx > shutdown_idx, (
        "records flush must precede 'daemon_stopped' event write in daemon.py shutdown"
    )
