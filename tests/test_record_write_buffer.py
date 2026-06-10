from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

import pytest


def _make_record(literal_surface: str = "test memory content"):
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
    from iai_mcp import store as store_mod

    store_mod._record_buffer.pop(id(store), None)
    store_mod._record_last_flush_at.pop(id(store), None)


def test_insert_sync_path_buffers_row_not_lancedb(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(RECORDS_TABLE)
        n_before = len(tbl.to_pandas())

        record = _make_record("buffered record test")
        store.insert(record)

        buf = store_mod._record_buffer.get(id(store), [])
        assert len(buf) >= 1, "expected _record_buffer to accumulate rows after insert"


def test_buffer_record_row_does_not_write_to_lancedb(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(RECORDS_TABLE)
        n_before = len(tbl.to_pandas())

        record = _make_record("direct buffer append test")
        row = store._to_row(record)
        store_mod._record_buffer.setdefault(id(store), []).append(row)

        tbl = store.db.open_table(RECORDS_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before, (
            f"buffer append changed store row count: {n_before} -> {n_after}"
        )

        assert len(store_mod._record_buffer.get(id(store), [])) >= 1


def test_flush_record_buffer_writes_batch_and_clears(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        tbl = store.db.open_table(RECORDS_TABLE)
        n_before = len(tbl.to_pandas())

        for i in range(3):
            record = _make_record(f"batch flush test {i}")
            row = store._to_row(record)
            store_mod._record_buffer.setdefault(id(store), []).append(row)

        assert len(store_mod._record_buffer.get(id(store), [])) == 3

        flushed = flush_record_buffer(store)
        assert flushed == 3

        assert not store_mod._record_buffer.get(id(store))

        tbl = store.db.open_table(RECORDS_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before + 3


def test_flush_record_buffer_empty_returns_zero(tmp_path):
    from iai_mcp.store import MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        flushed = flush_record_buffer(store)
        assert flushed == 0

        flushed2 = flush_record_buffer(store)
        assert flushed2 == 0


def test_should_flush_record_buffer_size_threshold(tmp_path, monkeypatch):
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, should_flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "10")

        assert should_flush_record_buffer(id(store)) is False

        for i in range(9):
            record = _make_record(f"threshold test {i}")
            row = store._to_row(record)
            store_mod._record_buffer.setdefault(id(store), []).append(row)
        assert should_flush_record_buffer(id(store)) is False

        record = _make_record("threshold test 9")
        row = store._to_row(record)
        store_mod._record_buffer.setdefault(id(store), []).append(row)
        assert should_flush_record_buffer(id(store)) is True

        assert should_flush_record_buffer(id(store), max_size=100) is False


def test_should_flush_record_buffer_by_time(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, should_flush_record_buffer_by_time

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        assert should_flush_record_buffer_by_time(id(store), None) is False
        assert should_flush_record_buffer_by_time(
            id(store), datetime.now(timezone.utc) - timedelta(seconds=60)
        ) is False

        record = _make_record("time threshold test")
        row = store._to_row(record)
        store_mod._record_buffer.setdefault(id(store), []).append(row)

        assert should_flush_record_buffer_by_time(id(store), None) is True

        recent = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert should_flush_record_buffer_by_time(id(store), recent) is False

        old = datetime.now(timezone.utc) - timedelta(seconds=6)
        assert should_flush_record_buffer_by_time(id(store), old) is True


def test_buffered_row_carries_ciphertext_prefix(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        record = _make_record("sensitive plaintext content that must be encrypted")
        store.insert(record)

        buf = store_mod._record_buffer.get(id(store), [])
        assert len(buf) >= 1, "expected row in _record_buffer after insert"

        buffered_row = buf[-1]
        literal_val = buffered_row["literal_surface"]
        assert literal_val.startswith("iai:enc:v1:"), (
            f"Expected ciphertext prefix 'iai:enc:v1:' on buffered row, "
            f"got plaintext: {literal_val[:50]!r}"
        )


def test_store_sync_path_uses_record_buffer():
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store.py"
    text = store_py.read_text(encoding="utf-8")

    count = text.count("_record_buffer.setdefault")
    assert count == 1, (
        f"expected exactly 1 '_record_buffer.setdefault' in store.py; got {count}"
    )

    assert "tbl.add([self._to_row" not in text, (
        "old direct tbl.add([self._to_row(record)]) call site was not removed from store.py"
    )


def test_store_has_three_flush_helpers():
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


def test_daemon_periodic_tick_calls_flush_record_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_record_buffer" in text, (
        "flush_record_buffer not found in daemon.py"
    )
    assert "should_flush_record_buffer_by_time" in text, (
        "periodic-tick wiring uses should_flush_record_buffer_by_time helper — missing from daemon.py"
    )

    tick_idx = text.find("should_flush_record_buffer_by_time")
    assert tick_idx > 0, "should_flush_record_buffer_by_time must appear in daemon.py"


def test_daemon_wake_drain_calls_flush_record_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_record_buffer" in text, (
        "flush_record_buffer not found in daemon.py — per-tick flush wiring missing"
    )
    assert "should_flush_record_buffer_by_time" in text, (
        "should_flush_record_buffer_by_time gate not found in daemon.py — per-tick time-threshold missing"
    )
    events_gate_idx = text.find("should_flush_by_time")
    records_gate_idx = text.find("should_flush_record_buffer_by_time")
    assert records_gate_idx > events_gate_idx, (
        "should_flush_record_buffer_by_time must appear after should_flush_by_time "
        "(events before records ordering); "
        f"events_gate_idx={events_gate_idx}, records_gate_idx={records_gate_idx}"
    )
    records_flush_idx = text.find("flush_record_buffer", records_gate_idx)
    assert records_flush_idx > records_gate_idx, (
        "flush_record_buffer must appear after should_flush_record_buffer_by_time; "
        f"records_gate_idx={records_gate_idx}, records_flush_idx={records_flush_idx}"
    )


def test_daemon_shutdown_calls_flush_record_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_py.read_text(encoding="utf-8")

    shutdown_idx = text.find("records buffer flushed on shutdown")
    assert shutdown_idx > 0, (
        "records buffer shutdown flush marker ('records buffer flushed on shutdown') not found in daemon.py"
    )

    daemon_stopped_idx = text.find("daemon_stopped", shutdown_idx)
    assert daemon_stopped_idx > shutdown_idx, (
        "records flush must precede 'daemon_stopped' event write in daemon.py shutdown"
    )
