"""F-05 regression: MemoryStore reader must see cross-connection writes.

Symptom that prompted this test:
    The sleep daemon ticks every 30 s and calls ``_store_is_empty(store)``
    against a connection it opened at process start. When short-lived MCP
    tool calls (e.g. ``memory_capture``) wrote new rows through a
    DIFFERENT connection to the same LanceDB directory, the daemon kept
    reporting ``last_tick_skipped_reason: empty_store`` forever — it had
    pinned the manifest snapshot at boot and LanceDB's default
    ``read_consistency_interval=None`` meant the handle never
    auto-refreshed.

Fix shape:
    ``MemoryStore`` gained a ``read_consistency_interval: timedelta | None``
    kwarg that is passed through to ``lancedb.connect``. Long-lived
    readers (the daemon) opt into ``timedelta(seconds=0)`` — strong
    consistency — so every read re-checks the latest committed
    version. Short-lived MCP callers keep the default ``None`` because
    they create a fresh connection per call and exit before staleness
    matters.

These tests pin both sides of the contract:
 1. With strong consistency the reader sees a writer's insert even
    when the reader opened first.
 2. With the default interval the reader's handle stays pinned to the
    snapshot it opened against (documents the knob the daemon now
    overrides, and guards against an accidental default change).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest


def _make_record(store):
    """Construct a minimal MemoryRecord sized to the store's embed dim."""
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="f-05 regression probe",
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
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
        language="en",
    )


@pytest.fixture
def tmp_store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    return tmp_path


def test_reader_with_strong_consistency_sees_writer_insert(tmp_store_env):
    """F-05 core regression: daemon-style reader (opened first, long-lived)
    must observe records written by a separate connection."""
    from iai_mcp.daemon import _store_is_empty
    from iai_mcp.store import MemoryStore

    # Reader opens first while store is empty — this is the daemon at boot.
    reader = MemoryStore(read_consistency_interval=timedelta(seconds=0))
    assert _store_is_empty(reader) is True  # baseline: nothing written yet

    # Writer is a distinct connection to the same directory — this is the
    # MCP tool call (memory_capture) running in a short-lived process.
    writer = MemoryStore()
    writer.insert(_make_record(writer))

    # With strong consistency the reader sees the commit on the next read.
    # Before the fix this stayed True forever.
    assert _store_is_empty(reader) is False


def test_default_connection_is_snapshot_pinned(tmp_store_env):
    """Documents Hippo/SQLite consistency: all reads see committed writes.

    SQLite WAL mode with autocommit provides read-committed isolation.
    Unlike LanceDB (which had snapshot pinning), every count_rows() call
    sees all previously committed rows. The read_consistency_interval kwarg
    is a no-op on the Hippo backend but is preserved for API compatibility.
    """
    from iai_mcp.store import MemoryStore

    reader = MemoryStore()  # no consistency kwarg — default None
    records_tbl = reader.db.open_table("records")
    assert records_tbl.count_rows() == 0

    writer = MemoryStore()
    writer.insert(_make_record(writer))

    # Hippo (SQLite WAL autocommit) provides immediate read visibility:
    # the write is committed and every subsequent read sees it.
    assert records_tbl.count_rows() == 1


def test_kwarg_is_persisted_for_introspection(tmp_store_env):
    """Callers (ops tooling, tests) can read the interval back."""
    from iai_mcp.store import MemoryStore

    default_store = MemoryStore()
    assert default_store._read_consistency_interval is None

    strong_store = MemoryStore(read_consistency_interval=timedelta(seconds=0))
    assert strong_store._read_consistency_interval == timedelta(seconds=0)

    eventual_store = MemoryStore(read_consistency_interval=timedelta(seconds=30))
    assert eventual_store._read_consistency_interval == timedelta(seconds=30)
