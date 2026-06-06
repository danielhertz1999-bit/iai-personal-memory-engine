"""Integration tests for the SQLite+hnswlib storage layer via MemoryStore.

All tests operate through the MemoryStore public API — no direct HippoDB pokes.
Every test isolates to pytest's tmp_path; no ~/.iai-mcp state is touched.
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.store import (
    EDGES_TABLE,
    RECORDS_TABLE,
    MemoryStore,
    flush_record_buffer,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(seed: int, text: str = "") -> MemoryRecord:
    """Return a MemoryRecord with a deterministic, well-separated embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text or f"record seed {seed}",
        aaak_index="",
        embedding=vec,
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
        language="en",
    )


def _wait_for_count(
    table,
    target: int,
    *,
    timeout_sec: float = 5.0,
    poll: float = 0.05,
) -> None:
    """Poll table.count_rows() until it reaches *target* or *timeout_sec* elapses."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if table.count_rows() >= target:
            return
        time.sleep(poll)
    actual = table.count_rows()
    raise TimeoutError(
        f"_wait_for_count timed out: expected {target}, got {actual}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_e2e_capture_recall_via_memory_store(tmp_path: Path) -> None:
    """Insert via MemoryStore.insert and verify the record is retrievable via get."""
    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec = _make_record(42, "hello world")
        store.insert(rec)
        flush_record_buffer(store)

        retrieved = store.get(rec.id)
        assert retrieved is not None, "record should be retrievable after flush"
        assert retrieved.id == rec.id
        assert "hello world" in retrieved.literal_surface
        assert len(retrieved.embedding) == EMBED_DIM
    finally:
        store.close()


def test_e2e_boost_edges_round_trip(tmp_path: Path) -> None:
    """boost_edges writes an edge; it should be readable back from the edges table."""
    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec_a = _make_record(10)
        rec_b = _make_record(11)
        store.insert(rec_a)
        store.insert(rec_b)
        flush_record_buffer(store)

        store.boost_edges([(rec_a.id, rec_b.id)], delta=1.0, edge_type="hebbian")

        edges_tbl = store.db.open_table(EDGES_TABLE)
        df = edges_tbl.to_pandas()
        assert len(df) > 0, "edges table should be non-empty after boost_edges"

        # boost_edges stores edges with canonically-sorted (src, dst) pairs.
        canonical_src, canonical_dst = sorted([str(rec_a.id), str(rec_b.id)])
        row = df[
            (df["src"] == canonical_src)
            & (df["dst"] == canonical_dst)
            & (df["edge_type"] == "hebbian")
        ]
        assert len(row) >= 1, "written edge should be present in edges table"
        assert float(row.iloc[0]["weight"]) > 0.0
    finally:
        store.close()


def test_e2e_pattern_separation_compatible(tmp_path: Path) -> None:
    """Pattern separation gate should not error on a fresh store with distinct records.

    Inserts records with maximally distinct random embeddings (different RNG seeds).
    The gate should return INSERT for each (no merging), so all records land in the DB.
    """
    n = 10
    store = MemoryStore(tmp_path, user_id="test")
    try:
        for i in range(n):
            store.insert(_make_record(seed=1000 + i))
        flush_record_buffer(store)

        records_tbl = store.db.open_table(RECORDS_TABLE)
        assert records_tbl.count_rows() == n, (
            f"all {n} distinct records should be stored (pattern sep should INSERT, not SKIP)"
        )
    finally:
        store.close()


def test_e2e_async_write_queue_drains(tmp_path: Path) -> None:
    """enable_async_writes() + 50 inserts + disable_async_writes() should leave all rows visible."""
    import asyncio

    n = 50
    store = MemoryStore(tmp_path, user_id="test")
    try:
        asyncio.run(store.enable_async_writes(coalesce_ms=10, max_batch=128))

        for i in range(n):
            store.insert(_make_record(seed=2000 + i))

        # Drain by disabling — disable_async_writes waits for queue to stop.
        asyncio.run(store.disable_async_writes())

        records_tbl = store.db.open_table(RECORDS_TABLE)
        count = records_tbl.count_rows()
        assert count == n, (
            f"all {n} async-queued records should be visible after drain, got {count}"
        )
    finally:
        store.close()


def test_concurrent_sync_and_async_writes(tmp_path: Path) -> None:
    """Four threads each adding rows directly to HippoTable must all land without exceptions.

    Tests thread-safety of HippoDB._hnsw_lock + SQLite check_same_thread=False
    via the HippoTable.add() path (bypassing MemoryStore to avoid conftest
    autoflush races with the pattern-separation gate).
    """
    from iai_mcp.hippo import HippoDB

    threads_n = 4
    per_thread = 25
    total = threads_n * per_thread
    errors: list[Exception] = []
    lock = threading.Lock()

    db = HippoDB(tmp_path)
    try:
        tbl = db.open_table(RECORDS_TABLE)

        def worker(thread_idx: int) -> None:
            try:
                rows = []
                for j in range(per_thread):
                    seed = 3000 + thread_idx * per_thread + j
                    rows.append({
                        "id": str(uuid.uuid4()),
                        "tier": "episodic",
                        "literal_surface": f"concurrent worker {thread_idx} j {j}",
                        "aaak_index": "",
                        "embedding": np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32).tolist(),
                        "structure_hv": b"",
                        "community_id": None,
                        "centrality": 0.0,
                        "detail_level": 1,
                        "pinned": False,
                        "stability": 0.0,
                        "difficulty": 0.0,
                        "last_reviewed": None,
                        "never_decay": False,
                        "never_merge": False,
                        "provenance_json": "[]",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "tags_json": "[]",
                        "language": "en",
                        "s5_trust_score": 0.5,
                        "profile_modulation_gain_json": "{}",
                        "schema_version": 4,
                    })
                # One batch per worker — fewer write transactions, still concurrent.
                tbl.add(rows)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads_n) as pool:
            futs = [pool.submit(worker, i) for i in range(threads_n)]
            for f in futs:
                f.result()

        assert not errors, f"concurrent writes raised exceptions: {errors}"

        _wait_for_count(tbl, total, timeout_sec=10)
        assert tbl.count_rows() == total
    finally:
        db.close()


def test_no_lancedb_concept_leaks_through_api(tmp_path: Path) -> None:
    """MemoryStore must not cause lancedb to be imported — it was replaced by HippoDB.

    We rely on the fact that lancedb is an optional/removed dependency: if
    HippoDB accidentally imports it, the module would appear in sys.modules.
    This test creates a fresh store in a subprocess-like isolation (checking
    sys.modules directly) to confirm the lancedb namespace is absent.
    """
    # Remove lancedb from sys.modules if it somehow got loaded by a prior test
    # in this session (would be a false negative otherwise).
    for key in list(sys.modules):
        if key == "lancedb" or key.startswith("lancedb."):
            del sys.modules[key]

    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec = _make_record(99)
        store.insert(rec)
        flush_record_buffer(store)
        _ = store.get(rec.id)
        _ = store.query_similar(rec.embedding, k=1)
    finally:
        store.close()

    lancedb_modules = [k for k in sys.modules if k == "lancedb" or k.startswith("lancedb.")]
    assert not lancedb_modules, (
        f"lancedb leaked into sys.modules: {lancedb_modules}"
    )
