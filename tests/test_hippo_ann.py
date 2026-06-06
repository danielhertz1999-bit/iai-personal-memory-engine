"""Regression tests for the hnswlib ANN integration in HippoDB / HippoTable.

All tests use pytest's ``tmp_path`` fixture — no ~/.iai-mcp/ touched.
Single-file target: pytest tests/test_hippo_ann.py -x
"""
from __future__ import annotations

import concurrent.futures
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_unit_vec(rng: np.random.Generator) -> list[float]:
    """Return a unit-normalised random vector of length EMBED_DIM."""
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


def _record_row(*, rid: str | None = None, embedding: list[float]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": rid or str(uuid4()),
        "tier": "episodic",
        "literal_surface": "test",
        "embedding": embedding,
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# 1. knn_query returns top-k
# ---------------------------------------------------------------------------

def test_knn_query_returns_top_k(tmp_path: Path) -> None:
    """Insert 100 records with distinct vectors; query returns the correct top-10."""
    rng = np.random.default_rng(42)
    rows = []
    query_idx = 7  # we will search for the record at this index

    vecs = [_rng_unit_vec(rng) for _ in range(100)]
    ids = [str(uuid4()) for _ in range(100)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for i, (rid, vec) in enumerate(zip(ids, vecs)):
            tbl.add([_record_row(rid=rid, embedding=vec)])

        # Query with the exact vector of record at query_idx — it must be #1 hit.
        query_vec = vecs[query_idx]
        df = tbl.search(query_vec).limit(10).to_pandas()

    assert len(df) == 10, f"Expected 10 results, got {len(df)}"
    assert "_distance" in df.columns, "_distance column missing from ANN result"
    top_hit_id = str(df.iloc[0]["id"])
    assert top_hit_id == ids[query_idx], (
        f"Top-1 should be the query record (id={ids[query_idx]}), got {top_hit_id}"
    )
    # Distances should be non-negative (cosine distance ∈ [0, 2]).
    assert (df["_distance"] >= 0).all(), "Negative distances in result"


# ---------------------------------------------------------------------------
# 2. Atomic save — tmp does not survive, final file exists
# ---------------------------------------------------------------------------

def test_atomic_save_tmp_rename(tmp_path: Path) -> None:
    """After close(), records.hnsw exists and records.hnsw.tmp does not."""
    rng = np.random.default_rng(1)
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        tbl.add([_record_row(embedding=_rng_unit_vec(rng))])
        # close() triggers _save_index_atomic which writes.tmp then os.replace.

    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    tmp_path_hnsw = tmp_path / "hippo" / "records.hnsw.tmp"

    assert hnsw_path.exists(), "records.hnsw should exist after close()"
    assert not tmp_path_hnsw.exists(), "records.hnsw.tmp should not exist after close()"
    assert hnsw_path.stat().st_size > 0, "records.hnsw should not be empty"


# ---------------------------------------------------------------------------
# 3. Rebuild from SQLite after records.hnsw corruption
# ---------------------------------------------------------------------------

def test_rebuild_from_sqlite_after_hnsw_corruption(tmp_path: Path) -> None:
    """Corrupt records.hnsw; reopen must rebuild from SQLite and still answer queries."""
    rng = np.random.default_rng(2)
    n = 20
    vecs = [_rng_unit_vec(rng) for _ in range(n)]
    ids = [str(uuid4()) for _ in range(n)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for rid, vec in zip(ids, vecs):
            tbl.add([_record_row(rid=rid, embedding=vec)])

    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    assert hnsw_path.exists(), "precondition: records.hnsw written on close"

    # Corrupt the file by overwriting with garbage bytes.
    hnsw_path.write_bytes(b"\x00" * 16)

    # Reopen — boot must survive the corrupt file and rebuild from SQLite.
    with HippoDB(tmp_path) as db2:
        assert db2._hnsw.get_current_count() == n, (
            f"After rebuild, index should have {n} items, "
            f"got {db2._hnsw.get_current_count()}"
        )
        # ANN query must work after rebuild.
        tbl2 = db2.open_table("records")
        query_vec = vecs[0]
        df = tbl2.search(query_vec).limit(5).to_pandas()

    assert len(df) == 5, "Should return 5 results after rebuild"
    assert str(df.iloc[0]["id"]) == ids[0], "Top hit should be the exact-match record"


# ---------------------------------------------------------------------------
# 4. _label_map repopulates on boot
# ---------------------------------------------------------------------------

def test_label_map_repopulates_on_boot(tmp_path: Path) -> None:
    """After close() + reopen, _label_map matches SELECT vec_label, id FROM records WHERE tombstoned_at IS NULL."""
    rng = np.random.default_rng(3)
    n = 15
    ids = [str(uuid4()) for _ in range(n)]
    vecs = [_rng_unit_vec(rng) for _ in range(n)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for rid, vec in zip(ids, vecs):
            tbl.add([_record_row(rid=rid, embedding=vec)])

    with HippoDB(tmp_path) as db2:
        # Ground truth from SQLite.
        rows = db2._conn.execute(
            "SELECT id, vec_label FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
        expected = {str(r["id"]): int(r["vec_label"]) for r in rows}
        actual = dict(db2._label_map)

    assert actual == expected, (
        f"_label_map mismatch after boot.\n"
        f"  expected {len(expected)} entries, got {len(actual)}"
    )


# ---------------------------------------------------------------------------
# 5. active_count excludes tombstoned
# ---------------------------------------------------------------------------

def test_active_count_excludes_tombstoned(tmp_path: Path) -> None:
    """Tombstoning a record via raw SQL removes it from _label_map after reboot.

    get_current_count() MAY stay the same (hnswlib soft-deletes keep the slot).
    len(_label_map) MUST decrease by 1.
    """
    rng = np.random.default_rng(4)
    n = 10
    ids = [str(uuid4()) for _ in range(n)]
    vecs = [_rng_unit_vec(rng) for _ in range(n)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for rid, vec in zip(ids, vecs):
            tbl.add([_record_row(rid=rid, embedding=vec)])
        initial_label_map_size = len(db._label_map)

        # Tombstone via raw SQL (not tbl.delete, which hard-deletes).
        now = datetime.now(timezone.utc).isoformat()
        db._conn.execute("BEGIN")
        db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, ids[0]),
        )
        db._conn.execute("COMMIT")

    assert initial_label_map_size == n, f"Expected {n} in _label_map before tombstone"

    # Reopen — boot repopulates _label_map from SQLite (active records only).
    with HippoDB(tmp_path) as db2:
        label_map_size_after = len(db2._label_map)

    assert label_map_size_after == n - 1, (
        f"After tombstone + reboot, _label_map should have {n - 1} entries, "
        f"got {label_map_size_after}"
    )
    # Tombstoned id must not be in the map.
    assert ids[0] not in db2._label_map  # db2 is closed but dict is still accessible


# ---------------------------------------------------------------------------
# 6. save_index fires every HNSW_SAVE_INTERVAL writes
# ---------------------------------------------------------------------------

def test_save_index_every_n_writes(tmp_path: Path) -> None:
    """records.hnsw is created when the write counter crosses HNSW_SAVE_INTERVAL (200)."""
    from iai_mcp.hippo import HNSW_SAVE_INTERVAL

    rng = np.random.default_rng(5)
    hnsw_path = tmp_path / "hippo" / "records.hnsw"

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")

        # Write HNSW_SAVE_INTERVAL - 1 records; no periodic save should have fired.
        for _ in range(HNSW_SAVE_INTERVAL - 1):
            tbl.add([_record_row(embedding=_rng_unit_vec(rng))])

        # After 199 writes the periodic save has not fired.
        # (The file may not exist yet — fresh DB has no prior hnsw file.)
        size_before = hnsw_path.stat().st_size if hnsw_path.exists() else -1

        # Write the 200th record — this must trigger _save_index_atomic.
        tbl.add([_record_row(embedding=_rng_unit_vec(rng))])

        assert hnsw_path.exists(), (
            f"records.hnsw should exist after {HNSW_SAVE_INTERVAL} writes"
        )
        size_after = hnsw_path.stat().st_size
        assert size_after > 0, "records.hnsw should not be empty after periodic save"
        # If the file existed before (e.g. from rebuild on open), it should have grown or stayed same.
        if size_before > 0:
            assert size_after >= size_before


# ---------------------------------------------------------------------------
# 7. Concurrent add is RLock-protected (no exceptions, all visible)
# ---------------------------------------------------------------------------

def test_concurrent_add_rlock_protected(tmp_path: Path) -> None:
    """4 parallel threads each add 25 records; no exceptions; all 100 records visible."""
    rng = np.random.default_rng(6)
    n_threads = 4
    records_per_thread = 25
    total = n_threads * records_per_thread

    all_vecs = [_rng_unit_vec(rng) for _ in range(total)]
    all_ids = [str(uuid4()) for _ in range(total)]

    errors: list[Exception] = []
    lock = threading.Lock()

    def _worker(thread_idx: int) -> None:
        start = thread_idx * records_per_thread
        end = start + records_per_thread
        try:
            with HippoDB(tmp_path) as db:
                tbl = db.open_table("records")
                for rid, vec in zip(all_ids[start:end], all_vecs[start:end]):
                    tbl.add([_record_row(rid=rid, embedding=vec)])
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    # HippoDB uses an fcntl exclusive lock — concurrent open from multiple
    # HippoDB instances in the same process will fail (HippoLockHeldError).
    # Use a single shared HippoDB and spawn threads that share it instead.
    errors_shared: list[Exception] = []
    errors_lock = threading.Lock()

    def _shared_worker(thread_idx: int, db: HippoDB) -> None:
        start = thread_idx * records_per_thread
        end = start + records_per_thread
        try:
            tbl = db.open_table("records")
            for rid, vec in zip(all_ids[start:end], all_vecs[start:end]):
                tbl.add([_record_row(rid=rid, embedding=vec)])
        except Exception as exc:  # noqa: BLE001
            with errors_lock:
                errors_shared.append(exc)

    with HippoDB(tmp_path) as db:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(_shared_worker, i, db)
                for i in range(n_threads)
            ]
            concurrent.futures.wait(futures)

        assert not errors_shared, f"Thread errors: {errors_shared}"

        # All 100 records must be present.
        tbl = db.open_table("records")
        count = tbl.count_rows(filter="tombstoned_at IS NULL")
        assert count == total, f"Expected {total} records, got {count}"

        # _label_map must track all 100.
        assert len(db._label_map) == total, (
            f"_label_map should have {total} entries, got {len(db._label_map)}"
        )
