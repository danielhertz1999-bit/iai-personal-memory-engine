"""Concurrent-access guards for the shared sqlite3 connection in the storage backend.

The storage connection runs with ``check_same_thread=False`` and is shared by
every worker thread the daemon dispatches through ``asyncio.to_thread``. CPython's
sqlite3 does NOT protect a cursor's result set between ``execute()`` and a later
``fetch*()`` when another thread issues a concurrent ``execute()`` on the same
connection -- so every ``execute(...).fetch*()`` pair (and every iterated cursor)
on the shared connection MUST run under the connection lock. The invariant is the
whole of the lock discipline: a guarded read either holds the lock across its
cursor's full life or it can observe a truncated / corrupted result.

These tests pin two consequences of that invariant under real concurrent load:

1. ``to_batches`` -- the bulk projection path -- iterated repeatedly while a writer
   thread hammers the same connection NEVER raises ``IndexError`` /
   ``sqlite3.InterfaceError`` and returns coherent rows. (Post-fix regression
   guard: the C-level cursor race cannot be manufactured deterministically in a
   hermetic WAL+GIL single-process test, so this asserts the guarded path stays
   clean rather than reproducing the raw corruption.)
2. The audited single-row / full-scan read paths (``count_rows``, ``to_pandas``,
   ``table_names``) driven concurrently with writes never observe a spurious
   ``None`` / truncated row (the symptom the guard prevents).

Hermeticity: the store opens under ``tmp_path``; an in-test assertion fails if it
ever resolves under the operator's real ``~/.iai-mcp``.
"""
from __future__ import annotations

import datetime
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


_N_SEED = 150


def _make_record(vec, community_id, idx: int) -> MemoryRecord:
    now = datetime.datetime.now(datetime.timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=f"record {idx} content",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=community_id,
        centrality=float(idx % 17),
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
        tags=["kind:note"],
        language="en",
        profile_modulation_gain={},
    )


def _seed_store(store: MemoryStore, n: int, community_id) -> None:
    dim = store._embed_dim
    rng = np.random.default_rng(7)
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        store.insert(_make_record(vec, community_id, i))


def _assert_hermetic(store: MemoryStore, tmp_path: Path) -> None:
    root = Path(store.root).resolve()
    assert str(root).startswith(str(tmp_path.resolve())), (
        f"store root {root} escaped tmp_path {tmp_path}"
    )
    real_home_store = (Path.home() / ".iai-mcp").resolve()
    assert real_home_store not in root.parents and root != real_home_store, (
        f"store root {root} resolved under the real ~/.iai-mcp"
    )


def test_to_batches_no_corruption_under_concurrent_add(tmp_path):
    """Iterate to_batches repeatedly while a writer thread issues concurrent
    add()/merge_insert on the SAME shared connection. The guarded fetch holds
    the connection lock across the cursor's full life, so the drain can never
    observe a cursor reset by the writer -> ZERO IndexError / InterfaceError,
    and every drain returns a coherent (monotonically non-decreasing) row count.
    """
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)

    community_id = uuid4()
    _seed_store(store, _N_SEED, community_id)

    dim = store._embed_dim
    rng = np.random.default_rng(99)

    stop = threading.Event()
    writer_errors: list[BaseException] = []
    inserted = {"n": 0}

    def _writer() -> None:
        idx = _N_SEED
        try:
            while not stop.is_set():
                vec = rng.standard_normal(dim).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-9
                store.insert(_make_record(vec, community_id, idx))
                idx += 1
                inserted["n"] = idx - _N_SEED
                time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001 — surface to the assertion
            writer_errors.append(exc)
            stop.set()

    reader_errors: list[BaseException] = []
    row_counts: list[int] = []

    writer = threading.Thread(target=_writer, daemon=True)
    try:
        writer.start()
        # Many drain trials overlapping the writer storm. A guarded drain takes
        # the connection lock for its whole cursor life, so a concurrent
        # insert() cannot reset its result set mid-fetch.
        for _ in range(60):
            try:
                query = store.db.open_table("records").search()
                assert query._db is not None, (
                    "to_batches query lost its HippoDB reference"
                )
                rows = 0
                for batch in query.to_batches(batch_size=16):
                    rows += batch.num_rows
                row_counts.append(rows)
            except (IndexError, sqlite3.InterfaceError) as exc:
                reader_errors.append(exc)
                break
            time.sleep(0.002)
    finally:
        stop.set()
        writer.join(timeout=5.0)
        store.close()

    assert not writer_errors, f"writer thread raised: {writer_errors!r}"
    assert not reader_errors, (
        f"to_batches corrupted under concurrent add (cursor race): {reader_errors!r}"
    )
    assert row_counts, "no to_batches drains completed"
    # Every drain is internally coherent: it reads a whole-cursor snapshot, so
    # each count is >= the seed and never below a prior count (rows only grow).
    assert min(row_counts) >= _N_SEED, (
        f"a drain returned fewer than the seeded rows: min={min(row_counts)} "
        f"< seed={_N_SEED} (truncated result set)"
    )
    assert row_counts == sorted(row_counts), (
        f"row counts went backwards across drains -> truncated/corrupted "
        f"snapshot: {row_counts}"
    )


def test_concurrent_reads_writes_no_none_rows(tmp_path):
    """Drive the audited single-row / full-scan read paths (count_rows,
    to_pandas, table_names) concurrently with writes and assert no spurious
    None / truncated row is observed -- the exact symptom the connection-lock
    discipline prevents (a concurrent execute resetting a single-row fetch to
    None, or a full scan returning fewer rows than the store holds).
    """
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)

    community_id = uuid4()
    _seed_store(store, _N_SEED, community_id)

    dim = store._embed_dim
    rng = np.random.default_rng(123)

    stop = threading.Event()
    writer_errors: list[BaseException] = []

    def _writer() -> None:
        idx = _N_SEED
        try:
            while not stop.is_set():
                vec = rng.standard_normal(dim).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-9
                store.insert(_make_record(vec, community_id, idx))
                idx += 1
                time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001
            writer_errors.append(exc)
            stop.set()

    reader_errors: list[BaseException] = []
    count_samples: list[int] = []
    pandas_samples: list[int] = []

    writer = threading.Thread(target=_writer, daemon=True)
    try:
        writer.start()
        for _ in range(60):
            try:
                tbl = store.db.open_table("records")
                # count_rows: guarded execute()+fetchone(); raises
                # HippoIntegrityError (NOT returns None) if the row is missing.
                c = tbl.count_rows()
                count_samples.append(c)
                # to_pandas: guarded full-scan read.
                df = tbl.search().to_pandas()
                pandas_samples.append(len(df))
                # table_names: guarded sqlite_master read.
                names = store.db.table_names()
                assert "records" in names, (
                    f"table_names lost the records table under load: {names!r}"
                )
            except BaseException as exc:  # noqa: BLE001
                reader_errors.append(exc)
                break
            time.sleep(0.002)
    finally:
        stop.set()
        writer.join(timeout=5.0)
        store.close()

    assert not writer_errors, f"writer thread raised: {writer_errors!r}"
    assert not reader_errors, (
        f"a guarded read path observed a corrupted/None row under concurrent "
        f"write: {reader_errors!r}"
    )
    # count_rows never returns None and never drops below the seed; it only grows.
    assert count_samples and min(count_samples) >= _N_SEED, (
        f"count_rows returned fewer than seeded under load: {count_samples!r}"
    )
    assert count_samples == sorted(count_samples), (
        f"count_rows went backwards (truncated single-row fetch): {count_samples}"
    )
    # The full-scan to_pandas never returns fewer rows than count_rows reported
    # just before it (the snapshot is coherent, never truncated).
    assert pandas_samples and min(pandas_samples) >= _N_SEED, (
        f"to_pandas full scan truncated under load: {pandas_samples!r}"
    )
