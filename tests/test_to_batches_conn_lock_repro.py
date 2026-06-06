"""Regression basis: HippoQuery.to_batches must hold the shared-connection lock
while it fetches from the shared connection.

Every ``conn.execute().fetch*`` against the shared ``sqlite3.Connection``
(``check_same_thread=False``) MUST run under the connection lock, because the
daemon's worker fan-out can reset cursor state between ``execute()`` and
``fetch*()`` on the shared connection. ``HippoQuery.to_batches`` (hippo.py) is the
one read path that runs ``cursor = self._conn.execute(sql)`` plus a
``cursor.fetchmany`` loop WITHOUT taking that lock — the gap that produces cursor
corruption (truncated rows / ``IndexError`` / ``InterfaceError``) under concurrent
shared-connection access in production.

WHY THIS TEST PINS THE LOCK-DISCIPLINE INVARIANT, NOT THE CORRUPTION SYMPTOM:
The corruption symptom is a genuine C-level data race that requires SQLite's
``step`` to release the GIL mid-fetch under real worker fan-out against a large
store. In a hermetic, fast, single-process test that race cannot be manufactured
deterministically: in WAL mode a SELECT reader takes a consistent snapshot and
never blocks on a writer, and CPython serialises each ``execute`` / ``fetchmany``
under the GIL with an independent per-statement cursor, so the public API exposes
no interleave window to corrupt. (Verified empirically across multiple disruptor
shapes: concurrent ``add()``, raw ``conn.execute`` loops, held write transactions,
busy_timeout=0 — all clean.) A test that asserts "to_batches RAISES" would
therefore XFAIL whether or not the bug exists — proving nothing.

Instead, this test pins the invariant the fix restores: while a worker holds the
real ``_conn_lock``, a ``to_batches`` drain MUST block until the lock is released.
Today ``to_batches`` is lock-free, so its ``conn.execute`` ignores the held Python
re-entrant lock and drains in milliseconds -> the assertion fails -> the test is
``xfail``. Once the ``to_batches`` fetch is guarded under ``_conn_lock``, the drain
blocks for the hold window and the assertion holds; the ``xfail`` marker is then
simply removed (no assertion rewrite).

Hermeticity: the store opens under ``tmp_path``; an in-test assertion fails if it
ever resolves under the operator's real ``~/.iai-mcp``. The worker thread holds
ONLY the Python ``_conn_lock`` via ``time.sleep`` (no open SQLite write
transaction), so today's lock-free reader is gated solely by the Python lock — a
clean discriminator.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


_N_SEED = 200
_HOLD_SEC = 1.5  # the worker holds _conn_lock for this window
# A guarded reader blocks ~the full hold; a lock-free reader drains in ~ms.
# Half the hold is a wide, unambiguous separator between the two regimes.
_BLOCK_THRESHOLD = _HOLD_SEC * 0.5


def _make_record(vec, community_id, idx: int) -> MemoryRecord:
    import datetime

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


def _seed_store(store: MemoryStore, n: int) -> None:
    dim = store._embed_dim
    cid = uuid4()
    rng = np.random.default_rng(11)
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        store.insert(_make_record(vec, cid, i))


def _assert_hermetic(store: MemoryStore, tmp_path: Path) -> None:
    root = Path(store.root).resolve()
    assert str(root).startswith(str(tmp_path.resolve())), (
        f"store root {root} escaped tmp_path {tmp_path}"
    )
    real_home_store = (Path.home() / ".iai-mcp").resolve()
    assert real_home_store not in root.parents and root != real_home_store, (
        f"store root {root} resolved under the real ~/.iai-mcp"
    )


def _hold_conn_lock(store: MemoryStore, hold_sec: float,
                    started: threading.Event, done: threading.Event) -> None:
    """Hold the real db._conn_lock for exactly hold_sec.

    ``started`` is set AFTER the lock is acquired so a waiter is guaranteed to
    contend. Only the Python re-entrant lock is held (via time.sleep) — no open
    SQLite write transaction — so a lock-disciplined reader is gated solely by
    the Python lock.
    """
    with store.db._conn_lock:
        started.set()
        time.sleep(hold_sec)
    done.set()


def test_to_batches_fetch_holds_conn_lock(tmp_path):
    """While a worker holds the real _conn_lock, a to_batches drain must block
    until the lock is released. The guarded fetch (post-fix) holds the
    connection lock across the cursor's full life, so the drain cannot complete
    while another thread holds the same lock -- it waits out the hold. This is
    now the post-fix regression guard for that lock-discipline invariant."""
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    _seed_store(store, _N_SEED)

    # The non-ANN query path must carry the HippoDB reference so the guarded
    # fetch (post-fix) can engage self._db._conn_lock.
    query = store.db.open_table("records").search()
    assert query._db is not None, "to_batches query lost its HippoDB reference"

    started = threading.Event()
    done = threading.Event()
    worker = threading.Thread(
        target=_hold_conn_lock,
        args=(store, _HOLD_SEC, started, done),
        daemon=True,
    )

    reader_elapsed = 0.0
    try:
        worker.start()
        assert started.wait(timeout=10.0), "worker never acquired _conn_lock"
        t0 = time.monotonic()
        rows = 0
        for batch in query.to_batches(batch_size=1):
            rows += batch.num_rows
        reader_elapsed = time.monotonic() - t0
        assert rows == _N_SEED, f"to_batches read {rows} rows, expected {_N_SEED}"
    finally:
        done.wait(timeout=_HOLD_SEC + 10.0)
        worker.join(timeout=5.0)
        store.close()

    # FIXED contract: a to_batches drain holds the connection lock, so it cannot
    # complete while a worker holds the same lock — it must wait out the hold.
    # The current lock-free fetch drains in ~ms regardless -> this fails -> xfail.
    assert reader_elapsed >= _BLOCK_THRESHOLD, (
        "to_batches drained without waiting for the held _conn_lock "
        f"(elapsed={reader_elapsed:.3f}s, hold={_HOLD_SEC}s) — the fetch did not "
        "hold the connection lock"
    )
