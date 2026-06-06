"""Regression tests for the daemon consolidation shared-connection race.

Two daemon consolidation drivers fan work to threads via asyncio.to_thread,
all sharing one ``HippoDB._conn`` (``check_same_thread=False``,
``isolation_level=None``). A co-occurrence writer (``boost_edges`` -> edge
merge_insert / add / search) running concurrently with the compaction VACUUM
(``optimize_hippo_storage``) used to corrupt the shared connection's
transaction state and starve the VACUUM:

  - "cannot start a transaction within a transaction"
  - "cannot commit - no transaction is active"
  - "bad parameter or other API misuse" (sqlite3.InterfaceError)
  - "database table is locked" / "cannot VACUUM - SQL statements in progress"

Root cause: edge writes (merge_insert.execute, non-records add) and table
reads (to_pandas / query to_pandas) issued BEGIN/COMMIT/SELECT on the shared
connection without holding ``_conn_lock``, so they interleaved with the
VACUUM that holds ``_conn_lock``. The fix serializes every shared-connection
statement site under ``_conn_lock``.

These tests are hermetic (pytest tmp_path; never touch the real store).
"""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from iai_mcp.hippo import AccessMode, HippoIntegrityError, _txn, _txn_owners, _txn_owners_lock
from iai_mcp.maintenance import optimize_hippo_storage
from iai_mcp.sleep import _decay_edges
from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

# Connection-corruption signatures (BUG A) and VACUUM-starvation signatures
# (the connection-level part of BUG B). NONE of these may appear under the fix.
_FORBIDDEN_SUBSTRINGS = (
    "cannot start a transaction within a transaction",
    "cannot commit - no transaction is active",
    "bad parameter or other api misuse",
    "database table is locked",
    "cannot vacuum",
    "sql statements in progress",
)


def _make_record(seed: int) -> MemoryRecord:
    rng = np.random.RandomState(seed)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"record seed {seed}",
        aaak_index="",
        embedding=rng.randn(EMBED_DIM).tolist(),
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


def _assert_no_forbidden(messages: list[str]) -> None:
    for msg in messages:
        low = msg.lower()
        for sig in _FORBIDDEN_SUBSTRINGS:
            assert sig not in low, (
                f"shared-connection consolidation race regressed: "
                f"forbidden signature {sig!r} in {msg!r}"
            )


def test_edge_writer_concurrent_with_vacuum_no_connection_corruption(
    tmp_path: Path,
) -> None:
    """A boost_edges writer thread running concurrently with the compaction
    VACUUM must not corrupt the shared connection nor starve the VACUUM.

    Reproduces the live daemon failure in a single process: thread A
    loops boost_edges (edge merge_insert + add + search on the shared
    connection) while thread B runs optimize_hippo_storage (WAL checkpoint +
    VACUUM on the SAME connection). Before the _conn_lock fix this raised the
    transaction-corruption and table-locked cascade.
    """
    store = MemoryStore(tmp_path, user_id="test")
    try:
        ids: list[uuid.UUID] = []
        for i in range(40):
            rec = _make_record(1000 + i)
            store.insert(rec)
            ids.append(rec.id)
        flush_record_buffer(store)

        errors: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            n = 0
            while not stop.is_set():
                try:
                    a = ids[n % len(ids)]
                    b = ids[(n + 1) % len(ids)]
                    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"writer: {type(exc).__name__}: {exc}")

        def vacuumer() -> None:
            for _ in range(15):
                if stop.is_set():
                    break
                report = optimize_hippo_storage(store)
                for table in ("records", "edges", "events"):
                    err = report.get(table, {}).get("error")
                    if err:
                        errors.append(f"vacuum[{table}]: {err}")
                time.sleep(0.005)

        ta = threading.Thread(target=writer, daemon=True)
        tb = threading.Thread(target=vacuumer, daemon=True)
        ta.start()
        tb.start()
        tb.join(timeout=30)
        stop.set()
        ta.join(timeout=5)

        _assert_no_forbidden(errors)
        assert not errors, f"unexpected errors under concurrent writer+VACUUM: {errors}"
    finally:
        store.close()


def test_escalate_sets_intent_flag_then_vacuum_clean(tmp_path: Path) -> None:
    """The daemon consolidation lock transition must set the
    consolidation-intent flag BEFORE the VACUUM runs, and the VACUUM must
    succeed under that EXCLUSIVE escalation.

    This asserts the BUG B wiring directly (intent flag present at VACUUM time),
    NOT the benign 'intent_missing' warning that fires when maintenance is
    invoked from a direct CLI-style call without escalation.
    """
    intent_path = tmp_path / "hippo" / ".consolidation-pending"

    # Open SHARED (mirrors the daemon's WAKE state) then escalate to EXCLUSIVE
    # for the consolidation window, exactly as the daemon FSM does.
    store = MemoryStore(tmp_path, user_id="test", access_mode=AccessMode.SHARED)
    try:
        for i in range(20):
            store.insert(_make_record(2000 + i))
        flush_record_buffer(store)

        assert not intent_path.exists(), "intent flag should be clear before escalate"

        store.db.escalate_to_exclusive()
        assert intent_path.exists(), (
            "escalate_to_exclusive must SET the consolidation-intent flag "
            "before the VACUUM window so SHARED clients back off"
        )
        assert store.db._access_mode is AccessMode.EXCLUSIVE

        report = optimize_hippo_storage(store)
        for table in ("records", "edges", "events"):
            assert report.get(table, {}).get("error") is None, (
                f"VACUUM under escalation must not fail for {table}: "
                f"{report.get(table, {}).get('error')}"
            )

        store.db.downgrade_to_shared()
        assert not intent_path.exists(), (
            "downgrade_to_shared must CLEAR the intent flag so clients proceed"
        )
        assert store.db._access_mode is AccessMode.SHARED
    finally:
        store.close()


def test_escalate_sets_intent_when_already_exclusive(tmp_path: Path) -> None:
    """escalate_to_exclusive must SET the consolidation-intent flag even when
    this process already holds EXCLUSIVE.

    The daemon can reach the consolidation window still EXCLUSIVE — e.g. it
    booted EXCLUSIVE (integrity rebuild) and went boot -> idle -> SLEEP without
    an intervening WAKE downgrade tick (the restart-into-idle path). The intent
    flag is a cross-process signal independent of this process's flock mode; if
    escalate early-returns on the already-EXCLUSIVE path without setting it, the
    compaction VACUUM runs with no consolidation-intent signal
    ("hippo_compact_intent_missing") and a racing client could open a fresh
    SHARED connection mid-VACUUM.
    """
    intent_path = tmp_path / "hippo" / ".consolidation-pending"

    # Boot EXCLUSIVE and never downgrade — mirror the restart-into-idle path.
    store = MemoryStore(tmp_path, user_id="test", access_mode=AccessMode.EXCLUSIVE)
    try:
        for i in range(10):
            store.insert(_make_record(3000 + i))
        flush_record_buffer(store)

        assert store.db._access_mode is AccessMode.EXCLUSIVE
        assert not intent_path.exists()

        store.db.escalate_to_exclusive()
        assert intent_path.exists(), (
            "escalate_to_exclusive must set the consolidation-intent flag even "
            "when already EXCLUSIVE (cross-process signal independent of flock)"
        )

        report = optimize_hippo_storage(store)
        for table in ("records", "edges", "events"):
            assert report.get(table, {}).get("error") is None, (
                f"VACUUM under already-EXCLUSIVE escalation must not fail for "
                f"{table}: {report.get(table, {}).get('error')}"
            )

        store.db.downgrade_to_shared()
        assert not intent_path.exists()
        assert store.db._access_mode is AccessMode.SHARED
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Variant A: endpoint-simulated M1/M2 race against canonical VACUUM/merge_insert
# ---------------------------------------------------------------------------

def test_legacy_edge_decay_endpoint_concurrent_with_vacuum_no_corruption(
    tmp_path: Path,
) -> None:
    """Endpoint-simulated M1 (tbl.update) + M2 (tbl.delete) on the edges table
    running concurrently with the canonical VACUUM and merge_insert must not
    corrupt the shared connection.

    The legacy thread directly exercises the HippoTable.update and
    HippoTable.delete paths that _decay_edges reaches in production. The
    canonical thread mirrors the daemon's compaction + co-occurrence boost.
    Before the M1/M2 _conn_lock fix this reproduced the transaction-corruption
    cascade on every run.

    Negative control: reverting the M1/M2 _conn_lock wrapping in hippo.py
    restores the race and causes this test to fail with a forbidden signature
    within the first few iterations.
    """
    store = MemoryStore(tmp_path, user_id="test")
    try:
        ids: list[uuid.UUID] = []
        for i in range(40):
            rec = _make_record(4000 + i)
            store.insert(rec)
            ids.append(rec.id)
        flush_record_buffer(store)
        # Seed initial edges so the legacy writer has rows to update/delete.
        for i in range(len(ids) - 1):
            store.boost_edges([(ids[i], ids[i + 1])], delta=0.3, edge_type="hebbian")

        errors: list[str] = []
        stop = threading.Event()

        def legacy_writer() -> None:
            tbl = store.db.open_table(EDGES_TABLE)
            n = 0
            while not stop.is_set():
                try:
                    a = str(ids[n % len(ids)])
                    b = str(ids[(n + 1) % len(ids)])
                    now = datetime.now(timezone.utc)
                    # M1: decay-style update (mirrors _decay_edges weight update)
                    tbl.update(
                        where=f"src = '{a}' AND dst = '{b}' AND edge_type = 'hebbian'",
                        values={"weight": 0.25, "updated_at": now},
                    )
                    # M2: prune-style delete then replenish so the table stays populated
                    tbl.delete(
                        f"src = '{a}' AND dst = '{b}' AND edge_type = 'hebbian'"
                    )
                    store.boost_edges(
                        [(ids[n % len(ids)], ids[(n + 1) % len(ids)])],
                        delta=0.3,
                        edge_type="hebbian",
                    )
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"legacy_writer: {type(exc).__name__}: {exc}")

        def canonical_thread() -> None:
            n = 0
            while not stop.is_set():
                try:
                    report = optimize_hippo_storage(store)
                    for table in ("records", "edges", "events"):
                        err = report.get(table, {}).get("error")
                        if err:
                            errors.append(f"vacuum[{table}]: {err}")
                    a = ids[n % len(ids)]
                    b = ids[(n + 1) % len(ids)]
                    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"canonical_thread: {type(exc).__name__}: {exc}")
                time.sleep(0.005)

        ta = threading.Thread(target=legacy_writer, daemon=True)
        tb = threading.Thread(target=canonical_thread, daemon=True)
        ta.start()
        tb.start()
        # Let canonical_thread drive the race window (~15 VACUUM cycles at 0.005 s).
        tb.join(timeout=15)
        stop.set()
        ta.join(timeout=5)
        tb.join(timeout=5)  # reap tb if it outlasted the first join

        _assert_no_forbidden(errors)
        assert not errors, (
            f"unexpected errors under concurrent M1/M2+VACUUM: {errors}"
        )

        # Post-race sanity: edges table must be queryable and consistent.
        df = store.db.open_table(EDGES_TABLE).to_pandas()
        assert df["weight"].notna().all(), "corrupt NULL weight after race"
        assert (df["weight"] >= 0).all(), "negative weight after race"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Variant B: REAL sleep._decay_edges concurrent with canonical VACUUM/merge_insert
# ---------------------------------------------------------------------------

def _insert_stale_edge(
    store: MemoryStore,
    src: uuid.UUID,
    dst: uuid.UUID,
    weight: float,
    days_old: int = 120,
    edge_type: str = "hebbian",
) -> None:
    """Insert an edge with a backdated updated_at so _decay_edges will act on it.

    boost_edges always sets updated_at=now, which places edges inside the
    90-day grace period — _decay_edges would skip them entirely. This helper
    writes directly into the edges table with an old timestamp so the sweep
    actually exercises the update (M1) and delete (M2) paths.
    """
    old_ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    tbl = store.db.open_table(EDGES_TABLE)
    # Use merge_insert so a pre-existing edge is overwritten cleanly.
    import pyarrow as pa
    rows = pa.Table.from_pylist(
        [{"src": str(src), "dst": str(dst), "edge_type": edge_type,
          "weight": weight, "updated_at": old_ts}],
        schema=pa.schema([
            ("src", pa.string()), ("dst", pa.string()),
            ("edge_type", pa.string()), ("weight", pa.float32()),
            ("updated_at", pa.string()),
        ]),
    )
    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(rows)
    )


def test_real_decay_edges_concurrent_with_vacuum_no_corruption(
    tmp_path: Path,
) -> None:
    """The REAL sleep._decay_edges (production decay/prune path) running
    concurrently with the canonical VACUUM + merge_insert must not corrupt
    the shared connection and must produce correct final edge counts.

    _decay_edges calls tbl.update (M1) for edges whose decayed weight stays
    above DECAY_EPSILON, and tbl.delete (M2) for edges that fall below it.
    Without M1/M2 _conn_lock serialization both sites race the VACUUM's
    BEGIN..COMMIT window.

    Edges are inserted with backdated updated_at (120 days ago, exceeding the
    90-day grace period) and split into two weight classes:
    - weight=0.5 → decays to ~0.021, stays above DECAY_EPSILON → M1 update
    - weight=0.02 → decays to ~0.00085, falls below DECAY_EPSILON → M2 delete

    The test asserts BOTH decayed>0 and pruned>0 in at least one iteration,
    confirming the real M1 and M2 paths were exercised (not skipped by grace).

    Negative control: reverting M1/M2 _conn_lock wrapping reproduces the
    'bad parameter or other API misuse' / 'cannot VACUUM' cascade.
    """
    store = MemoryStore(tmp_path, user_id="test")
    try:
        ids: list[uuid.UUID] = []
        for i in range(40):
            rec = _make_record(5000 + i)
            store.insert(rec)
            ids.append(rec.id)
        flush_record_buffer(store)

        # Seed stale edges: half decay-only (weight=0.5), half prune (weight=0.02).
        for i in range(0, len(ids) - 1, 2):
            _insert_stale_edge(store, ids[i], ids[i + 1], weight=0.5, days_old=120)
        for i in range(1, len(ids) - 1, 2):
            _insert_stale_edge(store, ids[i], ids[i + 1], weight=0.02, days_old=120)

        errors: list[str] = []
        stop = threading.Event()
        decay_results: list[dict] = []

        def legacy_decay_thread() -> None:
            n = 0
            while not stop.is_set():
                try:
                    result = _decay_edges(store)
                    decay_results.append(result)
                    # Re-seed stale edges so the next iteration has rows to act on.
                    pair_i = n % (len(ids) - 1)
                    _insert_stale_edge(
                        store, ids[pair_i], ids[pair_i + 1],
                        weight=0.5, days_old=120,
                    )
                    _insert_stale_edge(
                        store, ids[(pair_i + 1) % len(ids)],
                        ids[(pair_i + 2) % len(ids)],
                        weight=0.02, days_old=120,
                    )
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"decay_thread: {type(exc).__name__}: {exc}")

        def canonical_thread() -> None:
            n = 0
            while not stop.is_set():
                try:
                    report = optimize_hippo_storage(store)
                    for table in ("records", "edges", "events"):
                        err = report.get(table, {}).get("error")
                        if err:
                            errors.append(f"vacuum[{table}]: {err}")
                    a = ids[n % len(ids)]
                    b = ids[(n + 1) % len(ids)]
                    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"canonical_thread: {type(exc).__name__}: {exc}")
                time.sleep(0.01)

        ta = threading.Thread(target=legacy_decay_thread, daemon=True)
        tb = threading.Thread(target=canonical_thread, daemon=True)
        ta.start()
        tb.start()
        # Let both threads race for a bounded wall-clock window, then stop them.
        # canonical_thread controls the iteration pace (0.01 s sleep per cycle);
        # ~15 VACUUM cycles covers the race surface without running too long.
        tb.join(timeout=15)
        stop.set()
        ta.join(timeout=5)
        tb.join(timeout=5)  # reap tb if it outlasted the first join

        _assert_no_forbidden(errors)
        assert not errors, (
            f"unexpected errors under concurrent _decay_edges+VACUUM: {errors}"
        )

        # Confirm the real M1 and M2 paths were exercised (not just grace-skipped).
        total_decayed = sum(r.get("decayed", 0) for r in decay_results)
        total_pruned = sum(r.get("pruned", 0) for r in decay_results)
        assert total_decayed > 0, (
            "_decay_edges never exercised M1 (update) — "
            "edges may not have been stale enough or weight class misconfigured"
        )
        assert total_pruned > 0, (
            "_decay_edges never exercised M2 (delete/prune) — "
            "edges may not have fallen below DECAY_EPSILON"
        )

        # Post-race sanity: edges table must be consistent.
        df = store.db.open_table(EDGES_TABLE).to_pandas()
        if not df.empty:
            assert df["weight"].notna().all(), "corrupt NULL weight after race"
            assert (df["weight"] >= 0).all(), "negative weight after race"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tripwire-semantics unit tests
# ---------------------------------------------------------------------------

def test_txn_same_thread_reentry_does_not_raise(tmp_path: Path) -> None:
    """Nested _txn calls on the same thread must NOT raise HippoIntegrityError.

    This is the legitimate RLock re-entry case: a helper calls _txn while
    already inside an outer _txn on the same thread.
    """
    import sqlite3 as _sqlite3
    db_path = tmp_path / "test_reentry.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        with _txn(conn):
            # Same-thread nested _txn: must yield without raising.
            with _txn(conn):
                conn.execute("INSERT INTO t VALUES (1)")
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row is not None and row[0] == 1, "nested _txn did not commit"
    finally:
        conn.close()


def test_txn_no_owner_caller_managed_outer_txn_does_not_raise(tmp_path: Path) -> None:
    """A caller-managed outer BEGIN (no owner in _txn_owners) must NOT raise.

    Some callers issue their own BEGIN before entering _txn, expecting _txn to
    be a no-op nesting guard. The tripwire must not false-positive here.
    """
    import sqlite3 as _sqlite3
    db_path = tmp_path / "test_no_owner.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        # Caller-managed BEGIN — does NOT go through _txn, so no owner is recorded.
        conn.execute("BEGIN")
        with _txn(conn):
            # Must yield without raising (no-owner case 2).
            conn.execute("INSERT INTO t VALUES (99)")
        conn.execute("COMMIT")
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row is not None and row[0] == 99
        # Confirm no stale owner left in the map.
        with _txn_owners_lock:
            assert id(conn) not in _txn_owners, "stale owner entry after no-owner txn"
    finally:
        conn.close()


def test_txn_foreign_owner_raises_hippo_integrity_error(tmp_path: Path) -> None:
    """A foreign thread observing an in-progress _txn on the shared connection
    must raise HippoIntegrityError, NOT silently yield into the open transaction.

    Thread A acquires _conn_lock and enters _txn (the production pattern for a
    correctly-serialized site). Thread B enters _txn WITHOUT holding _conn_lock
    (the pattern of a missing-lock site). B must raise immediately.

    After A's _txn exits (barrier release), the owner entry must be cleared so
    a subsequent legitimate caller on any thread does NOT see a stale owner.
    """
    import sqlite3 as _sqlite3
    db_path = tmp_path / "test_foreign_owner.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    import threading as _threading

    barrier_txn_open = _threading.Event()   # A signals: txn is open
    barrier_release_a = _threading.Event()  # B signals: done, A may exit
    raised: list[Exception] = []
    a_errors: list[str] = []

    def thread_a() -> None:
        # Correctly-serialized site: holds _conn_lock around the full _txn window.
        import contextlib
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_txn(conn))
                # Signal that the transaction is open.
                barrier_txn_open.set()
                # Wait for thread B to finish its assertion.
                barrier_release_a.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            a_errors.append(f"thread_a unexpected error: {exc}")

    def thread_b() -> None:
        # Wait until A has an open transaction.
        barrier_txn_open.wait(timeout=5)
        try:
            # Missing-lock site: enters _txn WITHOUT _conn_lock.
            with _txn(conn):
                pass
            raised.append(None)  # sentinel: no exception raised
        except HippoIntegrityError as exc:
            raised.append(exc)
        except Exception as exc:  # noqa: BLE001
            raised.append(exc)
        finally:
            barrier_release_a.set()

    ta = _threading.Thread(target=thread_a, daemon=True)
    tb = _threading.Thread(target=thread_b, daemon=True)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not a_errors, f"thread_a failed unexpectedly: {a_errors}"
    assert len(raised) == 1, "thread_b did not complete"
    assert isinstance(raised[0], HippoIntegrityError), (
        f"expected HippoIntegrityError from foreign-owner thread, "
        f"got: {type(raised[0]).__name__}: {raised[0]}"
    )
    # Owner must be cleared after A's _txn exits.
    with _txn_owners_lock:
        assert id(conn) not in _txn_owners, (
            "owner-map entry NOT cleared after A's _txn exit — "
            "stale entry would false-positive the next legitimate caller"
        )
    conn.close()
