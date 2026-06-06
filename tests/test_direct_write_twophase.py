"""Scaffolds for two-phase direct write + deferred embedding +
boot with a pending row + migration back-compat.

Validation rows cover lost-write / dedup, index coherence, and the write SLO.

Tests marked xfail(strict=True) flip to pass when the corresponding
implementation lands.
"""
from __future__ import annotations

import sqlite3
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episodic_record(text: str = "generic user turn"):
    """Return a minimal episodic MemoryRecord with a real (random) embedding."""
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    rng = np.random.RandomState(seed=42)
    vec = rng.randn(EMBED_DIM).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
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
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
    )


def _zero_vector_blob(embed_dim: int) -> bytes:
    """Return an embed_dim zero-vector as a BLOB (4 bytes * embed_dim, little-endian float32)."""
    return struct.pack(f"<{embed_dim}f", *([0.0] * embed_dim))


# ---------------------------------------------------------------------------
# Test 1: direct write visible to recency (daemon down)
# ---------------------------------------------------------------------------


def test_direct_write_visible_to_recency_daemon_down(hermetic_store: Path) -> None:
    """Direct write with no daemon; row present in SQLite + recency in ≤1.5 s.

    The not-yet-existing direct write path bypasses the daemon socket and
    inserts the SQLite row immediately. This test imports that helper inside
    the body so a collection-time ImportError does not block other tests.
    """
    from iai_mcp.direct_write import write_turn_direct  # type: ignore[import]
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    t0 = time.monotonic()
    write_turn_direct(
        store_root=hermetic_store,
        text="direct write probe text",
        session_id="test-session",
        role="user",
    )
    elapsed_write = time.monotonic() - t0
    assert elapsed_write <= 1.5, f"direct write took {elapsed_write:.3f} s (SLO ≤1.5 s)"

    t1 = time.monotonic()
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    elapsed_read = time.monotonic() - t1
    assert elapsed_read <= 1.5, f"recency read after direct write took {elapsed_read:.3f} s"

    surfaces = [t.literal_surface for t in turns]
    assert any("direct write probe text" in s for s in surfaces), (
        "directly written turn not visible via recency immediately after write"
    )


# ---------------------------------------------------------------------------
# Test 2: no duplicate row on re-drain (idem-tag dedup)
# ---------------------------------------------------------------------------


def test_no_duplicate_row_on_redrain(hermetic_store: Path) -> None:
    """Idem-tag dedup — no duplicate row when the same turn is re-drained.

    Writes a turn via the direct path, then re-drains the same session/role/ts/text
    through the normal capture_turn path (same _idem_tag). Asserts exactly one
    row exists.
    """
    from iai_mcp.direct_write import write_turn_direct  # type: ignore[import]
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.capture import capture_turn

    ts_iso = datetime.now(timezone.utc).isoformat()
    text = "idem-dedup probe text"
    session_id = "idem-session"

    # Direct write (the new path).
    write_turn_direct(
        store_root=hermetic_store,
        text=text,
        session_id=session_id,
        role="user",
        ts_iso=ts_iso,
    )

    # Re-drain through capture_turn (the existing path, same idem key).
    store = MemoryStore(hermetic_store)
    try:
        capture_turn(
            store,
            cue=text,
            text=text,
            tier="episodic",
            session_id=session_id,
            role="user",
            ts=ts_iso,
        )
        flush_record_buffer(store)

        # Assert exactly one row for this idem key.
        from iai_mcp.capture import _idem_tag
        tag = _idem_tag(session_id, "user", ts_iso, text)
        record_id = store.find_record_by_tag(tag)
        assert record_id is not None, "idem-tagged row should exist after direct write"

        # Verify no duplicates by scanning all records.
        records = store.all_records()
        matching = [r for r in records if text in (r.literal_surface or "")]
        assert len(matching) == 1, (
            f"expected exactly 1 row for idem text, got {len(matching)} (duplicate on re-drain)"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3: mid-run integrity rebuild
# ---------------------------------------------------------------------------


def test_integrity_rebuild_triggers_mid_run(hermetic_store: Path) -> None:
    """Mid-run integrity rebuild reconciles index when SQLite leads.

    Constructs a state where SQLite has a record absent from the hnswlib
    index (active_label_count != sqlite_count) by inserting directly into
    SQLite after boot, then asserts the mid-run rebuild API reconciles the
    index without closing and reopening HippoDB.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp.types import EMBED_DIM

    import numpy as np

    hippo = HippoDB(hermetic_store)
    try:
        # Inject a record directly into SQLite, bypassing hnswlib.
        vec_blob = _zero_vector_blob(EMBED_DIM)
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with hippo._conn_lock:
            hippo._conn.execute(
                "INSERT INTO records "
                "(id, tier, literal_surface, aaak_index, embedding, "
                " created_at, updated_at, hv_tier, structure_hv_payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'bsc', x'')",
                (record_id, "episodic", "mid-run probe", "", vec_blob, now, now),
            )
            hippo._conn.commit()

        # The index is now stale (sqlite_count > active_label_count).
        active_before = len(hippo._label_map)
        with hippo._conn_lock:
            sqlite_count_row = hippo._conn.execute(
                "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
            ).fetchone()
        sqlite_count = sqlite_count_row[0]
        assert sqlite_count > active_before, (
            "test setup failed: SQLite count should exceed label-map count"
        )

        # Import the not-yet-existing mid-run reconcile API.
        from iai_mcp.hippo import reconcile_index_mid_run  # type: ignore[import]
        reconcile_index_mid_run(hippo)

        # After reconcile the label map should reflect the new row.
        assert record_id in hippo._label_map, (
            "injected record not in _label_map after mid-run rebuild"
        )
    finally:
        hippo.close()


# ---------------------------------------------------------------------------
# Test 4: H3 daemon-down deferred-embedding write SLO
# ---------------------------------------------------------------------------


def test_daemon_down_write_deferred_embedding_slo(hermetic_store: Path) -> None:
    """Daemon-down write — ≤1.5 s, pending zero-vector, recency-recallable,
    valid BLOB after simulated daemon re-embed.

    Asserts all four requirements:
    (1) write completes in ≤1.5 s (no cold embed call);
    (2) SQLite row present with embed_dim zero-vector BLOB + embedding_pending flag;
    (3) row is recency-recallable immediately (recency is embedding-independent);
    (4) after simulated re-embed, row carries a valid BLOB (length == embed_dim, non-zero).
    """
    from iai_mcp.direct_write import write_turn_direct  # type: ignore[import]
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]
    from iai_mcp.types import EMBED_DIM

    t0 = time.monotonic()
    write_turn_direct(
        store_root=hermetic_store,
        text="deferred embedding probe text",
        session_id="test-session",
        role="user",
        # Force deferred-embed mode: no embedder available.
        deferred_embedding=True,
    )
    elapsed = time.monotonic() - t0

    # (1) SLO check.
    assert elapsed <= 1.5, (
        f"deferred-embedding write took {elapsed:.3f} s (SLO ≤1.5 s) — "
        "write must complete fast without calling the embedder"
    )

    # (2) Verify the SQLite row has a zero-vector BLOB and embedding_pending=1.
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT embedding, embedding_pending FROM records "
            "WHERE literal_surface LIKE '%deferred embedding probe%' "
            "AND tombstoned_at IS NULL"
        ).fetchone()
        assert row is not None, "deferred-embedding row not found in SQLite"
        blob = row["embedding"]
        assert blob is not None and len(blob) > 0, (
            "embedding BLOB must not be NULL/empty (records.embedding is BLOB NOT NULL)"
        )
        floats = struct.unpack(f"<{EMBED_DIM}f", blob)
        assert len(floats) == EMBED_DIM, (
            f"BLOB length mismatch: got {len(floats)} floats, expected {EMBED_DIM}"
        )
        assert all(f == 0.0 for f in floats), (
            "pending row must carry a zero-vector BLOB (not a real embedding)"
        )
        pending_flag = row["embedding_pending"]
        assert pending_flag == 1, (
            f"embedding_pending flag must be 1 for a deferred-embed row, got {pending_flag}"
        )
    finally:
        conn.close()

    # (3) Recency-recallable immediately.
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    surfaces = [t.literal_surface for t in turns]
    assert any("deferred embedding probe" in s for s in surfaces), (
        "pending row not recency-recallable immediately (recency is embedding-independent)"
    )

    # (4) After simulated re-embed: valid non-zero BLOB, flag cleared.
    from iai_mcp.direct_write import simulate_daemon_reembed  # type: ignore[import]
    import numpy as np

    rng = np.random.RandomState(seed=99)
    real_embedding = rng.randn(EMBED_DIM).tolist()
    simulate_daemon_reembed(hermetic_store, text_fragment="deferred embedding probe", embedding=real_embedding)

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    try:
        row2 = conn2.execute(
            "SELECT embedding, embedding_pending FROM records "
            "WHERE literal_surface LIKE '%deferred embedding probe%' "
            "AND tombstoned_at IS NULL"
        ).fetchone()
        assert row2 is not None
        blob2 = row2["embedding"]
        floats2 = struct.unpack(f"<{EMBED_DIM}f", blob2)
        assert any(f != 0.0 for f in floats2), (
            "after daemon re-embed the BLOB must be non-zero (a real embedding)"
        )
        assert row2["embedding_pending"] == 0, (
            "embedding_pending flag must be cleared after daemon re-embed"
        )
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Test 5: boot with pending row — no crash, no churn
# ---------------------------------------------------------------------------


def test_boot_with_pending_row_no_crash_no_churn(hermetic_store: Path) -> None:
    """Daemon boot with a pending-embedding row — no crash, no churn.

    Constructs a store containing one pending-embedding row (zero-vector BLOB +
    embedding_pending=1) alongside one normal row, then opens a fresh HippoDB
    (simulating a daemon boot).

    Asserts all four requirements:
    (1) boot does NOT crash;
    (2) pending row is recency-recallable;
    (3) boot integrity check reports CONVERGED (no spurious rebuild);
    (4) after simulated re-embed, row carries a valid BLOB and is ANN-findable.
    """
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    # Build the store manually: create the hippo directory + SQLite schema by
    # opening one MemoryStore, inserting a normal row, closing it, then injecting
    # the pending row directly into SQLite.
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_episodic_record("normal embedded row")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    # Inject the pending row directly into SQLite.
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    pending_id = str(uuid.uuid4())
    zero_blob = _zero_vector_blob(EMBED_DIM)
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Add the embedding_pending column if absent (pre-migration store).
        cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "embedding_pending" not in cols:
            conn.execute(
                "ALTER TABLE records ADD COLUMN embedding_pending INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()

        conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, aaak_index, embedding, embedding_pending, "
            " created_at, updated_at, hv_tier, structure_hv_payload) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?, 'bsc', x'')",
            (pending_id, "episodic", "pending row text", "", zero_blob, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    # (1) Open a fresh HippoDB (daemon boot simulation) — must NOT crash.
    from iai_mcp.hippo import HippoDB
    hippo = HippoDB(hermetic_store)
    try:
        # (3) CONVERGED check: active_label_count should equal sqlite non-pending count.
        # Pending rows must be excluded from BOTH the ANN index AND the sqlite_count
        # comparison so they never trigger a spurious rebuild on every boot check.
        active_label_count = len(hippo._label_map)
        with hippo._conn_lock:
            non_pending_count_row = hippo._conn.execute(
                "SELECT COUNT(*) FROM records "
                "WHERE tombstoned_at IS NULL AND (embedding_pending IS NULL OR embedding_pending = 0)"
            ).fetchone()
        non_pending_count = non_pending_count_row[0]

        assert active_label_count == non_pending_count, (
            f"C2-H2 churn bug: active_label_count={active_label_count} != "
            f"non_pending_count={non_pending_count}; pending rows must be excluded "
            "from the ANN label map to prevent perpetual rebuild"
        )

        # (2) Pending row is recency-recallable.
        from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]
        turns = read_recent_user_turns_direct(hermetic_store, n=10)
        surfaces = [t.literal_surface for t in turns]
        assert any("pending row text" in s for s in surfaces), (
            "C2-H2: pending row must be recency-recallable immediately (embedding-independent)"
        )

        # (4) After re-embed: valid BLOB, ANN-findable.
        from iai_mcp.direct_write import simulate_daemon_reembed  # type: ignore[import]
        rng = np.random.RandomState(seed=77)
        real_vec = rng.randn(EMBED_DIM).tolist()
        simulate_daemon_reembed(hermetic_store, text_fragment="pending row", embedding=real_vec)

        # Reload the store and check ANN findability.
        hippo.close()
        hippo = HippoDB(hermetic_store)
        assert pending_id in hippo._label_map, (
            "pending row should be in the ANN label map after re-embed"
        )
    finally:
        hippo.close()


# ---------------------------------------------------------------------------
# Test 6: pre-migration store — opens cleanly + whitelist + old-reader
# ---------------------------------------------------------------------------


def test_pre_migration_store_opens_and_reconciles(hermetic_store: Path) -> None:
    """Pre-migration store (no embedding_pending column) opens cleanly.

    Constructs a pre-migration store (records table WITHOUT embedding_pending),
    then opens a fresh HippoDB which must run _reconcile_columns.

    Asserts:
    (1) the open succeeds — _reconcile_columns adds the column without crashing;
    (2) 'INTEGER NOT NULL DEFAULT 0' is ACCEPTED by the whitelist
        (hippo.py allowed_with_default set) — no RuntimeError;
    (3) old-reader tolerance: a row carrying the new embedding_pending
        column round-trips store._from_row without crashing.
    """
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    # Build a minimal store, then drop the embedding_pending column (pre-migration state).
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_episodic_record("migration test row")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    # Drop the embedding_pending column to simulate a pre-migration store.
    # SQLite < 3.35 has no DROP COLUMN; recreate the table using a fully literal
    # SQL string (no interpolation, no concatenation — avoids semgrep SQL-injection
    # false-positive on dynamic column lists).
    _CREATE_BACKUP_SQL = (
        "CREATE TABLE records_v4_backup AS SELECT"
        " vec_label, id, tier, literal_surface, aaak_index, embedding, structure_hv,"
        " community_id, centrality, detail_level, pinned, stability, difficulty,"
        " last_reviewed, never_decay, never_merge, tombstoned_at, schema_bypass,"
        " labile_until, provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version, wing, room,"
        " drawer, valence, hv_tier, structure_hv_payload"
        " FROM records"
    )
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "embedding_pending" in cols:
            conn.execute("BEGIN")
            conn.execute(_CREATE_BACKUP_SQL)
            conn.execute("DROP TABLE records")
            conn.execute("ALTER TABLE records_v4_backup RENAME TO records")
            conn.execute("COMMIT")
    finally:
        conn.close()

    # (1) Open fresh HippoDB — must NOT raise RuntimeError for the whitelist check.
    # The 'INTEGER NOT NULL DEFAULT 0' column must be accepted by the whitelist.
    from iai_mcp.hippo import HippoDB
    hippo = HippoDB(hermetic_store)
    try:
        # (2) embedding_pending column must now exist (added by _reconcile_columns).
        with hippo._conn_lock:
            cols_after = {row[1] for row in hippo._conn.execute("PRAGMA table_info(records)")}
        assert "embedding_pending" in cols_after, (
            "C3-H4: _reconcile_columns must add embedding_pending column to pre-migration store"
        )

        # (3) Old-reader named-column tolerance: a row with the new column must round-trip
        # store._from_row without crashing.
        # Re-open as MemoryStore to exercise _from_row.
        hippo.close()

        store2 = MemoryStore(hermetic_store)
        try:
            records = store2.all_records()
            assert len(records) >= 1, "migration test row should be present after reconcile"
            # Just accessing all_records() without crashing proves _from_row tolerates
            # the new column (it uses row.get("embedding_pending") not **row splat).
        finally:
            store2.close()

    except Exception:
        hippo.close()
        raise
    else:
        hippo.close()
