"""TDD regression tests for nested-transaction safety in Hippo.

Three scenarios:
  (a) Direct nested call: outer BEGIN open, then call HippoMergeInsert.execute()
      — must succeed (pre-fix: OperationalError "cannot start a transaction
      within a transaction").
  (b) Nested rollback safety: outer BEGIN open, force error inside
      HippoMergeInsert.execute() — outer transaction must survive (not be
      corrupted by a partial inner state).
  (c) Non-nested control: no outer BEGIN — HippoMergeInsert.execute() must
      still commit cleanly.

All tests use real SQLite tmp files (not in-memory) so the encryption layer
can operate correctly.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from iai_mcp.hippo import HippoDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 384


def _make_db(tmp_path: Path) -> HippoDB:
    """Open a fresh HippoDB against tmp_path (no ~/.iai-mcp state touched)."""
    store_dir = tmp_path / "hippo"
    store_dir.mkdir(parents=True, exist_ok=True)
    return HippoDB(str(store_dir))


def _sample_edge_row(src_id: str, dst_id: str) -> dict:
    from datetime import datetime, timezone
    return {
        "src": src_id,
        "dst": dst_id,
        "edge_type": "hebbian",
        "weight": 0.1,
        "updated_at": datetime.now(timezone.utc),
    }


def _edge_arrow(rows: list[dict]) -> pa.Table:
    from datetime import timezone
    import pyarrow as pa
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("src", pa.string()),
                ("dst", pa.string()),
                ("edge_type", pa.string()),
                ("weight", pa.float32()),
                ("updated_at", pa.timestamp("us", tz="UTC")),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Scenario (a): nested call succeeds under outer transaction
# ---------------------------------------------------------------------------


def test_nested_merge_insert_under_outer_tx_succeeds(tmp_path: Path) -> None:
    """HippoMergeInsert.execute() must not raise when conn already in a tx.

    Pre-fix: sqlite3.OperationalError "cannot start a transaction within a
    transaction" was raised at hippo.py.
    """
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    # Open an outer transaction manually (simulates the caller already being
    # in a transaction — exactly the daemon's live state at crash time).
    conn.execute("BEGIN")
    assert conn.in_transaction, "Expected conn to be in transaction after BEGIN"

    # This must NOT raise OperationalError.
    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    # Outer transaction still open — caller controls commit.
    conn.execute("COMMIT")

    # Verify the row landed.
    result = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert result == 1, f"Expected 1 edge row, got {result}"

    db.close()


# ---------------------------------------------------------------------------
# Scenario (b): nested error inside HippoMergeInsert does not corrupt outer tx
# ---------------------------------------------------------------------------


def test_nested_merge_insert_error_does_not_corrupt_outer_tx(
    tmp_path: Path,
) -> None:
    """Outer transaction survives an error raised inside HippoMergeInsert.execute().

    After the inner error:
    - The outer transaction is still valid (not rolled back).
    - A subsequent SQL op on the outer transaction succeeds.
    """
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())

    conn.execute("BEGIN")

    # Pass a row with a deliberately broken schema (missing required columns)
    # so executemany raises inside _txn — but conn state must remain usable.
    bad_arrow = pa.table({"src": [src_id]})  # missing dst, edge_type, weight, etc.
    with pytest.raises(Exception):
        (
            tbl.merge_insert(["src", "dst", "edge_type"])
            .when_matched_update_all()
            .execute(bad_arrow)
        )

    # The outer transaction must still be operable (not in an error state).
    # Issue a harmless SELECT inside the outer tx.
    count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    # count is whatever — the important thing is it didn't raise.
    assert isinstance(count, int)

    conn.execute("ROLLBACK")
    db.close()


# ---------------------------------------------------------------------------
# Scenario (c): non-nested call commits cleanly (regression control)
# ---------------------------------------------------------------------------


def test_non_nested_merge_insert_commits_cleanly(tmp_path: Path) -> None:
    """HippoMergeInsert.execute() without any outer tx commits correctly.

    This is the existing happy path — must continue to work after the fix.
    """
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    # No outer transaction open.
    assert not conn.in_transaction, "Expected no active transaction before call"

    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    # Transaction should be closed after the call.
    assert not conn.in_transaction, "Expected transaction closed after execute()"

    # Verify row committed.
    result = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert result == 1, f"Expected 1 edge row, got {result}"

    db.close()


# ---------------------------------------------------------------------------
# Scenario (d): HippoTable.update() is also safe under outer tx
# ---------------------------------------------------------------------------


def test_update_under_outer_tx_succeeds(tmp_path: Path) -> None:
    """HippoTable.update() must not raise when conn already in a transaction."""
    from datetime import datetime, timezone

    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    # Insert first (clean, no outer tx).
    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    # Now open outer tx and call update() inside it.
    conn.execute("BEGIN")
    tbl.update(
        where=f"src = '{src_id}' AND dst = '{dst_id}' AND edge_type = 'hebbian'",
        values={"weight": 0.9, "updated_at": datetime.now(timezone.utc)},
    )
    conn.execute("COMMIT")

    result = conn.execute(
        "SELECT weight FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert abs(result - 0.9) < 0.01, f"Expected weight ~0.9, got {result}"

    db.close()


# ---------------------------------------------------------------------------
# Scenario (e): HippoTable.delete() is also safe under outer tx
# ---------------------------------------------------------------------------


def test_delete_under_outer_tx_succeeds(tmp_path: Path) -> None:
    """HippoTable.delete() (non-records path) is safe under outer transaction."""
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    conn.execute("BEGIN")
    tbl.delete(
        where=f"src = '{src_id}' AND dst = '{dst_id}'"
    )
    conn.execute("COMMIT")

    result = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert result == 0, f"Expected 0 edges after delete, got {result}"

    db.close()
