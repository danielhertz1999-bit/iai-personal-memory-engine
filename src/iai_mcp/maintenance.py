"""Hippo storage maintenance: WAL checkpoint, VACUUM, hnswlib snapshot rebuild.

Runs from the sleep pipeline's compaction step and from the
``iai-mcp maintenance compact-hippo`` CLI command.

The VACUUM phase acquires SQLite's EXCLUSIVE lock and blocks all writers
for its duration — callers MUST invoke during quiescent windows only.

Two env overrides (read once at import):
- IAI_MCP_HIPPO_COMPACT_INTERVAL_SEC (default 3600s = 1 h cadence)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 1-hour periodic cadence (same order of magnitude as the maintenance work;
# far longer than a typical session; short enough that fragmentation stays bounded).
HIPPO_COMPACT_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_HIPPO_COMPACT_INTERVAL_SEC", "3600.0"),
)

# Daemon-owned tables; matches store.py constants
# (RECORDS_TABLE/EDGES_TABLE/EVENTS_TABLE) kept literal so this module
# does not pull MemoryStore at import time.
_HIPPO_TABLES_TO_COMPACT: tuple[str, ...] = ("records", "edges", "events")


def _measure_table_size_bytes(store: Any, table_name: str) -> int:
    """Estimate the byte size occupied by one table in brain.sqlite3.

    SQLite has a single database file for all tables. Per-table size is
    approximated via the ``dbstat`` virtual table (available in Python 3.11+
    stdlib builds which ship SQLite ≥ 3.31). Falls back to a row-count
    approximation (rows × 256 bytes) when dbstat is unavailable.

    Returns 0 on any measurement failure — size metrics are best-effort
    observability; a failure here MUST NOT cause the helper itself to raise.
    The total DB file size (returned once per full compaction pass) is the
    authoritative number; per-table figures are diagnostic only.
    """
    try:
        db = getattr(store, "db", None)
        if db is None:
            return 0
        conn = getattr(db, "_conn", None)
        if conn is None:
            return 0
        # dbstat virtual table: sum of page sizes per table (SQLite ≥ 3.31).
        try:
            row = conn.execute(
                "SELECT SUM(pgsize) FROM dbstat WHERE name = ?", (table_name,)
            ).fetchone()
            if row is not None and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
        # Fallback: approximate from row count using pre-built static SQL
        # (same pattern as hippo.py _TABLE_SQL; no dynamic SQL needed here
        # because this helper only runs on the three compactable tables).
        _COUNT_SQL: dict[str, str] = {
            "records": "SELECT COUNT(*) FROM records",
            "edges":   "SELECT COUNT(*) FROM edges",
            "events":  "SELECT COUNT(*) FROM events",
        }
        count_sql = _COUNT_SQL.get(table_name)
        if count_sql is not None:
            try:
                count_row = conn.execute(count_sql).fetchone()
                if count_row is not None:
                    return int(count_row[0]) * 256
            except Exception:
                pass
        return 0
    except (OSError, TypeError, AttributeError):
        return 0


def optimize_hippo_storage(
    store: Any,
    *,
    retention: timedelta | None = None,         # ignored — no MVCC in Hippo
    max_versions: int | None = None,            # ignored — no versions in Hippo
    delete_unverified: bool = False,            # ignored — no _transactions/ in Hippo
) -> dict[str, dict[str, Any]]:
    """Compact the Hippo storage.

    Three steps, in order:
      1. PRAGMA wal_checkpoint(TRUNCATE) — drains the WAL sidecar into the
         main brain.sqlite3 file. This is the quiescent window step; after
         it returns, no readers hold stale WAL snapshots. Runs FIRST so the
         WAL is empty before the EXCLUSIVE lock is acquired.
      2. VACUUM — reclaims space from tombstoned rows by rewriting the DB
         file in place. VACUUM acquires SQLite's EXCLUSIVE lock for its
         FULL DURATION; all other writers block. Wall-clock time scales
         with live data size (roughly ~1s per GB on local SSD).
      3. hnswlib rebuild + atomic save — drops mark_deleted slots from the
         in-memory index and persists a compact snapshot.

    BLOCKING WARNING: VACUUM blocks all SQLite writers. Callers MUST invoke
    this only during a quiescent window:
      - From the sleep pipeline OPTIMIZE step (caller guarantees no
        in-flight captures), OR
      - From the ``iai-mcp maintenance compact-hippo`` CLI (caller verifies
        daemon is stopped via pre-flight check).

    The legacy keyword arguments (retention, max_versions, delete_unverified)
    are preserved for API compatibility but have NO effect under SQLite +
    hnswlib (there are no versions to retain, no manifests to verify).

    Per-table dict keys: rows_before, rows_after, size_bytes_before,
    size_bytes_after, vacuum_elapsed_sec, hnswlib_rebuild_elapsed_sec,
    elapsed_sec. The optional 'error' key appears only on failure.

    Never raises; per-table failure is captured in the return dict's ``error``
    field (matching old contract). Per-table exception captured in report
    dict, helper continues with the next table.
    """
    report: dict[str, dict[str, Any]] = {}
    db = getattr(store, "db", None)

    overall_t0 = time.monotonic()

    # Collect pre-compaction measurements for each table.
    per_table_before: dict[str, tuple[int, int]] = {}
    for table_name in _HIPPO_TABLES_TO_COMPACT:
        try:
            if db is None:
                raise RuntimeError("store has no .db attribute")
            tbl = db.open_table(table_name)
            rows = int(tbl.count_rows())
            size = _measure_table_size_bytes(store, table_name)
            per_table_before[table_name] = (rows, size)
        except Exception:
            per_table_before[table_name] = (0, 0)

    # Intent-flag guard: VACUUM acquires a SQLite EXCLUSIVE lock that blocks
    # all readers. Callers MUST set the consolidation-intent flag BEFORE calling
    # this function so SHARED clients back off rather than opening new connections
    # during the VACUUM window.
    # This is a best-effort guard — not a hard prerequisite — because maintenance
    # may be invoked from the CLI without a running daemon.
    if db is not None:
        hippo_dir = getattr(db, "_hippo_dir", None)
        if hippo_dir is not None:
            intent_path = hippo_dir / ".consolidation-pending"
            if not intent_path.exists():
                logger.warning(
                    "hippo_compact_intent_missing: VACUUM called without "
                    "consolidation-intent flag set at %s — clients may open "
                    "a new SHARED connection during VACUUM",
                    intent_path,
                )

    # Step 1 + 2: WAL drain then VACUUM.
    # Error captured so it propagates to each table's report without raising.
    db_compaction_error: str | None = None
    vacuum_elapsed: float = 0.0
    try:
        conn = db._conn  # HippoDB's shared sqlite3.Connection.
        # Serialize the WAL-checkpoint + VACUUM against every other thread that
        # touches db._conn. The daemon runs consolidation across two concurrent
        # drivers that both fan work to threads via asyncio.to_thread (one writes
        # co-occurrence edges through boost_edges while this compaction runs).
        # Without holding _conn_lock for the whole checkpoint+VACUUM window a
        # sibling writer thread keeps an open transaction on the one shared
        # connection, so VACUUM fails with "cannot VACUUM - SQL statements in
        # progress" / "database table is locked", and the in_transaction probe
        # below races (TOCTOU) into "cannot commit - no transaction is active".
        # _conn_lock is the existing re-entrant RLock; this never takes
        # _hnsw_lock so the _hnsw_lock-before-_conn_lock order is preserved (the
        # hnswlib rebuild below is a SEPARATE try-block that takes _hnsw_lock on
        # its own, never while holding _conn_lock).
        conn_lock = getattr(db, "_conn_lock", None)
        _lock_ctx = conn_lock if conn_lock is not None else nullcontext()
        with _lock_ctx:
            # Step 1: WAL drain (quiescence gate). MUST come BEFORE VACUUM so the
            # WAL file is empty and the EXCLUSIVE lock can be acquired without
            # contention from readers holding WAL snapshots.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # Step 2: VACUUM — actual reclaim. EXCLUSIVE lock held for the
            # duration; see BLOCKING WARNING in docstring above.
            # VACUUM cannot run inside a transaction. Commit any open outer tx
            # first (now race-free: _conn_lock is held, so no sibling thread can
            # end the transaction between this probe and the COMMIT/VACUUM).
            if conn.in_transaction:
                logger.warning(
                    "hippo_compact_implicit_commit: VACUUM requires no open transaction; "
                    "committing outer transaction before VACUUM."
                )
                conn.execute("COMMIT")
            vac_t0 = time.monotonic()
            conn.execute("VACUUM")
            vacuum_elapsed = round(time.monotonic() - vac_t0, 3)
    except Exception as exc:
        db_compaction_error = f"{type(exc).__name__}: {exc}"[:500]
        logger.warning("hippo_compact_failed: %s", db_compaction_error)

    # Step 3: hnswlib rebuild + atomic save (drops mark_deleted slots).
    # _rebuild_index_from_sqlite internally calls _save_index_atomic.
    hnsw_error: str | None = None
    hnsw_elapsed: float = 0.0
    try:
        if db is not None:
            hnsw_t0 = time.monotonic()
            with db._hnsw_lock:
                db._rebuild_index_from_sqlite()
            hnsw_elapsed = round(time.monotonic() - hnsw_t0, 3)
    except Exception as exc:
        hnsw_error = f"{type(exc).__name__}: {exc}"[:500]
        logger.warning("hippo_hnsw_rebuild_failed: %s", hnsw_error)

    # Build per-table report with post-compaction measurements.
    for table_name in _HIPPO_TABLES_TO_COMPACT:
        rows_before, size_before = per_table_before.get(table_name, (0, 0))
        per_table: dict[str, Any] = {
            "rows_before": rows_before,
            "rows_after": rows_before,  # updated below if measurement succeeds
            "size_bytes_before": size_before,
            "size_bytes_after": 0,
            "vacuum_elapsed_sec": vacuum_elapsed,
            "hnswlib_rebuild_elapsed_sec": hnsw_elapsed,
            "elapsed_sec": round(time.monotonic() - overall_t0, 3),
        }
        try:
            if db is None:
                raise RuntimeError("store has no .db attribute")
            tbl = db.open_table(table_name)
            per_table["rows_after"] = int(tbl.count_rows())
            per_table["size_bytes_after"] = _measure_table_size_bytes(
                store, table_name,
            )
        except Exception as exc:
            per_table["error"] = (
                f"post_compact_measurement_failed: "
                f"{type(exc).__name__}: {exc}"
            )[:500]
        if db_compaction_error and "error" not in per_table:
            per_table["error"] = f"db_compact_failed: {db_compaction_error}"
        if hnsw_error and "error" not in per_table:
            per_table["error"] = f"hnsw_rebuild_failed: {hnsw_error}"
        report[table_name] = per_table

    return report


def optimize_lance_storage(store: Any, **kwargs: Any) -> "dict[str, dict[str, Any]]":
    """Deprecated alias for optimize_hippo_storage. Kept one release for compatibility."""
    import warnings
    warnings.warn(
        "optimize_lance_storage is deprecated; use optimize_hippo_storage",
        DeprecationWarning,
        stacklevel=2,
    )
    return optimize_hippo_storage(store, **kwargs)


def symmetrize_self_loops(
    store: Any, *, dry_run: bool = True,
) -> dict[str, Any]:
    """Backfill missing hebbian self-loops on existing records.

    Pre-existing stores have a per-record asymmetry: records that hit the
    dedup SKIP branch in ``MemoryStore.insert`` accumulated ``(rid, rid)``
    hebbian self-loops via ``reinforce_record(existing_id)``, while records
    that took the fresh-INSERT branch did not. The 30/40 bench ratio is
    the dedup-rate signature.

    A later write-path fix closes the source at insert time; this helper
    backfills existing stores so degree-norm becomes symmetric across all
    records. After running with ``dry_run=False``,
    ``count(records with self-loop) ∈ {0, N}`` — never partial.

    Args:
        store: MemoryStore-shaped (.db connection,.boost_edges).
            Duck-typed to match the ``optimize_hippo_storage`` precedent.
        dry_run: When True, reports counts without writing. Default True.

    Returns:
        {
          "records_total": <int>,
          "self_loops_present": <int>,
          "self_loops_pending": <int>,
          "self_loops_inserted": <int>, # 0 when dry_run=True
          "dry_run": <bool>,
        }

    No-op when every record already has a self-loop (apply path skips
    boost_edges; ``self_loops_pending == 0`` -> ``self_loops_inserted == 0``).
    """
    from uuid import UUID

    db = store.db
    records_tbl = db.open_table("records")
    edges_tbl = db.open_table("edges")

    records_df = records_tbl.to_pandas()
    edges_df = edges_tbl.to_pandas()

    records_ids = set(records_df["id"].astype(str))
    if len(edges_df) > 0:
        self_loops_df = edges_df[
            (edges_df["edge_type"] == "hebbian")
            & (edges_df["src"] == edges_df["dst"])
        ]
        self_loop_ids = set(self_loops_df["src"].astype(str))
    else:
        self_loop_ids = set()

    missing = sorted(records_ids - self_loop_ids)
    result: dict[str, Any] = {
        "records_total": len(records_ids),
        "self_loops_present": len(self_loop_ids & records_ids),
        "self_loops_pending": len(missing),
        "self_loops_inserted": 0,
        "dry_run": bool(dry_run),
    }

    if dry_run or not missing:
        return result

    pairs = [(UUID(rid), UUID(rid)) for rid in missing]
    # delta=0.1 mirrors the default + matches the initial link weight so
    # backfilled self-loops are weight-comparable to fresh-INSERT loops.
    store.boost_edges(pairs, delta=0.1, edge_type="hebbian")
    result["self_loops_inserted"] = len(missing)
    return result


def optimize_lance_storage(store: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Deprecated alias for optimize_hippo_storage.

    Routes to optimize_hippo_storage and emits a DeprecationWarning.
    """
    import warnings

    warnings.warn(
        "optimize_lance_storage is deprecated; use optimize_hippo_storage instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return optimize_hippo_storage(store, **kwargs)
