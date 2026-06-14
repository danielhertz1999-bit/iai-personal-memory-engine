from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)

HIPPO_COMPACT_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_HIPPO_COMPACT_INTERVAL_SEC", "3600.0"),
)

_HIPPO_TABLES_TO_COMPACT: tuple[str, ...] = ("records", "edges", "events")


def _measure_table_size_bytes(store: Any, table_name: str) -> int:
    try:
        db = getattr(store, "db", None)
        if db is None:
            return 0
        conn = getattr(db, "_conn", None)
        if conn is None:
            return 0
        try:
            row = conn.execute(
                "SELECT SUM(pgsize) FROM dbstat WHERE name = ?", (table_name,)
            ).fetchone()
            if row is not None and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
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
    retention: timedelta | None = None,
    max_versions: int | None = None,
    delete_unverified: bool = False,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    db = getattr(store, "db", None)

    overall_t0 = time.monotonic()

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

    db_compaction_error: str | None = None
    vacuum_elapsed: float = 0.0
    try:
        conn = db._conn
        conn_lock = getattr(db, "_conn_lock", None)
        _lock_ctx = conn_lock if conn_lock is not None else nullcontext()
        with _lock_ctx:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
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

    for table_name in _HIPPO_TABLES_TO_COMPACT:
        rows_before, size_before = per_table_before.get(table_name, (0, 0))
        per_table: dict[str, Any] = {
            "rows_before": rows_before,
            "rows_after": rows_before,
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
    store.boost_edges(pairs, delta=0.1, edge_type="hebbian")
    result["self_loops_inserted"] = len(missing)
    return result


def optimize_lance_storage(store: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    import warnings

    warnings.warn(
        "optimize_lance_storage is deprecated; use optimize_hippo_storage instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return optimize_hippo_storage(store, **kwargs)
