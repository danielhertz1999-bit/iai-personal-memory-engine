"""periodic Lance storage maintenance.

Forensic trigger (2026-04-27): the daemon was running 248% CPU sustained for
1h14min because `records.lance` had grown to 10,841 versions / 3.66 GB for
only 7,130 rows over 9 days. There has never been a `table.optimize()` call
site in production code. Offline `optimize(cleanup_older_than=timedelta(days=1))`
reclaimed 84% disk and dropped `build_runtime_graph` cold latency 13.3s ->
0.13s (102x). codifies that fix as a daemon-managed periodic job
so version manifests + soft-deleted rows do not re-accumulate.

Architecture:
- D7.3-01: periodic + startup, NOT write-triggered (post-write hook would
  amplify write latency unboundedly).
- D7.3-02: single-process inside the daemon (no worker process).
- D7.3-03: helper is SYNC; callers wrap in `asyncio.to_thread`. Phase 7.2's
  AST fence (tests/test_no_bare_sync_in_async.py) enforces this discipline
  via `BLOCKING_NAMES` (D7.3-26).
- D7.3-09: helper NEVER raises. Per-table failures captured in the per-table
  dict's `error` field. The daemon must not die from an optimize failure.
- D7.3-13/D7.3-21: 1-day default retention matches Lance docs FAQ.

Two env overrides (read once at import per D7.3-22):
- IAI_MCP_LANCE_OPTIMIZE_INTERVAL_SEC (default 3600s = 1h cadence)
- IAI_MCP_LANCE_OPTIMIZE_RETENTION_SEC (default 86400s = 1 day)
"""
from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

# D7.3-20: 1-hour periodic cadence (12x the cascade-poll cadence; same order
# of magnitude as the maintenance work itself; far longer than typical session
# length so optimize rarely interferes; short enough that bloat stays bounded).
LANCE_OPTIMIZE_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_LANCE_OPTIMIZE_INTERVAL_SEC", "3600.0"),
)

# D7.3-21: 1-day retention matches Lance's documented `cleanup_older_than`
# example. Aggressive enough to free disk fast; conservative enough for
# point-in-time time-travel reads within the same day.
LANCE_OPTIMIZE_RETENTION_SEC: float = float(
    os.environ.get("IAI_MCP_LANCE_OPTIMIZE_RETENTION_SEC", "86400.0"),
)

# Daemon-owned tables; matches src/iai_mcp/store.py constants
# (RECORDS_TABLE/EDGES_TABLE/EVENTS_TABLE) but kept literal so this module
# does not pull MemoryStore at import time.
_TABLES_TO_OPTIMIZE: tuple[str, ...] = ("records", "edges", "events")


def _measure_table_size_bytes(store: Any, table_name: str) -> int:
    """Sum the size of every file under <storage_root>/lancedb/<table>.lance/.

    Returns 0 on any measurement failure so size metrics are best-effort:
    a measurement failure must NOT cause the helper itself to raise. The
    actual `tbl.optimize()` call is independent — disk-size telemetry is
    purely observational and exists for the operator-facing event payload.
    """
    try:
        # MemoryStore.root is the user-supplied (or env-derived) storage
        # root; the LanceDB connection lives at root/lancedb (see store.py
        # line 202). Each table is a `<name>.lance` directory underneath.
        root = getattr(store, "root", None)
        if root is None:
            return 0
        table_dir = Path(root) / "lancedb" / f"{table_name}.lance"
        if not table_dir.exists():
            return 0
        total = 0
        for p in table_dir.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                # File could be unlinked mid-scan during an active optimize;
                # skip it, keep counting the rest.
                continue
        return total
    except Exception:
        return 0


def optimize_lance_storage(
    store: Any,
    *,
    retention: timedelta | None = None,
) -> dict[str, dict[str, Any]]:
    """Run `tbl.optimize(cleanup_older_than=retention)` on each daemon-owned
    LanceDB table (records, edges, events).

    Args:
        store: MemoryStore-shaped object exposing `.db` (lancedb.Connection).
            Duck-typed so test fixtures can pass a stub. The function only
            reads `store.db` and `store.root` (latter optional for size
            telemetry).
        retention: timedelta passed to LanceDB's `cleanup_older_than`. If
            None, defaults to `timedelta(seconds=LANCE_OPTIMIZE_RETENTION_SEC)`
            which is 1 day in production.

    Returns:
        Flat dict keyed by table name (`records`, `edges`, `events`). Each
        value is a per-table dict::

            {
                "rows_before": int,        # tbl.count_rows() pre-optimize
                "rows_after": int,         # tbl.count_rows() post-optimize
                "versions_before": int,    # len(tbl.list_versions()) pre
                "versions_after": int,     # len(tbl.list_versions()) post
                "size_bytes_before": int,  # du -sb on .lance/ pre, 0 on err
                "size_bytes_after": int,   # du -sb on .lance/ post, 0 on err
                "elapsed_sec": float,      # wall-clock for optimize()
                "error": str,              # ONLY present on failure
            }

    Per D7.3-09: this helper NEVER raises. Per-table failure captured in
    the table's `error` field; the other tables are still processed.
    """
    if retention is None:
        retention = timedelta(seconds=LANCE_OPTIMIZE_RETENTION_SEC)

    report: dict[str, dict[str, Any]] = {}
    db = getattr(store, "db", None)

    for table_name in _TABLES_TO_OPTIMIZE:
        per_table: dict[str, Any] = {
            "rows_before": 0,
            "rows_after": 0,
            "versions_before": 0,
            "versions_after": 0,
            "size_bytes_before": 0,
            "size_bytes_after": 0,
            "elapsed_sec": 0.0,
        }
        try:
            if db is None:
                raise RuntimeError("store has no .db attribute")
            tbl = db.open_table(table_name)
            try:
                per_table["rows_before"] = int(tbl.count_rows())
            except Exception:
                per_table["rows_before"] = 0
            try:
                per_table["versions_before"] = len(tbl.list_versions())
            except Exception:
                per_table["versions_before"] = 0
            per_table["size_bytes_before"] = _measure_table_size_bytes(
                store, table_name,
            )

            t0 = time.monotonic()
            tbl.optimize(cleanup_older_than=retention)
            per_table["elapsed_sec"] = round(time.monotonic() - t0, 3)

            # Re-open the table after optimize: some LanceDB versions return
            # cached metadata on the original handle until refresh.
            try:
                tbl_after = db.open_table(table_name)
            except Exception:
                tbl_after = tbl
            try:
                per_table["rows_after"] = int(tbl_after.count_rows())
            except Exception:
                per_table["rows_after"] = per_table["rows_before"]
            try:
                per_table["versions_after"] = len(tbl_after.list_versions())
            except Exception:
                per_table["versions_after"] = per_table["versions_before"]
            per_table["size_bytes_after"] = _measure_table_size_bytes(
                store, table_name,
            )
        except Exception as exc:  # noqa: BLE001 -- helper MUST NOT raise (D7.3-09)
            per_table["error"] = str(exc)[:500]

        report[table_name] = per_table

    return report
