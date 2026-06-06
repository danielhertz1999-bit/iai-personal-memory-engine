"""TDD regression for flush_edge_buffer duplicate-edge handling.

Scenario (a): Insert N distinct edges → all land (regression control).
Scenario (b): Insert batch where M of N edges already exist → exactly N-M new
              rows added, existing M rows have weight/updated_at updated, no
              exception raised. PRE-FIX: `.add()` raises IntegrityError and
              the whole batch is dropped. POST-FIX: merge_insert upsert
              handles conflict; new rows land, existing rows updated in place.
Scenario (c): Insert empty pending list → no-op, returns 0.

The critical discriminator for scenario (b) is: no exception raised, and
row count increases by exactly (N - M). The existing rows also get their
weight/updated_at overwritten by the buffered values (upsert latest wins).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# helpers shared across scenarios
# ---------------------------------------------------------------------------


def _clear_edge_buffer(store) -> None:
    from iai_mcp import store as store_mod

    store_mod._edge_buffer.pop(id(store), None)
    store_mod._edge_last_flush_at.pop(id(store), None)


def _make_edge_row(
    src: str | None = None,
    dst: str | None = None,
    edge_type: str = "hebbian",
    weight: float = 0.5,
) -> dict:
    return {
        "src": src or str(uuid4()),
        "dst": dst or str(uuid4()),
        "edge_type": edge_type,
        "weight": weight,
        "updated_at": datetime.now(timezone.utc),
    }


def _table_row_count(store) -> int:
    from iai_mcp.store import EDGES_TABLE

    return len(store.db.open_table(EDGES_TABLE).to_pandas())


# ---------------------------------------------------------------------------
# Scenario (a): all distinct — regression control
# ---------------------------------------------------------------------------


def test_flush_edge_buffer_all_distinct_land(tmp_path, monkeypatch):
    """Scenario (a): N distinct edges buffered and flushed → all N land in DB."""
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        n_before = _table_row_count(store)

        rows = [_make_edge_row() for _ in range(5)]
        for row in rows:
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        flushed = flush_edge_buffer(store)
        assert flushed == 5, f"expected 5 rows flushed; got {flushed}"

        n_after = _table_row_count(store)
        assert n_after == n_before + 5, (
            f"expected {n_before + 5} rows after flush of 5 distinct edges; got {n_after}"
        )


# ---------------------------------------------------------------------------
# Scenario (b): partial duplicate batch — the bug scenario
#
# PRE-FIX:.add(pending) raises sqlite3.IntegrityError on duplicate (src,dst,edge_type).
# The except clause only catches (OSError, RuntimeError, ValueError) so
# IntegrityError propagates. flush_edge_buffer raises and returns no count.
#
# POST-FIX: merge_insert(["src","dst","edge_type"]).execute(pending) with
# non_key=[weight,updated_at] hits the "ON CONFLICT DO UPDATE SET" branch
# in HippoMergeInsert.execute(). Conflicts are handled: existing rows
# get weight/updated_at updated, new rows inserted. No exception.
# ---------------------------------------------------------------------------


def test_flush_edge_buffer_partial_duplicate_no_exception(tmp_path, monkeypatch):
    """Scenario (b): batch with M pre-existing + N-M new edges flushes without exception.

    RED on pre-fix code (IntegrityError propagates).
    GREEN on post-fix code (merge_insert handles conflict).
    """
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        # Seed 3 edges directly into DB (bypass buffer so they're committed).
        pre_existing = [_make_edge_row(weight=0.1) for _ in range(3)]
        store.db.open_table(EDGES_TABLE).add(pre_existing)

        n_after_seed = _table_row_count(store)
        assert n_after_seed == 3, f"seed failed: expected 3 rows, got {n_after_seed}"

        # Buffer: same 3 edges (duplicates) + 2 new distinct edges.
        new_rows = [_make_edge_row(weight=0.9) for _ in range(2)]
        # Re-use the same src/dst/edge_type as pre-existing rows, but update weight.
        dup_rows = [
            {**row, "weight": 0.8, "updated_at": datetime.now(timezone.utc)}
            for row in pre_existing
        ]
        pending = dup_rows + new_rows  # 5 total: 3 dup + 2 new

        for row in pending:
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        # This must NOT raise (IntegrityError on pre-fix, silent on post-fix).
        flushed = flush_edge_buffer(store)

        # Post-fix: flushed == 5 (all rows processed via merge_insert).
        assert flushed == 5, f"expected 5 rows processed; got {flushed}"

        # Row count: 3 pre-existing + 2 new = 5 total.
        n_final = _table_row_count(store)
        assert n_final == 5, (
            f"expected 5 total rows after upsert (3 existing + 2 new); got {n_final}"
        )

        # Existing rows were updated: weight should now be 0.8 (from dup_rows).
        tbl = store.db.open_table(EDGES_TABLE)
        df = tbl.to_pandas()

        # Find the pre-existing rows by src value.
        pre_srcs = {row["src"] for row in pre_existing}
        updated_rows = df[df["src"].isin(pre_srcs)]
        assert len(updated_rows) == 3, (
            f"expected 3 updated rows for pre-existing srcs; got {len(updated_rows)}"
        )
        for _, row in updated_rows.iterrows():
            assert abs(row["weight"] - 0.8) < 1e-9, (
                f"expected weight=0.8 (upsert latest wins) on existing edge; got {row['weight']}"
            )


# ---------------------------------------------------------------------------
# Scenario (c): empty buffer — no-op
# ---------------------------------------------------------------------------


def test_flush_edge_buffer_empty_is_noop(tmp_path, monkeypatch):
    """Scenario (c): flush on empty buffer returns 0 and does not raise."""
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        n_before = _table_row_count(store)

        flushed = flush_edge_buffer(store)
        assert flushed == 0, f"expected 0 from empty flush; got {flushed}"

        n_after = _table_row_count(store)
        assert n_after == n_before, "empty flush changed row count"
