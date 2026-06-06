"""Tests for 02- (SQL predicate injection in sleep._decay_edges)
and (migrate.py delete predicate). Both findings share one root cause:
raw UUIDs interpolated into store WHERE/DELETE f-strings without
_uuid_literal validation. bundles into the fix.

Defence-in-depth contract:
    EVERY raw-UUID-WHERE/DELETE site MUST pass through _uuid_literal before
    f-string interpolation. Poisoned inputs raise ValueError; callers wrap
    per-row bodies in try/except ValueError: continue so the whole sweep
    does not crash on one corrupt row.

RED assertions encoded here:
    - test_decay_edges_rejects_malformed_uuid: a poisoned edge row is skipped,
      not executed as a SQL wildcard. Surrounding clean rows still decay.
    - test_decay_edges_uses_uuid_literal_helper: module-scope import check.
    - test_migrate_delete_uses_uuid_literal: module-scope import check.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM


# ---------------------------------------------------------------- helpers

def _insert_raw_edge(
    store, src: str, dst: str, edge_type: str, weight: float, days_old: int,
) -> None:
    """Insert an aged edge directly, bypassing boost_edges (which always stamps
    now() as updated_at). Accepts arbitrary src/dst strings so callers can
    inject poisoned UUIDs for RED assertions."""
    tbl = store.db.open_table("edges")
    old = datetime.now(timezone.utc) - timedelta(days=days_old)
    tbl.add([
        {
            "src": src,
            "dst": dst,
            "edge_type": edge_type,
            "weight": float(weight),
            "updated_at": old,
        }
    ])


# ====================================================: _decay_edges hardening


def test_decay_edges_rejects_malformed_uuid(tmp_path):
    """a poisoned src value must NOT reach the store SQL dialect.

    Seed 3 stale hebbian edges:
      row 0: clean (canonical UUIDs)
      row 1: poisoned (src = "xxxx' OR '1'='1" -- classic predicate injection)
      row 2: clean (canonical UUIDs)

    After _decay_edges:
      - Clean rows 0 and 2 are decayed/pruned correctly.
      - Poisoned row 1 is skipped (per-row try/except ValueError).
      - Total edge count reflects only the clean-row operations.
    """
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    # Row 0: clean, should decay
    clean_src_a = str(uuid4())
    clean_dst_a = str(uuid4())
    _insert_raw_edge(store, clean_src_a, clean_dst_a, "hebbian", weight=0.8, days_old=100)

    # Row 1: poisoned -- classic predicate injection payload
    poisoned_src = "00000000-0000-0000-0000-000000000000' OR '1'='1"
    poisoned_dst = str(uuid4())
    _insert_raw_edge(store, poisoned_src, poisoned_dst, "hebbian", weight=0.5, days_old=100)

    # Row 2: clean, should decay
    clean_src_b = str(uuid4())
    clean_dst_b = str(uuid4())
    _insert_raw_edge(store, clean_src_b, clean_dst_b, "hebbian", weight=0.8, days_old=100)

    # Pre-condition: all 3 rows present
    df_before = store.db.open_table("edges").to_pandas()
    assert len(df_before) == 3

    # Should NOT raise even with a poisoned row
    result = _decay_edges(store)

    # Clean rows were decayed
    assert result["decayed"] >= 2, (
        f"expected >= 2 clean rows decayed, got {result}"
    )

    # Poisoned row was skipped -- still in the table at original weight
    df_after = store.db.open_table("edges").to_pandas()
    poisoned_row = df_after[df_after["src"] == poisoned_src]
    assert len(poisoned_row) == 1, "poisoned row must not be deleted"
    assert float(poisoned_row.iloc[0]["weight"]) == 0.5, (
        "poisoned row weight must not be decayed -- _uuid_literal rejected it"
    )

    # Assert that the injection payload was not executed as a wildcard: no
    # row matching clean_src_a with updated_at == old survived unchanged (i.e.
    # the decay actually ran on row 0), confirming rows were processed
    # individually.
    row_a = df_after[df_after["src"] == clean_src_a]
    assert len(row_a) == 1
    assert float(row_a.iloc[0]["weight"]) < 0.8, (
        "clean row 0 should have been decayed"
    )


def test_decay_edges_imports_uuid_literal_at_module_scope():
    """structural check (defence-in-depth): _uuid_literal must be
    imported into sleep.py at module scope, not re-inlined."""
    from iai_mcp import sleep as sleep_mod

    # The helper must be reachable via the sleep module's namespace
    assert hasattr(sleep_mod, "_uuid_literal"), (
        "sleep.py must `from iai_mcp.store import _uuid_literal` at module scope"
    )


def test_decay_edges_single_poisoned_row_does_not_kill_sweep(tmp_path):
    """per-row try/except ValueError must wrap the body, not the whole
    function. One poisoned row skipped != entire pass aborted.
    """
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    # Poisoned row with weight that would definitely prune if it decayed
    poisoned_src = "not-a-uuid-at-all"
    poisoned_dst = str(uuid4())
    _insert_raw_edge(store, poisoned_src, poisoned_dst, "hebbian", weight=0.02, days_old=500)

    # Legit row with high weight
    clean_src = str(uuid4())
    clean_dst = str(uuid4())
    _insert_raw_edge(store, clean_src, clean_dst, "hebbian", weight=0.8, days_old=100)

    # Must not raise
    result = _decay_edges(store)

    # Clean row processed (decayed or pruned)
    assert (result["decayed"] + result["pruned"]) >= 1, (
        "sweep must continue past poisoned row"
    )

    # Poisoned row still present
    df = store.db.open_table("edges").to_pandas()
    assert len(df[df["src"] == poisoned_src]) == 1


# ====================================================: migrate delete predicate


def test_migrate_imports_uuid_literal_at_module_scope():
    """structural check: migrate.py must import _uuid_literal so its
    tbl.delete() call cannot carry SQL injection content even if record.id
    shape drifts."""
    from iai_mcp import migrate as migrate_mod

    assert hasattr(migrate_mod, "_uuid_literal"), (
        "migrate.py must `from iai_mcp.store import _uuid_literal`"
    )


def test_migrate_delete_predicate_uses_uuid_literal(tmp_path, monkeypatch):
    """the migration path's tbl.delete(f\"id = '{record.id}'\") must be
    replaced with a _uuid_literal-wrapped form. We assert the source text
    contains the safe pattern and does NOT contain the raw interpolation.
    """
    import inspect
    from iai_mcp import migrate as migrate_mod

    src = inspect.getsource(migrate_mod)

    # Safe pattern must be present
    assert "_uuid_literal(record.id)" in src, (
        "migrate.py tbl.delete call must wrap record.id via _uuid_literal"
    )
    # Unsafe pattern must be gone
    assert "id = '{record.id}'" not in src, (
        "migrate.py still contains raw f-string interpolation of record.id"
    )
