"""Tests for FSRS-style edge decay sweep inside sleep._decay_edges.

Behaviour:
- hebbian edges with last updated > 90d ago and weight < ε after decay are pruned.
- hebbian edges above ε are updated with the decayed weight.
- NON-hebbian edges (contradicts, invariant_anchor, consolidated_from, etc.)
  are NEVER pruned by the sweep. This is load-bearing for S5 identity protection
 : invariant anchors must survive decay.
- never_decay records are unaffected on the records side (Plan 02-01 __post_init__
  already enforces this on detail_level>=3; decay loop here targets edges only).
- DECAY_EPSILON defaults to 0.01.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest


def _insert_stale_edge(store, edge_type: str, weight: float, days_old: int):
    """Directly insert an aged edge for decay testing. Bypasses boost_edges
    which always stamps now() as updated_at."""
    import pandas as pd

    tbl = store.db.open_table("edges")
    old = datetime.now(timezone.utc) - timedelta(days=days_old)
    src_id, dst_id = str(uuid4()), str(uuid4())
    tbl.add([
        {
            "src": src_id,
            "dst": dst_id,
            "edge_type": edge_type,
            "weight": float(weight),
            "updated_at": old,
        }
    ])
    return src_id, dst_id


# ---- constants


def test_decay_epsilon_default():
    from iai_mcp import sleep as sleep_mod

    assert sleep_mod.DECAY_EPSILON == 0.01


# ---- sweep behaviour


def test_decay_edges_preserves_fresh_hebbian_edges(tmp_path):
    """Edges <= 90d old are untouched by the sweep."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.5, days_old=30)

    result = _decay_edges(store)
    assert result["decayed"] == 0
    assert result["pruned"] == 0

    # Edge still exists at original weight
    df = store.db.open_table("edges").to_pandas()
    row = df[(df["src"] == src) & (df["dst"] == dst)]
    assert not row.empty
    assert float(row.iloc[0]["weight"]) == 0.5


def test_decay_edges_decays_stale_hebbian_edges(tmp_path):
    """Edge >90d old and weight above ε is decayed, not pruned."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.8, days_old=100)

    result = _decay_edges(store)
    assert result["decayed"] >= 1

    df = store.db.open_table("edges").to_pandas()
    row = df[(df["src"] == src) & (df["dst"] == dst)]
    assert not row.empty
    assert float(row.iloc[0]["weight"]) < 0.8


def test_decay_edges_prunes_below_epsilon(tmp_path):
    """Edge decayed to weight < ε is removed."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # Very old + already tiny weight -> decays below 0.01
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.02, days_old=200)

    result = _decay_edges(store)
    assert result["pruned"] >= 1

    df = store.db.open_table("edges").to_pandas()
    gone = df[(df["src"] == src) & (df["dst"] == dst) & (df["edge_type"] == "hebbian")]
    assert gone.empty


def test_decay_edges_spares_contradicts(tmp_path):
    """Decay sweep only touches hebbian edges; contradicts edges survive forever."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "contradicts", weight=0.5, days_old=1000)

    _decay_edges(store)

    df = store.db.open_table("edges").to_pandas()
    row = df[
        (df["src"] == src)
        & (df["dst"] == dst)
        & (df["edge_type"] == "contradicts")
    ]
    assert not row.empty
    assert float(row.iloc[0]["weight"]) == 0.5


def test_decay_edges_spares_invariant_anchor(tmp_path):
    """S5 invariant_anchor edges MUST NOT be pruned."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "invariant_anchor", weight=0.001, days_old=5000)

    _decay_edges(store)
    df = store.db.open_table("edges").to_pandas()
    row = df[
        (df["src"] == src)
        & (df["dst"] == dst)
        & (df["edge_type"] == "invariant_anchor")
    ]
    assert not row.empty  # survived



def test_decay_edges_spares_consolidated_from(tmp_path):
    """consolidated_from (semantic<-episode) edges must survive decay."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "consolidated_from", weight=0.01, days_old=2000)

    _decay_edges(store)
    df = store.db.open_table("edges").to_pandas()
    row = df[
        (df["src"] == src)
        & (df["dst"] == dst)
        & (df["edge_type"] == "consolidated_from")
    ]
    assert not row.empty


def test_decay_edges_custom_epsilon(tmp_path):
    """Epsilon can be overridden per-call."""
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    src, dst = _insert_stale_edge(store, "hebbian", weight=0.05, days_old=95)

    # Default ε=0.01 -> likely not pruned after only 5 days of decay beyond 90
    result_default = _decay_edges(store, epsilon=0.01)
    # High ε=0.5 -> should prune anything below 0.5
    # Re-insert since we may have been decayed
    df = store.db.open_table("edges").to_pandas()
    remaining = df[(df["src"] == src) & (df["dst"] == dst) & (df["edge_type"] == "hebbian")]
    # Reset for clean experiment
    if not remaining.empty:
        store.db.open_table("edges").delete(
            f"src = '{src}' AND dst = '{dst}' AND edge_type = 'hebbian'"
        )

    src2, dst2 = _insert_stale_edge(store, "hebbian", weight=0.3, days_old=95)
    result_custom = _decay_edges(store, epsilon=0.5)
    df2 = store.db.open_table("edges").to_pandas()
    row = df2[(df2["src"] == src2) & (df2["dst"] == dst2) & (df2["edge_type"] == "hebbian")]
    # With epsilon=0.5 and starting weight 0.3, prune should happen immediately.
    assert row.empty
    assert result_custom["pruned"] >= 1
