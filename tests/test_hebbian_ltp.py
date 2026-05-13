"""Tests for 02-REVIEW.md H-03 (CLS heavy cycle missing Hebbian LTP).

Bug: run_heavy_consolidation creates `consolidated_from` edges for cluster
members (LTD-side write) but does NOT strengthen existing hebbian edges
between co-retrieved cluster members (LTP). The spec requires both
sides -- frequently-traversed edges strengthen; old rarely-traversed fade.
Pre-fix, the only LTP source was store.boost_edges inside pipeline_recall,
which fires on explicit user retrieval, never during offline consolidation.

Fix:
    - Add module constant HEAVY_LTP_DELTA = 0.05 in sleep.py.
    - In run_heavy_consolidation, after _create_semantic_summary runs for a
      cluster, call store.boost_edges(combinations(cluster_ids, 2),
      edge_type="hebbian", delta=HEAVY_LTP_DELTA) so existing hebbian edges
      between co-cluster members are potentiated.
    - Non-cluster edges remain untouched.

Constitutional contract ( biological fidelity + symmetry):
    Hebbian LTP/LTD symmetry is the core Hebbian-learning invariant. Without
    LTP during consolidation the graph drifts monotonically weaker. Matches
    Woz 2022 SRS reinforcement on co-retrieval.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------- helpers


def _record(
    *,
    text: str = "n",
    language: str = "en",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language=language,
    )


def _hebbian_weight(store, a: UUID, b: UUID) -> float | None:
    """Look up the current hebbian edge weight for (a, b), canonicalised."""
    from iai_mcp.store import EDGES_TABLE

    key = sorted([str(a), str(b)])
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    if df.empty:
        return None
    mask = (
        (df["src"] == key[0])
        & (df["dst"] == key[1])
        & (df["edge_type"] == "hebbian")
    )
    if not mask.any():
        return None
    return float(df.loc[mask, "weight"].iloc[0])


# ==================================================== H-03: named constant


def test_heavy_ltp_delta_is_named_constant():
    """The LTP increment must be a module-scope constant (HEAVY_LTP_DELTA=0.05)
    so maintainers can tune it without hunting for magic numbers, matching the
    DECAY_BASE / DECAY_EPSILON pattern already used for the LTD side."""
    from iai_mcp import sleep as sleep_mod

    assert hasattr(sleep_mod, "HEAVY_LTP_DELTA"), (
        "sleep.py must define HEAVY_LTP_DELTA at module scope"
    )
    assert sleep_mod.HEAVY_LTP_DELTA == pytest.approx(0.05, abs=1e-6), (
        f"HEAVY_LTP_DELTA must equal 0.05, got {sleep_mod.HEAVY_LTP_DELTA}"
    )


# ==================================================== H-03: LTP on cluster members


def test_heavy_cycle_strengthens_existing_hebbian_edges(tmp_path):
    """4-member cluster with pre-existing hebbian edges: after heavy
    consolidation every pairwise edge weight increases by >= HEAVY_LTP_DELTA.

    Pre-fix: weights stayed at 0.3 (decay-only behaviour).
    Post-fix: weights >= 0.35 (every pair potentiated once by LTP).
    """
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import HEAVY_LTP_DELTA, SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    # 4 records A B C D all cohesive
    recs = [_record(text=f"fact_{i}") for i in range(4)]
    for r in recs:
        store.insert(r)

    # Pre-seed pairwise hebbian edges at 0.3 each
    ids = [r.id for r in recs]
    pairs = [
        (ids[i], ids[j])
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ]
    for a, b in pairs:
        store.boost_edges([(a, b)], edge_type="hebbian", delta=0.3)

    # Sanity: all 6 pairs at 0.3
    for a, b in pairs:
        w = _hebbian_weight(store, a, b)
        assert w == pytest.approx(0.3, abs=1e-3), (
            f"pre-condition: {a}/{b} weight must be 0.3, got {w}"
        )

    # Run heavy consolidation, Tier-0 path
    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    run_heavy_consolidation(
        store,
        session_id="ltp-test",
        config=cfg,
        budget=budget,
        rate=rate,
        has_api_key=False,
    )

    # Every pairwise edge weight must have grown by at least HEAVY_LTP_DELTA
    for a, b in pairs:
        w = _hebbian_weight(store, a, b)
        assert w is not None, f"edge {a}/{b} must still exist"
        assert w >= 0.3 + HEAVY_LTP_DELTA - 1e-3, (
            f"hebbian edge {a}/{b} not potentiated: expected >= "
            f"{0.3 + HEAVY_LTP_DELTA}, got {w}"
        )


def test_heavy_cycle_does_not_touch_non_cluster_edges(tmp_path):
    """An edge between a cluster member and an unrelated record must NOT be
    boosted by the heavy cycle LTP path. Only co-cluster edges receive the
    potentiation."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    # Cluster A B C (all 3 hebbian-linked)
    cluster = [_record(text=f"c{i}") for i in range(3)]
    for r in cluster:
        store.insert(r)
    cluster_ids = [r.id for r in cluster]
    cluster_pairs = [
        (cluster_ids[0], cluster_ids[1]),
        (cluster_ids[1], cluster_ids[2]),
        (cluster_ids[0], cluster_ids[2]),
    ]
    for a, b in cluster_pairs:
        store.boost_edges([(a, b)], edge_type="hebbian", delta=0.3)

    # Extra record X with a hebbian edge to an UNRELATED record E
    rec_x = _record(text="x")
    rec_e = _record(text="e")
    store.insert(rec_x)
    store.insert(rec_e)
    # Only X<->E, not connected to the cluster
    store.boost_edges([(rec_x.id, rec_e.id)], edge_type="hebbian", delta=0.4)
    x_e_before = _hebbian_weight(store, rec_x.id, rec_e.id)
    assert x_e_before == pytest.approx(0.4, abs=1e-3)

    # Run heavy
    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    run_heavy_consolidation(
        store,
        session_id="ltp-isolate",
        config=cfg,
        budget=budget,
        rate=rate,
        has_api_key=False,
    )

    # X-E edge untouched because it is its own isolated 2-node component
    # (below CLUSTER_MIN_SIZE=3), so no LTP fires on it.
    x_e_after = _hebbian_weight(store, rec_x.id, rec_e.id)
    assert x_e_after == pytest.approx(0.4, abs=1e-3), (
        f"non-cluster edge must stay at 0.4, got {x_e_after}"
    )


def test_heavy_cycle_boost_edges_uses_hebbian_type(tmp_path):
    """Structural check: run_heavy_consolidation source MUST call
    boost_edges with edge_type='hebbian' (not consolidated_from). Prevents a
    regression where someone 'fixes' this by just reusing the consolidated_from
    write path."""
    import inspect
    from iai_mcp import sleep as sleep_mod

    src = inspect.getsource(sleep_mod.run_heavy_consolidation)
    assert "edge_type=\"hebbian\"" in src or "edge_type='hebbian'" in src, (
        "run_heavy_consolidation must boost hebbian edges (LTP), not only "
        "create consolidated_from edges"
    )
    assert "HEAVY_LTP_DELTA" in src, (
        "run_heavy_consolidation must use the named HEAVY_LTP_DELTA constant"
    )
