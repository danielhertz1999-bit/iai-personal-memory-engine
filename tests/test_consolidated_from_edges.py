"""Tests for the consolidated_from edge type.

After run_heavy_consolidation:
- `consolidated_from` edges link the semantic summary record to each source
  episodic record in its cluster.
- src = summary record (tier=semantic); dst = source episode.
- Source episodes keep their literal_surface verbatim (preservation).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(text: str, tier: str = "episodic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _run_heavy(store):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    return run_heavy_consolidation(
        store,
        session_id="s-cfr",
        config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store),
        rate=RateLimitLedger(store),
        has_api_key=False,
    )


def test_consolidated_from_edge_created_on_heavy_run(tmp_path):
    """Cohesive cluster of 3 -> at least one consolidated_from edge."""
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    recs = [_record(f"rec {i}") for i in range(3)]
    for r in recs:
        store.insert(r)
    # Triangle: all three connected
    store.boost_edges(
        [(recs[0].id, recs[1].id), (recs[1].id, recs[2].id), (recs[0].id, recs[2].id)],
        edge_type="hebbian", delta=0.5,
    )

    _run_heavy(store)

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    cf = df[df["edge_type"] == "consolidated_from"]
    assert len(cf) >= 3


def test_consolidated_from_edge_points_semantic_to_episodes(tmp_path):
    """src of consolidated_from is the summary record (tier=semantic);
    dst is a source episode (tier=episodic)."""
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    recs = [_record(f"rec {i}") for i in range(3)]
    for r in recs:
        store.insert(r)
    store.boost_edges(
        [(recs[0].id, recs[1].id), (recs[1].id, recs[2].id), (recs[0].id, recs[2].id)],
        edge_type="hebbian", delta=0.5,
    )

    _run_heavy(store)

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    cf = df[df["edge_type"] == "consolidated_from"]
    assert not cf.empty

    source_ids = {str(r.id) for r in recs}
    for _, row in cf.iterrows():
        # Either src or dst is a summary (not in our original source_ids);
        # the other should be one of our source episodes.
        if row["src"] not in source_ids and row["dst"] in source_ids:
            # Fetch the summary record
            summary = store.get(UUID(row["src"]))
            assert summary is not None
            assert summary.tier == "semantic"
            dst_rec = store.get(UUID(row["dst"]))
            assert dst_rec is not None
            assert dst_rec.tier == "episodic"
        elif row["dst"] not in source_ids and row["src"] in source_ids:
            # boost_edges canonicalises (src, dst) as sorted -- either direction
            summary = store.get(UUID(row["dst"]))
            assert summary is not None
            assert summary.tier == "semantic"
        else:
            # Edge between two source records -- that's wrong for consolidated_from.
            pytest.fail(
                f"consolidated_from edge without a summary endpoint: "
                f"{row['src']} -> {row['dst']}"
            )


def test_consolidated_from_edges_preserve_literal_in_episodes(tmp_path):
    """source episodes' literal_surface unchanged after consolidation."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literals = ["alpha", "beta", "gamma"]
    recs = [_record(t) for t in literals]
    for r in recs:
        store.insert(r)
    store.boost_edges(
        [(recs[0].id, recs[1].id), (recs[1].id, recs[2].id), (recs[0].id, recs[2].id)],
        edge_type="hebbian", delta=0.5,
    )

    _run_heavy(store)

    for rec, expected in zip(recs, literals):
        reloaded = store.get(rec.id)
        assert reloaded is not None
        assert reloaded.literal_surface == expected
