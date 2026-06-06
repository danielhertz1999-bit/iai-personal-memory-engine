"""Tests for temporal_next edges (Task 3).

temporal_next edges:
- Created on record insert when a previous insert event exists in the same
  session within the last 5 minutes.
- Not created across different sessions.
- Fade past 30d (soft decay applied in sleep.py's decay sweep).
- Build a navigable chain (A->B->C) traversable in the graph.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


def _rec(text: str, tags=None) -> MemoryRecord:
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
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language="en",
    )


# ---------------------------------------------------------------- creation


def test_temporal_next_created_on_insert(tmp_path):
    """Two records inserted in same session within 5min -> temporal_next edge."""
    from iai_mcp.retrieve import link_temporal_next

    store = MemoryStore(path=tmp_path)
    a = _rec("a")
    store.insert(a)
    link_temporal_next(store, a, session_id="s1")

    b = _rec("b")
    store.insert(b)
    link_temporal_next(store, b, session_id="s1")

    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    tn = edges[edges["edge_type"] == "temporal_next"]
    assert len(tn) >= 1
    # One of the edges should involve both a and b
    ids = {str(a.id), str(b.id)}
    matches = tn[(tn["src"].isin(ids)) & (tn["dst"].isin(ids))]
    assert len(matches) >= 1


def test_temporal_next_not_created_across_sessions(tmp_path):
    """Record A in session 1, B in session 2 -> no temporal_next."""
    from iai_mcp.retrieve import link_temporal_next

    store = MemoryStore(path=tmp_path)
    a = _rec("a")
    store.insert(a)
    link_temporal_next(store, a, session_id="s1")

    b = _rec("b")
    store.insert(b)
    link_temporal_next(store, b, session_id="s2")

    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    tn = edges[edges["edge_type"] == "temporal_next"]
    # No cross-session edges
    for _, row in tn.iterrows():
        assert not (row["src"] == str(a.id) and row["dst"] == str(b.id))
        assert not (row["src"] == str(b.id) and row["dst"] == str(a.id))


def test_temporal_next_navigable_chain(tmp_path):
    """A->B->C->D creates 3 temporal_next edges traversable via graph."""
    from iai_mcp.retrieve import link_temporal_next

    store = MemoryStore(path=tmp_path)
    records = [_rec(f"r{i}") for i in range(4)]
    for r in records:
        store.insert(r)
        link_temporal_next(store, r, session_id="s-chain")

    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    tn = edges[edges["edge_type"] == "temporal_next"]
    # 3 sequential edges (r0->r1, r1->r2, r2->r3) expected
    assert len(tn) >= 3


def test_temporal_next_event_logged(tmp_path):
    """Each insert emits a record_inserted event that drives temporal_next."""
    from iai_mcp.events import query_events
    from iai_mcp.retrieve import link_temporal_next

    store = MemoryStore(path=tmp_path)
    a = _rec("first")
    store.insert(a)
    link_temporal_next(store, a, session_id="s-ev")
    events = query_events(store, kind="record_inserted")
    assert len(events) >= 1
