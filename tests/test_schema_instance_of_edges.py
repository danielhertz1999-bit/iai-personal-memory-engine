"""Tests for schema_instance_of edge semantics (Task 3).

schema_instance_of edges:
- Point from an evidence episode record to a schema hub record.
- Never decay (edge-type exempt from FSRS sweep).
- Make the schema record a first-class hub: pipeline retrieval should
  surface schema records when evidence is activated.
"""
from __future__ import annotations

from datetime import datetime, timezone
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


def _rec(*, text: str = "t", tags: list[str] | None = None) -> MemoryRecord:
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


# ---------------------------------------------------------------- edge creation


def test_schema_instance_of_edge_created_on_persist(tmp_path):
    """persist_schema creates schema_instance_of edges."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev = [_rec(text=f"x{i}", tags=["m", "n"]) for i in range(5)]
    for r in ev:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:m+n",
        confidence=0.9,
        evidence_count=5,
        evidence_ids=[r.id for r in ev],
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    sio = edges[edges["edge_type"] == "schema_instance_of"]
    assert len(sio) == 5


def test_schema_instance_of_edge_never_decays(tmp_path):
    """schema_instance_of edges survive FSRS decay sweep."""
    from iai_mcp.schema import SchemaCandidate, persist_schema
    from iai_mcp.sleep import _decay_edges

    store = MemoryStore(path=tmp_path)
    ev = [_rec(text=f"x{i}", tags=["a", "b"]) for i in range(3)]
    for r in ev:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:a+b", confidence=0.9, evidence_count=3,
        evidence_ids=[r.id for r in ev], status="auto",
    )
    persist_schema(store, cand)

    # Backdate the schema_instance_of edges to 500d ago
    import lancedb
    edges_tbl = store.db.open_table(EDGES_TABLE)
    # Update all schema_instance_of edges to have an ancient updated_at
    from datetime import timedelta
    ancient = datetime.now(timezone.utc) - timedelta(days=500)
    edges_tbl.update(
        where="edge_type = 'schema_instance_of'",
        values={"updated_at": ancient, "weight": 0.0001},
    )
    # Run the decay sweep
    _decay_edges(store)

    # schema_instance_of edges must still exist
    df = edges_tbl.to_pandas()
    sio = df[df["edge_type"] == "schema_instance_of"]
    assert len(sio) == 3


def test_schema_record_becomes_hub(tmp_path):
    """After persist, the schema record has detail_level=3 (never_decay) and
    many schema_instance_of edges (hub property)."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev = [_rec(text=f"x{i}", tags=["p", "q"]) for i in range(5)]
    for r in ev:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:p+q", confidence=0.9, evidence_count=5,
        evidence_ids=[r.id for r in ev], status="auto",
    )
    schema_id = persist_schema(store, cand)

    rec = store.get(schema_id)
    assert rec is not None
    assert rec.detail_level == 3
    assert rec.never_decay is True
    # Hub: 5 incoming schema_instance_of edges (one per evidence)
    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    sio = edges[edges["edge_type"] == "schema_instance_of"]
    assert len(sio) == 5
