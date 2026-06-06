"""Tests for MCP-08 schema_list dispatch (Task 1).

schema_list returns induced schemas with confidence + evidence + status.
Supports domain + confidence_min filters.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore
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


def _make_record(
    *,
    text: str = "r",
    tags: list[str] | None = None,
    detail_level: int = 2,
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
        detail_level=detail_level,
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
        language=language,
    )


def test_schema_list_empty(tmp_path):
    store = MemoryStore(path=tmp_path)
    out = dispatch(store, "schema_list", {})
    assert out == {"schemas": [], "total": 0}


def test_schema_list_returns_persisted(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    evidence = [_make_record(tags=["python", "web"]) for _ in range(3)]
    for r in evidence:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:python+web",
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in evidence],
        status="auto",
    )
    persist_schema(store, cand)

    out = dispatch(store, "schema_list", {})
    assert out["total"] >= 1
    s0 = out["schemas"][0]
    assert "pattern" in s0
    assert "confidence" in s0
    assert "evidence_count" in s0
    assert "status" in s0


def test_schema_list_filter_confidence_min(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev_a = [_make_record(tags=["python"]) for _ in range(2)]
    for r in ev_a:
        store.insert(r)
    persist_schema(
        store,
        SchemaCandidate(
            pattern="low-confidence",
            confidence=0.7,
            evidence_count=2,
            evidence_ids=[r.id for r in ev_a],
            status="pending_user_approval",
        ),
    )
    ev_b = [_make_record(tags=["web"]) for _ in range(5)]
    for r in ev_b:
        store.insert(r)
    persist_schema(
        store,
        SchemaCandidate(
            pattern="high-confidence",
            confidence=0.95,
            evidence_count=5,
            evidence_ids=[r.id for r in ev_b],
            status="auto",
        ),
    )
    out = dispatch(store, "schema_list", {"confidence_min": 0.85})
    assert out["total"] == 1
    assert out["schemas"][0]["pattern"] == "high-confidence"


def test_schema_list_shape_has_exceptions_count(tmp_path):
    """Schema entries always carry an exceptions_count key (0 when no exceptions)."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev = [_make_record(tags=["x"]) for _ in range(3)]
    for r in ev:
        store.insert(r)
    persist_schema(
        store,
        SchemaCandidate(
            pattern="tags:x",
            confidence=0.9,
            evidence_count=3,
            evidence_ids=[r.id for r in ev],
            status="auto",
        ),
    )
    out = dispatch(store, "schema_list", {})
    assert out["total"] >= 1
    for s in out["schemas"]:
        assert "exceptions_count" in s
