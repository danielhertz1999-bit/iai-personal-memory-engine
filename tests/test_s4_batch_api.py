"""Tests for s4.on_read_check_batch (D-SPEED gap closure).

D-SPEED contract: bench/neural_map p95<100ms at N=100. Root cause:
`s4.on_read_check` called per-hit inside pipeline_recall with no records_cache,
forcing N+1 store.get() round-trips. Fix: new `on_read_check_batch` that accepts
an optional records_cache from the caller and does ONE store.all_records() (or
zero if cache provided).

Equivalence contract: on_read_check_batch returns semantically identical hint
output to on_read_check for the same (store, hits, session_id) input. The
source_id contents of the returned hints must be a set-equal match; orderings
may differ because event-write side effects are intermingled.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord


def _make_record(
    *,
    text: str = "hello",
    vec: list[float] | None = None,
    tags: list[str] | None = None,
    detail_level: int = 2,
    tier: str = "episodic",
    language: str = "en",
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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


def _hit_for(rec: MemoryRecord, score: float = 0.9) -> MemoryHit:
    return MemoryHit(
        record_id=rec.id,
        score=score,
        reason="test",
        literal_surface=rec.literal_surface,
        adjacent_suggestions=[],
    )


# ------------------------------------------------------------- contract


def test_s4_exports_on_read_check_batch():
    """The batch variant exists and is callable."""
    from iai_mcp import s4

    assert hasattr(s4, "on_read_check_batch")
    assert callable(s4.on_read_check_batch)


# ------------------------------------------------------------- behaviour


def test_on_read_check_batch_uses_records_cache(tmp_path):
    """When records_cache is passed, store.get is NOT called (zero round-trips).

    This is the core D-SPEED fix: the caller (pipeline_recall) builds
    records_cache at stage 1, so S4 must not re-fetch via store.get.
    Monkeypatch store.get to raise; the call MUST succeed without exception.
    """
    from iai_mcp.s4 import on_read_check_batch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v1 = [0.0] * EMBED_DIM; v1[0] = 1.0
    v2 = [0.0] * EMBED_DIM; v2[1] = 1.0
    r1 = _make_record(text="X is true", vec=v1, tags=["claim"])
    r2 = _make_record(text="X is false", vec=v2, tags=["claim"])
    store.insert(r1)
    store.insert(r2)
    store.add_contradicts_edge(r1.id, r2.id)

    records_cache = {r1.id: r1, r2.id: r2}
    hits = [_hit_for(r1), _hit_for(r2)]

    # If store.get is invoked at all, this test will raise.
    def _boom(*args, **kwargs):
        raise RuntimeError("store.get must not be called when records_cache is provided")

    original_get = store.get
    store.get = _boom  # type: ignore[assignment]
    try:
        result = on_read_check_batch(
            store, hits, session_id="test", records_cache=records_cache,
        )
    finally:
        store.get = original_get  # type: ignore[assignment]

    # Contradicts-edge detection still fires.
    assert len(result) == 1
    assert set(result[0]["source_ids"]) == {str(r1.id), str(r2.id)}


def test_on_read_check_batch_fallback_no_cache(tmp_path):
    """Without records_cache, falls back to exactly one store.all_records() call.

    Counts invocations via monkeypatched counter. store.get must not be called;
    all_records must be called exactly once.
    """
    from iai_mcp.s4 import on_read_check_batch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v1 = [0.0] * EMBED_DIM; v1[0] = 1.0
    v2 = [0.0] * EMBED_DIM; v2[1] = 1.0
    r1 = _make_record(text="x", vec=v1)
    r2 = _make_record(text="y", vec=v2)
    store.insert(r1)
    store.insert(r2)

    get_calls = [0]
    all_calls = [0]
    original_get = store.get
    original_all = store.all_records

    def _counting_get(*a, **kw):
        get_calls[0] += 1
        return original_get(*a, **kw)

    def _counting_all(*a, **kw):
        all_calls[0] += 1
        return original_all(*a, **kw)

    store.get = _counting_get  # type: ignore[assignment]
    store.all_records = _counting_all  # type: ignore[assignment]
    try:
        hits = [_hit_for(r1), _hit_for(r2)]
        _ = on_read_check_batch(store, hits, session_id="test")
    finally:
        store.get = original_get  # type: ignore[assignment]
        store.all_records = original_all  # type: ignore[assignment]

    assert get_calls[0] == 0, f"store.get called {get_calls[0]} times (should be 0)"
    assert all_calls[0] == 1, f"store.all_records called {all_calls[0]} times (should be 1)"


def test_batch_api_equivalence_on_detection(tmp_path):
    """on_read_check and on_read_check_batch return semantically-identical
    hint output over the same (store, hits, session_id) input.

    Comparison is over the (kind, frozenset(source_ids)) pair so that event
    ordering / text wording differences don't invalidate parity.
    """
    from iai_mcp.s4 import on_read_check, on_read_check_batch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # Near-identical vectors + opposite polarity tags (cosine > 0.97)
    v1 = [1.0] + [0.0] * (EMBED_DIM - 1)
    v2 = [0.99] + [0.01] + [0.0] * (EMBED_DIM - 2)
    r1 = _make_record(text="X good", vec=v1, tags=["topic", "positive"])
    r2 = _make_record(text="X bad", vec=v2, tags=["topic", "negative"])
    # Additionally a contradicts pair
    v3 = [0.0] * EMBED_DIM; v3[2] = 1.0
    v4 = [0.0] * EMBED_DIM; v4[3] = 1.0
    r3 = _make_record(text="Y true", vec=v3, tags=["claim"])
    r4 = _make_record(text="Y false", vec=v4, tags=["claim"])
    for r in (r1, r2, r3, r4):
        store.insert(r)
    store.add_contradicts_edge(r3.id, r4.id)

    hits = [_hit_for(r) for r in (r1, r2, r3, r4)]

    single = on_read_check(store, hits, session_id="eq_test")
    batch = on_read_check_batch(store, hits, session_id="eq_test")

    def _key(h: dict) -> tuple[str, frozenset[str]]:
        return (h["kind"], frozenset(h["source_ids"]))

    assert {_key(h) for h in single} == {_key(h) for h in batch}
    # Both should have detected at least 2 hints: polarity + contradicts.
    assert len(batch) >= 2


def test_on_read_check_batch_empty_hits(tmp_path):
    """Empty hits list -> empty hints, no exception."""
    from iai_mcp.s4 import on_read_check_batch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = on_read_check_batch(store, [], session_id="test")
    assert result == []
