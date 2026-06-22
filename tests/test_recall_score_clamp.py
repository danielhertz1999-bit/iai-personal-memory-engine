"""Falsifiable tests for BUG (4a): displayed recall score must be clamped to
[0,1] WITHOUT changing the relative ordering of hits.

The recall pipeline applies *multiplicative* boosts (trigram*2, FTS*3,
valence*(1+v)) to the base similarity score (pipeline.py:786-789). With no
clamp the serialized `score` can exceed 1.0 -- a leaky internal quantity
surfaced as if it were a probability/confidence. The contract under test:

  1. _hit_to_json(hit)["score"] is always within [0.0, 1.0].
  2. The order of hits sorted by their *internal* sort key is preserved even
     after the displayed score is clamped (two equally-clamped 1.0 hits must
     keep their pre-clamp ranking).

These tests are hermetic: they build MemoryHit objects directly, no store,
no daemon, no embedder.

RUN:
  cd <REPO> && .venv/bin/python -m pytest /tmp/iai_fix_recall/test_recall_score_clamp.py -v
"""
from __future__ import annotations

from iai_mcp.types import MemoryHit
from iai_mcp.core._serializers import _hit_to_json


def _hit(score: float, sort_score: float | None = None, rid_int: int = 1) -> MemoryHit:
    from uuid import UUID
    return MemoryHit(
        record_id=UUID(int=rid_int),
        score=score,
        sort_score=sort_score,
        reason="t",
        literal_surface="x",
        adjacent_suggestions=[],
    )


def test_displayed_score_is_clamped_to_unit_interval():
    # A boosted hit (trigram*2 then FTS*3 => 6x a 0.4 cosine = 2.4) must NOT
    # leak a >1 score to the client.
    h = _hit(score=2.4, sort_score=2.4)
    out = _hit_to_json(h)
    assert 0.0 <= out["score"] <= 1.0, f"score leaked out of [0,1]: {out['score']}"


def test_negative_score_clamped_to_zero():
    h = _hit(score=-0.3, sort_score=-0.3)
    out = _hit_to_json(h)
    assert out["score"] == 0.0


def test_clamp_preserves_internal_ordering():
    # Two hits both boost past 1.0 (2.4 and 1.8). Displayed scores collapse to
    # 1.0/1.0, but the internal sort_score must still distinguish them so the
    # ranking the engine computed is preserved.
    strong = _hit(score=2.4, sort_score=2.4, rid_int=1)
    weaker = _hit(score=1.8, sort_score=1.8, rid_int=2)
    mid = _hit(score=0.5, sort_score=0.5, rid_int=3)

    hits = [mid, weaker, strong]
    # Sort the way the pipeline does post-rank: by the internal key.
    hits.sort(key=lambda h: (h.sort_score if h.sort_score is not None else h.score),
              reverse=True)

    order = [h.record_id.int for h in hits]
    assert order == [1, 2, 3], f"internal ordering not preserved: {order}"

    # And after serialization the two boosted ones are both clamped but still
    # appear in the engine's order (list order, not score value).
    serialized = [_hit_to_json(h) for h in hits]
    assert serialized[0]["record_id"].endswith("0001")
    assert serialized[1]["record_id"].endswith("0002")
    assert all(0.0 <= s["score"] <= 1.0 for s in serialized)
    # Top two collapsed to the ceiling -- proves ordering can't rely on the
    # displayed score alone, which is exactly why sort_score must exist.
    assert serialized[0]["score"] == 1.0
    assert serialized[1]["score"] == 1.0


def test_sort_score_falls_back_to_score_when_absent():
    # Backward compat: a hit built the old way (no sort_score) still sorts and
    # serializes sanely.
    h = _hit(score=0.42, sort_score=None)
    key = h.sort_score if h.sort_score is not None else h.score
    assert key == 0.42
    assert _hit_to_json(h)["score"] == 0.42
