from __future__ import annotations

from iai_mcp.types import EMBED_DIM
from iai_mcp.write import VIGILANCE_RHO, apply_art_gate, cosine
from tests.test_store import _make


def test_vigilance_rho_is_0_95():
    assert VIGILANCE_RHO == 0.95


def test_empty_store_creates():
    new = _make()
    action, target = apply_art_gate([], new)
    assert action == "create"
    assert target == new.id


def test_high_similarity_merges():
    existing = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    candidate = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    action, target = apply_art_gate([existing], candidate)
    assert action == "merge"
    assert target == existing.id


def test_low_similarity_creates():
    existing = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    candidate = _make(vec=[0.0] * (EMBED_DIM - 1) + [1.0])
    action, target = apply_art_gate([existing], candidate)
    assert action == "create"
    assert target == candidate.id


def test_moderate_similarity_below_rho_creates():
    existing = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    import math
    y = math.sqrt(1 - 0.9 * 0.9)
    candidate = _make(vec=[0.9, y] + [0.0] * (EMBED_DIM - 2))
    sim = cosine(existing.embedding, candidate.embedding)
    assert abs(sim - 0.9) < 1e-6
    action, target = apply_art_gate([existing], candidate)
    assert action == "create"
    assert target == candidate.id


def test_never_merge_record_skipped():
    pinned = _make(
        vec=[1.0] + [0.0] * (EMBED_DIM - 1),
        pinned=True,
        never_merge=True,
    )
    candidate = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    action, target = apply_art_gate([pinned], candidate)
    assert action == "create"
    assert target == candidate.id


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0
