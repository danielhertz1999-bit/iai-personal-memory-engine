"""ART vigilance gate tests (MEM-03, D-07, D-14)."""
from __future__ import annotations

from iai_mcp.types import EMBED_DIM
from iai_mcp.write import VIGILANCE_RHO, apply_art_gate, cosine
from tests.test_store import _make


def test_vigilance_rho_is_0_95():
    """ρ fixed at 0.95 for Phase 1."""
    assert VIGILANCE_RHO == 0.95


def test_empty_store_creates():
    new = _make()
    action, target = apply_art_gate([], new)
    assert action == "create"
    assert target == new.id


def test_high_similarity_merges():
    """Nearly-identical vectors -> merge target is the existing record."""
    existing = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    candidate = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))  # same vector
    action, target = apply_art_gate([existing], candidate)
    assert action == "merge"
    assert target == existing.id


def test_low_similarity_creates():
    """Orthogonal vectors -> cosine 0 < 0.95 -> create new."""
    existing = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    candidate = _make(vec=[0.0] * (EMBED_DIM - 1) + [1.0])
    action, target = apply_art_gate([existing], candidate)
    assert action == "create"
    assert target == candidate.id


def test_moderate_similarity_below_rho_creates():
    """cos = 0.90 < 0.95 -> create."""
    existing = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))
    # Construct a vector with cosine exactly 0.90 to the existing one.
    # If we take [0.9, sqrt(1 - 0.81), 0, 0, ...] with unit norm, cosine = 0.9
    import math
    y = math.sqrt(1 - 0.9 * 0.9)
    candidate = _make(vec=[0.9, y] + [0.0] * (EMBED_DIM - 2))
    sim = cosine(existing.embedding, candidate.embedding)
    assert abs(sim - 0.9) < 1e-6
    action, target = apply_art_gate([existing], candidate)
    assert action == "create"
    assert target == candidate.id


def test_never_merge_record_skipped():
    """records with never_merge=True (L0 identity) are never merge targets."""
    pinned = _make(
        vec=[1.0] + [0.0] * (EMBED_DIM - 1),
        pinned=True,
        never_merge=True,
    )
    candidate = _make(vec=[1.0] + [0.0] * (EMBED_DIM - 1))  # identical vector
    action, target = apply_art_gate([pinned], candidate)
    assert action == "create"
    assert target == candidate.id


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0
