"""Tests for the Sparse VSA tier backend.

Verifies K-sparsity invariant, determinism, pack/unpack round-trip, APPROXIMATE
binding contract, bundle, permute, and Jaccard similarity.
"""
from __future__ import annotations

import pytest
from iai_mcp.lilli.tiers.sparse_vsa import (
    LILLI_SPARSE_DIM,
    SPARSE_K,
    TIER_INFO,
    bind,
    bundle,
    filler_hv,
    pack_indices,
    permute,
    role_hv,
    similarity,
    unbind,
    unpack_indices,
)


# ---------------------------------------------------------------------------
# role_hv
# ---------------------------------------------------------------------------

def test_role_hv_length_K() -> None:
    """role_hv must return exactly SPARSE_K=20 indices."""
    result = role_hv("WHEN")
    assert len(result) == SPARSE_K == 20


def test_role_hv_sorted_and_in_range() -> None:
    """All indices must be sorted ascending and within [0, LILLI_SPARSE_DIM)."""
    result = role_hv("WHO")
    assert result == sorted(result), "Indices must be sorted ascending"
    assert all(0 <= i < LILLI_SPARSE_DIM for i in result), "All indices must be in [0, D)"


def test_role_hv_deterministic() -> None:
    """Same role string must produce identical index list across calls."""
    assert role_hv("WHEN") == role_hv("WHEN")
    assert role_hv("WHERE") == role_hv("WHERE")


def test_role_hv_no_duplicates() -> None:
    """Active-bit indices must be unique (no duplicate bits set)."""
    result = role_hv("WHEN")
    assert len(result) == len(set(result)), "role_hv must return deduplicated indices"


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------

def test_pack_unpack_round_trip() -> None:
    """pack then unpack must reproduce the original index list."""
    indices = [5, 100, 1000]
    assert unpack_indices(pack_indices(indices)) == indices


def test_pack_length_40_bytes() -> None:
    """pack_indices must always produce exactly 40 bytes (SPARSE_K × uint16)."""
    assert len(pack_indices(role_hv("WHEN"))) == 40


def test_pack_full_hv_round_trip() -> None:
    """A full SPARSE_K-element HV must survive pack/unpack unchanged."""
    hv = role_hv("ENTITY")
    assert unpack_indices(pack_indices(hv)) == hv


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------

def test_bind_cdt_basic() -> None:
    """bind must return exactly SPARSE_K sorted integers in range."""
    a = role_hv("a")
    b = role_hv("b")
    result = bind(a, b)
    assert len(result) == SPARSE_K
    assert result == sorted(result), "bind result must be sorted"
    assert all(0 <= i < LILLI_SPARSE_DIM for i in result), "All bound indices must be in range"


def test_bind_commutative() -> None:
    """Approximate union-truncate binding must be symmetric: bind(a,b) == bind(b,a)."""
    a = role_hv("apple")
    b = role_hv("orange")
    assert bind(a, b) == bind(b, a)


def test_bind_length_always_sparse_k() -> None:
    """bind must return exactly SPARSE_K elements even when inputs fully overlap."""
    # Same HV bound with itself -- union == the original set
    hv = role_hv("overlap_test")
    result = bind(hv, hv)
    assert len(result) == SPARSE_K


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def test_bundle_empty() -> None:
    """bundle([]) must return an empty list."""
    assert bundle([]) == []


def test_bundle_single_preserves_K() -> None:
    """bundle of a single HV must return that HV unchanged."""
    hv = role_hv("SINGLE")
    assert bundle([hv]) == hv


def test_bundle_two_identical_preserves_input() -> None:
    """bundle of two identical HVs must return the input HV."""
    hv = role_hv("CLONE")
    assert bundle([hv, hv]) == hv


def test_bundle_three_distinct_keeps_K() -> None:
    """bundle of three distinct HVs must return exactly SPARSE_K indices."""
    result = bundle([role_hv("a"), role_hv("b"), role_hv("c")])
    assert len(result) == SPARSE_K


# ---------------------------------------------------------------------------
# permute
# ---------------------------------------------------------------------------

def test_permute_round_trip() -> None:
    """permute(permute(hv, k), -k) must reconstruct the original HV."""
    hv = role_hv("WHEN")
    assert permute(permute(hv, 7), -7) == hv


def test_permute_preserves_length() -> None:
    """permute must not change the number of active indices."""
    hv = role_hv("ENTITY")
    for k in (0, 1, -1, 100, LILLI_SPARSE_DIM - 1):
        result = permute(hv, k)
        assert len(result) == SPARSE_K, f"permute({k}) changed length"


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

def test_similarity_identical() -> None:
    """Jaccard of a HV with itself must be exactly 1.0."""
    hv = role_hv("WHEN")
    assert similarity(hv, hv) == 1.0


def test_similarity_disjoint() -> None:
    """Two HVs with no shared indices must have similarity 0.0."""
    a = list(range(0, 20))
    b = list(range(100, 120))
    assert similarity(a, b) == 0.0


def test_similarity_half_overlap() -> None:
    """Jaccard for 10-index overlap over 30-index union must be ~0.333."""
    # shared: [0..9], a-only: [10..19], b-only: [100..109]
    shared = list(range(10))
    a = shared + list(range(10, 20))
    b = shared + list(range(100, 110))
    expected = 10 / 30  # |intersection| / |union|
    assert abs(similarity(a, b) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_tier_info_metadata() -> None:
    """TIER_INFO must carry the canonical backend metadata."""
    assert TIER_INFO == {
        "backend": "sparse_vsa",
        "D": 2048,
        "bytes_per_hv": 40,
        "use_case": "procedural",
    }
