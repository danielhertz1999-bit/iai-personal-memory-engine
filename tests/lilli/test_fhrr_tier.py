"""FHRR tier backend verification tests.

Tests cover: HV generation, bind/unbind round-trip, bundle (circular mean),
permute, similarity, and tier metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.lilli.tiers.fhrr import (
    LILLI_FHRR_DIM,
    TIER_INFO,
    bind,
    bundle,
    filler_hv,
    permute,
    random_hv,
    role_hv,
    similarity,
    unbind,
)


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------


def test_role_hv_length_10000() -> None:
    """role_hv must return exactly 10000 bytes."""
    assert len(role_hv("WHEN")) == 10000


def test_role_hv_deterministic() -> None:
    """Same role string always produces identical bytes."""
    hv1 = role_hv("WHEN")
    hv2 = role_hv("WHEN")
    assert hv1 == hv2


def test_filler_hv_length_10000() -> None:
    """filler_hv must return exactly 10000 bytes."""
    assert len(filler_hv("today")) == 10000


def test_random_hv_seeded_deterministic() -> None:
    """Same seed always produces identical output from random_hv."""
    hv1 = random_hv(42)
    hv2 = random_hv(42)
    assert hv1 == hv2


# ---------------------------------------------------------------------------
# bind / unbind tests
# ---------------------------------------------------------------------------


def test_bind_round_trip_exact() -> None:
    """unbind(bind(filler, role), role) must equal filler exactly (bytewise)."""
    role = role_hv("WHEN")
    filler = filler_hv("today")
    bound = bind(filler, role)
    recovered = unbind(bound, role)
    assert recovered == filler, "Round-trip bind->unbind must be exact bytewise"


def test_bind_length_mismatch_raises() -> None:
    """bind must raise ValueError when lengths differ."""
    a = role_hv("WHEN")   # 10000 bytes
    b = b"\x00" * 5       # 5 bytes
    with pytest.raises(ValueError):
        bind(a, b)


def test_bind_phase_addition_mod_256() -> None:
    """0xFF + 0x01 must wrap to 0x00 (mod 256 phase addition)."""
    a = b"\xFF" * 10000
    b = b"\x01" * 10000
    result = bind(a, b)
    assert result == b"\x00" * 10000, "Phase addition 0xFF + 0x01 must wrap to 0x00"


def test_unbind_phase_subtraction() -> None:
    """0x00 - 0x01 must wrap to 0xFF (mod 256 phase subtraction)."""
    bound = b"\x00" * 10000
    key = b"\x01" * 10000
    result = unbind(bound, key)
    assert result == b"\xFF" * 10000, "Phase subtraction 0x00 - 0x01 must wrap to 0xFF"


# ---------------------------------------------------------------------------
# bundle tests
# ---------------------------------------------------------------------------


def test_bundle_empty_returns_zero_bytes() -> None:
    """bundle([]) must return 10000 zero bytes."""
    result = bundle([])
    assert result == bytes(LILLI_FHRR_DIM)
    assert len(result) == 10000


def test_bundle_single_returns_input() -> None:
    """bundle([hv]) must return exactly the input hypervector."""
    hv = role_hv("WHAT")
    result = bundle([hv])
    assert result == hv, "bundle of a single HV must return that HV unchanged"


def test_bundle_two_identical() -> None:
    """bundle([hv, hv]) must return hv within ±1 rounding per byte.

    The circular mean of two identical phase vectors is the phase itself.
    Float64 precision through atan2 introduces at most ±1 quantisation step.
    """
    hv = role_hv("WHO")
    result = bundle([hv, hv])
    a = np.frombuffer(hv, dtype=np.uint8)
    b = np.frombuffer(result, dtype=np.uint8)
    diff = a.astype(np.int16) - b.astype(np.int16)
    assert np.all(np.abs(diff) <= 1), (
        "bundle([hv, hv]) must match hv within ±1 quantisation step per byte"
    )


# ---------------------------------------------------------------------------
# permute tests
# ---------------------------------------------------------------------------


def test_permute_round_trip() -> None:
    """permute(permute(hv, k), -k) must return the original hv."""
    hv = role_hv("WHERE")
    assert permute(permute(hv, 7), -7) == hv


# ---------------------------------------------------------------------------
# similarity tests
# ---------------------------------------------------------------------------


def test_similarity_identical() -> None:
    """similarity(hv, hv) must equal 1.0 within 1e-6."""
    hv = role_hv("WHEN")
    sim = similarity(hv, hv)
    assert abs(sim - 1.0) < 1e-6, f"Self-similarity must be 1.0, got {sim}"


def test_similarity_random_pair_near_zero() -> None:
    """Two independent random HVs at D=10000 must have |similarity| < 0.05."""
    hv1 = role_hv("WHEN")
    hv2 = role_hv("WHY")  # unrelated role
    sim = similarity(hv1, hv2)
    assert abs(sim) < 0.05, (
        f"Similarity of unrelated HVs at D=10000 should be near zero, got {sim}"
    )


def test_similarity_orthogonal_phases() -> None:
    """Vectors with 90-degree phase offset (0x00 vs 0x40) must have similarity ~0.0."""
    # 0x00 encodes angle 0.0; 0x40 = 64 encodes angle 64/256 * 2pi = pi/2 (90 degrees)
    a = b"\x00" * 10000
    b = b"\x40" * 10000
    sim = similarity(a, b)
    assert abs(sim) < 1e-6, f"90-degree offset must give similarity ~0.0, got {sim}"


# ---------------------------------------------------------------------------
# Tier metadata
# ---------------------------------------------------------------------------


def test_tier_info_metadata() -> None:
    """TIER_INFO must match the canonical semantic tier descriptor."""
    assert TIER_INFO == {
        "backend": "fhrr",
        "D": 10000,
        "bytes_per_hv": 10000,
        "use_case": "semantic",
    }
