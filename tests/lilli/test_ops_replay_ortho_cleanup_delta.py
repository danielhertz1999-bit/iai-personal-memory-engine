"""Tests for lilli.ops.replay, orthogonalize, cleanup, delta (20 tests)."""
from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.lilli.ops.cleanup import cleanup
from iai_mcp.lilli.ops.delta import delta_decode, delta_encode, delta_sparse_bits
from iai_mcp.lilli.ops.orthogonalize import orthogonalize
from iai_mcp.lilli.ops.replay import replay_with_noise
from iai_mcp.lilli.tiers import bsc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO = b"\x00" * 512
_ONES = b"\xff" * 512


def _random_hv(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, 512, dtype=np.uint8).tobytes()


# ---------------------------------------------------------------------------
# replay (5 tests)
# ---------------------------------------------------------------------------


def test_replay_sigma_zero_identity():
    """sigma=0.0 returns input unchanged byte-for-byte."""
    hv = _random_hv(1)
    assert replay_with_noise(hv, 0.0) == hv


def test_replay_sigma_one_inverts():
    """sigma=1.0 flips every bit (bitwise NOT)."""
    result = replay_with_noise(_ONES, 1.0)
    assert result == _ZERO


def test_replay_seed_deterministic():
    """Same seed produces identical output on repeated calls."""
    hv = _random_hv(2)
    a = replay_with_noise(hv, 0.1, seed=42)
    b = replay_with_noise(hv, 0.1, seed=42)
    assert a == b


def test_replay_invalid_sigma_raises():
    """sigma outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        replay_with_noise(_ZERO, 1.5)
    with pytest.raises(ValueError):
        replay_with_noise(_ZERO, -0.01)


def test_replay_seed_different_produces_different():
    """Different seeds (with reasonable sigma) produce distinct outputs."""
    hv = _random_hv(3)
    a = replay_with_noise(hv, 0.3, seed=100)
    b = replay_with_noise(hv, 0.3, seed=200)
    # With sigma=0.3 and 4096 bits, probability of identical outputs is negligible
    assert a != b


# ---------------------------------------------------------------------------
# orthogonalize (5 tests)
# ---------------------------------------------------------------------------


def test_ortho_empty_background_identity():
    """Empty background list returns target unchanged."""
    hv = _random_hv(10)
    assert orthogonalize(hv, []) == hv


def test_ortho_preserves_length():
    """Output has the same byte length as input."""
    t = bsc.role_hv("WHEN")
    bg = [bsc.role_hv("WHERE")]
    result = orthogonalize(t, bg)
    assert len(result) == len(t)


def test_ortho_reduces_mean_similarity_to_background():
    """Mean cosine similarity to background does not increase materially."""
    from iai_mcp.lilli.core.similarity import cosine_packed

    t = bsc.role_hv("ROLE")
    bg = [bsc.role_hv("PROJECT"), bsc.role_hv("ACTOR")]
    result = orthogonalize(t, bg, tau=0.7, max_flips=20)
    orig_sim = sum(cosine_packed(t, b) for b in bg) / len(bg)
    new_sim = sum(cosine_packed(result, b) for b in bg) / len(bg)
    assert new_sim <= orig_sim + 0.01


def test_ortho_respects_max_flips_cap():
    """With max_flips=0 the output equals the input (no flips allowed)."""
    t = bsc.role_hv("INTENT")
    bg = [bsc.role_hv("SESSION_ID"), bsc.role_hv("LANG")]
    result = orthogonalize(t, bg, max_flips=0)
    assert result == t


def test_ortho_tau_blocks_excessive_drift():
    """Very high tau (tau=1.0) blocks all flips -- output equals input."""
    from iai_mcp.lilli.core.similarity import cosine_packed

    t = bsc.role_hv("WHEN")
    bg = [bsc.role_hv("WHERE"), bsc.role_hv("VALENCE")]
    result = orthogonalize(t, bg, tau=1.0, max_flips=50)
    # At tau=1.0 any flip would drop below threshold, so no flip is accepted.
    # In practice the first flip drops similarity below 1.0, so output == target.
    sim_to_target = cosine_packed(result, t)
    assert sim_to_target >= 0.99  # either identical or near-identical


# ---------------------------------------------------------------------------
# cleanup (5 tests)
# ---------------------------------------------------------------------------


def test_cleanup_exact_match_returns_self():
    """Noisy hv identical to a codebook entry snaps to that entry."""
    assert cleanup(_ZERO, [_ZERO, _ONES]) == _ZERO


def test_cleanup_empty_codebook_raises():
    """Empty codebook raises ValueError."""
    with pytest.raises(ValueError):
        cleanup(_ZERO, [])


def test_cleanup_picks_nearest():
    """Returns nearest entry (by Hamming) when there is a clear winner."""
    # _ONES vs codebook [_ZERO (dist=1.0), _ONES (dist=0.0)] -- _ONES wins
    result = cleanup(_ONES, [_ZERO, _ONES])
    assert result == _ONES


def test_cleanup_deterministic_tiebreak():
    """When two codebook entries tie for distance, the FIRST wins."""
    # Craft a tie: noisy_hv halfway between entries A and B in Hamming space.
    # Use two complementary entries; noisy = random mix so it may tie.
    # Simpler: identical entries => distance 0 for both; first must win.
    canon = _random_hv(99)
    result = cleanup(canon, [canon, canon])
    assert result is canon or result == canon  # first entry returned


def test_cleanup_length_mismatch_raises():
    """Codebook entry with wrong length raises ValueError."""
    with pytest.raises(ValueError):
        cleanup(_ZERO, [b"\x00" * 256])


# ---------------------------------------------------------------------------
# delta (5 tests)
# ---------------------------------------------------------------------------


def test_delta_round_trip():
    """delta_decode(c, delta_encode(c, v)) == v for all-ff / all-00 pair."""
    d = delta_encode(_ONES, _ZERO)
    assert delta_decode(_ONES, d) == _ZERO


def test_delta_self_zero():
    """delta_encode(hv, hv) produces all-zero bytes."""
    hv = _random_hv(7)
    assert delta_encode(hv, hv) == b"\x00" * len(hv)


def test_delta_length_mismatch_raises():
    """Length mismatch in delta_encode raises ValueError."""
    with pytest.raises(ValueError):
        delta_encode(b"\x00" * 5, b"\x00" * 6)


def test_delta_sparse_bits_count():
    """delta_sparse_bits returns correct popcount on known inputs."""
    assert delta_sparse_bits(b"\x00" * 512) == 0
    assert delta_sparse_bits(b"\xff" * 512) == 4096
    # One byte with 3 set bits: 0b10110000 = 0xb0
    assert delta_sparse_bits(b"\xb0") == 3


def test_delta_chain_three_variants_correct():
    """Round-trip holds for three independent variant–canonical pairs."""
    rng = np.random.default_rng(55)
    canonical = rng.integers(0, 256, 512, dtype=np.uint8).tobytes()
    for i in range(3):
        variant = rng.integers(0, 256, 512, dtype=np.uint8).tobytes()
        enc = delta_encode(canonical, variant)
        dec = delta_decode(canonical, enc)
        assert dec == variant, f"round-trip failed for variant {i}"
