from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.lilli.ops.cleanup import cleanup
from iai_mcp.lilli.ops.delta import delta_decode, delta_encode, delta_sparse_bits
from iai_mcp.lilli.ops.orthogonalize import orthogonalize
from iai_mcp.lilli.ops.replay import replay_with_noise
from iai_mcp.lilli.tiers import bsc

_ZERO = b"\x00" * 512
_ONES = b"\xff" * 512

def _random_hv(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, 512, dtype=np.uint8).tobytes()

def test_replay_sigma_zero_identity():
    hv = _random_hv(1)
    assert replay_with_noise(hv, 0.0) == hv

def test_replay_sigma_one_inverts():
    result = replay_with_noise(_ONES, 1.0)
    assert result == _ZERO

def test_replay_seed_deterministic():
    hv = _random_hv(2)
    a = replay_with_noise(hv, 0.1, seed=42)
    b = replay_with_noise(hv, 0.1, seed=42)
    assert a == b

def test_replay_invalid_sigma_raises():
    with pytest.raises(ValueError):
        replay_with_noise(_ZERO, 1.5)
    with pytest.raises(ValueError):
        replay_with_noise(_ZERO, -0.01)

def test_replay_seed_different_produces_different():
    hv = _random_hv(3)
    a = replay_with_noise(hv, 0.3, seed=100)
    b = replay_with_noise(hv, 0.3, seed=200)
    assert a != b

def test_ortho_empty_background_identity():
    hv = _random_hv(10)
    assert orthogonalize(hv, []) == hv

def test_ortho_preserves_length():
    t = bsc.role_hv("WHEN")
    bg = [bsc.role_hv("WHERE")]
    result = orthogonalize(t, bg)
    assert len(result) == len(t)

def test_ortho_reduces_mean_similarity_to_background():
    from iai_mcp.lilli.core.similarity import cosine_packed

    t = bsc.role_hv("ROLE")
    bg = [bsc.role_hv("PROJECT"), bsc.role_hv("ACTOR")]
    result = orthogonalize(t, bg, tau=0.7, max_flips=20)
    orig_sim = sum(cosine_packed(t, b) for b in bg) / len(bg)
    new_sim = sum(cosine_packed(result, b) for b in bg) / len(bg)
    assert new_sim <= orig_sim + 0.01

def test_ortho_respects_max_flips_cap():
    t = bsc.role_hv("INTENT")
    bg = [bsc.role_hv("SESSION_ID"), bsc.role_hv("LANG")]
    result = orthogonalize(t, bg, max_flips=0)
    assert result == t

def test_ortho_tau_blocks_excessive_drift():
    from iai_mcp.lilli.core.similarity import cosine_packed

    t = bsc.role_hv("WHEN")
    bg = [bsc.role_hv("WHERE"), bsc.role_hv("VALENCE")]
    result = orthogonalize(t, bg, tau=1.0, max_flips=50)
    sim_to_target = cosine_packed(result, t)
    assert sim_to_target >= 0.99

def test_cleanup_exact_match_returns_self():
    assert cleanup(_ZERO, [_ZERO, _ONES]) == _ZERO

def test_cleanup_empty_codebook_raises():
    with pytest.raises(ValueError):
        cleanup(_ZERO, [])

def test_cleanup_picks_nearest():
    result = cleanup(_ONES, [_ZERO, _ONES])
    assert result == _ONES

def test_cleanup_deterministic_tiebreak():
    canon = _random_hv(99)
    result = cleanup(canon, [canon, canon])
    assert result is canon or result == canon

def test_cleanup_length_mismatch_raises():
    with pytest.raises(ValueError):
        cleanup(_ZERO, [b"\x00" * 256])

def test_delta_round_trip():
    d = delta_encode(_ONES, _ZERO)
    assert delta_decode(_ONES, d) == _ZERO

def test_delta_self_zero():
    hv = _random_hv(7)
    assert delta_encode(hv, hv) == b"\x00" * len(hv)

def test_delta_length_mismatch_raises():
    with pytest.raises(ValueError):
        delta_encode(b"\x00" * 5, b"\x00" * 6)

def test_delta_sparse_bits_count():
    assert delta_sparse_bits(b"\x00" * 512) == 0
    assert delta_sparse_bits(b"\xff" * 512) == 4096
    assert delta_sparse_bits(b"\xb0") == 3

def test_delta_chain_three_variants_correct():
    rng = np.random.default_rng(55)
    canonical = rng.integers(0, 256, 512, dtype=np.uint8).tobytes()
    for i in range(3):
        variant = rng.integers(0, 256, 512, dtype=np.uint8).tobytes()
        enc = delta_encode(canonical, variant)
        dec = delta_decode(canonical, enc)
        assert dec == variant, f"round-trip failed for variant {i}"
