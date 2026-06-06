"""Tests for lilli.ops.separation -- rejection-sampling pattern separation MVP."""
from __future__ import annotations

import pytest

from iai_mcp.lilli.core.similarity import hamming
from iai_mcp.lilli.ops.separation import (
    MAX_RETRIES_DEFAULT,
    SIMILARITY_THRESHOLD_DEFAULT,
    pattern_separate,
)
from iai_mcp.lilli.tiers import bsc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_sim(hv: bytes, background: list[bytes]) -> float:
    """Return max BSC similarity between hv and any background vector."""
    if not background:
        return 0.0
    return max(1.0 - hamming(hv, b) for b in background)


def _distinct_hvs(n: int) -> list[bytes]:
    """Return n deterministically distinct BSC role hypervectors."""
    roles = list(bsc.BSC_ROLE_VOCABULARY[:n])
    return [bsc.role_hv(r) for r in roles]


# ---------------------------------------------------------------------------
# Test 1: empty background returns target unchanged
# ---------------------------------------------------------------------------


def test_empty_background_returns_target_unchanged() -> None:
    target = bsc.role_hv("WHEN")
    result = pattern_separate(target, [])
    assert result == target, "Empty background must return target unchanged"


# ---------------------------------------------------------------------------
# Test 2: output length matches input for various background sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_background", [1, 5, 50])
def test_returns_bytes_of_same_length(n_background: int) -> None:
    # Build n_background distinct background HVs using filler seeds.
    background = [bsc.filler_hv(f"bg-{i}") for i in range(n_background)]
    target = bsc.role_hv("WHEN")
    result = pattern_separate(target, background)
    assert len(result) == len(target), (
        f"Output length {len(result)} != input length {len(target)} "
        f"for background size {n_background}"
    )


# ---------------------------------------------------------------------------
# Test 3: deterministic output -- same inputs produce identical bytes
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    target = bsc.role_hv("PROJECT")
    background = _distinct_hvs(3)
    result_a = pattern_separate(target, background, similarity_threshold=0.5)
    result_b = pattern_separate(target, background, similarity_threshold=0.5)
    assert result_a == result_b, "pattern_separate must be deterministic"


# ---------------------------------------------------------------------------
# Test 4: dissimilar target is early-accepted (attempt 0 passes)
# ---------------------------------------------------------------------------


def test_dissimilar_target_early_accepted() -> None:
    # WHEN and OBJECT role HVs are seeded differently; their Hamming distance
    # at D=4096 is ~50% (random independent seeds) → similarity ~0.5 < 0.85.
    target = bsc.role_hv("WHEN")
    background = [bsc.role_hv("OBJECT")]

    sim_score = 1.0 - hamming(target, background[0])
    # Confirm precondition: background is not dangerously close to target.
    assert sim_score < SIMILARITY_THRESHOLD_DEFAULT, (
        f"Precondition failed: role HVs have sim={sim_score:.3f} >= threshold; "
        "test needs independent seeds"
    )

    result = pattern_separate(target, background)
    # Early-accept means we get back the original target bytes.
    assert result == target, (
        "Dissimilar target should be returned unchanged (early accept on attempt 0)"
    )


# ---------------------------------------------------------------------------
# Test 5: identical target to background forces at least one bit flip
# ---------------------------------------------------------------------------


def test_identical_target_to_background_perturbed() -> None:
    target = bsc.role_hv("WHEN")
    # Background contains a clone of target -- similarity == 1.0 > any threshold.
    background = [target]
    result = pattern_separate(target, background, similarity_threshold=0.99)
    assert result != target, (
        "Target identical to background must be perturbed by at least one bit flip"
    )


# ---------------------------------------------------------------------------
# Test 6: rejection sampling reduces max-similarity to background
# ---------------------------------------------------------------------------


def test_rejection_sampling_decorrelates() -> None:
    # Construct a target very similar (but not identical) to a background HV.
    # Flip only 2 bits of a role HV so similarity > 0.95.
    import numpy as np

    base = bsc.role_hv("WHEN")
    bits = np.unpackbits(np.frombuffer(base, dtype=np.uint8))
    bits[0] ^= 1
    bits[1] ^= 1
    target = np.packbits(bits).tobytes()

    background = [base]  # original role HV as background

    input_max_sim = _max_sim(target, background)
    assert input_max_sim > 0.95, f"Precondition: input sim={input_max_sim:.4f} must be > 0.95"

    result = pattern_separate(target, background)
    output_max_sim = _max_sim(result, background)

    assert output_max_sim < input_max_sim, (
        f"Rejection sampling must reduce max-similarity: "
        f"input={input_max_sim:.4f}, output={output_max_sim:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: bounded retries -- no raise on exhaustion, returns bytes of correct length
# ---------------------------------------------------------------------------


def test_bounded_retries_returns_best_on_exhaustion() -> None:
    # target identical to all 5 background entries -- worst possible overlap.
    # With max_retries=1 and threshold=0.999, exhaustion is guaranteed.
    target = bsc.role_hv("WHEN")
    background = [target] * 5

    result = pattern_separate(target, background, max_retries=1, similarity_threshold=0.999)
    # Must not raise; must return bytes of the correct length.
    assert isinstance(result, bytes), "Result must be bytes"
    assert len(result) == len(target), f"Result length {len(result)} != {len(target)}"


# ---------------------------------------------------------------------------
# Test 8: progressive flip count increases separation with more retries
# ---------------------------------------------------------------------------


def test_progressive_flip_count_increases_separation() -> None:
    """More retries allowed means more bits can be flipped, increasing separation.

    Strategy: use a target identical to background so the threshold can never be
    met (threshold=0.999). Compare Hamming distance from original target between
    max_retries=1 (flips attempt-1 * 16 = 16 bits) and max_retries=3 (flips up
    to 48 bits). The result with more retries must be at least as far from the
    original (larger Hamming distance or equal).
    """
    target = bsc.role_hv("WHEN")
    background = [target] * 3  # always above threshold

    result_1 = pattern_separate(target, background, max_retries=1, similarity_threshold=0.999)
    result_3 = pattern_separate(target, background, max_retries=3, similarity_threshold=0.999)

    # "Best" candidate: lowest max-similarity to background.
    # More retries explores further; best found with max_retries=3 should have
    # at least as many bits flipped as with max_retries=1.
    ham_1 = hamming(target, result_1)  # hamming distance from original
    ham_3 = hamming(target, result_3)

    # With max_retries=3 the best candidate explores attempts 1,2,3 with
    # flip counts 16, 32, 48. With max_retries=1 only attempt 1 (16 flips).
    # Best-candidate selection means ham_3 >= ham_1 (more flips available).
    assert ham_3 >= ham_1, (
        f"More retries should explore larger perturbations: "
        f"ham(max_retries=1)={ham_1:.4f}, ham(max_retries=3)={ham_3:.4f}"
    )
