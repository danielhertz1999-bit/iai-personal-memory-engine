"""Temporal decay primitives.

decay_structure_edge implements the FSRS-derived multiplier (no decay in 90-day grace,
then 0.9 ** (days - 90)). This formula is behaviourally identical to the ancestor
implementation in tem.py.

temporal_decay applies structured noise injection to a hypervector -- older hvs
accumulate bit flips proportional to age, biasing similarity comparison natively.
temporal_decay is BSC-tier-specific: packed-bit binary hvs only. FHRR/sparse decay
would require different noise models (phase-space noise / active-index perturbation)
and is deferred to a future plan.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# FSRS decay constants
# ---------------------------------------------------------------------------

DECAY_GRACE_DAYS: int = 90
"""Number of days with no decay applied. Edge weights stay at 1.0 during this window."""

DECAY_BASE: float = 0.9
"""Exponential decay base applied after the grace window: weight *= DECAY_BASE ** (days - DECAY_GRACE_DAYS)."""


# ---------------------------------------------------------------------------
# FSRS decay formula
# ---------------------------------------------------------------------------


def decay_structure_edge(stability: float, difficulty: float, dt_days: float) -> float:
    """FSRS decay multiplier for structure edges.

    Returns 1.0 during the grace window (0..DECAY_GRACE_DAYS days), then applies
    ``DECAY_BASE ** (dt_days - DECAY_GRACE_DAYS)``.

    Behaviourally identical to the ancestor implementation in tem.py for all
    (stability, difficulty, dt_days) inputs. The stability and difficulty
    parameters are accepted for forward compatibility but are not used in the
    current formula.

    Args:
        stability: Edge stability score (float, unused in current formula).
        difficulty: Edge difficulty score (float, unused in current formula).
        dt_days: Age in days (negative values clamped to 0.0).

    Returns:
        Float multiplier in (0.0, 1.0]: 1.0 = no decay; near-0.0 = heavily decayed.
    """
    age_days = max(0.0, float(dt_days))
    if age_days <= DECAY_GRACE_DAYS:
        return 1.0
    return DECAY_BASE ** (age_days - DECAY_GRACE_DAYS)


# ---------------------------------------------------------------------------
# HV-side structured noise decay
# ---------------------------------------------------------------------------


def temporal_decay(
    hv: bytes,
    dt_days: float,
    *,
    D: int | None = None,
    seed: int | None = None,
) -> bytes:
    """Apply structured noise injection to a BSC packed-bit hypervector.

    Computes ``flip_probability = 1.0 - decay_structure_edge(0, 0, dt_days)``
    and independently flips each bit with that probability. Older hvs accumulate
    more bit flips, biasing downstream similarity comparisons natively.

    Within the 90-day grace window (dt_days <= DECAY_GRACE_DAYS), flip_probability
    is 0.0 and the hv is returned unchanged.

    This function is BSC-tier-specific (packed binary bits). FHRR and Sparse VSA
    decay require different noise models (phase-space perturbation / active-index
    shuffling) and are deferred.

    Args:
        hv: Packed-bit BSC hypervector (bytes).
        dt_days: Age in days. Values <= DECAY_GRACE_DAYS return hv unchanged.
        D: Unused parameter (accepted for API symmetry with other ops). The
                 dimensionality is inferred from len(hv) * 8.
        seed: Optional integer seed for numpy PCG64 RNG. When provided, output
                 is fully deterministic across calls with the same (hv, dt_days, seed).
                 When None, uses a non-seeded RNG (non-deterministic).

    Returns:
        Packed bytes of the same length as hv with bit flips applied.
    """
    decay_multiplier = decay_structure_edge(0, 0, dt_days)
    flip_probability = 1.0 - decay_multiplier

    # Fast path: no decay in grace window.
    if flip_probability == 0.0:
        return hv

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))  # length len(hv)*8
    flip_mask = rng.random(bits.shape) < flip_probability
    bits = bits ^ flip_mask.astype(np.uint8)
    return np.packbits(bits).tobytes()
