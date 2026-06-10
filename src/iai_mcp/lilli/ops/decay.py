from __future__ import annotations

import numpy as np


DECAY_GRACE_DAYS: int = 90
"""Number of days with no decay applied. Edge weights stay at 1.0 during this window."""

DECAY_BASE: float = 0.9
"""Exponential decay base applied after the grace window: weight *= DECAY_BASE ** (days - DECAY_GRACE_DAYS)."""


def decay_structure_edge(stability: float, difficulty: float, dt_days: float) -> float:
    age_days = max(0.0, float(dt_days))
    if age_days <= DECAY_GRACE_DAYS:
        return 1.0
    return DECAY_BASE ** (age_days - DECAY_GRACE_DAYS)


def temporal_decay(
    hv: bytes,
    dt_days: float,
    *,
    D: int | None = None,
    seed: int | None = None,
) -> bytes:
    decay_multiplier = decay_structure_edge(0, 0, dt_days)
    flip_probability = 1.0 - decay_multiplier

    if flip_probability == 0.0:
        return hv

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
    flip_mask = rng.random(bits.shape) < flip_probability
    bits = bits ^ flip_mask.astype(np.uint8)
    return np.packbits(bits).tobytes()
