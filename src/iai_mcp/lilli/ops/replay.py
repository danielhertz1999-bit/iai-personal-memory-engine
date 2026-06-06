"""Replay-with-noise primitive for the CLUSTER_REPLAY sleep step.

Generates a noisy variant of a canonical hypervector by flipping each bit
with probability sigma. Used to test memory robustness -- if the replayed
variant still resonates with the canonical via similarity > threshold, the
memory is reinforced.

Imports only numpy and stdlib; no lilli tier imports needed for this module.
"""
from __future__ import annotations

import numpy as np


def replay_with_noise(
    hv: bytes,
    sigma: float = 0.05,
    seed: int | None = None,
) -> bytes:
    """Return a noisy variant of *hv* by flipping each bit with probability *sigma*.

    Parameters
    ----------
    hv:
        Packed binary hypervector (any length in bytes).
    sigma:
        Bit-flip probability in [0.0, 1.0]. 0.0 returns *hv* unchanged;
        1.0 returns the bitwise inverse.
    seed:
        Optional integer seed for the RNG. When provided, the result is
        deterministic. When ``None`` (default), a new non-seeded RNG is used.

    Returns
    -------
    bytes
        Packed binary hypervector of the same length as *hv*.

    Raises
    ------
    ValueError
        If *sigma* is outside [0.0, 1.0].
    """
    if not (0.0 <= sigma <= 1.0):
        raise ValueError(
            f"sigma must be in [0.0, 1.0], got {sigma!r}"
        )

    if sigma == 0.0:
        return hv

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
    flip_mask = (rng.random(bits.shape) < sigma).astype(np.uint8)
    noisy = bits ^ flip_mask
    return np.packbits(noisy).tobytes()
