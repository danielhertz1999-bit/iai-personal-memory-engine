"""HV orthogonalization for the CRISIS_RECLUSTER sleep step.

orthogonalize(target, background_hvs) returns a hv that minimizes mean
similarity to background while preserving target structure. Greedy bit-flip
algorithm: for each bit position where flipping reduces mean similarity to
the background AND keeps similarity to the target above threshold tau
(default 0.7), flip it. Bounded by max_flips (default 50 bit flips).

MVP implementation -- upgrade to Gram-Schmidt-style at scale is deferred
to a future extension per consensus decision (rejection sampling / greedy
is sufficient at current record counts).

Imports only numpy, lilli.core.similarity, and stdlib.
"""
from __future__ import annotations

import numpy as np

from iai_mcp.lilli.core import similarity as sim


def orthogonalize(
    target: bytes,
    background_hvs: list[bytes],
    *,
    tau: float = 0.7,
    max_flips: int = 50,
) -> bytes:
    """Return a variant of *target* with reduced mean similarity to *background_hvs*.

    Uses a greedy bit-flip search: at each step it tries every candidate bit
    position, picks the single flip that maximally reduces mean cosine similarity
    to the background, and accepts it only if the resulting hv retains cosine
    similarity >= tau to the original target. Stops when no beneficial flip
    remains or *max_flips* have been applied.

    Parameters
    ----------
    target:
        Packed binary hypervector to orthogonalize.
    background_hvs:
        List of packed binary hypervectors to push away from. Empty list
        returns *target* unchanged.
    tau:
        Minimum cosine similarity to *target* to preserve structural identity.
        Default 0.7.
    max_flips:
        Maximum number of bit flips to apply. Default 50.

    Returns
    -------
    bytes
        Packed binary hypervector of the same length as *target*.
    """
    if not background_hvs:
        return target

    # Working copy as mutable uint8 bit array
    bits = np.unpackbits(np.frombuffer(target, dtype=np.uint8)).copy()
    n_bits = bits.shape[0]

    def _mean_sim_to_background(b: bytes) -> float:
        total = sum(sim.cosine_packed(b, bg) for bg in background_hvs)
        return total / len(background_hvs)

    def _sim_to_target(b: bytes) -> float:
        return sim.cosine_packed(b, target)

    for _ in range(max_flips):
        current_packed = np.packbits(bits).tobytes()
        current_bg_sim = _mean_sim_to_background(current_packed)

        best_delta = 0.0   # we only accept flips that REDUCE mean sim
        best_idx = -1

        # Evaluate each candidate bit flip
        for i in range(n_bits):
            bits[i] ^= 1
            candidate = np.packbits(bits).tobytes()
            new_bg_sim = _mean_sim_to_background(candidate)
            delta = current_bg_sim - new_bg_sim  # positive = improvement
            if delta > best_delta:
                best_delta = delta
                best_idx = i
            bits[i] ^= 1  # revert

        if best_idx == -1:
            # No flip reduces background similarity further
            break

        # Apply the best flip, but check tau constraint
        bits[best_idx] ^= 1
        candidate = np.packbits(bits).tobytes()
        if _sim_to_target(candidate) < tau:
            # Revert: would drift too far from original structure
            bits[best_idx] ^= 1
            break

    return np.packbits(bits).tobytes()
