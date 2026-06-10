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
    if not background_hvs:
        return target

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

        best_delta = 0.0
        best_idx = -1

        for i in range(n_bits):
            bits[i] ^= 1
            candidate = np.packbits(bits).tobytes()
            new_bg_sim = _mean_sim_to_background(candidate)
            delta = current_bg_sim - new_bg_sim
            if delta > best_delta:
                best_delta = delta
                best_idx = i
            bits[i] ^= 1

        if best_idx == -1:
            break

        bits[best_idx] ^= 1
        candidate = np.packbits(bits).tobytes()
        if _sim_to_target(candidate) < tau:
            bits[best_idx] ^= 1
            break

    return np.packbits(bits).tobytes()
