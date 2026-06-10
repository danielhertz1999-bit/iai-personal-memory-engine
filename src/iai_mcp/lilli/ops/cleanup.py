from __future__ import annotations

from iai_mcp.lilli.core import similarity as sim


def cleanup(noisy_hv: bytes, codebook: list[bytes]) -> bytes:
    if not codebook:
        raise ValueError("codebook must not be empty")

    for idx, entry in enumerate(codebook):
        if len(entry) != len(noisy_hv):
            raise ValueError(
                f"codebook entry {idx} length {len(entry)} does not match "
                f"noisy_hv length {len(noisy_hv)}"
            )

    best_dist, best_entry = min(
        (sim.hamming(noisy_hv, entry), i)
        for i, entry in enumerate(codebook)
    )
    return codebook[best_entry]
