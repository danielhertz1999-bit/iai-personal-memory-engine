"""Auto-associative cleanup memory.

cleanup(noisy_hv, codebook) returns the codebook entry whose hamming distance
is minimum -- snaps a noisy hypervector back to its nearest canonical form.
Used by the CLUSTER_REPLAY sleep step to prevent bit-rot accumulation in
repeated replay cycles.

Imports only lilli.core.similarity and stdlib.
"""
from __future__ import annotations

from iai_mcp.lilli.core import similarity as sim


def cleanup(noisy_hv: bytes, codebook: list[bytes]) -> bytes:
    """Snap *noisy_hv* to the nearest entry in *codebook* by Hamming distance.

    1-NN lookup: returns the codebook entry with minimum normalized Hamming
    distance to *noisy_hv*. Ties are broken by input order (first entry wins),
    making the result deterministic.

    Parameters
    ----------
    noisy_hv:
        Packed binary hypervector to snap (query).
    codebook:
        Non-empty list of packed binary hypervectors (canonical forms). All
        entries must have the same byte length as *noisy_hv*.

    Returns
    -------
    bytes
        The codebook entry with minimum Hamming distance to *noisy_hv*.

    Raises
    ------
    ValueError
        If *codebook* is empty.
    ValueError
        If any codebook entry has a different length from *noisy_hv*.
    """
    if not codebook:
        raise ValueError("codebook must not be empty")

    for idx, entry in enumerate(codebook):
        if len(entry) != len(noisy_hv):
            raise ValueError(
                f"codebook entry {idx} length {len(entry)} does not match "
                f"noisy_hv length {len(noisy_hv)}"
            )

    # Pair each entry with its index so ties resolve to the first occurrence.
    best_dist, best_entry = min(
        (sim.hamming(noisy_hv, entry), i)
        for i, entry in enumerate(codebook)
    )
    return codebook[best_entry]
