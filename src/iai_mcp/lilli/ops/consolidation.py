"""Episodic-to-semantic consolidation.

consolidate(hvs, tier) bundles N episodic hypervectors into one schema hypervector.
Tier-polymorphic -- BSC uses majority vote, FHRR uses circular mean, Sparse VSA uses
frequency-thinned union. This is the SCHEMA_MINE primitive: the sleep cycle calls it
to distill recurring patterns into stable schema records.

For BSC tier consolidation, raw packed-bit hvs are bundled directly via per-bit
majority vote (NOT via bsc.bundle(), which expects role-filler pairs and applies
the saturation guard). Tiebreak: bit=1 on even ties, matching bsc.bundle() semantics.
"""
from __future__ import annotations

import numpy as np

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa


def consolidate(
    hvs: list,
    tier: str = "bsc",
    *,
    D: int | None = None,
) -> bytes | list:
    """Bundle N episodic hypervectors into one schema hypervector.

    Tier-polymorphic dispatch:
    - ``"bsc"``: list[bytes] → bytes. Per-bit majority vote on raw packed-bit hvs.
                        D defaults to bsc.LILLI_BSC_DEFAULT_DIM; inferred from hvs[0] if
                        hvs is non-empty and D is None.
    - ``"fhrr"``: list[bytes] → bytes. Circular mean via fhrr.bundle(). D ignored.
    - ``"sparse_vsa"``: list[list[int]] → list[int]. Frequency-thinned union via
                        sparse_vsa.bundle(). D ignored.

    Empty hvs list behaviour:
    - bsc: returns bytes(D // 8) [default D=bsc.LILLI_BSC_DEFAULT_DIM → 512 bytes]
    - fhrr: returns bytes(fhrr.LILLI_FHRR_DIM)
    - sparse_vsa: returns []

    Args:
        hvs: List of hypervectors. Type depends on tier (see above).
        tier: Tier name string. One of ``"bsc"``, ``"fhrr"``, ``"sparse_vsa"``.
        D: BSC dimensionality override. Ignored for FHRR and Sparse VSA.
              If None (default) and tier is ``"bsc"``, inferred from hvs[0] length
              when hvs is non-empty; defaults to LILLI_BSC_DEFAULT_DIM when empty.

    Returns:
        Schema hypervector in the format appropriate for the tier.

    Raises:
        ValueError: If tier is not one of the three recognised tier names.
    """
    if tier == "bsc":
        return _consolidate_bsc(hvs, D=D)
    elif tier == "fhrr":
        return fhrr.bundle(hvs)
    elif tier == "sparse_vsa":
        return sparse_vsa.bundle(hvs)
    else:
        raise ValueError(
            f"consolidate: unknown tier '{tier}'. "
            f"Expected one of: 'bsc', 'fhrr', 'sparse_vsa'."
        )


def _consolidate_bsc(hvs: list[bytes], *, D: int | None) -> bytes:
    """Per-bit majority vote on raw BSC packed-bit hvs.

    This is a direct majority-vote operation on the raw hv bytes, NOT a call to
    bsc.bundle() (which expects role-filler pairs). The semantics are identical
    to bsc.bundle()'s voting step: bit=1 when more than half the inputs have
    bit=1; ties go to 1.

    Args:
        hvs: List of packed-bit BSC hypervectors (bytes). All must have the same length.
        D: Dimensionality override. Inferred from hvs[0] if None and hvs is non-empty.

    Returns:
        Packed bytes of length D // 8.
    """
    if not hvs:
        effective_D = D if D is not None else bsc.LILLI_BSC_DEFAULT_DIM
        return bytes(effective_D // 8)

    # Infer D from the first hv if not explicitly provided.
    effective_D = D if D is not None else len(hvs[0]) * 8

    # Fast path: single vector.
    if len(hvs) == 1:
        return hvs[0]

    # Per-bit majority vote on raw packed-bit hvs.
    arr = np.stack([np.frombuffer(hv, np.uint8) for hv in hvs])  # (N, D//8)
    bits = np.unpackbits(arr, axis=1).astype(np.int32)            # (N, D)
    sums = bits.sum(0)
    n = arr.shape[0]
    # Tiebreak: bit=1 when sums * 2 >= n (same as bsc.bundle majority contract).
    voted = (sums * 2 >= n).astype(np.uint8)
    return np.packbits(voted).tobytes()
