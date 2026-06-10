from __future__ import annotations

import numpy as np

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa


def consolidate(
    hvs: list,
    tier: str = "bsc",
    *,
    D: int | None = None,
) -> bytes | list:
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
    if not hvs:
        effective_D = D if D is not None else bsc.LILLI_BSC_DEFAULT_DIM
        return bytes(effective_D // 8)

    effective_D = D if D is not None else len(hvs[0]) * 8

    if len(hvs) == 1:
        return hvs[0]

    arr = np.stack([np.frombuffer(hv, np.uint8) for hv in hvs])
    bits = np.unpackbits(arr, axis=1).astype(np.int32)
    sums = bits.sum(0)
    n = arr.shape[0]
    voted = (sums * 2 >= n).astype(np.uint8)
    return np.packbits(voted).tobytes()
