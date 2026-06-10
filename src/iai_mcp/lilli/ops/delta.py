from __future__ import annotations

import numpy as np


def delta_encode(canonical: bytes, variant: bytes) -> bytes:
    if len(canonical) != len(variant):
        raise ValueError(
            f"delta_encode requires equal-length hypervectors, "
            f"got {len(canonical)} and {len(variant)}"
        )
    aa = np.frombuffer(canonical, dtype=np.uint8)
    bb = np.frombuffer(variant, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


def delta_decode(canonical: bytes, delta_bytes: bytes) -> bytes:
    return delta_encode(canonical, delta_bytes)


def delta_sparse_bits(delta_bytes: bytes) -> int:
    arr = np.frombuffer(delta_bytes, dtype=np.uint8)
    try:
        return int(np.bitwise_count(arr).sum())
    except AttributeError:
        return int(np.unpackbits(arr).sum())
