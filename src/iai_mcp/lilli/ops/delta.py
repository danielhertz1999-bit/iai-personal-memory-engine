"""Delta encoding for HV compaction chains.

delta_encode(canonical, variant) -> bytes(canonical XOR variant).

Storage saves on consolidation chains where many variants are minor edits of
one canonical -- only the delta bytes need to be persisted alongside a single
canonical_id reference. Round-trip identity holds because XOR is self-inverse:

    delta_decode(canonical, delta_encode(canonical, variant)) == variant

Imports only numpy and stdlib.
"""
from __future__ import annotations

import numpy as np


def delta_encode(canonical: bytes, variant: bytes) -> bytes:
    """Return the XOR difference between *canonical* and *variant*.

    Parameters
    ----------
    canonical:
        Reference packed binary hypervector.
    variant:
        Modified packed binary hypervector of the same length.

    Returns
    -------
    bytes
        XOR delta of the same length as *canonical*.

    Raises
    ------
    ValueError
        If *canonical* and *variant* have different lengths.
    """
    if len(canonical) != len(variant):
        raise ValueError(
            f"delta_encode requires equal-length hypervectors, "
            f"got {len(canonical)} and {len(variant)}"
        )
    aa = np.frombuffer(canonical, dtype=np.uint8)
    bb = np.frombuffer(variant, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


def delta_decode(canonical: bytes, delta_bytes: bytes) -> bytes:
    """Reconstruct *variant* from *canonical* and a previously computed delta.

    Because XOR is self-inverse, decoding is identical to encoding:
    canonical XOR (canonical XOR variant) == variant.

    Parameters
    ----------
    canonical:
        Reference packed binary hypervector.
    delta_bytes:
        Delta produced by:func:`delta_encode` (same length as *canonical*).

    Returns
    -------
    bytes
        Reconstructed variant hypervector.

    Raises
    ------
    ValueError
        If *canonical* and *delta_bytes* have different lengths.
    """
    return delta_encode(canonical, delta_bytes)


def delta_sparse_bits(delta_bytes: bytes) -> int:
    """Return the count of nonzero bits in *delta_bytes* (popcount).

    Useful for deciding whether a delta is small enough to be worth storing
    instead of the full variant. A delta with few set bits indicates a minor
    edit; a fully-set delta is no cheaper than storing the variant directly.

    Parameters
    ----------
    delta_bytes:
        Packed bytes, typically the output of:func:`delta_encode`.

    Returns
    -------
    int
        Number of bits set to 1 in *delta_bytes*.
    """
    arr = np.frombuffer(delta_bytes, dtype=np.uint8)
    try:
        return int(np.bitwise_count(arr).sum())
    except AttributeError:
        # numpy < 1.25 fallback
        return int(np.unpackbits(arr).sum())
