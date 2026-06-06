"""Tier-agnostic similarity primitives -- Hamming distance, cosine similarity
for packed binary hypervectors, and Jaccard similarity for sparse VSA index sets."""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

log = logging.getLogger(__name__)


def hamming(a: bytes, b: bytes) -> float:
    """Return normalized Hamming distance between two packed binary hypervectors.

    Computes the fraction of differing bits in [0.0, 1.0]. A value of 0.0
    means the vectors are identical; 1.0 means every bit differs.

    Length-mismatch returns 1.0 (maximally distant) to degrade gracefully on
    corrupt or cross-tier comparisons rather than raising.

    Args:
        a: First packed-bits buffer.
        b: Second packed-bits buffer (must be same length for a valid result).

    Returns:
        Float in [0.0, 1.0].
    """
    if len(a) != len(b):
        return 1.0
    if len(a) == 0:
        return 0.0
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    xor = np.bitwise_xor(aa, bb)
    try:
        ham_bits = int(np.bitwise_count(xor).sum())
    except AttributeError:
        ham_bits = int(np.unpackbits(xor).sum())
    total_bits = len(a) * 8
    return ham_bits / total_bits


def cosine_packed(a: bytes, b: bytes) -> float:
    """Return cosine similarity between two packed binary hypervectors.

    Treats packed bits as a {-1, +1} BSC encoding: bit 0 maps to -1,
    bit 1 maps to +1. Returns the normalized dot product in [-1.0, 1.0].

    A value of 1.0 means identical vectors; -1.0 means perfectly opposite;
    0.0 means maximally orthogonal.

    Length-mismatch returns 0.0 (orthogonal) to degrade gracefully.

    Args:
        a: First packed-bits buffer.
        b: Second packed-bits buffer.

    Returns:
        Float in [-1.0, 1.0].
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    bits_a = np.unpackbits(np.frombuffer(a, dtype=np.uint8)).astype(np.float32)
    bits_b = np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.float32)
    # Map 0 -> -1, 1 -> +1
    bsc_a = bits_a * 2.0 - 1.0
    bsc_b = bits_b * 2.0 - 1.0
    D = len(bsc_a)
    return float(np.dot(bsc_a, bsc_b) / D)


def jaccard(
    a_indices: Union[set, list],
    b_indices: Union[set, list],
) -> float:
    """Return Jaccard similarity between two sparse VSA active-bit index sets.

    Computes |intersection| / |union|. Returns 0.0 if both sets are empty.

    Args:
        a_indices: Active bit indices for the first hypervector (set or list).
        b_indices: Active bit indices for the second hypervector (set or list).

    Returns:
        Float in [0.0, 1.0].
    """
    set_a = set(a_indices)
    set_b = set(b_indices)
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    intersection_size = len(set_a & set_b)
    return intersection_size / union_size
