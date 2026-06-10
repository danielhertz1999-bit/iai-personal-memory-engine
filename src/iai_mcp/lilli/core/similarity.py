from __future__ import annotations

import logging
from typing import Union

import numpy as np

log = logging.getLogger(__name__)


def hamming(a: bytes, b: bytes) -> float:
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
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    bits_a = np.unpackbits(np.frombuffer(a, dtype=np.uint8)).astype(np.float32)
    bits_b = np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.float32)
    bsc_a = bits_a * 2.0 - 1.0
    bsc_b = bits_b * 2.0 - 1.0
    D = len(bsc_a)
    return float(np.dot(bsc_a, bsc_b) / D)


def jaccard(
    a_indices: Union[set, list],
    b_indices: Union[set, list],
) -> float:
    set_a = set(a_indices)
    set_b = set(b_indices)
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    intersection_size = len(set_a & set_b)
    return intersection_size / union_size
