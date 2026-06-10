from __future__ import annotations

import struct
from collections import Counter
from typing import Sequence

import numpy as np

from iai_mcp.lilli.core.seed import seed_from_str


LILLI_SPARSE_DIM: int = 2048
"""Hypervector dimension.  All active-bit indices live in [0, LILLI_SPARSE_DIM)."""

SPARSE_K: int = 20
"""Number of active bits per HV (~1 % sparsity at D=2048)."""

SPARSE_ROLE_SEED_PREFIX: str = "lilli-sparse-role"
SPARSE_FILLER_SEED_PREFIX: str = "lilli-sparse-filler"

TIER_INFO: dict = {
    "backend": "sparse_vsa",
    "D": 2048,
    "bytes_per_hv": 40,
    "use_case": "procedural",
}
"""Tier metadata dictionary (read-only; consumed by tier_info() dispatcher)."""

_SENTINEL: int = 0xFFFF
"""Padding sentinel value used when pack_indices receives fewer than SPARSE_K
indices.  Stripped on unpack."""


def random_indices(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(LILLI_SPARSE_DIM, size=SPARSE_K, replace=False)
    return sorted(int(x) for x in indices)


def role_hv(role: str) -> list[int]:
    seed = seed_from_str(SPARSE_ROLE_SEED_PREFIX, role)
    return random_indices(seed)


def filler_hv(value: str) -> list[int]:
    seed = seed_from_str(SPARSE_FILLER_SEED_PREFIX, value)
    return random_indices(seed)


def pack_indices(indices: Sequence[int]) -> bytes:
    idx_list = list(indices)[:SPARSE_K]
    while len(idx_list) < SPARSE_K:
        idx_list.append(_SENTINEL)
    return struct.pack(f"<{SPARSE_K}H", *idx_list)


def unpack_indices(packed: bytes) -> list[int]:
    values = struct.unpack(f"<{SPARSE_K}H", packed)
    return [v for v in values if v != _SENTINEL]


def _pad_to_k(indices: list[int], seed_a: list[int], seed_b: list[int]) -> list[int]:
    existing = set(indices)
    needed = SPARSE_K - len(existing)
    if needed <= 0:
        return sorted(existing)[:SPARSE_K]

    a_int = hash(tuple(seed_a)) & 0xFFFF_FFFF_FFFF_FFFF
    b_int = hash(tuple(seed_b)) & 0xFFFF_FFFF_FFFF_FFFF
    xor_seed = a_int ^ b_int
    rng = np.random.default_rng(xor_seed)

    candidates = list(range(LILLI_SPARSE_DIM))
    rng.shuffle(candidates)
    for c in candidates:
        if len(existing) >= SPARSE_K:
            break
        existing.add(c)
    return sorted(existing)[:SPARSE_K]


def bind(a: list[int], b: list[int]) -> list[int]:
    merged = sorted(set(a) | set(b))
    if len(merged) >= SPARSE_K:
        return merged[:SPARSE_K]
    return _pad_to_k(merged, a, b)


def unbind(bound: list[int], key: list[int]) -> list[int]:
    remainder = sorted(set(bound) - set(key))
    if len(remainder) >= SPARSE_K:
        return remainder[:SPARSE_K]
    return _pad_to_k(remainder, bound, key)


def bundle(hvs: list[list[int]]) -> list[int]:
    if not hvs:
        return []
    counts: Counter = Counter()
    for hv in hvs:
        for idx in hv:
            counts[idx] += 1
    ranked = sorted(counts.keys(), key=lambda i: (-counts[i], i))
    return sorted(ranked[:SPARSE_K])


def permute(hv: list[int], shift: int) -> list[int]:
    return sorted((i + shift) % LILLI_SPARSE_DIM for i in hv)


def similarity(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    union_size = len(sa | sb)
    if union_size == 0:
        return 0.0
    return len(sa & sb) / union_size
