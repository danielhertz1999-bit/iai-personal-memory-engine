"""Sparse VSA tier -- procedural memory backend.

Rachkovskij-style SBDR at D=2048, ~1% sparsity (default SPARSE_K=20 active
bits). HVs are sorted lists of active-bit indices stored as 20 × uint16
little-endian (40 bytes per HV -- the smallest of the three tiers).

Binding is an APPROXIMATE union-truncate utility (sorted-union truncation; NOT
true Rachkovskij Context-Dependent Thinning -- true CDT requires
context-key-driven re-sampling with deterministic thinning that this module
does not implement). A future plan may upgrade to true CDT if procedural-tier
capacity becomes a constraint.

Operations at a glance
----------------------
* ``role_hv(role)`` -- deterministic codebook HV for a role symbol
* ``filler_hv(value)`` -- deterministic codebook HV for a filler value
* ``bind(a, b)`` -- APPROXIMATE binding (union-truncate)
* ``unbind(bound, key)`` -- APPROXIMATE, lossy inverse of bind
* ``bundle(hvs)`` -- frequency-weighted superposition
* ``permute(hv, shift)`` -- index-shift permutation
* ``similarity(a, b)`` -- Jaccard on active-index sets
* ``pack_indices(indices)`` -- encode as 40-byte uint16 LE blob
* ``unpack_indices(packed)``-- decode blob back to sorted index list

Designed for fast updates and sparse activation; matches motor-skill rehearsal
cycles.
"""
from __future__ import annotations

import struct
from collections import Counter
from typing import Sequence

import numpy as np

from iai_mcp.lilli.core.seed import seed_from_str

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LILLI_SPARSE_DIM: int = 2048
"""Hypervector dimension.  All active-bit indices live in [0, LILLI_SPARSE_DIM)."""

SPARSE_K: int = 20
"""Number of active bits per HV (~1 % sparsity at D=2048)."""

SPARSE_ROLE_SEED_PREFIX: str = "lilli-sparse-role"
SPARSE_FILLER_SEED_PREFIX: str = "lilli-sparse-filler"

TIER_INFO: dict = {
    "backend": "sparse_vsa",
    "D": 2048,
    "bytes_per_hv": 40,   # SPARSE_K × sizeof(uint16) = 20 × 2
    "use_case": "procedural",
}
"""Tier metadata dictionary (read-only; consumed by tier_info() dispatcher)."""

_SENTINEL: int = 0xFFFF
"""Padding sentinel value used when pack_indices receives fewer than SPARSE_K
indices.  Stripped on unpack."""


# ---------------------------------------------------------------------------
# Core HV generation
# ---------------------------------------------------------------------------

def random_indices(seed: int) -> list[int]:
    """Sample SPARSE_K unique indices from [0, LILLI_SPARSE_DIM) for a seed.

    Uses ``numpy.random.default_rng`` (PCG64) for cross-process stability.
    Output is always sorted ascending with no duplicates.

    Args:
        seed: 64-bit unsigned integer seed (e.g. from ``seed_from_str``).

    Returns:
        Sorted list of SPARSE_K unique integers in [0, LILLI_SPARSE_DIM).
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(LILLI_SPARSE_DIM, size=SPARSE_K, replace=False)
    return sorted(int(x) for x in indices)


def role_hv(role: str) -> list[int]:
    """Return the deterministic codebook HV for a role symbol.

    The returned list has exactly SPARSE_K elements, sorted ascending, all in
    [0, LILLI_SPARSE_DIM). Deterministic across processes.

    Args:
        role: Role name string (e.g. ``"WHEN"``, ``"WHO"``).

    Returns:
        Sorted list of SPARSE_K unique integers.
    """
    seed = seed_from_str(SPARSE_ROLE_SEED_PREFIX, role)
    return random_indices(seed)


def filler_hv(value: str) -> list[int]:
    """Return the deterministic codebook HV for a filler value.

    Args:
        value: Filler string (e.g. a memory content fragment).

    Returns:
        Sorted list of SPARSE_K unique integers.
    """
    seed = seed_from_str(SPARSE_FILLER_SEED_PREFIX, value)
    return random_indices(seed)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def pack_indices(indices: Sequence[int]) -> bytes:
    """Encode active-bit indices as a fixed-length 40-byte blob (SPARSE_K × uint16 LE).

    Truncates to SPARSE_K if more indices are given. Pads with 0xFFFF
    sentinels if fewer.

    Args:
        indices: Iterable of non-negative integers, each < LILLI_SPARSE_DIM.

    Returns:
        Exactly 40 bytes.
    """
    idx_list = list(indices)[:SPARSE_K]
    # Pad with sentinels if needed
    while len(idx_list) < SPARSE_K:
        idx_list.append(_SENTINEL)
    return struct.pack(f"<{SPARSE_K}H", *idx_list)


def unpack_indices(packed: bytes) -> list[int]:
    """Decode a 40-byte blob back to a sorted list of active-bit indices.

    Strips 0xFFFF sentinel values produced by ``pack_indices``.

    Args:
        packed: Exactly 40 bytes (SPARSE_K × uint16 little-endian).

    Returns:
        Sorted list of unique integers; length may be < SPARSE_K if padded.
    """
    values = struct.unpack(f"<{SPARSE_K}H", packed)
    return [v for v in values if v != _SENTINEL]


# ---------------------------------------------------------------------------
# Algebraic operations
# ---------------------------------------------------------------------------

def _pad_to_k(indices: list[int], seed_a: list[int], seed_b: list[int]) -> list[int]:
    """Deterministically pad *indices* up to SPARSE_K when union is small.

    This happens only when set(a) | set(b) has < SPARSE_K elements, which
    occurs when a and b share many indices. A deterministic re-sample
    derived from the XOR-hash of (a, b) fills the gap without introducing
    process-level randomness.

    Returns exactly SPARSE_K indices (sorted, deduplicated, in range).
    """
    existing = set(indices)
    needed = SPARSE_K - len(existing)
    if needed <= 0:
        return sorted(existing)[:SPARSE_K]

    # Build a deterministic seed from the bitwise XOR of both lists' integer
    # representations so the padding is symmetric and reproducible.
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
    """Approximate binding utility for sparse VSA HVs.

    Returns ``sorted(set(a) | set(b))[:SPARSE_K]``. NOT Rachkovskij
    Context-Dependent Thinning -- true CDT uses context-key-driven thinning
    with deterministic re-sampling. The simpler union-truncate preserves
    K-sparsity and symmetry without the context-key machinery. Use this for
    procedural-tier scratch storage; consult ``TIER_INFO`` for capacity
    guidance.

    The result always has exactly SPARSE_K elements. When the union has fewer
    than SPARSE_K elements (high overlap between a and b), the gap is filled
    via a deterministic re-sample seeded by xor-hash of (a, b) so the output
    is symmetric and reproducible.

    Args:
        a: First HV as sorted list of active-bit indices.
        b: Second HV as sorted list of active-bit indices.

    Returns:
        Sorted list of exactly SPARSE_K integers in [0, LILLI_SPARSE_DIM).
    """
    merged = sorted(set(a) | set(b))
    if len(merged) >= SPARSE_K:
        return merged[:SPARSE_K]
    return _pad_to_k(merged, a, b)


def unbind(bound: list[int], key: list[int]) -> list[int]:
    """Approximate, lossy inverse -- see bind() for the limits of this operation.

    Removes key indices from *bound*, then pads back to SPARSE_K via
    deterministic re-sample when needed. Because the union-truncate bind
    discards information (indices beyond position SPARSE_K in the union), this
    inverse is inherently lossy and is provided only as a best-effort
    approximation.

    Args:
        bound: HV produced by ``bind(filler, key)`` or similar.
        key: Key HV used during binding.

    Returns:
        Best-effort recovery HV of exactly SPARSE_K indices.
    """
    remainder = sorted(set(bound) - set(key))
    if len(remainder) >= SPARSE_K:
        return remainder[:SPARSE_K]
    return _pad_to_k(remainder, bound, key)


def bundle(hvs: list[list[int]]) -> list[int]:
    """Frequency-weighted superposition of a list of HVs.

    Each index in the union is counted across all input HVs. The top SPARSE_K
    by frequency are returned (ties broken by smaller index). An empty list
    returns ``[]``.

    Args:
        hvs: List of HVs (each a sorted list of active-bit indices).

    Returns:
        Sorted list of SPARSE_K indices, or ``[]`` if *hvs* is empty.
    """
    if not hvs:
        return []
    counts: Counter = Counter()
    for hv in hvs:
        for idx in hv:
            counts[idx] += 1
    # Sort by (-count, index) to get highest frequency, smallest index as tiebreak
    ranked = sorted(counts.keys(), key=lambda i: (-counts[i], i))
    return sorted(ranked[:SPARSE_K])


def permute(hv: list[int], shift: int) -> list[int]:
    """Index-shift permutation: return sorted((i + shift) % D for i in hv).

    Length-preserving cyclic shift of all active-bit indices modulo
    LILLI_SPARSE_DIM.

    Args:
        hv: HV as sorted list of active-bit indices.
        shift: Integer offset; may be negative.

    Returns:
        Sorted list of the same length as *hv*.
    """
    return sorted((i + shift) % LILLI_SPARSE_DIM for i in hv)


def similarity(a: list[int], b: list[int]) -> float:
    """Jaccard similarity between two sparse HVs.

    ``|intersection| / max(1, |union)``. Returns 0.0 when both inputs
    are empty.

    Args:
        a: First HV as sorted list of active-bit indices.
        b: Second HV as sorted list of active-bit indices.

    Returns:
        Float in [0.0, 1.0].
    """
    sa, sb = set(a), set(b)
    union_size = len(sa | sb)
    if union_size == 0:
        return 0.0
    return len(sa & sb) / union_size
