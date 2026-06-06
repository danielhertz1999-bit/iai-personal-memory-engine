"""Deterministic seeding helpers -- SHA256-derived 64-bit ints,
cross-process reproducible. Used by every tier backend to materialize
codebook vectors without randomness drift."""
from __future__ import annotations

import hashlib
import logging
import math

import numpy as np

log = logging.getLogger(__name__)


def seed_from_str(prefix: str, value: str) -> int:
    """Return a stable 64-bit unsigned seed derived from (prefix, value).

    Uses SHA256 of the UTF-8 encoding of ``prefix:value`` and takes the
    first 8 bytes big-endian. Deterministic across processes and platforms.

    Args:
        prefix: Namespace string (e.g. "tier-bsc-v1").
        value: Unique value within that namespace (e.g. a role symbol).

    Returns:
        Unsigned 64-bit integer in [0, 2**64).
    """
    digest = hashlib.sha256(f"{prefix}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def hv_from_seed(seed: int, D: int) -> bytes:
    """Generate a D-dimensional binary hypervector packed to ceil(D/8) bytes.

    Produces a uniformly random bit string seeded from ``seed``, packed via
    numpy packbits. The same (seed, D) always produces the same bytes in any
    process, on any platform, across numpy versions (numpy.random.default_rng
    guarantees cross-version stability for PCG64).

    Args:
        seed: Unsigned 64-bit seed (e.g. from seed_from_str).
        D: Hypervector dimensionality (positive integer).

    Returns:
        Packed bytes of length ceil(D / 8).

    Raises:
        ValueError: If D <= 0.

    Examples:
        At D=4096 returns 512 bytes.
        At D=10000 returns 1250 bytes.
        At D=2048 returns 256 bytes.
    """
    if D <= 0:
        raise ValueError(f"D must be a positive integer, got {D}")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=D, dtype=np.uint8)
    return np.packbits(bits).tobytes()
