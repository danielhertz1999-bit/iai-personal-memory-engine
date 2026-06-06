"""Fourier HRR tier -- semantic memory backend. Continuous phase representation at
D=10000, quantized to uint8 (256 phase bins) for storage efficiency.

bind = element-wise phase addition mod 256.
unbind = phase subtraction mod 256.
bundle = circular mean via cos/sin accumulation.

Round-trip (bind then unbind by the same key) is exact bytewise.
Designed for smooth similarity gradients -- closer concepts produce closer
hypervectors.
"""
from __future__ import annotations

import math

import numpy as np

from iai_mcp.lilli.core.seed import seed_from_str

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

LILLI_FHRR_DIM: int = 10000
"""Hypervector dimensionality for the FHRR semantic tier."""

FHRR_PHASE_BINS: int = 256
"""Number of phase quantisation bins (uint8 = 256 bins covering [0, 2pi))."""

FHRR_ROLE_SEED_PREFIX: str = "lilli-fhrr-role"
"""Namespace prefix used when seeding role hypervectors."""

FHRR_FILLER_SEED_PREFIX: str = "lilli-fhrr-filler"
"""Namespace prefix used when seeding filler (value) hypervectors."""

TIER_INFO: dict[str, object] = {
    "backend": "fhrr",
    "D": 10000,
    "bytes_per_hv": 10000,
    "use_case": "semantic",
}
"""Tier metadata dictionary consumed by lilli.tier_info() dispatcher."""

# Precomputed constant for radian conversion.
_TWO_PI: float = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def random_hv(seed: int) -> bytes:
    """Return a random uint8 phase hypervector of length LILLI_FHRR_DIM.

    Uses numpy's PCG64 generator seeded from ``seed`` for cross-process
    reproducibility. Each byte encodes a phase angle uniformly distributed
    in [0, 256) (i.e. [0, 2pi) in FHRR_PHASE_BINS quantisation).

    Args:
        seed: Unsigned 64-bit seed (e.g. from seed_from_str).

    Returns:
        bytes of length 10000.
    """
    rng = np.random.default_rng(seed)
    phases = rng.integers(0, 256, size=LILLI_FHRR_DIM, dtype=np.uint8)
    return phases.tobytes()


def role_hv(role: str) -> bytes:
    """Return a deterministic uint8 phase hypervector for the given role symbol.

    Args:
        role: Role name (e.g. "WHEN", "WHO", "WHAT").

    Returns:
        bytes of length 10000.
    """
    seed = seed_from_str(FHRR_ROLE_SEED_PREFIX, role)
    return random_hv(seed)


def filler_hv(value: str) -> bytes:
    """Return a deterministic uint8 phase hypervector for the given value string.

    Args:
        value: Value string to encode.

    Returns:
        bytes of length 10000.
    """
    seed = seed_from_str(FHRR_FILLER_SEED_PREFIX, value)
    return random_hv(seed)


# ---------------------------------------------------------------------------
# Algebraic operations
# ---------------------------------------------------------------------------


def bind(a: bytes, b: bytes) -> bytes:
    """Element-wise phase addition modulo 256.

    Implements FHRR binding: each byte position adds phase angles modulo 2pi
    (represented as modulo 256 in uint8 quantisation).

    Args:
        a: First hypervector as bytes.
        b: Second hypervector as bytes.

    Returns:
        Bound hypervector as bytes, same length as inputs.

    Raises:
        ValueError: If len(a) != len(b).
    """
    if len(a) != len(b):
        raise ValueError(
            f"bind: length mismatch -- len(a)={len(a)}, len(b)={len(b)}"
        )
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    result = (aa.astype(np.uint16) + bb.astype(np.uint16)) & 0xFF
    return result.astype(np.uint8).tobytes()


def unbind(bound: bytes, key: bytes) -> bytes:
    """Element-wise phase subtraction modulo 256.

    Inverse of bind: ``unbind(bind(f, k), k) == f`` exactly for all
    hypervectors of the same length.

    Args:
        bound: Bound hypervector (output of bind).
        key: Key hypervector used in bind.

    Returns:
        Recovered hypervector as bytes.

    Raises:
        ValueError: If len(bound) != len(key).
    """
    if len(bound) != len(key):
        raise ValueError(
            f"unbind: length mismatch -- len(bound)={len(bound)}, len(key)={len(key)}"
        )
    bb = np.frombuffer(bound, dtype=np.uint8)
    kk = np.frombuffer(key, dtype=np.uint8)
    result = (bb.astype(np.int16) - kk.astype(np.int16)) & 0xFF
    return result.astype(np.uint8).tobytes()


def bundle(hvs: list[bytes]) -> bytes:
    """Phase-coherent superposition of hypervectors via circular mean.

    For each position i, computes the circular mean of the phase angles
    across all input hypervectors using cos/sin accumulation then atan2:

        angle_rad_j = (uint8[i][j] / 256) * 2pi
        c_i = sum_j cos(angle_rad_j)
        s_i = sum_j sin(angle_rad_j)
        result_rad_i = atan2(s_i, c_i) # in (-pi, pi]
        result_rad_i normalised to [0, 2pi)
        result_uint8_i = int((result_rad_i / 2pi) * 256) & 0xFF

    Args:
        hvs: List of hypervectors as bytes. All must have the same length.

    Returns:
        Bundled hypervector as bytes. Empty list returns bytes(LILLI_FHRR_DIM).
    """
    if not hvs:
        return bytes(LILLI_FHRR_DIM)

    # Fast path: single vector — circular mean of one phase is the phase itself.
    if len(hvs) == 1:
        return hvs[0]

    # Stack into 2-D array: shape (N, D), dtype uint8.
    mat = np.frombuffer(b"".join(hvs), dtype=np.uint8).reshape(len(hvs), -1)
    D = mat.shape[1]

    # Convert to radians: shape (N, D), float64.
    radians = (mat.astype(np.float64) / 256.0) * _TWO_PI

    # Circular mean via cos/sin accumulation.
    c = np.cos(radians).sum(axis=0)  # shape (D)
    s = np.sin(radians).sum(axis=0)  # shape (D)
    mean_angle = np.arctan2(s, c)    # shape (D), range (-pi, pi]

    # Normalise to [0, 2pi).
    mean_angle = mean_angle % _TWO_PI

    # Quantise to uint8.
    quantised = (mean_angle / _TWO_PI * 256.0).astype(np.int32) & 0xFF
    return quantised.astype(np.uint8).tobytes()


def permute(hv: bytes, shift: int) -> bytes:
    """Cyclic byte-permutation of a hypervector.

    Equivalent to numpy.roll: positive shift moves bytes towards higher indices
    (right roll); negative shift moves them towards lower indices (left roll).
    Round-trip: ``permute(permute(hv, k), -k) == hv`` for any k.

    Args:
        hv: Hypervector as bytes.
        shift: Number of positions to roll.

    Returns:
        Permuted hypervector as bytes, same length as input.
    """
    arr = np.frombuffer(hv, dtype=np.uint8)
    return np.roll(arr, shift).tobytes()


def similarity(a: bytes, b: bytes) -> float:
    """Mean cosine of element-wise phase differences.

    Returns a value in [-1.0, 1.0]:
    - 1.0 for identical hypervectors (zero phase difference everywhere).
    - ~0.0 for random unrelated hypervectors at D=10000.
    - -1.0 for perfectly anti-phase hypervectors.

    Returns 0.0 on length mismatch (graceful degradation).

    Args:
        a: First hypervector as bytes.
        b: Second hypervector as bytes.

    Returns:
        float in [-1.0, 1.0].
    """
    if len(a) != len(b):
        return 0.0
    aa_rad = (np.frombuffer(a, dtype=np.uint8).astype(np.float64) / 256.0) * _TWO_PI
    bb_rad = (np.frombuffer(b, dtype=np.uint8).astype(np.float64) / 256.0) * _TWO_PI
    diff = aa_rad - bb_rad
    return float(np.cos(diff).mean())
