"""Fixed random projection matrix. P maps 384-dim embeddings into the
10000-dim hypervector space via SimHash (sign of emb @ P). Seed is SHA256-derived
from the string 'lilli-projection-v1'. NEVER REGENERATED -- cross-session identity
persistence requires HV(today) == HV(2 years ago) for the same input embedding.
Any change to the seed string or the generation procedure invalidates every stored
hypervector across every Hippo store on the planet. Treat this module as append-only."""
from __future__ import annotations

import hashlib
import logging

import numpy as np

from iai_mcp.lilli.core.seed import seed_from_str

log = logging.getLogger(__name__)

# Embedding dimensionality -- matches bge-small-en-v1.5 output size.
EMBED_DIM: int = 384

# Hypervector dimensionality for cross-modal projection. This is the ONE place
# in lilli/ where 10000 is hardcoded for cross-modal compatibility reasons:
# the FHRR semantic tier uses D=10000, and every stored HV was projected through
# this same matrix. Changing this value invalidates all stored hypervectors.
HV_DIM: int = 10000

# Seed name string. SHA256 of this string (with "lilli-projection" as prefix)
# determines the entire P matrix. NEVER change this string.
_PROJECTION_SEED_NAME: str = "lilli-projection-v1"

# ---------------------------------------------------------------------------
# P matrix generation (runs ONCE at module import, then is frozen).
#
# Bootstrap procedure (one-time, already performed):
# 1. The module was initially shipped with P_SHA256_HASH = "BOOTSTRAP_PENDING".
# 2. The hash was computed by running:
# python -c "from iai_mcp.lilli.core.projection import P; \
# import hashlib; print(hashlib.sha256(P.tobytes()).hexdigest())"
# 3. The printed 64-hex-char digest was pasted in as the literal below.
# This value is now LOCKED. Any regeneration that produces a different hash
# means P has drifted.
# ---------------------------------------------------------------------------

_seed = seed_from_str("lilli-projection", _PROJECTION_SEED_NAME)
_rng = np.random.default_rng(_seed)

#: SHA256-locked random projection matrix.
#: Shape (384, 10000), dtype float32.
#: Generated ONCE from a fixed SHA256-derived seed. Never modified.
P: np.ndarray = _rng.standard_normal((EMBED_DIM, HV_DIM)).astype(np.float32)

assert P.shape == (384, 10000), f"P shape mismatch: {P.shape}"
assert P.dtype == np.float32, f"P dtype mismatch: {P.dtype}"

# P_SHA256_HASH is the locked full-tensor digest of P.tobytes().
# This constant was computed on first bootstrap and is locked forever.
# Any regeneration that produces a different 64-hex digest indicates drift.
P_SHA256_HASH: str = "df97cc72a960567da17edbba16107881340349bd47b69f9b58d3091d96eb4e4e"

# ---------------------------------------------------------------------------
# Drift guard: every import recomputes the hash and asserts
# equality with the locked constant. If P drifts (different numpy version,
# different seed string, different RNG sequence), this raises immediately.
# ---------------------------------------------------------------------------
if P_SHA256_HASH != "BOOTSTRAP_PENDING":
    _actual_hash = hashlib.sha256(P.tobytes()).hexdigest()
    if _actual_hash != P_SHA256_HASH:
        raise RuntimeError(
            f"P matrix has drifted. "
            f"Expected hash {P_SHA256_HASH}, got {_actual_hash}. "
            f"This invalidates every stored hypervector across every Hippo store."
        )


def project(emb: np.ndarray) -> np.ndarray:
    """Apply the fixed projection to a 384-dim embedding.

    Returns the raw dot product ``emb @ P`` as a float32 array of shape (10000).
    The caller applies sign + packbits to obtain a binary hypervector (that
    conversion lives in lilli/crossmodal).

    Args:
        emb: 1-D float32 array of shape (384).

    Returns:
        1-D float32 array of shape (10000).

    Raises:
        ValueError: If emb.shape != (384) or emb.dtype != float32.
    """
    if emb.shape != (384,):
        raise ValueError(
            f"project() expects emb.shape == (384,), got {emb.shape}"
        )
    if emb.dtype != np.float32:
        raise ValueError(
            f"project() expects emb.dtype == float32, got {emb.dtype}"
        )
    return emb @ P
