from __future__ import annotations

import hashlib
import logging

import numpy as np

from iai_mcp.lilli.core.seed import seed_from_str

log = logging.getLogger(__name__)

EMBED_DIM: int = 384

HV_DIM: int = 10000

_PROJECTION_SEED_NAME: str = "lilli-projection-v1"


_seed = seed_from_str("lilli-projection", _PROJECTION_SEED_NAME)
_rng = np.random.default_rng(_seed)

P: np.ndarray = _rng.standard_normal((EMBED_DIM, HV_DIM)).astype(np.float32)

assert P.shape == (384, 10000), f"P shape mismatch: {P.shape}"
assert P.dtype == np.float32, f"P dtype mismatch: {P.dtype}"

P_SHA256_HASH: str = "df97cc72a960567da17edbba16107881340349bd47b69f9b58d3091d96eb4e4e"

if P_SHA256_HASH != "BOOTSTRAP_PENDING":
    _actual_hash = hashlib.sha256(P.tobytes()).hexdigest()
    if _actual_hash != P_SHA256_HASH:
        raise RuntimeError(
            f"P matrix has drifted. "
            f"Expected hash {P_SHA256_HASH}, got {_actual_hash}. "
            f"This invalidates every stored hypervector across every Hippo store."
        )


def project(emb: np.ndarray) -> np.ndarray:
    if emb.shape != (384,):
        raise ValueError(
            f"project() expects emb.shape == (384,), got {emb.shape}"
        )
    if emb.dtype != np.float32:
        raise ValueError(
            f"project() expects emb.dtype == float32, got {emb.dtype}"
        )
    return emb @ P
