"""Pattern-separation primitives: HV-level rejection sampling + embedding-level
orthogonalization and hubness detection.

HV-level (``pattern_separate``):
    Re-salts a target hypervector via deterministic SHA256 perturbation until
    its maximum similarity to any background hypervector falls below threshold
    (default 0.85), or until max_retries (default 5) is exhausted.

    On exhaustion the function returns the BEST variant seen across all retry
    attempts — the candidate with the lowest maximum similarity to the
    background — rather than raising. This graceful degradation stores a
    modified trace rather than discarding the record.

    Upgrade path: at high record counts (>100K), Gram-Schmidt orthogonalization
    delivers guaranteed decorrelation without the probabilistic retry loop. The
    rejection-sampling approach here is efficient and sufficient at the
    current store scale.

    Telemetry: when a store kwarg is supplied and at least one retry was
    consumed, the function emits a ``separation_retries_used`` event.
    Pure-function callers (no store) see no I/O side-effects — same pattern as
    the BSC bundle saturation telemetry.

Embedding-level (``orthogonalize_for_routing``, ``detect_hubness``):
    Operates on dense float vectors before community assignment. These functions
    do NOT modify stored embeddings — the verbatim invariant on storage is
    preserved. ``OrthogonalizationResult`` is the companion dataclass returned
    by ``orthogonalize_for_routing``.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from iai_mcp.lilli.core import similarity as sim

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MAX_RETRIES_DEFAULT: int = 5
"""Maximum number of re-salting attempts before returning the best seen candidate."""

SIMILARITY_THRESHOLD_DEFAULT: float = 0.85
"""Maximum allowed similarity between the output and any background hypervector."""

# Telemetry event kind string — kept as a local constant so this module has no
# hard dependency on iai_mcp.events at import time (avoids circular imports).
_TELEMETRY_SEPARATION_RETRIES_KIND: str = "separation_retries_used"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def pattern_separate(
    target: bytes,
    background_hvs: list[bytes],
    *,
    max_retries: int = MAX_RETRIES_DEFAULT,
    similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    store: "Optional[MemoryStore]" = None,
) -> bytes:
    """Return a version of target decorrelated from background_hvs.

    Uses rejection sampling with deterministic SHA256-salted perturbation.
    Each retry flips progressively more bits (16 * attempt_number) to increase
    the probability of crossing the similarity threshold.

    Args:
        target: Packed binary hypervector to separate.
        background_hvs: List of packed binary hypervectors representing
                              already-stored patterns. Empty list returns target
                              unchanged (no separation needed).
        max_retries: Maximum number of perturbation attempts before
                              returning the best candidate. Default 5.
        similarity_threshold: Similarity ceiling. A candidate is accepted when
                              its maximum similarity to all background_hvs is
                              strictly less than this value. Default 0.85.
        store: Optional MemoryStore. When supplied, emits a
                              `separation_retries_used` telemetry event if any
                              retries were consumed. Pass None (default) for
                              pure-function callers.

    Returns:
        Packed bytes of the same length as target. Equal to target when
        background_hvs is empty or when target already satisfies the threshold
        on attempt 0. Otherwise a perturbed variant with lower maximum
        similarity to the background.
    """
    if not background_hvs:
        return target

    def _max_sim(hv: bytes) -> float:
        """Return max(1 - hamming(hv, b)) across all background vectors."""
        return max(1.0 - sim.hamming(hv, b) for b in background_hvs)

    # Attempt 0: check the original target before any perturbation.
    candidate = target
    best_candidate = target
    best_max_sim = _max_sim(target)

    if best_max_sim < similarity_threshold:
        # Early accept — target is already sufficiently decorrelated.
        return target

    retries_used = 0

    for attempt in range(1, max_retries + 1):
        # Deterministic salt: SHA256 of attempt index string concatenated with
        # the raw target bytes. Produces reproducible perturbations for the same
        # (attempt, target) pair across calls.
        salt_seed = hashlib.sha256(
            f"separate-attempt-{attempt}".encode() + target
        ).digest()[:8]
        seed_int = int.from_bytes(salt_seed, "big")
        rng = np.random.default_rng(seed_int)

        # Progressive flip count: earlier attempts flip fewer bits (smaller
        # perturbation → higher chance of preserving useful structure);
        # later attempts flip more to escape tight background clusters.
        flip_count = attempt * 16

        bits = np.unpackbits(np.frombuffer(target, dtype=np.uint8))
        total_bits = bits.shape[0]
        # Clamp flip_count so we never request more indices than available bits.
        actual_flips = min(flip_count, total_bits)
        flip_indices = rng.choice(total_bits, size=actual_flips, replace=False)
        bits[flip_indices] ^= 1
        candidate = np.packbits(bits).tobytes()

        retries_used = attempt
        candidate_max_sim = _max_sim(candidate)

        if candidate_max_sim < best_max_sim:
            best_candidate = candidate
            best_max_sim = candidate_max_sim

        if candidate_max_sim < similarity_threshold:
            # Accepted — emit telemetry if store is attached.
            _emit_retries_telemetry(store, retries_used, best_max_sim)
            return candidate

    # Exhausted all retries. Return the best candidate found (lowest
    # max-similarity to background). Graceful degradation: never raises.
    _emit_retries_telemetry(store, retries_used, best_max_sim)
    return best_candidate


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _emit_retries_telemetry(
    store: "Optional[MemoryStore]",
    retries_used: int,
    final_max_sim: float,
) -> None:
    """Emit separation_retries_used telemetry when store is attached.

    Wrapped in a broad try/except — telemetry must never crash the separation
    path. Matches the same safety contract used in BSC bundle saturation telemetry.
    """
    if store is None or retries_used == 0:
        return
    try:
        from iai_mcp import events  # deferred import — avoids hard dependency loop

        events.write_event(
            store,
            _TELEMETRY_SEPARATION_RETRIES_KIND,
            {
                "retries_used": retries_used,
                "final_max_sim": final_max_sim,
            },
            domain="lilli.ops.separation",
        )
    except Exception:  # noqa: BLE001 — telemetry must never crash separation
        log.warning("separation_retries_used telemetry emit failed (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Embedding-level separation helpers
# ---------------------------------------------------------------------------


@dataclass
class OrthogonalizationResult:
    """Metrics returned by ``orthogonalize_for_routing``."""

    original_cos_mean: float
    orthogonalized_cos_mean: float
    separation_gain: float
    neighbors_used: int


def orthogonalize_for_routing(
    vec: list[float],
    neighbor_vecs: list[list[float]],
    strength: float = 0.3,
) -> tuple[list[float], OrthogonalizationResult]:
    """Subtract a fraction of the neighbor mean from a routing vector.

    Does NOT modify the stored embedding (verbatim invariant on storage).
    Returns a routing-only vector used for community assignment.

    The strength parameter controls how much of the neighbor mean is
    subtracted. Higher values produce more separation but risk semantic drift.
    0.3 is empirically safe for 384d bge-small embeddings.

    Args:
        vec: Dense float vector to orthogonalize (routing copy, not
                       the stored embedding).
        neighbor_vecs: Nearest-neighbor vectors used to compute the mean
                       direction to suppress. Empty list returns vec unchanged.
        strength: Fraction of the projected component to subtract.
                       Default 0.3.

    Returns:
        Tuple of (routing_vector, OrthogonalizationResult). The routing vector
        is unit-normalized. If the input list is empty or the computed mean
        norm is below 1e-8, the original vec is returned unchanged.
    """
    if not neighbor_vecs:
        return vec, OrthogonalizationResult(0.0, 0.0, 0.0, 0)

    v = np.array(vec, dtype=np.float32)
    neighbors = np.array(neighbor_vecs, dtype=np.float32)

    neighbor_mean = neighbors.mean(axis=0)
    norm_nm = np.linalg.norm(neighbor_mean)
    if norm_nm < 1e-8:
        return vec, OrthogonalizationResult(0.0, 0.0, 0.0, len(neighbor_vecs))

    neighbor_mean_unit = neighbor_mean / norm_nm

    original_cos = float(np.dot(v, neighbor_mean_unit))

    projection = np.dot(v, neighbor_mean_unit) * neighbor_mean_unit
    orthogonal = v - strength * projection

    norm_o = np.linalg.norm(orthogonal)
    if norm_o < 1e-8:
        return vec, OrthogonalizationResult(original_cos, original_cos, 0.0, len(neighbor_vecs))

    orthogonal_unit = orthogonal / norm_o
    new_cos = float(np.dot(orthogonal_unit, neighbor_mean_unit))

    return orthogonal_unit.tolist(), OrthogonalizationResult(
        original_cos_mean=original_cos,
        orthogonalized_cos_mean=new_cos,
        separation_gain=original_cos - new_cos,
        neighbors_used=len(neighbor_vecs),
    )


def detect_hubness(
    community_embeddings: list[list[float]],
    threshold: float = 0.85,
) -> dict:
    """Detect hubness in a community by measuring intra-cluster similarity.

    High mean cosine similarity within a community indicates potential
    information collapse — the community is attracting too many diverse
    memories, acting as a retrieval bottleneck.

    Args:
        community_embeddings: List of dense float vectors for all members of
                              a community. Fewer than 2 vectors returns zeros.
        threshold: Mean similarity above which the community is
                              flagged as a hub. Default 0.85.

    Returns:
        dict with keys: mean_similarity (float), max_similarity (float),
        is_hub (bool), size (int).
    """
    if len(community_embeddings) < 2:
        return {
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "is_hub": False,
            "size": len(community_embeddings),
        }

    vecs = np.array(community_embeddings, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = vecs / norms

    sim_matrix = normalized @ normalized.T
    np.fill_diagonal(sim_matrix, 0.0)

    n = len(community_embeddings)
    mean_sim = float(sim_matrix.sum() / (n * (n - 1)))
    max_sim = float(sim_matrix.max())

    return {
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,
        "is_hub": mean_sim > threshold,
        "size": n,
    }
