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


MAX_RETRIES_DEFAULT: int = 5
"""Maximum number of re-salting attempts before returning the best seen candidate."""

SIMILARITY_THRESHOLD_DEFAULT: float = 0.85
"""Maximum allowed similarity between the output and any background hypervector."""

_TELEMETRY_SEPARATION_RETRIES_KIND: str = "separation_retries_used"


def pattern_separate(
    target: bytes,
    background_hvs: list[bytes],
    *,
    max_retries: int = MAX_RETRIES_DEFAULT,
    similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
    store: "Optional[MemoryStore]" = None,
) -> bytes:
    if not background_hvs:
        return target

    def _max_sim(hv: bytes) -> float:
        return max(1.0 - sim.hamming(hv, b) for b in background_hvs)

    candidate = target
    best_candidate = target
    best_max_sim = _max_sim(target)

    if best_max_sim < similarity_threshold:
        return target

    retries_used = 0

    for attempt in range(1, max_retries + 1):
        salt_seed = hashlib.sha256(
            f"separate-attempt-{attempt}".encode() + target
        ).digest()[:8]
        seed_int = int.from_bytes(salt_seed, "big")
        rng = np.random.default_rng(seed_int)

        flip_count = attempt * 16

        bits = np.unpackbits(np.frombuffer(target, dtype=np.uint8))
        total_bits = bits.shape[0]
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
            _emit_retries_telemetry(store, retries_used, best_max_sim)
            return candidate

    _emit_retries_telemetry(store, retries_used, best_max_sim)
    return best_candidate


def _emit_retries_telemetry(
    store: "Optional[MemoryStore]",
    retries_used: int,
    final_max_sim: float,
) -> None:
    if store is None or retries_used == 0:
        return
    try:
        from iai_mcp import events

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


@dataclass
class OrthogonalizationResult:

    original_cos_mean: float
    orthogonalized_cos_mean: float
    separation_gain: float
    neighbors_used: int


def orthogonalize_for_routing(
    vec: list[float],
    neighbor_vecs: list[list[float]],
    strength: float = 0.3,
) -> tuple[list[float], OrthogonalizationResult]:
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
