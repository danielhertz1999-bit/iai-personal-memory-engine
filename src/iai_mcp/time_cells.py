from __future__ import annotations

import hashlib
import logging
import struct
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

TEMPORAL_DIM = 128
TEMPORAL_RESOLUTION_SEC = 60


def _hash_to_bipolar(seed: bytes, dim: int = TEMPORAL_DIM) -> np.ndarray:
    rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
    return (rng.random(dim) > 0.5).astype(np.float32) * 2 - 1


def _time_slot(ts: datetime) -> int:
    epoch = ts.timestamp()
    return int(epoch // TEMPORAL_RESOLUTION_SEC)


def compute_temporal_hash(
    session_id: str,
    timestamp: datetime,
    turn_number: int = 0,
) -> list[float]:
    session_vec = _hash_to_bipolar(
        hashlib.sha256(session_id.encode()).digest()
    )

    slot = _time_slot(timestamp)
    slot_bytes = struct.pack("<q", slot)
    time_vec = _hash_to_bipolar(
        hashlib.sha256(slot_bytes).digest()
    )

    turn_bytes = struct.pack("<i", turn_number)
    turn_vec = _hash_to_bipolar(
        hashlib.sha256(turn_bytes).digest()
    )

    combined = 0.5 * session_vec + 0.3 * time_vec + 0.2 * turn_vec

    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined = combined / norm

    return combined.tolist()


def temporal_similarity(hash_a: list[float], hash_b: list[float]) -> float:
    a = np.array(hash_a, dtype=np.float32)
    b = np.array(hash_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_temporal_neighbors(
    query_hash: list[float],
    candidate_hashes: list[tuple[str, list[float]]],
    top_k: int = 10,
    min_similarity: float = 0.3,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for record_id, hash_vec in candidate_hashes:
        sim = temporal_similarity(query_hash, hash_vec)
        if sim >= min_similarity:
            scored.append((record_id, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def reconstruct_sequence(
    seed_hash: list[float],
    all_hashes: list[tuple[str, list[float], datetime]],
    window_minutes: int = 30,
) -> list[tuple[str, datetime, float]]:
    results: list[tuple[str, datetime, float]] = []
    for record_id, hash_vec, ts in all_hashes:
        sim = temporal_similarity(seed_hash, hash_vec)
        if sim >= 0.2:
            results.append((record_id, ts, sim))

    results.sort(key=lambda x: x[1])
    return results
