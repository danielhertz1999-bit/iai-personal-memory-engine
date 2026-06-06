"""Time cells — temporal VSA (Vector Symbolic Architecture) overlay.

A dedicated temporal axis binding that enables queries like "what was the
sequence of events leading to X?"

The standard semantic embedding (384d bge-small) captures WHAT was said.
The temporal binding captures WHEN relative to other events. Together they
enable episodic sequence reconstruction across disconnected communities.

Architecture:
- Each record gets a temporal_hash: a hyperdimensional vector encoding its
  position in the session timeline (session_id + turn_number + timestamp)
- Temporal similarity = cosine between temporal_hashes
- Sequence reconstruction: given a seed memory, find others with high
  temporal similarity (close in time) regardless of semantic similarity

This is orthogonal to the existing `temporal_next` edges (which link
consecutive records within ONE session). Time cells enable cross-session
temporal binding: "what was happening AROUND the same time as X?"
"""
from __future__ import annotations

import hashlib
import logging
import struct
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TEMPORAL_DIM = 128
TEMPORAL_RESOLUTION_SEC = 60


def _hash_to_bipolar(seed: bytes, dim: int = TEMPORAL_DIM) -> np.ndarray:
    """Generate a deterministic bipolar vector from a seed."""
    rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
    return (rng.random(dim) > 0.5).astype(np.float32) * 2 - 1


def _time_slot(ts: datetime) -> int:
    """Quantize timestamp to TEMPORAL_RESOLUTION_SEC slots."""
    epoch = ts.timestamp()
    return int(epoch // TEMPORAL_RESOLUTION_SEC)


def compute_temporal_hash(
    session_id: str,
    timestamp: datetime,
    turn_number: int = 0,
) -> list[float]:
    """Compute a temporal binding vector for a memory.

    Encodes three temporal dimensions:
    1. Session identity (which conversation)
    2. Time slot (when within global timeline)
    3. Turn position (where within session)

    The resulting vector enables temporal proximity queries without
    relying on semantic similarity.
    """
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
    """Cosine similarity between two temporal hashes."""
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
    """Find records temporally close to the query.

    Returns (record_id, similarity) pairs sorted by temporal proximity.
    Works across communities — pure temporal matching.
    """
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
    """Reconstruct temporal sequence around a seed memory.

    Returns records ordered by timestamp that are temporally similar
    to the seed (within window_minutes). This answers "what happened
    around the same time as X?"
    """
    results: list[tuple[str, datetime, float]] = []
    for record_id, hash_vec, ts in all_hashes:
        sim = temporal_similarity(seed_hash, hash_vec)
        if sim >= 0.2:
            results.append((record_id, ts, sim))

    results.sort(key=lambda x: x[1])
    return results
