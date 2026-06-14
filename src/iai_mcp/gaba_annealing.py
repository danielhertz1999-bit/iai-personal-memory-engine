from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

K_INITIAL = 20
K_FINAL = 5
ANNEAL_CYCLES = 30


@dataclass
class AnnealingState:
    current_k: int
    cycle_count: int
    k_initial: int = K_INITIAL
    k_final: int = K_FINAL
    total_cycles: int = ANNEAL_CYCLES


def compute_annealed_k(cycle_count: int) -> int:
    if cycle_count >= ANNEAL_CYCLES:
        return K_FINAL
    progress = cycle_count / ANNEAL_CYCLES
    k = K_INITIAL - progress * (K_INITIAL - K_FINAL)
    return max(K_FINAL, int(round(k)))


def normalize_edge_weights(
    weights: dict[str, float],
    target_norm: float = 1.0,
) -> dict[str, float]:
    if not weights:
        return weights

    values = np.array(list(weights.values()), dtype=np.float32)
    current_norm = float(np.linalg.norm(values))

    if current_norm < 1e-8:
        return weights

    scale = target_norm / current_norm
    return {k: float(v * scale) for k, v in weights.items()}


def should_normalize(cycle_count: int, normalize_every: int = 3) -> bool:
    return cycle_count > 0 and cycle_count % normalize_every == 0
