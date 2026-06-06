"""GABA-switch k-annealing + weight normalization.

Implements two SDM-inspired mechanisms:
1. k-annealing: The number of active neurons (k) in the sparse retrieval
   decreases over sleep cycles, mimicking GABA-mediated inhibition increase
   during consolidation. Young memories get broad activation (high k);
   old consolidated memories get precise activation (low k).

2. Weight normalization: L2 normalization on Hebbian edge weights after
   each sleep cycle prevents unbounded weight growth (biological synaptic
   homeostasis). Without this, frequently co-retrieved pairs accumulate
   arbitrarily high weights, creating artificial hubs.

Together these enable organic continual learning without explicit replay
by ensuring the retrieval landscape naturally sharpens over time.
"""
from __future__ import annotations

import logging
import math
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
    """Linear k-annealing from K_INITIAL to K_FINAL over ANNEAL_CYCLES.

    After ANNEAL_CYCLES, k stays at K_FINAL permanently.
    This models the GABA-switch: early learning is broad (high k),
    mature memory is precise (low k).
    """
    if cycle_count >= ANNEAL_CYCLES:
        return K_FINAL
    progress = cycle_count / ANNEAL_CYCLES
    k = K_INITIAL - progress * (K_INITIAL - K_FINAL)
    return max(K_FINAL, int(round(k)))


def normalize_edge_weights(
    weights: dict[str, float],
    target_norm: float = 1.0,
) -> dict[str, float]:
    """L2-normalize a weight dictionary to prevent unbounded growth.

    Applies synaptic homeostasis: total weight budget is fixed,
    individual edges compete for share of that budget.
    Strong edges stay relatively strong; weak ones stay weak;
    but the total magnitude is bounded.
    """
    if not weights:
        return weights

    values = np.array(list(weights.values()), dtype=np.float32)
    current_norm = float(np.linalg.norm(values))

    if current_norm < 1e-8:
        return weights

    scale = target_norm / current_norm
    return {k: float(v * scale) for k, v in weights.items()}


def should_normalize(cycle_count: int, normalize_every: int = 3) -> bool:
    """Normalize every N cycles to avoid over-correction."""
    return cycle_count > 0 and cycle_count % normalize_every == 0
