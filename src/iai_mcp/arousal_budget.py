"""Arousal-based dynamic budget + Basta constraint.

Connects the token budget for recall to an internal "stress" variable,
implementing Beer's VSM arousal-based resource allocation:

- High stress (many failed recalls, errors, rapid queries):
  Monotropic tunneling — 1 hop, high rank, narrow focus
- Low stress (successful recalls, idle, stable):
  Associative dreaming — 2 hops, low rank threshold, broad exploration

The Basta constraint (S5 says "no") limits write throughput when the
system's variety exceeds its capacity to absorb new information —
preventing information overload that degrades retrieval quality.

Somatic Markers: The arousal level IS the somatic marker — it encodes
the system's "gut feeling" about whether to explore broadly or focus
narrowly based on accumulated experience.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

AROUSAL_DECAY_RATE = 0.95
AROUSAL_MAX = 1.0
AROUSAL_MIN = 0.0
STRESS_THRESHOLD_HIGH = 0.7
STRESS_THRESHOLD_LOW = 0.3

BUDGET_MIN_TOKENS = 800
BUDGET_MAX_TOKENS = 3000
HOPS_HIGH_STRESS = 1
HOPS_LOW_STRESS = 2
RANK_THRESHOLD_HIGH = 0.6
RANK_THRESHOLD_LOW = 0.3

BASTA_WRITE_CAP_PER_MINUTE = 10
BASTA_CAPACITY_RATIO = 0.8


@dataclass
class ArousalState:
    level: float = 0.5
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0
    success_count: int = 0
    queries_last_minute: int = 0


@dataclass
class RetrievalParams:
    budget_tokens: int
    max_hops: int
    rank_threshold: float
    mode: str


def update_arousal(state: ArousalState, event: str) -> ArousalState:
    """Update arousal level based on system event.

    Events that increase arousal (stress):
    - 'recall_failed': retrieval returned empty
    - 'error': exception in hot path
    - 'rapid_query': queries arriving faster than 1/sec

    Events that decrease arousal (calm):
    - 'recall_success': retrieval found hits
    - 'idle': no activity for >30s
    - 'sleep_complete': consolidation finished
    """
    now = time.time()
    elapsed = now - state.last_updated
    state.level *= AROUSAL_DECAY_RATE ** elapsed
    state.last_updated = now

    if event == "recall_failed":
        state.level = min(AROUSAL_MAX, state.level + 0.15)
        state.error_count += 1
    elif event == "error":
        state.level = min(AROUSAL_MAX, state.level + 0.2)
        state.error_count += 1
    elif event == "rapid_query":
        state.level = min(AROUSAL_MAX, state.level + 0.1)
        state.queries_last_minute += 1
    elif event == "recall_success":
        state.level = max(AROUSAL_MIN, state.level - 0.05)
        state.success_count += 1
    elif event == "idle":
        state.level = max(AROUSAL_MIN, state.level - 0.1)
    elif event == "sleep_complete":
        state.level = max(AROUSAL_MIN, state.level - 0.2)

    state.level = max(AROUSAL_MIN, min(AROUSAL_MAX, state.level))
    return state


def compute_retrieval_params(arousal: ArousalState) -> RetrievalParams:
    """Derive retrieval parameters from current arousal level.

    High arousal → monotropic tunneling (focused, shallow, high threshold)
    Low arousal → associative exploration (broad, deep, low threshold)
    """
    level = arousal.level

    if level >= STRESS_THRESHOLD_HIGH:
        return RetrievalParams(
            budget_tokens=BUDGET_MIN_TOKENS,
            max_hops=HOPS_HIGH_STRESS,
            rank_threshold=RANK_THRESHOLD_HIGH,
            mode="monotropic_tunnel",
        )
    elif level <= STRESS_THRESHOLD_LOW:
        return RetrievalParams(
            budget_tokens=BUDGET_MAX_TOKENS,
            max_hops=HOPS_LOW_STRESS,
            rank_threshold=RANK_THRESHOLD_LOW,
            mode="associative_dream",
        )
    else:
        progress = (level - STRESS_THRESHOLD_LOW) / (STRESS_THRESHOLD_HIGH - STRESS_THRESHOLD_LOW)
        budget = int(BUDGET_MAX_TOKENS - progress * (BUDGET_MAX_TOKENS - BUDGET_MIN_TOKENS))
        rank = RANK_THRESHOLD_LOW + progress * (RANK_THRESHOLD_HIGH - RANK_THRESHOLD_LOW)
        return RetrievalParams(
            budget_tokens=budget,
            max_hops=HOPS_LOW_STRESS,
            rank_threshold=rank,
            mode="balanced",
        )


def basta_check(
    writes_last_minute: int,
    total_records: int,
    community_count: int,
) -> bool:
    """S5 Basta constraint: should the system refuse new writes?

    Returns True when variety exceeds capacity:
    - Too many writes per minute (flooding)
    - Records/community ratio too high (communities can't absorb)
    """
    if writes_last_minute > BASTA_WRITE_CAP_PER_MINUTE:
        logger.info("Basta: write rate %d/min exceeds cap %d", writes_last_minute, BASTA_WRITE_CAP_PER_MINUTE)
        return True

    if community_count > 0:
        ratio = total_records / community_count
        if ratio > (1.0 / BASTA_CAPACITY_RATIO) * 100:
            logger.info("Basta: records/community ratio %.1f exceeds capacity", ratio)
            return True

    return False
