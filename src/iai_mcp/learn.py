"""Learning layer (LEARN-01/02/05/06, Task 2).

Four mechanisms live here:

1. LEARN-01 (Bayesian profile update) is implemented in `iai_mcp.profile`
   as `bayesian_update`; this module re-exports the RetrievalFeedback and
   policy utilities used by the pipeline + core dispatch.

2. LEARN-02 retrieval-policy RL -- simple tabular gradient on score
   weights. Feedback sources:
   - user acted on hit (used)           -> boost W_COSINE
   - user issued contradict (corrected) -> reduce W_COSINE
   - user re-asked same cue (re_asked)  -> reduce W_COSINE

3. LEARN-05 meta-learning -- ε-greedy bandit over retrieval strategies
   keyed by query type.

4. LEARN-06 identity refinement -- reads s5_invariant_update /
   s5_invariant_proposal events and drifts s5_trust_score up for
   consistently-agreeing anchors, down for frequently-rejected ones.

All writes go through the D-STORAGE events table; no .jsonl files.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore


# ---------------------------------------------------------------- constants

LEARN_RATE: float = 0.05
MAX_WEIGHT: float = 5.0
MIN_WEIGHT: float = 0.0
EPSILON_EXPLORE: float = 0.1  # LEARN-05 bandit exploration probability


# ---------------------------------------------------------------- feedback


@dataclass
class RetrievalFeedback:
    """Implicit feedback signal on a memory_recall response."""

    query_type: str                # e.g. "fact_lookup" | "open_ended" | "contradiction_check"
    hit_ids: list[UUID]
    used_ids: list[UUID] = field(default_factory=list)
    corrected: bool = False        # user issued memory_contradict on a hit
    re_asked: bool = False         # user re-issued the same cue within 5 turns


# ---------------------------------------------------------------- LEARN-02


def update_retrieval_weights(
    feedback: RetrievalFeedback,
    current_weights: dict[str, float],
) -> dict[str, float]:
    """LEARN-02 tabular gradient on score weights.

    Primary signal: use-rate = |used_ids ∩ hit_ids| / |hit_ids|.
    delta = (use_rate - 0.5) * LEARN_RATE
    Correction penalty:  -LEARN_RATE
    Re-ask penalty:      -LEARN_RATE * 0.5

    All weights clamped to [MIN_WEIGHT, MAX_WEIGHT].
    Returns a new dict (does not mutate the input).
    """
    w = dict(current_weights)
    delta = 0.0
    if feedback.hit_ids:
        hits_set = set(feedback.hit_ids)
        used_set = set(feedback.used_ids)
        use_rate = len(hits_set & used_set) / len(feedback.hit_ids)
        delta = (use_rate - 0.5) * LEARN_RATE
    if feedback.corrected:
        delta -= LEARN_RATE
    if feedback.re_asked:
        delta -= LEARN_RATE * 0.5

    w_cos = w.get("W_COSINE", 1.0)
    w["W_COSINE"] = max(MIN_WEIGHT, min(MAX_WEIGHT, w_cos + delta))

    # Clamp other weights in case of external mutation.
    for k in ("W_AAAK", "W_DEGREE", "W_AGE"):
        if k in w:
            w[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, w[k]))
    return w


# ---------------------------------------------------------------- LEARN-05


def pick_retrieval_strategy(
    query_type: str,
    history: dict,
    strategies: list[str] | None = None,
) -> str:
    """ε-greedy bandit over retrieval strategies per query type.

    `history` shape:
        {
            "<query_type>": {
                "<strategy>": {"mean": float, "n": int},
                ...
            },
            ...
        }

    Returns the strategy with the highest mean for this query_type except on
    the ε fraction of calls where a random strategy is explored.
    """
    strategies = strategies or ["pipeline_default", "greedy_2hop", "rich_club_first"]
    if random.random() < EPSILON_EXPLORE:
        return random.choice(strategies)
    rewards = history.get(query_type, {})
    if not rewards:
        return strategies[0]
    return max(
        strategies,
        key=lambda s: rewards.get(s, {}).get("mean", 0.0),
    )


# ---------------------------------------------------------------- LEARN-06


TRUST_INCREMENT_PER_COMMIT: float = 0.02
TRUST_DECREMENT_PER_REJECT: float = 0.01


def refine_s5_trust_score(
    store: MemoryStore,
    record_id: UUID,
    current: float,
) -> float:
    """LEARN-06: trust score drifts based on consensus history.

    +TRUST_INCREMENT per s5_invariant_update event with agree_count >= 3
    -TRUST_DECREMENT per s5_invariant_proposal with passes_vigilance == False

    Clamped to [0, 1].
    """
    updates = query_events(store, kind="s5_invariant_update", limit=200)
    commits = sum(
        1 for e in updates
        if e["data"].get("anchor_id") == str(record_id)
        and int(e["data"].get("agree_count", 0)) >= 3
    )
    rejects_events = query_events(store, kind="s5_invariant_proposal", limit=500)
    rejects = sum(
        1 for e in rejects_events
        if e["data"].get("anchor_id") == str(record_id)
        and not e["data"].get("passes_vigilance", True)
    )
    new_score = (
        current
        + TRUST_INCREMENT_PER_COMMIT * commits
        - TRUST_DECREMENT_PER_REJECT * rejects
    )
    return max(0.0, min(1.0, new_score))
