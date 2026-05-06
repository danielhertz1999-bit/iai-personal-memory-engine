"""Tests for LEARN-02 retrieval-policy RL + LEARN-05 meta-learning.

LEARN-02: implicit user feedback (used/corrected/re_asked) updates the D-13
score weights (W_COSINE / W_AAAK / W_DEGREE / W_AGE).

LEARN-05: ε-greedy bandit over strategies picks best strategy per query type.
"""
from __future__ import annotations

import random
from uuid import uuid4

import pytest


# ---------------------------------------------------------------- feedback shape


def test_retrieval_feedback_dataclass():
    from iai_mcp.learn import RetrievalFeedback

    fb = RetrievalFeedback(
        query_type="fact_lookup",
        hit_ids=[uuid4(), uuid4()],
        used_ids=[],
        corrected=False,
        re_asked=False,
    )
    assert fb.query_type == "fact_lookup"
    assert len(fb.hit_ids) == 2


# ---------------------------------------------------------------- update_retrieval_weights


def test_retrieval_feedback_used_boosts_weights():
    """Higher use-rate -> W_COSINE goes up."""
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    fb = RetrievalFeedback(
        query_type="lookup",
        hit_ids=ids,
        used_ids=ids[:3],        # used all hits
        corrected=False,
        re_asked=False,
    )
    before = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    after = update_retrieval_weights(fb, before)
    assert after["W_COSINE"] > before["W_COSINE"]


def test_retrieval_feedback_corrected_reduces_weights():
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    fb = RetrievalFeedback(
        query_type="lookup",
        hit_ids=ids,
        used_ids=[],
        corrected=True,
        re_asked=False,
    )
    before = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    after = update_retrieval_weights(fb, before)
    assert after["W_COSINE"] < before["W_COSINE"]


def test_retrieval_feedback_re_asked_reduces_weights():
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    fb = RetrievalFeedback(
        query_type="lookup",
        hit_ids=ids,
        used_ids=[],
        corrected=False,
        re_asked=True,
    )
    before = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    after = update_retrieval_weights(fb, before)
    assert after["W_COSINE"] < before["W_COSINE"]


def test_retrieval_weights_bounded():
    """After many updates, weights stay in [0, 5]."""
    from iai_mcp.learn import MAX_WEIGHT, MIN_WEIGHT, RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    # 1000 "used" feedbacks (continually boost)
    weights = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    for _ in range(1000):
        fb = RetrievalFeedback(
            query_type="x", hit_ids=ids, used_ids=ids,
            corrected=False, re_asked=False,
        )
        weights = update_retrieval_weights(fb, weights)
    assert weights["W_COSINE"] <= MAX_WEIGHT
    assert weights["W_COSINE"] >= MIN_WEIGHT


# ---------------------------------------------------------------- epsilon-greedy strategy


def test_pick_retrieval_strategy_returns_string():
    from iai_mcp.learn import pick_retrieval_strategy

    random.seed(42)
    s = pick_retrieval_strategy("fact_lookup", history={})
    assert isinstance(s, str)


def test_pick_retrieval_strategy_epsilon_greedy():
    """Over 200 calls, mostly picks the highest-mean strategy."""
    from iai_mcp.learn import pick_retrieval_strategy

    random.seed(7)
    history = {
        "fact_lookup": {
            "pipeline_default": {"mean": 0.9, "n": 10},
            "greedy_2hop": {"mean": 0.1, "n": 10},
            "rich_club_first": {"mean": 0.2, "n": 10},
        }
    }
    picks = {"pipeline_default": 0, "greedy_2hop": 0, "rich_club_first": 0}
    for _ in range(200):
        s = pick_retrieval_strategy("fact_lookup", history)
        picks[s] = picks.get(s, 0) + 1
    # The best strategy (pipeline_default) should dominate at >= 60%.
    assert picks["pipeline_default"] > 120


def test_pick_retrieval_strategy_no_history():
    """Fresh query_type with no history -> returns a strategy anyway."""
    from iai_mcp.learn import pick_retrieval_strategy

    random.seed(42)
    s = pick_retrieval_strategy("unseen", history={})
    assert isinstance(s, str)
    assert s in ("pipeline_default", "greedy_2hop", "rich_club_first")


def test_pick_retrieval_strategy_custom_strategies():
    """Caller can pass custom strategy list."""
    from iai_mcp.learn import pick_retrieval_strategy

    random.seed(1)
    s = pick_retrieval_strategy("x", history={}, strategies=["a", "b", "c"])
    assert s in ("a", "b", "c")


def test_retrieval_policy_per_query_type():
    """Different query_types accumulate separate weights."""
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4()]
    w1 = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    w2 = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    # Query type A: user uses everything
    fb_a = RetrievalFeedback("A", ids, ids, False, False)
    w1 = update_retrieval_weights(fb_a, w1)
    # Query type B: user corrects
    fb_b = RetrievalFeedback("B", ids, [], True, False)
    w2 = update_retrieval_weights(fb_b, w2)
    assert w1["W_COSINE"] > w2["W_COSINE"]
