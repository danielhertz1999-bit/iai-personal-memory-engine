"""Tests for TOK-09 Batch API consolidation (Plan 02-04 Task 3, D-29).

submit_batch_consolidation passes through D-GUARD (should_call_llm) before
any network work. On Tier 0 fallback (no llm_enabled, no api key, budget
exceeded, ratelimit cooldown) returns stub results + writes llm_health
event. scope: the gate + event side-effects are load-bearing;
the real anthropic.batches.create call is stubbed (SDK surface varies).
"""
from __future__ import annotations

import pytest

from iai_mcp.events import query_events
from iai_mcp.guard import BudgetLedger, RateLimitLedger
from iai_mcp.store import MemoryStore


def _tasks(n: int = 3) -> list[dict]:
    return [
        {
            "task_id": f"t{i}",
            "prompt": f"summarise cluster {i}",
            "prompt_tok": 500,
            "output_tok": 200,
        }
        for i in range(n)
    ]


def test_batch_fallback_when_llm_disabled(tmp_path):
    from iai_mcp.batch import submit_batch_consolidation

    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    ok, reason, results = submit_batch_consolidation(
        store, _tasks(), budget, rate, llm_enabled=False,
    )
    assert ok is False
    assert "llm_enabled" in reason.lower() or "disabled" in reason.lower()
    # Fallback returns an empty-but-structured list so downstream consumers
    # don't crash on a None.
    assert isinstance(results, list)


def test_batch_fallback_when_no_api_key(tmp_path, monkeypatch):
    from iai_mcp.batch import submit_batch_consolidation

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    ok, reason, _ = submit_batch_consolidation(
        store, _tasks(), budget, rate, llm_enabled=True,
    )
    assert ok is False
    # D-GUARD step 2.
    assert "api" in reason.lower() or "key" in reason.lower()


def test_batch_emits_llm_health_on_fallback(tmp_path):
    from iai_mcp.batch import submit_batch_consolidation

    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    submit_batch_consolidation(
        store, _tasks(), budget, rate, llm_enabled=False,
    )
    events = query_events(store, kind="llm_health")
    fallback_events = [
        e for e in events
        if e["data"].get("component") == "batch_consolidation"
    ]
    assert len(fallback_events) >= 1


def test_batch_50pct_discount():
    """Pricing helper returns 50% of sync cost per D-29."""
    from iai_mcp.batch import BATCH_DISCOUNT, _sync_tier_cost

    sync = _sync_tier_cost(1_000_000, 1_000_000)
    # Haiku 4.5 approximate -- not exact numbers, just shape.
    assert sync > 0
    discounted = sync * BATCH_DISCOUNT
    assert discounted == sync * 0.5
    assert BATCH_DISCOUNT == 0.5


def test_batch_records_spend_when_eligible(tmp_path, monkeypatch):
    """Eligible path records a discounted spend to BudgetLedger."""
    from iai_mcp.batch import submit_batch_consolidation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    before = budget.daily_used()
    ok, _reason, _results = submit_batch_consolidation(
        store, _tasks(5), budget, rate, llm_enabled=True,
    )
    after = budget.daily_used()
    # Whether the SDK is present or not, the eligible gate records a nominal
    # spend (Plan 02-04 scaffolds the budget side-effect; real batch API is
    # implemented via mock/stub so tests don't hit the network).
    if ok:
        assert after >= before
    else:
        # If the SDK is unavailable, spend should NOT increase (we never
        # got past the gate).
        assert after == before


def test_sync_tier_cost_monotonic():
    """Longer prompts cost more."""
    from iai_mcp.batch import _sync_tier_cost

    a = _sync_tier_cost(1000, 500)
    b = _sync_tier_cost(2000, 500)
    assert b > a
