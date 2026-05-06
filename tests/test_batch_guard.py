"""Tests for 02-REVIEW.md H-02 (batch scaffold silently debits budget +
flips effective_tier=tier1 on a stub that produces no output).

Bug: submit_batch_consolidation called budget.record_spend BEFORE the real
SDK call and returned (True, "ok", []). run_heavy_consolidation then saw
ok_batch=True and set effective_tier="tier1", logging it in the
consolidation event. Users inspecting `iai-mcp audit` saw Tier-1 events
that were factually false.

Fix:
    - Scaffold path returns (False, "stub: batch API not yet wired", []).
    - NO budget.record_spend call during the stub period.
    - Emit one info-severity llm_health event documenting the gap so the
      audit CLI reflects honest state.
    - run_heavy_consolidation sees ok_batch=False and keeps tier0; the
      cls_consolidation_run event payload carries batch_submitted=False.

Constitutional contract (D-GUARD budget honesty + audit repudiability):
    Budget ledger rows MUST correspond to real API spend. Tier flags in
    the event log MUST correspond to real Tier-1 output. Both invariants
    were silently violated by the scaffold.
"""
from __future__ import annotations

import pytest

from iai_mcp.events import query_events
from iai_mcp.guard import BudgetLedger, RateLimitLedger
from iai_mcp.store import MemoryStore


def _tasks(n: int = 1) -> list[dict]:
    return [
        {
            "task_id": f"t{i}",
            "prompt": f"summarise cluster {i}",
            "prompt_tok": 500,
            "output_tok": 200,
        }
        for i in range(n)
    ]


# ==================================================== H-02: batch scaffold guard


def test_batch_stub_returns_false_with_scaffold_reason(tmp_path, monkeypatch):
    """Stub path must return (False, "stub: batch API not yet wired", [])
    even when all D-GUARD steps pass (API key + llm_enabled + budget + rate
    all clean). This is the load-bearing assertion that neutralises the
    tier1 flip."""
    from iai_mcp.batch import submit_batch_consolidation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    ok, reason, results = submit_batch_consolidation(
        store, _tasks(3), budget, rate, llm_enabled=True,
    )

    assert ok is False, "scaffold must return ok=False until real SDK wire-up lands"
    assert reason.startswith("stub:"), (
        f"reason must advertise scaffold status, got {reason!r}"
    )
    assert "batch API not yet wired" in reason
    assert results == [], "scaffold produces empty result list"


def test_batch_stub_does_not_debit_budget(tmp_path, monkeypatch):
    """Budget MUST NOT increase during the scaffold period. Only a real
    successful anthropic.batches.create response may record spend."""
    from iai_mcp.batch import submit_batch_consolidation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    before_daily = budget.daily_used()
    before_monthly = budget.monthly_used()

    submit_batch_consolidation(
        store, _tasks(5), budget, rate, llm_enabled=True,
    )

    after_daily = budget.daily_used()
    after_monthly = budget.monthly_used()

    assert after_daily == before_daily, (
        f"daily spend changed during stub: {before_daily} -> {after_daily}"
    )
    assert after_monthly == before_monthly


def test_batch_stub_emits_info_llm_health_event(tmp_path, monkeypatch):
    """Observability contract: scaffold state must be visible in the events
    table so `iai-mcp audit` observers can see the gap explicitly.
    Severity=info (not warning/critical) because this is intentional
    scaffold behaviour, not an error."""
    from iai_mcp.batch import submit_batch_consolidation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    submit_batch_consolidation(
        store, _tasks(), budget, rate, llm_enabled=True,
    )

    events = query_events(store, kind="llm_health")
    batch_events = [
        e for e in events
        if e["data"].get("component") == "batch_consolidation"
    ]
    assert len(batch_events) >= 1, "must emit llm_health for batch stub"
    ev = batch_events[0]
    assert ev["severity"] == "info", (
        f"scaffold event must be info-severity, got {ev['severity']!r}"
    )
    note = ev["data"].get("note") or ""
    assert "scaffold" in note.lower() or "not yet wired" in note.lower(), (
        f"event note must advertise scaffold/not-yet-wired status, got {note!r}"
    )


def test_run_heavy_does_not_flip_tier1_on_stub(tmp_path, monkeypatch):
    """run_heavy_consolidation must not set effective_tier='tier1' while
    submit_batch_consolidation is a stub. Even when the D-GUARD ladder
    greenlights Tier-1 (key + enabled + budget + rate), ok_batch=False so
    the caller stays on Tier-0."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    cfg = SleepConfig(llm_enabled=True)
    result = run_heavy_consolidation(
        store,
        session_id="h-stub",
        config=cfg,
        budget=budget,
        rate=rate,
        has_api_key=True,
    )

    assert result["tier"] == "tier0", (
        f"effective_tier must stay tier0 during scaffold, got {result['tier']!r}"
    )

    # cls_consolidation_run event has batch_submitted=False
    events = query_events(store, kind="cls_consolidation_run")
    heavy = [e for e in events if e["data"].get("mode") == "heavy"]
    assert len(heavy) >= 1
    assert heavy[0]["data"]["batch_submitted"] is False, (
        "batch_submitted flag must honestly reflect stub state"
    )
    # tier_eligible still records that the D-GUARD ladder was CONSULTED (tier1)
    # even though effective_tier is tier0 -- lets auditors see the gap.
    assert heavy[0]["data"].get("tier") == "tier0"


def test_run_heavy_does_not_debit_budget_during_stub(tmp_path, monkeypatch):
    """End-to-end: running heavy consolidation with full Tier-1 eligibility
    must leave the budget untouched because submit_batch_consolidation is a
    stub."""
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    store = MemoryStore(path=tmp_path)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    before = budget.daily_used()

    cfg = SleepConfig(llm_enabled=True)
    run_heavy_consolidation(
        store,
        session_id="h-no-debit",
        config=cfg,
        budget=budget,
        rate=rate,
        has_api_key=True,
    )

    # Note: schema_induction_tier1 also records a small spend when eligible.
    # We assert the batch_consolidation row specifically is NOT present.
    tbl = store.db.open_table("budget_ledger")
    df = tbl.to_pandas()
    if not df.empty:
        batch_rows = df[df["kind"] == "batch_consolidation"]
        assert len(batch_rows) == 0, (
            "stub must not record a batch_consolidation spend row"
        )
