""" Batch API consolidation (Task 3, ).

 (unified daily process): when Tier 1 is enabled + credentials + budget
+ rate-limit all green (D-GUARD ladder via should_call_llm), submit a batch
to Anthropic's Batch API at 50% discount vs synchronous calls. Falls back
to Tier 0 stub results on any gate failure or SDK absence.

scope: the D-GUARD gate + budget side-effect + llm_health event
emission are load-bearing. The actual anthropic.batches.create call is
scaffolded behind a lazy import; when the SDK surface differs from what the
Python core expects (e.g. version skew), the stub returns an empty result
list and records llm_health fallback. / future phases own the real
wire-up once the SDK API settles.

Pricing model:
- Haiku 4.5 approx sync cost: prompt $0.25 / 1M tokens + output $1.25 / 1M
- Batch discount: 50% off sync cost.
"""
from __future__ import annotations

import os
from typing import Any

from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm


# 50% discount vs sync tier.
BATCH_DISCOUNT = 0.5

# scope: we do not poll in-process. Real-world Batch API can take
# up to ~24h. The dispatch path is "submit -> return (True, 'ok', stub)" with
# the actual results arriving via a future polling job. Tests assert the
# gate + side-effects; the stub list is empty in
BATCH_POLL_TIMEOUT_SEC = 60

# Haiku 4.5 approximate sync pricing (USD per 1M tokens).
_HAIKU_PROMPT_USD_PER_MTOK = 0.25
_HAIKU_OUTPUT_USD_PER_MTOK = 1.25


def _sync_tier_cost(prompt_tokens: int, output_tokens: int) -> float:
    """Approximate sync-tier USD cost for Haiku 4.5.

    uses Haiku 4.5 for consolidation. Pricing is approximate and may
    drift; the gate uses this only for budget-cap decisions (D-GUARD step
    3+4), never for billing reconciliation.
    """
    p = (float(prompt_tokens) / 1_000_000.0) * _HAIKU_PROMPT_USD_PER_MTOK
    o = (float(output_tokens) / 1_000_000.0) * _HAIKU_OUTPUT_USD_PER_MTOK
    return float(p + o)


def _aggregate_estimated_usd(tasks: list[dict]) -> float:
    total_sync = 0.0
    for t in tasks:
        total_sync += _sync_tier_cost(
            int(t.get("prompt_tok", 0)),
            int(t.get("output_tok", 0)),
        )
    return total_sync * BATCH_DISCOUNT


def submit_batch_consolidation(
    store,
    tasks: list[dict],
    budget: BudgetLedger,
    rate: RateLimitLedger,
    llm_enabled: bool = True,
) -> tuple[bool, str, list[dict]]:
    """Submit a batch of consolidation tasks to the Anthropic Batch API.

    Returns (ok, reason, results). On any D-GUARD fallback, ok=False and
    results is an empty list; the caller falls back to local Tier 0 output.

    Gate ordering (D-GUARD):
      1. llm_enabled toggle
      2. API key present
      3. Budget daily + monthly caps (can_spend)
      4. Rate-limit cooldown (last 429 < 15 min)
      5. SDK import path
      6. Real batch submission (stub; see module docstring)
    """
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    estimated_usd = _aggregate_estimated_usd(tasks)

    ok, reason = should_call_llm(
        budget=budget,
        rate=rate,
        llm_enabled=llm_enabled,
        has_api_key=has_key,
        estimated_usd=estimated_usd,
    )
    if not ok:
        write_event(
            store,
            kind="llm_health",
            data={
                "component": "batch_consolidation",
                "tier": "fallback",
                "reason": reason,
                "task_count": len(tasks),
                "estimated_usd": estimated_usd,
            },
            severity="warning",
        )
        return False, reason, []

    # Eligible path: lazy import the SDK. On ImportError or any runtime
    # failure, log critical and fall back. This is also how the current Plan
    # 02-04 scaffold returns -- the real batch submission is stubbed (the
    # SDK surface for batches.create has changed across minor versions).
    try:
        import anthropic  # noqa: F401
    except Exception as exc:
        write_event(
            store,
            kind="llm_health",
            data={
                "component": "batch_consolidation",
                "tier": "fallback",
                "error": f"import anthropic: {exc}",
            },
            severity="critical",
        )
        return False, f"SDK unavailable: {exc}", []

    # H-02 FIX (gap closure): budget stays untouched and
    # effective_tier stays tier0 until a REAL successful anthropic.batches.create
    # response lands. The previous behaviour called budget.record_spend + returned
    # (True, "ok", []), which caused run_heavy_consolidation to flip
    # effective_tier=tier1 and debit the BudgetLedger on a stub producing zero
    # output -- corrupts D-GUARD audit honesty + cost accounting.
    #
    # Real SDK wire-up is scope. Until then the scaffold is honestly
    # documented via an info-severity llm_health event so `iai-mcp audit`
    # observers can see the gap explicitly.
    write_event(
        store,
        kind="llm_health",
        data={
            "component": "batch_consolidation",
            "tier": "fallback",
            "task_count": len(tasks),
            "estimated_usd": estimated_usd,
            "note": (
                "disables the scaffold-true return; "
                "real anthropic.batches.create wire-up is Budget "
                "stays untouched and effective_tier stays tier0 until a "
                "real successful SDK response lands."
            ),
        },
        severity="info",
    )
    return False, "stub: batch API not yet wired", []
