"""REM cycle orchestrator. CALLS existing modules -- does not reimplement.

Biological mapping:
- NREM-2 (Hebbian binding)      = existing hebbian LTP inside sleep.py cluster pass
- NREM-3 (hippocampal replay)   = sleep.run_heavy_consolidation Tier-0 path
- REM   (cross-community)       = schema.induce_schemas_tier1(llm_enabled=False)
- REM lucid moment (last cycle) = insight.generate_overnight_insight

Constitutional guard:
- LOCAL primary worker; llm_enabled ALWAYS False when calling sleep/schema.
- has_api_key=False always for daemon (zero paid-API path).
- 15-minute hard cap per cycle (asyncio.timeout context manager).
- C1: daemon must already hold the fcntl exclusive lock BEFORE calling
      run_rem_cycle -- this module does NOT acquire locks, that is _tick_body's
      job. This module is called under the lock.
- C3: ZERO API cost. The single nightly Claude call is a subprocess, wired
      by in insight.py. No paid-API env var is referenced here.
- C5: literal preservation -- we only call modules that modify metadata
      (FSRS state, edge weights, schema tags). Never assigns to literal_surface.
"""
from __future__ import annotations

import asyncio

from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger
from iai_mcp.schema import induce_schemas_tier1
from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

# hard cap per REM cycle.
REM_CYCLE_MAX_SEC: int = 15 * 60


async def _emit(store, kind: str, data: dict, severity: str | None = None) -> None:
    """Emit an event off the main loop so LanceDB writes don't block asyncio."""
    if severity is None:
        await asyncio.to_thread(write_event, store, kind, data)
    else:
        await asyncio.to_thread(write_event, store, kind, data, severity=severity)


async def run_rem_cycle(
    store,
    cycle_num: int,
    total_cycles: int,
    session_id: str,
    *,
    is_last: bool,
    claude_enabled: bool,
) -> dict:
    """One REM cycle. Runs to completion or hits 15min cap.

    Returns dict consumed by the morning digest:
      {cycle, summaries_created, schemas_induced, schema_candidates,
       claude_call_used, main_insight_text, timed_out}

    Never raises. All failure modes (timeout, module exception) surface as
    event emissions + a partial result dict so the daemon's outer loop
    cannot crash on cycle-internal exceptions (T-04-12 mitigation).
    """
    await _emit(store, "rem_cycle_started", {"n": cycle_num, "of": total_cycles})

    result: dict = {
        "cycle": cycle_num,
        "summaries_created": 0,
        "schemas_induced": 0,
        "schema_candidates": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out": False,
    }

    try:
        async with asyncio.timeout(REM_CYCLE_MAX_SEC):
            # NREM-3 equivalent: heavy consolidation, Tier-0 only in daemon.
            cfg = SleepConfig(llm_enabled=False)
            heavy = await asyncio.to_thread(
                run_heavy_consolidation,
                store, session_id, cfg,
                BudgetLedger(store), RateLimitLedger(store),
                False,  # has_api_key=False always for daemon
            )
            if isinstance(heavy, dict):
                result["summaries_created"] = int(heavy.get("summaries_created", 0) or 0)
                result["schemas_induced"] = int(heavy.get("schemas_induced", 0) or 0)

            # REM cross-community schema induction (explicit Tier-0).
            # Signature: induce_schemas_tier1(store, budget, rate, llm_enabled=True)
            # -- we force llm_enabled=False so the D-GUARD ladder falls through to
            # the pure-local Tier-0 path.
            candidates = await asyncio.to_thread(
                induce_schemas_tier1,
                store, BudgetLedger(store), RateLimitLedger(store), False,
            )
            result["schema_candidates"] = len(candidates) if candidates else 0

            # Lucid moment -- ONLY on last cycle, budget-gated by caller.
            if is_last and claude_enabled:
                from iai_mcp.insight import generate_overnight_insight

                insight = await generate_overnight_insight(store, session_id)
                if isinstance(insight, dict) and insight.get("ok"):
                    result["claude_call_used"] = True
                    result["main_insight_text"] = insight.get("text")

    except asyncio.TimeoutError:
        result["timed_out"] = True
        await _emit(
            store,
            "rem_cycle_timeout",
            {"cycle": cycle_num},
            severity="warning",
        )
    except Exception as exc:  # noqa: BLE001 -- daemon must never die on cycle error
        await _emit(
            store,
            "rem_cycle_error",
            {"cycle": cycle_num, "error": str(exc)[:500]},
            severity="critical",
        )

    await _emit(store, "rem_cycle_completed", result)
    return result
