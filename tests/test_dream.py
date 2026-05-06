"""Tests for iai_mcp.dream -- Task 1.

Covers 9 behaviours from the plan:
1. run_rem_cycle calls sleep.run_heavy_consolidation with SleepConfig(llm_enabled=False)
   and has_api_key=False.
2. run_rem_cycle calls schema.induce_schemas_tier1 with llm_enabled=False (Tier-0).
3. Non-last cycle does NOT invoke insight.generate_overnight_insight even if
   claude_enabled=True.
4. Last cycle WITH claude_enabled=True invokes insight.generate_overnight_insight
   and surfaces text into result.
5. Last cycle with claude_enabled=False does NOT invoke insight.
6. rem_cycle_started + rem_cycle_completed events emitted.
7. 15min cap enforced via asyncio.timeout; emits rem_cycle_timeout and returns
   timed_out=True.
8. Exception inside run_heavy_consolidation is caught; rem_cycle_error event
   emitted; function returns a partial result dict (daemon never dies).
9. literal preservation -- no daemon-side code path mutates
   MemoryRecord.literal_surface during a cycle (static assertion on dream.py).
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# helpers: lightweight store stub + event capture
# ---------------------------------------------------------------------------


class _EventLog:
    """In-memory capture of write_event calls for test assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def capture(self, store, kind, data, *, severity=None, **kwargs):
        self.events.append((kind, dict(data), severity))
        return None

    def kinds(self) -> list[str]:
        return [k for (k, _d, _s) in self.events]


def _fresh_store(tmp_path, monkeypatch):
    """Minimal MemoryStore tied to a tmp path (pattern reused from tests)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _install_stubs(
    monkeypatch,
    *,
    heavy_return=None,
    heavy_raises=None,
    heavy_sleep_sec: float | None = None,
    candidates_return=None,
    insight_return=None,
    event_log: _EventLog | None = None,
):
    """Monkeypatch the three external callables dream.run_rem_cycle invokes.

    Returns the (heavy_calls, schema_calls, insight_calls) recorders.
    """
    heavy_calls: list[tuple] = []
    schema_calls: list[tuple] = []
    insight_calls: list[tuple] = []

    def fake_heavy(store, session_id, cfg, budget, rate, has_api_key):
        heavy_calls.append((session_id, cfg, has_api_key))
        if heavy_sleep_sec is not None:
            time.sleep(heavy_sleep_sec)
        if heavy_raises is not None:
            raise heavy_raises
        return heavy_return if heavy_return is not None else {
            "mode": "heavy", "tier": "tier0",
            "summaries_created": 3, "schemas_induced": 1,
            "decay_result": {"decayed": 0, "pruned": 0},
            "schema_candidates": [],
        }

    def fake_induce(store, budget, rate, llm_enabled):
        schema_calls.append((llm_enabled,))
        return candidates_return if candidates_return is not None else []

    async def fake_insight(store, session_id):
        insight_calls.append((session_id,))
        return insight_return if insight_return is not None else {
            "ok": True, "text": "test insight"
        }

    monkeypatch.setattr("iai_mcp.dream.run_heavy_consolidation", fake_heavy)
    monkeypatch.setattr("iai_mcp.dream.induce_schemas_tier1", fake_induce)
    monkeypatch.setattr("iai_mcp.insight.generate_overnight_insight", fake_insight)

    if event_log is not None:
        monkeypatch.setattr("iai_mcp.dream.write_event", event_log.capture)

    # Stub BudgetLedger / RateLimitLedger ctors so a bare store object works.
    class _NoOp:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr("iai_mcp.dream.BudgetLedger", _NoOp)
    monkeypatch.setattr("iai_mcp.dream.RateLimitLedger", _NoOp)

    return heavy_calls, schema_calls, insight_calls


# ---------------------------------------------------------------------------
# Test 1: heavy consolidation called with llm_enabled=False + has_api_key=False
# ---------------------------------------------------------------------------

def test_rem_cycle_invokes_heavy(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    heavy_calls, _schema_calls, _insight_calls = _install_stubs(
        monkeypatch, event_log=event_log,
    )

    store = object()  # dream.py never touches store directly; stubs handle it.

    async def runner():
        return await dream.run_rem_cycle(
            store, 1, 4, "sess-X",
            is_last=False, claude_enabled=False,
        )

    result = asyncio.run(runner())

    assert len(heavy_calls) == 1, "run_heavy_consolidation not called"
    session_id, cfg, has_api_key = heavy_calls[0]
    assert session_id == "sess-X"
    assert has_api_key is False, "daemon must pass has_api_key=False"
    assert getattr(cfg, "llm_enabled", None) is False, "llm_enabled must be False"

    # The heavy result stub returns summaries_created=3.
    assert result["summaries_created"] == 3
    assert result["timed_out"] is False


# ---------------------------------------------------------------------------
# Test 2: Tier-0 schema induction (llm_enabled=False)
# ---------------------------------------------------------------------------

def test_rem_cycle_invokes_tier0_induction(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    _h, schema_calls, _i = _install_stubs(
        monkeypatch, event_log=event_log,
        candidates_return=[{"pattern": "foo"}, {"pattern": "bar"}],
    )

    store = object()

    async def runner():
        return await dream.run_rem_cycle(
            store, 2, 4, "sess-Y",
            is_last=False, claude_enabled=False,
        )

    result = asyncio.run(runner())

    assert len(schema_calls) == 1, "induce_schemas_tier1 not called"
    (llm_enabled,) = schema_calls[0]
    assert llm_enabled is False, "Tier-0 path requires llm_enabled=False"
    assert result["schema_candidates"] == 2


# ---------------------------------------------------------------------------
# Test 3: non-last cycle with claude_enabled=True does NOT invoke insight
# ---------------------------------------------------------------------------

def test_non_last_cycle_does_not_invoke_insight(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    _h, _s, insight_calls = _install_stubs(
        monkeypatch, event_log=event_log,
    )

    store = object()

    async def runner():
        return await dream.run_rem_cycle(
            store, 2, 4, "sess-Y",
            is_last=False, claude_enabled=True,
        )

    result = asyncio.run(runner())

    assert insight_calls == [], "insight called on non-last cycle (D-08 violation)"
    assert result["claude_call_used"] is False


# ---------------------------------------------------------------------------
# Test 4: last cycle with claude_enabled=True invokes insight and surfaces text
# ---------------------------------------------------------------------------

def test_last_cycle_triggers_insight(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    _h, _s, insight_calls = _install_stubs(
        monkeypatch, event_log=event_log,
        insight_return={"ok": True, "text": "unified insight about patterns"},
    )

    store = object()

    async def runner():
        return await dream.run_rem_cycle(
            store, 4, 4, "sess-Z",
            is_last=True, claude_enabled=True,
        )

    result = asyncio.run(runner())

    assert len(insight_calls) == 1, "last cycle must invoke insight"
    assert insight_calls[0] == ("sess-Z",)
    assert result["claude_call_used"] is True
    assert result["main_insight_text"] == "unified insight about patterns"


# ---------------------------------------------------------------------------
# Test 5: last cycle with claude_enabled=False does NOT invoke insight
# ---------------------------------------------------------------------------

def test_last_cycle_respects_host_disabled(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    _h, _s, insight_calls = _install_stubs(
        monkeypatch, event_log=event_log,
    )

    store = object()

    async def runner():
        return await dream.run_rem_cycle(
            store, 4, 4, "sess-W",
            is_last=True, claude_enabled=False,
        )

    result = asyncio.run(runner())

    assert insight_calls == [], "claude_enabled=False must gate insight call"
    assert result["claude_call_used"] is False
    assert result["main_insight_text"] is None


# ---------------------------------------------------------------------------
# Test 6: rem_cycle_started + rem_cycle_completed events emitted
# ---------------------------------------------------------------------------

def test_cycle_start_and_completed_events(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    _install_stubs(monkeypatch, event_log=event_log)

    store = object()

    async def runner():
        return await dream.run_rem_cycle(
            store, 1, 4, "sess-E",
            is_last=False, claude_enabled=False,
        )

    asyncio.run(runner())

    kinds = event_log.kinds()
    assert "rem_cycle_started" in kinds
    assert "rem_cycle_completed" in kinds
    assert kinds.index("rem_cycle_started") < kinds.index("rem_cycle_completed")

    # rem_cycle_started payload shape
    started = next(e for e in event_log.events if e[0] == "rem_cycle_started")
    assert started[1] == {"n": 1, "of": 4}


# ---------------------------------------------------------------------------
# Test 7: 15min cap enforced; timeout emits rem_cycle_timeout, timed_out=True
# ---------------------------------------------------------------------------

def test_rem_cycle_respects_15min_cap(tmp_path, monkeypatch):
    from iai_mcp import dream

    # Shrink the cap so the test is fast; make run_heavy_consolidation slow
    # enough (sleep 0.3s) to trigger the timeout.
    monkeypatch.setattr(dream, "REM_CYCLE_MAX_SEC", 0.1)

    event_log = _EventLog()
    _install_stubs(
        monkeypatch, event_log=event_log,
        heavy_sleep_sec=0.3,
    )

    store = object()

    async def runner():
        return await dream.run_rem_cycle(
            store, 3, 4, "sess-T",
            is_last=False, claude_enabled=False,
        )

    result = asyncio.run(runner())

    assert result["timed_out"] is True
    kinds = event_log.kinds()
    assert "rem_cycle_timeout" in kinds, f"missing rem_cycle_timeout; kinds={kinds}"
    # Timeout still completes with rem_cycle_completed (non-crashing).
    assert "rem_cycle_completed" in kinds


# ---------------------------------------------------------------------------
# Test 8: exception inside heavy-consolidation is caught, error event emitted
# ---------------------------------------------------------------------------

def test_rem_cycle_exception_does_not_crash_daemon(tmp_path, monkeypatch):
    from iai_mcp import dream

    event_log = _EventLog()
    _install_stubs(
        monkeypatch, event_log=event_log,
        heavy_raises=RuntimeError("boom from heavy"),
    )

    store = object()

    async def runner():
        # Must NOT raise -- daemon's outer loop relies on this invariant.
        return await dream.run_rem_cycle(
            store, 1, 4, "sess-X",
            is_last=False, claude_enabled=False,
        )

    result = asyncio.run(runner())

    kinds = event_log.kinds()
    assert "rem_cycle_error" in kinds, (
        f"rem_cycle_error must be emitted on exception; got {kinds}"
    )
    err_event = next(e for e in event_log.events if e[0] == "rem_cycle_error")
    assert "boom from heavy" in err_event[1]["error"]
    # Partial result still returned (no exception propagates).
    assert "cycle" in result
    assert result["cycle"] == 1


# ---------------------------------------------------------------------------
# Test 9: literal preservation -- dream.py does not mutate literal_surface
# ---------------------------------------------------------------------------

def test_dream_does_not_mutate_literal_surface():
    """C5 static guard. dream.py must contain zero writes to
    record.literal_surface (read-access is fine but assignment is forbidden)."""
    dream_src = (
        Path(__file__).resolve().parent.parent
        / "src" / "iai_mcp" / "dream.py"
    ).read_text()
    pattern = re.compile(r"\.literal_surface\s*=")
    assert not pattern.search(dream_src), (
        "C5 violation: dream.py assigns to literal_surface"
    )
