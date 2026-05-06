"""Tests for _tick_body honoring socket control flags (Plan 04-gap-1).

The dispatcher (tests/test_daemon_dispatcher.py) proves the flags are
SET correctly on the daemon state. These tests prove the scheduler
READS those flags and acts on them:

  - scheduler_paused=True   -> _tick_body emits daemon_tick_skipped and
                               returns without acquiring the lock.
  - user_sleep_request.pending=True + empty quiet_window -> _tick_body
                               still bypasses the gate, enters SLEEP,
                               clears the flag.
  - force_rem_request.pending=True -> ONE REM cycle runs out of schedule
                               (total_cycles=1), flag cleared.
  - force_wake_request.pending=True set mid-night -> REM loop breaks
                               early with daemon_yielded reason=
                               force_wake_requested; flag cleared.

All REM cycles are mocked with a coroutine that sleeps 0.01s to avoid
the real 15-minute cap + real consolidation pipeline.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tick_env(tmp_path, monkeypatch):
    """Isolate LOCK_PATH / STATE_PATH to tmp_path; mock REM cycle.

    Returns (store, lock, state_path, rem_calls_list).

    `state_path` points at the tmp_path state file so tests can verify
    flag persistence via load_state().
    """
    from iai_mcp import concurrency, daemon_state
    from iai_mcp.concurrency import ProcessLock
    from iai_mcp.store import MemoryStore

    lock_path = tmp_path / ".lock"
    state_path = tmp_path / ".daemon-state.json"

    monkeypatch.setattr(concurrency, "LOCK_PATH", lock_path)
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")

    store = MemoryStore()

    # Seed a single record so _store_is_empty returns False (we want the
    # scheduler to reach the flag-gate, not the empty-store shortcut).
    from iai_mcp.types import MemoryRecord
    from uuid import uuid4
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="seed record so the store is not empty",
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    store.insert(rec)

    lock = ProcessLock(lock_path)
    yield store, lock, state_path, tmp_path
    try:
        lock.release()
    except Exception:
        pass
    lock.close()


async def _fast_rem_cycle(
    store, cycle_num, total_cycles, session_id, *, is_last, claude_enabled,
):
    """Stand-in for dream.run_rem_cycle -- completes in 0.01s."""
    await asyncio.sleep(0.01)
    return {
        "cycle": cycle_num,
        "summaries_created": 1,
        "schemas_induced": 0,
        "schema_candidates": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out": False,
    }


def _window_covering_now() -> list[int]:
    """A quiet_window [start_bucket, duration] that contains the current local time."""
    from iai_mcp.tz import load_user_tz
    tz = load_user_tz()
    now_local = datetime.now(timezone.utc).astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // 30
    start = (cur_bucket - 2) % 48
    return [start, 8]


# ---------------------------------------------------------------------------
# Test 1: scheduler_paused=True short-circuits the tick
# ---------------------------------------------------------------------------


def test_scheduler_paused_emits_skip_event_and_returns(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.daemon_state import load_state
    from iai_mcp.events import query_events

    store, lock, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "scheduler_paused": True,
        "quiet_window": _window_covering_now(),
    }

    # If the body reaches the REM loop, this mock fails the test.
    monkeypatch.setattr(daemon_mod, "run_rem_cycle", AsyncMock(
        side_effect=AssertionError("REM loop must not run when paused")
    ))

    asyncio.run(daemon_mod._tick_body(store, lock, state))

    # State reports the pause reason.
    assert state.get("last_tick_skipped_reason") == "paused"
    # Event recorded.
    events = query_events(store, kind="daemon_tick_skipped", limit=1)
    assert len(events) == 1
    assert events[0]["data"]["reason"] == "paused"
    # FSM stayed at WAKE.
    assert state["fsm_state"] == "WAKE"


# ---------------------------------------------------------------------------
# Test 2: user_sleep_request bypasses quiet-window gate
# ---------------------------------------------------------------------------


def test_user_sleep_request_bypasses_quiet_window(tick_env, monkeypatch):
    """Empty quiet_window + no recent sessions should normally skip the tick
    (outside_window). A pending user_sleep_request must override that gate
    and actually run the REM loop + clear the flag.
    """
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.daemon_state import load_state

    store, lock, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "quiet_window": None,  # Empty quiet window -- gate would normally skip.
        "user_sleep_request": {
            "reason": "I am going to bed now",
            "ts": "2026-04-18T23:00:00+00:00",
            "pending": True,
        },
        # Ensure the bootstrap idle check ALSO fails (recent session marker).
        "last_session_ts": datetime.now(timezone.utc).isoformat(),
    }

    monkeypatch.setattr(daemon_mod, "run_rem_cycle", _fast_rem_cycle)
    # Skip quiet-window relearn path entirely.
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    asyncio.run(daemon_mod._tick_body(store, lock, state))

    # Flag cleared after honoring the request.
    assert state["user_sleep_request"]["pending"] is False
    assert "honored_at" in state["user_sleep_request"]
    # FSM returned to WAKE after the full cycle loop.
    assert state["fsm_state"] == "WAKE"
    # At least one cycle completed.
    assert state.get("last_completed_cycles", 0) >= 1

    # State was persisted.
    loaded = load_state()
    assert loaded["user_sleep_request"]["pending"] is False


# ---------------------------------------------------------------------------
# Test 3: force_rem_request runs EXACTLY ONE REM cycle out of schedule
# ---------------------------------------------------------------------------


def test_force_rem_request_runs_single_cycle(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, lock, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "quiet_window": None,
        "force_rem_request": {
            "ts": "2026-04-18T10:00:00+00:00",
            "pending": True,
        },
        # rem_cycle_count=4 -- we want to confirm force_rem overrides this
        # with total_cycles=1 (NOT 4).
        "rem_cycle_count": 4,
        "last_session_ts": datetime.now(timezone.utc).isoformat(),
    }

    cycle_calls: list[int] = []

    async def _tracking_rem(
        store, cycle_num, total_cycles, session_id, *, is_last, claude_enabled,
    ):
        cycle_calls.append(cycle_num)
        await asyncio.sleep(0.005)
        return {
            "cycle": cycle_num,
            "summaries_created": 0,
            "schemas_induced": 0,
            "schema_candidates": 0,
            "claude_call_used": False,
            "main_insight_text": None,
            "timed_out": False,
        }

    monkeypatch.setattr(daemon_mod, "run_rem_cycle", _tracking_rem)
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    asyncio.run(daemon_mod._tick_body(store, lock, state))

    # Exactly ONE cycle fired despite rem_cycle_count=4 being set.
    assert cycle_calls == [1], (
        f"force_rem must bound the loop to 1 cycle, got {cycle_calls}"
    )
    # Flag cleared.
    assert state["force_rem_request"]["pending"] is False
    assert state["fsm_state"] == "WAKE"


# ---------------------------------------------------------------------------
# Test 4: force_wake_request mid-night breaks the REM loop early
# ---------------------------------------------------------------------------


def test_force_wake_request_breaks_rem_loop_early(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.events import query_events

    store, lock, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "quiet_window": _window_covering_now(),
        "rem_cycle_count": 5,
    }

    cycle_calls: list[int] = []

    async def _rem_sets_force_wake_on_second_cycle(
        store, cycle_num, total_cycles, session_id, *, is_last, claude_enabled,
    ):
        cycle_calls.append(cycle_num)
        await asyncio.sleep(0.005)
        # Halfway into the night, simulate the dispatcher flipping the flag.
        # The _tick_body loop checks force_wake_request.pending AFTER each
        # cycle completes -- so setting it on cycle 2 breaks before cycle 3.
        if cycle_num == 2:
            state["force_wake_request"] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "pending": True,
            }
        return {
            "cycle": cycle_num,
            "summaries_created": 0,
            "schemas_induced": 0,
            "schema_candidates": 0,
            "claude_call_used": False,
            "main_insight_text": None,
            "timed_out": False,
        }

    monkeypatch.setattr(daemon_mod, "run_rem_cycle", _rem_sets_force_wake_on_second_cycle)
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    asyncio.run(daemon_mod._tick_body(store, lock, state))

    # Loop broke after cycle 2; cycles 3/4/5 never ran.
    assert cycle_calls == [1, 2], (
        f"force_wake must break the loop after cycle 2, got {cycle_calls}"
    )
    # Flag cleared.
    assert state["force_wake_request"]["pending"] is False
    assert "honored_at" in state["force_wake_request"]
    # daemon_yielded event emitted with the correct reason.
    yield_events = query_events(store, kind="daemon_yielded", limit=5)
    reasons = [e["data"].get("reason") for e in yield_events]
    assert "force_wake_requested" in reasons, (
        f"expected force_wake_requested in {reasons}"
    )
    # FSM returned cleanly to WAKE.
    assert state["fsm_state"] == "WAKE"


# ---------------------------------------------------------------------------
# Test 5: flags work under concurrent state changes (realistic race)
# ---------------------------------------------------------------------------


def test_user_sleep_plus_force_rem_still_bounds_one_cycle(tick_env, monkeypatch):
    """If both user_sleep_request AND force_rem_request are pending (e.g.
    the user sent both MCP messages in quick succession), force_rem still
    constrains the loop to 1 cycle, and BOTH flags get cleared.
    """
    from iai_mcp import daemon as daemon_mod

    store, lock, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "quiet_window": None,
        "user_sleep_request": {
            "reason": "bedtime",
            "ts": "2026-04-18T23:00:00+00:00",
            "pending": True,
        },
        "force_rem_request": {
            "ts": "2026-04-18T23:00:01+00:00",
            "pending": True,
        },
        "rem_cycle_count": 4,
    }

    cycle_calls: list[int] = []

    async def _tracking_rem(
        store, cycle_num, total_cycles, session_id, *, is_last, claude_enabled,
    ):
        cycle_calls.append(cycle_num)
        await asyncio.sleep(0.005)
        return {
            "cycle": cycle_num,
            "summaries_created": 0,
            "schemas_induced": 0,
            "schema_candidates": 0,
            "claude_call_used": False,
            "main_insight_text": None,
            "timed_out": False,
        }

    monkeypatch.setattr(daemon_mod, "run_rem_cycle", _tracking_rem)
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    asyncio.run(daemon_mod._tick_body(store, lock, state))

    # force_rem bounded to 1 cycle even though rem_cycle_count=4.
    assert cycle_calls == [1]
    # Both pending flags cleared.
    assert state["user_sleep_request"]["pending"] is False
    assert state["force_rem_request"]["pending"] is False


# ---------------------------------------------------------------------------
# Test 6: paused=True state persisted AND surfaced via load_state
# ---------------------------------------------------------------------------


def test_paused_skip_persists_to_disk(tick_env, monkeypatch):
    """save_state must persist scheduler_paused+last_tick_skipped_reason so
    a daemon restart observes the same state.
    """
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.daemon_state import load_state

    store, lock, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "scheduler_paused": True,
    }

    asyncio.run(daemon_mod._tick_body(store, lock, state))

    loaded = load_state()
    assert loaded["last_tick_skipped_reason"] == "paused"
    assert loaded["scheduler_paused"] is True
    # last_tick_at is an ISO string.
    datetime.fromisoformat(loaded["last_tick_at"])
