"""Phase 07.2-03 R1 / A1 regression test — cascade body must not block the event loop.

Mechanism: stub `retrieve.build_runtime_graph` with a sync function that
`time.sleep(5.0)`. With Plan 03's `await asyncio.to_thread(...)` wrap,
the cascade-body sleep runs in a worker thread and a concurrent
`asyncio.sleep(0)` + small coroutine on the same event loop completes
in <100ms. Without the wrap, the event loop is pinned for 5s.

Project async-test idiom (mandatory): sync `def test_*` body wraps
`asyncio.run(_async_body())`. The project does NOT depend on
`pytest-asyncio`; `@pytest.mark.asyncio` markers silently pass without
running. See tests/test_daemon_tick_flags.py:144 for the canonical pattern.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch


def test_concurrent_coroutine_completes_under_100ms_while_cascade_sleeps_5s(monkeypatch):
    """R1 acceptance: concurrent async work runs while cascade body is mid-sleep."""
    asyncio.run(_concurrent_coroutine_completes_under_100ms_body(monkeypatch))


async def _concurrent_coroutine_completes_under_100ms_body(monkeypatch):
    # Patch retrieve.build_runtime_graph at the module the cascade imports
    # from (cascade does `from iai_mcp import retrieve`; so we patch
    # `iai_mcp.retrieve.build_runtime_graph` — that's what the local-import
    # name resolution lands on inside the function body).
    sleep_duration = 5.0
    sentinel_assignment = type("Asgmt", (), {"top_communities": [], "mid_regions": {}})()

    def slow_blocking_stub(store):
        time.sleep(sleep_duration)
        # Return a 3-tuple matching real signature: (graph, assignment, rich_club).
        return (None, sentinel_assignment, [])

    # Stub run_cascade to instantly return — we only care about the heavy
    # build_runtime_graph step blocking-or-not.
    async def fast_cascade_stub(store, assignment, **kwargs):
        return {"communities_selected": 0, "records_warmed": 0}

    # Stub state I/O so the cascade body sees pending=true once.
    state_holder = {
        "fsm_state": "WAKE",
        "hippea_cascade_request": {"pending": True, "session_id": "test"},
    }

    def load_state_stub():
        return dict(state_holder)

    def save_state_stub(state):
        state_holder.clear()
        state_holder.update(state)

    # Stub write_event (called inside the cascade body via to_thread).
    def write_event_stub(*args, **kwargs):
        return None

    # Build a shutdown event that we'll set after a moment to terminate the loop.
    shutdown = asyncio.Event()

    # Reset module-level cooldown state to 0.0 so first iteration runs body.
    import iai_mcp.daemon as daemon_mod
    monkeypatch.setattr(daemon_mod, "_last_cascade_completed_at", 0.0)

    # Patch the names the cascade body resolves at call time.
    with patch("iai_mcp.retrieve.build_runtime_graph", slow_blocking_stub), \
         patch("iai_mcp.hippea_cascade.run_cascade", fast_cascade_stub), \
         patch("iai_mcp.daemon_state.load_state", load_state_stub), \
         patch("iai_mcp.daemon_state.save_state", save_state_stub), \
         patch("iai_mcp.daemon.write_event", write_event_stub):

        # Start the cascade loop as a background task.
        cascade_task = asyncio.create_task(
            daemon_mod._hippea_cascade_loop(store=None, shutdown=shutdown),
        )

        # Give the cascade a moment to enter the body and start sleeping.
        # We need cascade to BE INSIDE the to_thread sleep when we measure.
        await asyncio.sleep(0.2)

        # Now race a small coroutine that should complete in <100ms if the
        # event loop isn't blocked.
        t_start = time.monotonic()
        await asyncio.sleep(0.01)  # 10ms — basic loop responsiveness probe
        await asyncio.sleep(0.01)
        elapsed = time.monotonic() - t_start

        # Cleanup: shut down the cascade loop.
        shutdown.set()
        try:
            await asyncio.wait_for(cascade_task, timeout=sleep_duration + 2.0)
        except asyncio.TimeoutError:
            cascade_task.cancel()
            try:
                await cascade_task
            except (asyncio.CancelledError, Exception):
                pass

        # The two `asyncio.sleep(0.01)` calls + coroutine overhead should
        # land WELL under 100ms if the wrap is in place. Without the wrap
        # (bare `retrieve.build_runtime_graph(store)` call), this elapsed
        # would be ≥ 5.0s.
        assert elapsed < 0.1, (
            f"R1 FAIL: event loop pinned for {elapsed:.3f}s while cascade body "
            f"was running. Expected <100ms (wrap working). Did Plan 03's "
            f"`await asyncio.to_thread(retrieve.build_runtime_graph, store)` "
            f"land in src/iai_mcp/daemon.py::_hippea_cascade_loop?"
        )
