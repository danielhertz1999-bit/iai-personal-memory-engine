"""Regression test — the cascade body must not block the event loop.

Mechanism: stub ``retrieve.build_runtime_graph`` with a sync function that
``time.sleep(5.0)``. With the ``await asyncio.to_thread(...)`` wrap,
the cascade-body sleep runs in a worker thread and a concurrent
``asyncio.sleep(0)`` + small coroutine on the same event loop completes
in <100ms. Without the wrap, the event loop is pinned for 5s.

After the off-loop refactor the daemon calls ``compute_and_fetch_warm``
(a SYNC callable) on the dedicated executor — NOT the async ``run_cascade``
coroutine. The stub is repointed to ``iai_mcp.hippea_cascade.compute_and_fetch_warm``
so the test continues to intercept the daemon's real cascade dispatch.
The stub shape changes accordingly: ``compute_and_fetch_warm`` is sync and
returns ``(records, top)`` — a plain ``def`` returning ``([], [])`` is correct.

Project async-test idiom (mandatory): sync ``def test_*`` body wraps
``asyncio.run(_async_body())``. The project does NOT depend on
``pytest-asyncio``; ``@pytest.mark.asyncio`` markers silently pass without
running.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import time
from unittest.mock import patch


def test_concurrent_coroutine_completes_under_100ms_while_cascade_sleeps_5s(monkeypatch):
    """Concurrent async work runs while cascade body is mid-sleep."""
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

    # Stub compute_and_fetch_warm — the SYNC callable the daemon now submits to
    # the dedicated executor. Returns (records, top) matching the post-refactor
    # signature. We only care about the heavy build_runtime_graph step being
    # off-loop; an instant stub here keeps the test focused.
    def fast_cascade_stub(store, assignment, **kwargs):
        return ([], [])  # (records, top) — empty warm set, no communities

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

    # Provide a real executor so the cascade loop can call run_in_executor.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="cascade-test"
    )
    monkeypatch.setattr(daemon_mod, "_cascade_executor", executor)

    # Patch the names the cascade body resolves at call time.
    # The daemon now calls compute_and_fetch_warm (SYNC) on the executor,
    # NOT run_cascade. Patch that symbol to intercept the daemon's dispatch.
    with patch("iai_mcp.retrieve.build_runtime_graph", slow_blocking_stub), \
         patch("iai_mcp.hippea_cascade.compute_and_fetch_warm", fast_cascade_stub), \
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

    executor.shutdown(wait=False)

    # The two `asyncio.sleep(0.01)` calls + coroutine overhead should
    # land WELL under 100ms if the to_thread wrap is in place. Without the
    # wrap (bare `retrieve.build_runtime_graph(store)` call), elapsed ≥ 5.0s.
    assert elapsed < 0.1, (
        f"R1 FAIL: event loop pinned for {elapsed:.3f}s while cascade body "
        f"was running. Expected <100ms. Did the "
        f"`await asyncio.to_thread(retrieve.build_runtime_graph, store)` "
        f"wrap land in src/iai_mcp/daemon.py::_hippea_cascade_loop?"
    )
