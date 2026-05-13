"""W1 / tests for the startup grace before the first
`_s4_offline_loop` iteration.

Defends against the regression where a freshly-spawned daemon immediately
runs the heavy S4 viability scan (sigma.compute_and_emit ->
retrieve.build_runtime_graph -> runtime_graph_cache.save -> json.dumps),
materialising a multi-GB intermediate Python string (CONTEXT.md D-01:
py-spy 2026-04-29 PID 7959 RSS 7.6GB).

Project async-test idiom (mandatory): sync `def test_X(...)` body wraps
`asyncio.run(_async_body(...))`. The project does NOT depend on
`pytest-asyncio`; `@pytest.mark.asyncio` markers silently pass without
running. See tests/test_cpu_watchdog.py:12, tests/test_cascade_no_block.py:11
for the canonical pattern. The plan template prescribed pytest-asyncio
markers; this file deviates (Rule 1 — fake-GREEN avoidance) per project
precedent.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_store():
    """_s4_offline_loop only forwards `store` to s4.run_offline_pass and
    write_event; both are stubbed in these tests, so a SimpleNamespace
    placeholder is enough — never touches LanceDB.
    """
    return SimpleNamespace()


# ---------------------------------------------------------------------------
# Test 1: grace=0 fast-path — first iter runs within ≤100ms
# ---------------------------------------------------------------------------

def test_grace_zero_runs_first_iter_within_100ms(monkeypatch):
    """D-06 (a): grace=0 => stubbed run_offline_pass invoked within ≤100ms."""
    asyncio.run(_grace_zero_fast_path_body(monkeypatch))


async def _grace_zero_fast_path_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "S4_FIRST_ITER_GRACE_SEC", 0.0)
    called = asyncio.Event()
    call_count = {"n": 0}

    def _stub_run_offline_pass(_store):
        call_count["n"] += 1
        called.set()

    monkeypatch.setattr(daemon_mod.s4, "run_offline_pass", _stub_run_offline_pass)
    shutdown = asyncio.Event()
    store = _fake_store()
    t0 = time.monotonic()
    task = asyncio.create_task(daemon_mod._s4_offline_loop(store, shutdown))
    try:
        await asyncio.wait_for(called.wait(), timeout=0.1)
        elapsed = time.monotonic() - t0
        assert elapsed <= 0.15, (
            f"first run_offline_pass took {elapsed*1000:.1f}ms; expected <=100ms "
            f"(plus ~50ms slack for to_thread schedule)"
        )
    finally:
        shutdown.set()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# Test 2: grace>0 deferred-path — no call before grace, ≥1 call after
# ---------------------------------------------------------------------------

def test_grace_positive_defers_first_iter(monkeypatch):
    """D-06 (b): grace=0.5 => no call before 0.4s; ≥1 call after 0.7s."""
    asyncio.run(_grace_positive_deferred_body(monkeypatch))


async def _grace_positive_deferred_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "S4_FIRST_ITER_GRACE_SEC", 0.5)
    call_count = {"n": 0}

    def _stub_run_offline_pass(_store):
        call_count["n"] += 1

    monkeypatch.setattr(daemon_mod.s4, "run_offline_pass", _stub_run_offline_pass)
    shutdown = asyncio.Event()
    store = _fake_store()
    task = asyncio.create_task(daemon_mod._s4_offline_loop(store, shutdown))
    try:
        await asyncio.sleep(0.4)
        assert call_count["n"] == 0, (
            f"S4 ran before 0.5s grace elapsed: call_count={call_count['n']}"
        )
        # Total ~0.7s — past 0.5s grace + to_thread schedule slack.
        await asyncio.sleep(0.3)
        assert call_count["n"] >= 1, (
            f"S4 did not run after grace elapsed: call_count={call_count['n']}"
        )
    finally:
        shutdown.set()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Test 3: shutdown during grace — clean return, no run, no exception
# ---------------------------------------------------------------------------

def test_shutdown_during_grace_returns_cleanly(monkeypatch):
    """shutdown set during grace => loop returns cleanly, 0 calls."""
    asyncio.run(_shutdown_during_grace_body(monkeypatch))


async def _shutdown_during_grace_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "S4_FIRST_ITER_GRACE_SEC", 5.0)
    call_count = {"n": 0}

    def _stub_run_offline_pass(_store):
        call_count["n"] += 1

    monkeypatch.setattr(daemon_mod.s4, "run_offline_pass", _stub_run_offline_pass)
    shutdown = asyncio.Event()
    store = _fake_store()
    task = asyncio.create_task(daemon_mod._s4_offline_loop(store, shutdown))
    await asyncio.sleep(0.05)
    shutdown.set()
    # raises if loop did not return cleanly within 1s.
    await asyncio.wait_for(task, timeout=1.0)
    assert call_count["n"] == 0, (
        f"S4 ran despite shutdown during grace: call_count={call_count['n']}"
    )
    assert task.done(), "loop task did not finish"
    assert task.exception() is None, (
        f"loop raised during shutdown-in-grace: {task.exception()!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: existing s4_offline_pass_error event-emit preserved
# ---------------------------------------------------------------------------

def test_run_offline_pass_error_still_emits_event(monkeypatch):
    """Existing layered-defense preserved: run_offline_pass raises => write_event
    called with kind='s4_offline_pass_error' + severity='warning'.
    """
    asyncio.run(_error_event_preserved_body(monkeypatch))


async def _error_event_preserved_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "S4_FIRST_ITER_GRACE_SEC", 0.0)
    events: list[tuple[str, dict, str]] = []

    def _stub_run_offline_pass(_store):
        raise RuntimeError("boom")

    def _stub_write_event(_store, kind, payload, severity="info", **_kwargs):
        events.append((kind, dict(payload) if isinstance(payload, dict) else payload, severity))

    monkeypatch.setattr(daemon_mod.s4, "run_offline_pass", _stub_run_offline_pass)
    monkeypatch.setattr(daemon_mod, "write_event", _stub_write_event)
    shutdown = asyncio.Event()
    store = _fake_store()
    task = asyncio.create_task(daemon_mod._s4_offline_loop(store, shutdown))
    # Give the loop time to: enter while-body, hit run_offline_pass raise,
    # emit s4_offline_pass_error, then await the inter-iteration wait_for.
    await asyncio.sleep(0.1)
    shutdown.set()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    matching = [
        e for e in events
        if e[0] == "s4_offline_pass_error"
        and e[2] == "warning"
        and "boom" in str(e[1])
    ]
    assert matching, f"expected s4_offline_pass_error event with severity=warning + 'boom' payload, got: {events}"
