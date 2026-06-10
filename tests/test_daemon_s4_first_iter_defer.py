from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace


def _fake_store():
    return SimpleNamespace()


def test_grace_zero_runs_first_iter_within_100ms(monkeypatch):
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


def test_grace_positive_defers_first_iter(monkeypatch):
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


def test_shutdown_during_grace_returns_cleanly(monkeypatch):
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
    await asyncio.wait_for(task, timeout=1.0)
    assert call_count["n"] == 0, (
        f"S4 ran despite shutdown during grace: call_count={call_count['n']}"
    )
    assert task.done(), "loop task did not finish"
    assert task.exception() is None, (
        f"loop raised during shutdown-in-grace: {task.exception()!r}"
    )


def test_run_offline_pass_error_still_emits_event(monkeypatch):
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
