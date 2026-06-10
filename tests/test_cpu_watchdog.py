from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


def test_sustained_overload_emits_exactly_one_daemon_cpu_overload_event(monkeypatch):
    asyncio.run(_sustained_overload_body(monkeypatch))


async def _sustained_overload_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    captured_events: list[tuple[str, dict, str]] = []

    def write_event_capture(store, kind, data, severity="info", **kwargs):
        captured_events.append((kind, dict(data), severity))

    monkeypatch.setattr(daemon_mod, "WATCHDOG_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_THRESHOLD_PERCENT", 50.0)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_EVENT_COOLDOWN_SEC", 300.0)
    monkeypatch.setattr(daemon_mod, "_last_overload_event_at", 0.0)
    monkeypatch.setattr(daemon_mod, "_daemon_started_monotonic", 0.0)

    sample_seq = iter([80.0, 80.0, 30.0, 80.0, 80.0, 80.0])

    class FakeProc:
        def cpu_percent(self, interval=None):
            if not getattr(self, "_primed", False):
                self._primed = True
                return 0.0
            try:
                return next(sample_seq)
            except StopIteration:
                return 0.0

    with patch("psutil.Process", return_value=FakeProc()), \
         patch("iai_mcp.daemon.write_event", write_event_capture), \
         patch("iai_mcp.daemon.load_state", lambda: {"fsm_state": "DREAMING"}):

        shutdown = asyncio.Event()
        store = MagicMock()
        task = asyncio.create_task(daemon_mod._cpu_watchdog_loop(store, shutdown))

        await asyncio.sleep(1.5)
        shutdown.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    overload_events = [e for e in captured_events if e[0] == "daemon_cpu_overload"]

    assert len(overload_events) == 1, (
        f"Expected exactly 1 daemon_cpu_overload event; got "
        f"{len(overload_events)}: {overload_events}"
    )

    kind, data, severity = overload_events[0]
    assert severity == "critical"
    assert data["fsm_state"] == "DREAMING"
    assert data["threshold_pct"] == 50.0
    assert data["sustained_sec"] == int(0.05 * 2)
    assert "cpu_samples_pct" in data
    assert all(s >= 0 for s in data["cpu_samples_pct"])
    assert "active_tasks" in data
    assert "uptime_sec" in data


def test_below_threshold_emits_no_event(monkeypatch):
    asyncio.run(_below_threshold_body(monkeypatch))


async def _below_threshold_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    captured_events: list[tuple[str, dict, str]] = []

    def write_event_capture(store, kind, data, severity="info", **kwargs):
        captured_events.append((kind, dict(data), severity))

    monkeypatch.setattr(daemon_mod, "WATCHDOG_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_THRESHOLD_PERCENT", 50.0)
    monkeypatch.setattr(daemon_mod, "_last_overload_event_at", 0.0)

    class FakeProc:
        def cpu_percent(self, interval=None):
            if not getattr(self, "_primed", False):
                self._primed = True
                return 0.0
            return 30.0

    with patch("psutil.Process", return_value=FakeProc()), \
         patch("iai_mcp.daemon.write_event", write_event_capture), \
         patch("iai_mcp.daemon.load_state", lambda: {"fsm_state": "WAKE"}):

        shutdown = asyncio.Event()
        store = MagicMock()
        task = asyncio.create_task(daemon_mod._cpu_watchdog_loop(store, shutdown))
        await asyncio.sleep(1.0)
        shutdown.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    overload_events = [e for e in captured_events if e[0] == "daemon_cpu_overload"]
    assert overload_events == [], (
        f"Expected zero daemon_cpu_overload events under sub-threshold "
        f"samples; got {overload_events}"
    )


def test_event_cooldown_prevents_ledger_flood(monkeypatch):
    asyncio.run(_event_cooldown_body(monkeypatch))


async def _event_cooldown_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    captured_events: list[tuple[str, dict, str]] = []

    def write_event_capture(store, kind, data, severity="info", **kwargs):
        captured_events.append((kind, dict(data), severity))

    monkeypatch.setattr(daemon_mod, "WATCHDOG_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_THRESHOLD_PERCENT", 50.0)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_EVENT_COOLDOWN_SEC", 300.0)
    monkeypatch.setattr(daemon_mod, "_last_overload_event_at", 0.0)

    class FakeProc:
        def cpu_percent(self, interval=None):
            if not getattr(self, "_primed", False):
                self._primed = True
                return 0.0
            return 90.0

    with patch("psutil.Process", return_value=FakeProc()), \
         patch("iai_mcp.daemon.write_event", write_event_capture), \
         patch("iai_mcp.daemon.load_state", lambda: {"fsm_state": "DREAMING"}):

        shutdown = asyncio.Event()
        store = MagicMock()
        task = asyncio.create_task(daemon_mod._cpu_watchdog_loop(store, shutdown))
        await asyncio.sleep(1.5)
        shutdown.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    overload_events = [e for e in captured_events if e[0] == "daemon_cpu_overload"]
    assert len(overload_events) == 1, (
        f"D7.2-20 cooldown failed: expected 1 event under persistent "
        f"overload; got {len(overload_events)}"
    )
