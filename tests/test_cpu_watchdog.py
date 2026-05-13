"""-05 R5 / A5 regression test — CPU watchdog emits one event under sustained overload.

Mock psutil.Process.cpu_percent with a scripted sequence so the test runs
in seconds instead of 75s wall time. D7.2-23 explicitly allows mocks for
heavy-dep tests. The synthetic-CPU-burner approach (real 80% CPU thread)
is documented in SPEC A5 but is impractical for the unit suite; we test
the SAME contract (sustained > threshold => one event) with deterministic
sample injection.

Project async-test idiom (mandatory): sync `def test_X(...)` body wraps
`asyncio.run(_async_body())`. The project does NOT depend on
`pytest-asyncio`; `@pytest.mark.asyncio` markers silently pass without
running. See tests/test_daemon_tick_flags.py:144 for the canonical pattern.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


def test_sustained_overload_emits_exactly_one_daemon_cpu_overload_event(monkeypatch):
    """A5 acceptance: 2 consecutive samples > threshold => 1 critical event."""
    asyncio.run(_sustained_overload_body(monkeypatch))


async def _sustained_overload_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    captured_events: list[tuple[str, dict, str]] = []

    def write_event_capture(store, kind, data, severity="info", **kwargs):
        captured_events.append((kind, dict(data), severity))

    # Reduce poll cadence so the test loop completes in <2 seconds.
    monkeypatch.setattr(daemon_mod, "WATCHDOG_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_THRESHOLD_PERCENT", 50.0)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_EVENT_COOLDOWN_SEC", 300.0)
    monkeypatch.setattr(daemon_mod, "_last_overload_event_at", 0.0)
    monkeypatch.setattr(daemon_mod, "_daemon_started_monotonic", 0.0)

    # Scripted CPU samples: prime call returns 0.0 (psutil first-call rule),
    # then 80, 80, 30, 80, 80 — should trigger ONCE on the second 80
    # (after cooldown the next two-80 burst would NOT trigger since we
    # only run ~2s and cooldown is 300s).
    sample_seq = iter([80.0, 80.0, 30.0, 80.0, 80.0, 80.0])

    class FakeProc:
        def cpu_percent(self, interval=None):
            # Prime call (the first call returns 0.0 per psutil docs).
            # We mimic this: first call = 0.0; subsequent calls = next()
            # from the scripted sequence.
            if not getattr(self, "_primed", False):
                self._primed = True
                return 0.0
            try:
                return next(sample_seq)
            except StopIteration:
                return 0.0

    # Patch psutil.Process to return our fake proc.
    # Watchdog body uses `import psutil` locally; patch the underlying class.
    with patch("psutil.Process", return_value=FakeProc()), \
         patch("iai_mcp.daemon.write_event", write_event_capture), \
         patch("iai_mcp.daemon.load_state", lambda: {"fsm_state": "DREAMING"}):

        shutdown = asyncio.Event()
        store = MagicMock()
        task = asyncio.create_task(daemon_mod._cpu_watchdog_loop(store, shutdown))

        # Run the watchdog for ~1.5s — at 0.05s poll, that's ~30 samples,
        # plenty for the scripted 6-sample sequence + trigger.
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

    # Filter to overload events only.
    overload_events = [e for e in captured_events if e[0] == "daemon_cpu_overload"]

    # A5: exactly one event.
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
    """Control: samples below threshold => no event."""
    asyncio.run(_below_threshold_body(monkeypatch))


async def _below_threshold_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    captured_events: list[tuple[str, dict, str]] = []

    def write_event_capture(store, kind, data, severity="info", **kwargs):
        captured_events.append((kind, dict(data), severity))

    monkeypatch.setattr(daemon_mod, "WATCHDOG_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_THRESHOLD_PERCENT", 50.0)
    monkeypatch.setattr(daemon_mod, "_last_overload_event_at", 0.0)

    # All samples below threshold.
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
    """D7.2-20: at most one event per WATCHDOG_EVENT_COOLDOWN_SEC."""
    asyncio.run(_event_cooldown_body(monkeypatch))


async def _event_cooldown_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    captured_events: list[tuple[str, dict, str]] = []

    def write_event_capture(store, kind, data, severity="info", **kwargs):
        captured_events.append((kind, dict(data), severity))

    monkeypatch.setattr(daemon_mod, "WATCHDOG_POLL_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "WATCHDOG_THRESHOLD_PERCENT", 50.0)
    # Long cooldown so a 2nd trigger is blocked.
    monkeypatch.setattr(daemon_mod, "WATCHDOG_EVENT_COOLDOWN_SEC", 300.0)
    monkeypatch.setattr(daemon_mod, "_last_overload_event_at", 0.0)

    # Persistent overload — every post-prime sample = 90.
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
        await asyncio.sleep(1.5)  # plenty of time for 30 samples
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
    # Cooldown should clamp it to exactly 1.
    assert len(overload_events) == 1, (
        f"D7.2-20 cooldown failed: expected 1 event under persistent "
        f"overload; got {len(overload_events)}"
    )
