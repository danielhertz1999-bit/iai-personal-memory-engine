"""Tests for iai_mcp.identity_audit -- Task 2.

Covers 6 behaviours from the plan:
1. continuous_audit runs s5.detect_drift_anomaly + sigma.compute_and_emit on
   each tick.
2. Audit runs regardless of daemon pause state.
3. Audit does NOT acquire the fcntl exclusive lock -- never instantiates
   ProcessLock inside the loop.
4. Audit shuts down cleanly when the shutdown event is set; task completes
   without hanging.
5. Exception inside detect_drift_anomaly is caught, identity_audit_error
   event emitted, loop continues on next tick.
6. Short interval patched -- several ticks within a fraction of a second
   produce multiple detect_drift_anomaly calls.
"""
from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# Test 1: continuous_audit calls s5.detect_drift_anomaly + sigma.compute_and_emit
# ---------------------------------------------------------------------------

def test_continuous_audit_invokes_both_underlying_calls(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []
    sigma_calls: list = []

    def fake_s5(store, window):
        s5_calls.append((store, window))
        return []

    def fake_sigma(store):
        sigma_calls.append((store,))
        return {"phase": "healthy"}

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", fake_sigma)
    # Very short tick so the test finishes quickly.
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.02)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        # Let at least one tick run.
        await asyncio.sleep(0.05)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    assert len(s5_calls) >= 1, "detect_drift_anomaly never called"
    assert len(sigma_calls) >= 1, "compute_and_emit never called"
    # window_sessions=5 as specified in the action.
    assert s5_calls[0][1] == 5


# ---------------------------------------------------------------------------
# Test 2: audit runs regardless of daemon pause state
# ---------------------------------------------------------------------------

def test_audit_runs_even_when_paused(monkeypatch):
    """C6: the daemon may be paused (state['paused_until'] in the future) but
    the audit loop does NOT consult that state and continues to tick."""
    from iai_mcp import identity_audit

    s5_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(1)
        return []

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda store: {})
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.02)

    # "Paused" daemon state is just a dict the audit does not consult --
    # still, we set it to be explicit about what C6 means.
    daemon_state = {
        "paused_until": "2099-01-01T00:00:00+00:00",
        "fsm_state": "WAKE",
    }
    # The audit does not take state at all; this is the point of the test.
    assert "paused_until" in daemon_state

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.05)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    assert len(s5_calls) >= 1, "audit did NOT fire while daemon was 'paused' (C6 violation)"


# ---------------------------------------------------------------------------
# Test 3: audit does NOT acquire fcntl exclusive (C6 MVCC-only)
# ---------------------------------------------------------------------------

def test_audit_never_acquires_exclusive_lock(monkeypatch):
    """C6 grep + runtime guard: ProcessLock.try_acquire_exclusive must never
    be called from within continuous_audit."""
    from iai_mcp import identity_audit, concurrency

    def raiser(self):
        raise AssertionError(
            "C6 violation: continuous_audit acquired ProcessLock exclusive"
        )

    monkeypatch.setattr(
        concurrency.ProcessLock, "try_acquire_exclusive", raiser
    )
    # Same for acquire_shared and holds_exclusive_nb -- audit must not touch
    # the lock at all.
    monkeypatch.setattr(concurrency.ProcessLock, "acquire_shared", raiser)
    monkeypatch.setattr(concurrency.ProcessLock, "holds_exclusive_nb", raiser)

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", lambda s, w: [])
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.02)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.05)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    # If the audit touched the lock, the raisers would fire and surface here.
    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Test 4: audit shuts down cleanly when the shutdown event is set
# ---------------------------------------------------------------------------

def test_audit_shuts_down_cleanly(monkeypatch):
    from iai_mcp import identity_audit

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", lambda s, w: [])
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    # Long interval so we rely on shutdown to break out.
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 3600)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        # Give one tick a chance to fire.
        await asyncio.sleep(0.02)
        shutdown.set()
        # Task MUST complete quickly once shutdown is set -- no 1h hang.
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Test 5: exception inside detect_drift_anomaly is caught; event emitted;
#          audit continues on next tick
# ---------------------------------------------------------------------------

def test_audit_survives_s5_exception_and_emits_event(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []
    emitted: list = []

    def flaky_s5(store, window):
        s5_calls.append(1)
        if len(s5_calls) == 1:
            raise RuntimeError("simulated s5 failure")
        return []

    def capture_event(store, kind, data, *, severity=None, **kwargs):
        emitted.append((kind, dict(data), severity))
        return None

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", flaky_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.setattr(identity_audit, "write_event", capture_event)
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.01)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.25)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    # identity_audit_error with stage=s5 must appear.
    s5_err = [e for e in emitted if e[0] == "identity_audit_error" and e[1].get("stage") == "s5"]
    assert len(s5_err) >= 1, f"no s5 error event emitted; emitted={emitted}"
    assert "simulated s5 failure" in s5_err[0][1]["error"]
    # Loop kept going -- at least 2 ticks.
    assert len(s5_calls) >= 2, (
        f"audit did not continue after s5 exception; calls={len(s5_calls)}"
    )


# ---------------------------------------------------------------------------
# Test 6: short interval -> multiple ticks in a short real time window
# ---------------------------------------------------------------------------

def test_audit_fires_multiple_times_with_short_interval(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(1)
        return []

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.03)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.25)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())
    assert len(s5_calls) >= 3, (
        f"expected >=3 ticks in 0.25s @ 0.03s interval; got {len(s5_calls)}"
    )
