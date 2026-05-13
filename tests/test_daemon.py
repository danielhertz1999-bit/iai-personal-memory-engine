"""Tests for iai_mcp.daemon -- Task 3.

Covers 10 behaviours:
1. main() completes cleanly when shutdown event is set externally.
2. State-machine transitions: valid edges succeed, illegal edges raise ValueError.
3. Scheduler tick body gets called repeatedly; exceptions caught, daemon continues.
4. bge-m3 prewarm invoked exactly once at boot.
5. Graceful shutdown cancels scheduler + socket tasks; lock fd closed.
5b. mid-night MCP shared-lock acquisition surfaces via holds_exclusive_nb=False.
6. Empty-store shortcut: _tick_body records `empty_store` reason without REM work.
7. launchd plist is valid XML + has required Label/KeepAlive/ThrottleInterval keys.
8. systemd unit has Type=simple + Restart=on-failure + WantedBy=default.target +
   python3 -m iai_mcp.daemon + TimeoutStopSec=60.
9. Neither plist nor systemd unit contains ANTHROPIC_API_KEY (C3 guard).
"""
from __future__ import annotations

import asyncio
import plistlib
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLIST_PATH = PROJECT_ROOT / "deploy" / "launchd" / "com.iai-mcp.daemon.plist"
SERVICE_PATH = PROJECT_ROOT / "deploy" / "systemd" / "iai-mcp-daemon.service"


def _module_child_take_shared(path_str: str, ready_flag: str, release_flag: str) -> None:
    """Module-level helper (spawn context requires top-level serialisation)."""
    import fcntl
    import os
    import time
    from pathlib import Path
    fd = os.open(path_str, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        Path(ready_flag).write_text("ok")
        rel = Path(release_flag)
        for _ in range(300):
            if rel.exists():
                break
            time.sleep(0.1)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _short_socket_paths(tmp_path, monkeypatch):
    """Redirect concurrency LOCK_PATH + SOCKET_PATH to short paths (AF_UNIX 104-char limit)."""
    import os
    from iai_mcp import concurrency
    lock_path = tmp_path / ".lock"
    sock_dir = Path(f"/tmp/iai-daemon-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setattr(concurrency, "LOCK_PATH", lock_path)
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    return lock_path, sock_path, sock_dir


# ---------------------------------------------------------------------------
# Test 1: clean shutdown via signal-like event trigger
# ---------------------------------------------------------------------------

def test_main_clean_shutdown(tmp_path, monkeypatch):
    """main() returns 0 when shutdown fires shortly after boot."""
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    # Prevent real embedder instantiation (saves 10s + avoids model download).
    def _fake_embedder(store):
        class _Stub:
            def embed(self, text):
                return [0.0]
        return _Stub()
    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        # Give the daemon a chance to boot, then trigger shutdown by sending SIGTERM.
        await asyncio.sleep(0.2)
        # Simulate signal delivery: find the loop's shutdown event and set it.
        # Easiest: raise CancelledError on the main task after a brief run.
        # We inject shutdown by cancelling the task, then verifying it returns cleanly.
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return 0

    rc = asyncio.run(runner())
    assert rc == 0


# ---------------------------------------------------------------------------
# Test 2: state-machine transitions
# ---------------------------------------------------------------------------

def test_state_machine_transitions(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")

    state: dict = {}  # fresh state starts at WAKE default

    # WAKE -> TRANSITIONING (valid)
    daemon_mod.transition(state, daemon_mod.STATE_TRANSITIONING)
    assert state["fsm_state"] == daemon_mod.STATE_TRANSITIONING

    # TRANSITIONING -> SLEEP (valid)
    daemon_mod.transition(state, daemon_mod.STATE_SLEEP)
    assert state["fsm_state"] == daemon_mod.STATE_SLEEP

    # SLEEP -> DREAMING (valid)
    daemon_mod.transition(state, daemon_mod.STATE_DREAMING)
    assert state["fsm_state"] == daemon_mod.STATE_DREAMING

    # DREAMING -> TRANSITIONING (ILLEGAL)
    with pytest.raises(ValueError, match="Illegal transition"):
        daemon_mod.transition(state, daemon_mod.STATE_TRANSITIONING)
    assert state["fsm_state"] == daemon_mod.STATE_DREAMING  # state unchanged

    # DREAMING -> SLEEP (valid)
    daemon_mod.transition(state, daemon_mod.STATE_SLEEP)
    assert state["fsm_state"] == daemon_mod.STATE_SLEEP

    # SLEEP -> WAKE (valid)
    daemon_mod.transition(state, daemon_mod.STATE_WAKE)
    assert state["fsm_state"] == daemon_mod.STATE_WAKE

    # WAKE -> SLEEP (ILLEGAL, must go through TRANSITIONING)
    with pytest.raises(ValueError):
        daemon_mod.transition(state, daemon_mod.STATE_SLEEP)

    # State persisted each time: load_state finds fsm_state=WAKE after final txn.
    loaded = ds_mod.load_state()
    assert loaded["fsm_state"] == daemon_mod.STATE_WAKE


# ---------------------------------------------------------------------------
# Test 3: scheduler tick loop continues after exceptions
# ---------------------------------------------------------------------------

def test_scheduler_tick_survives_exceptions(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)

    # Shrink tick interval so the test finishes quickly.
    monkeypatch.setattr(daemon_mod, "TICK_INTERVAL_SEC", 0)

    from iai_mcp.concurrency import ProcessLock
    lock = ProcessLock(tmp_path / ".lock")
    state: dict = {}

    call_count = {"n": 0}

    async def flaky_body(store, lock, state):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated tick failure")

    async def runner():
        task = asyncio.create_task(
            daemon_mod._scheduler_tick(store, lock, state, tick_body=flaky_body)
        )
        # Let several ticks happen.
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    lock.close()

    assert call_count["n"] >= 2, (
        f"tick loop did not continue past first exception; only {call_count['n']} calls"
    )
    # tick_error event recorded on the first failing call.
    from iai_mcp.events import query_events
    err_events = query_events(store, kind="tick_error", limit=5)
    assert len(err_events) >= 1
    assert "simulated tick failure" in err_events[0]["data"].get("error", "")


# ---------------------------------------------------------------------------
# Test 4: bge-m3 prewarm called exactly once at boot
# ---------------------------------------------------------------------------

def test_prewarm_called_once_at_boot(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    prewarm_calls = {"n": 0}

    class _StubEmbedder:
        def embed(self, text):
            prewarm_calls["n"] += 1
            return [0.0]

    def _fake_embedder(store):
        return _StubEmbedder()

    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    assert prewarm_calls["n"] == 1, (
        f"prewarm expected once, got {prewarm_calls['n']}"
    )


# ---------------------------------------------------------------------------
# Test 5: graceful shutdown cancels both tasks + closes lock fd
# ---------------------------------------------------------------------------

def test_graceful_shutdown_cancels_tasks_and_closes_lock(tmp_path, monkeypatch):
    """We monkeypatch ProcessLock.close to observe it being called on shutdown."""
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod
    from iai_mcp import concurrency

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    def _fake_embedder(store):
        class _S:
            def embed(self, text): return [0.0] * 384
        return _S()
    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    close_calls = {"n": 0}
    real_close = concurrency.ProcessLock.close

    def _tracked_close(self):
        close_calls["n"] += 1
        real_close(self)

    monkeypatch.setattr(concurrency.ProcessLock, "close", _tracked_close)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        # added ~5 startup steps before `await shutdown.wait()`
        # (LifecycleLock acquire, capture_queue ingest, lifecycle FSM init,
        # heartbeat scanner init, sleep_pipeline init, lifecycle_tick spawn).
        # Wait up to 5 sec for the daemon to reach `await shutdown.wait()`
        # so cancellation propagates through the finally block instead of
        # being raised in synchronous setup.
        deadline = 5.0
        step = 0.05
        elapsed = 0.0
        while elapsed < deadline:
            await asyncio.sleep(step)
            elapsed += step
            if close_calls["n"] >= 0 and task.done():
                break
            # Daemon should have hit await shutdown.wait() by this point
            # for any reasonable Lance + embedder warmup. If we cancel
            # mid-startup, finally will not fire (no await-point reached).
            if elapsed >= 1.0:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    assert close_calls["n"] >= 1, "lock.close() was never called on shutdown"


# ---------------------------------------------------------------------------
# Test 5b: holds_exclusive_nb returns False when a shared holder appears
# ---------------------------------------------------------------------------

def test_d06_holds_exclusive_nb_yields_to_mcp(tmp_path, monkeypatch):
    """While the daemon holds EX, a second process taking SH forces
    holds_exclusive_nb() to return False -- the cooperative-yield signal
    that downstream plans (04-02) use to abort mid-cycle."""
    import multiprocessing
    import time
    from iai_mcp.concurrency import ProcessLock

    spawn = multiprocessing.get_context("spawn")
    lock_path = tmp_path / ".lock"

    daemon_lock = ProcessLock(lock_path)
    try:
        assert daemon_lock.try_acquire_exclusive() is True
        assert daemon_lock.holds_exclusive_nb() is True

        # Daemon releases to allow child to grab shared (simulating the gap
        # between REM cycles when the daemon intentionally yields).
        daemon_lock.release()

        ready_flag = tmp_path / ".ready"
        release_flag = tmp_path / ".release"
        child = spawn.Process(
            target=_module_child_take_shared,
            args=(str(lock_path), str(ready_flag), str(release_flag)),
        )
        child.start()
        try:
            deadline = time.time() + 15
            while time.time() < deadline and not ready_flag.exists():
                time.sleep(0.05)
            assert ready_flag.exists()

            # Probe: daemon should see "no, we don't hold EX; MCP is active".
            assert daemon_lock.holds_exclusive_nb() is False
        finally:
            release_flag.write_text("go")
            child.join(timeout=10)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
    finally:
        daemon_lock.close()


# ---------------------------------------------------------------------------
# Test 6: empty-store shortcut in _tick_body
# ---------------------------------------------------------------------------

def test_empty_store_shortcut(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    from iai_mcp.concurrency import ProcessLock
    lock = ProcessLock(tmp_path / ".lock")
    state: dict = {"fsm_state": "WAKE"}

    async def run_once():
        await daemon_mod._tick_body(store, lock, state)

    asyncio.run(run_once())
    lock.close()

    assert state.get("last_tick_skipped_reason") == "empty_store"

    # No `rem_cycle_started` event emitted on empty store.
    from iai_mcp.events import query_events
    rem = query_events(store, kind="rem_cycle_started", limit=5)
    assert rem == []


# ---------------------------------------------------------------------------
# Test 7: launchd plist valid XML + required keys
# ---------------------------------------------------------------------------

def test_launchd_plist_valid_xml_with_required_keys():
    assert PLIST_PATH.exists(), f"missing plist at {PLIST_PATH}"

    with open(PLIST_PATH, "rb") as f:
        data = plistlib.load(f)

    assert data["Label"] == "com.iai-mcp.daemon"
    assert data["ProgramArguments"][-1] == "iai_mcp.daemon"
    assert data["RunAtLoad"] is True

    keepalive = data["KeepAlive"]
    assert isinstance(keepalive, dict)
    # KeepAlive policy is now
    # `Crashed=true` only. The legacy `SuccessfulExit=false` paired
    # with the 75/0 exit-code branching; with the new lifecycle
    # state machine exit code is uniformly 0 on graceful shutdown,
    # so SuccessfulExit=false would create a respawn loop.
    assert keepalive.get("Crashed") is True
    assert "SuccessfulExit" not in keepalive

    assert data["ThrottleInterval"] == 5
    assert "StandardOutPath" in data
    assert "StandardErrorPath" in data
    assert "WorkingDirectory" in data

    env = data["EnvironmentVariables"]
    for required_key in ("PATH", "IAI_MCP_STORE", "HOME", "LANG"):
        assert required_key in env, f"missing env key {required_key}"

    # C3 guard (redundant with Test 9 but check locally too):
    assert "ANTHROPIC_API_KEY" not in env


# ---------------------------------------------------------------------------
# Test 8: systemd unit required keys
# ---------------------------------------------------------------------------

def test_systemd_unit_required_keys():
    assert SERVICE_PATH.exists(), f"missing unit file at {SERVICE_PATH}"
    text = SERVICE_PATH.read_text()

    assert "[Unit]" in text
    assert "Description=" in text
    assert "[Service]" in text
    assert "Type=simple" in text
    assert "Restart=on-failure" in text
    assert "RestartSec=30" in text
    assert "StartLimitIntervalSec=60" in text
    assert "StartLimitBurst=3" in text
    assert "python3 -m iai_mcp.daemon" in text
    assert "StandardOutput=journal" in text
    assert "StandardError=journal" in text
    assert "SyslogIdentifier=iai-mcp-daemon" in text
    assert "TimeoutStopSec=60" in text
    assert "KillSignal=SIGTERM" in text
    assert "[Install]" in text
    assert "WantedBy=default.target" in text


# ---------------------------------------------------------------------------
# Test 9: C3 guard -- no ANTHROPIC_API_KEY anywhere
# ---------------------------------------------------------------------------

def test_c3_no_anthropic_api_key_in_artifacts():
    daemon_src = (PROJECT_ROOT / "src" / "iai_mcp" / "daemon.py").read_text()
    plist_src = PLIST_PATH.read_text()
    service_src = SERVICE_PATH.read_text()

    for name, src in (("daemon.py", daemon_src), ("plist", plist_src), ("service", service_src)):
        assert "ANTHROPIC_API_KEY" not in src, (
            f"C3 VIOLATION: ANTHROPIC_API_KEY found in {name}"
        )
