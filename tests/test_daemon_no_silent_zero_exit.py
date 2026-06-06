"""Daemon shutdown exit-code contract tests.

Contract:
    Daemon main() exits 0 uniformly on graceful shutdown, regardless
    of who triggered it. The plist's `KeepAlive={"Crashed": true}`
    ensures graceful exit 0 stays DEAD until wrapper kickstart fires.
    The only path returning a non-zero exit is `LifecycleLockConflict`
    (a same-host live-PID conflict) which returns 1.

Cross-process invariant:
    The CLI `iai-mcp daemon stop` runs in a SEPARATE process from
    the daemon. CLI writes the `user_requested_shutdown=True`
    sentinel to `.daemon-state.json` BEFORE sending SIGTERM. The
    daemon's main() finally block calls
    `_clear_user_shutdown_sentinel(state)` which:
      1. Reads the on-disk state file (the source of truth, since
         the in-memory state was loaded at boot).
      2. Pops the sentinel from disk + memory.
      3. Re-saves the cleaned state record.

The sentinel is informational rather than control: its presence
on disk does not change the exit code. The write-before-SIGTERM
ordering is what makes the daemon's later cleanup symmetric across boots.
"""
from __future__ import annotations

import platform
from pathlib import Path

import pytest

from iai_mcp import cli as cli_mod
from iai_mcp import daemon as daemon_mod
from iai_mcp import daemon_state as state_mod


# ---------------------------------------------------------------------------
# Test A -- _clear_user_shutdown_sentinel: clean state -> in-memory pop only
# ---------------------------------------------------------------------------


def test_clear_sentinel_no_disk_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No sentinel on disk + no in-memory flag -> helper is a no-op.

    Locks the regression where a clean shutdown without an explicit
    `iai-mcp daemon stop` must leave the on-disk record consistent
    (no spurious sentinel write, no exception).
    """
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    state: dict = {"fsm_state": "WAKE", "daemon_pid": 12345}
    snapshot = dict(state)
    daemon_mod._clear_user_shutdown_sentinel(state)
    # In-memory dict shape is preserved (no spurious keys / drops).
    assert state == snapshot


# ---------------------------------------------------------------------------
# Test B -- sentinel True on disk -> cleared from disk + memory
# ---------------------------------------------------------------------------


def test_clear_sentinel_true_on_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Production flow: CLI process wrote sentinel to disk; daemon
    clears it on graceful exit so it does not leak across boots.
    """
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)
    state_mod.save_state(
        {"user_requested_shutdown": True, "fsm_state": "WAKE"}
    )

    daemon_in_memory: dict = {
        "fsm_state": "DREAMING",
        "daemon_pid": 999,
        # No "user_requested_shutdown" key here -- production reality.
    }
    daemon_mod._clear_user_shutdown_sentinel(daemon_in_memory)

    # Disk-side sentinel is gone.
    on_disk = state_mod.load_state()
    assert "user_requested_shutdown" not in on_disk
    # In-memory dict picked up no spurious flag.
    assert "user_requested_shutdown" not in daemon_in_memory


# ---------------------------------------------------------------------------
# Test C -- helper does not mutate unrelated keys
# ---------------------------------------------------------------------------


def test_clear_sentinel_preserves_unrelated_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The helper does exactly one in-memory mutation
    (`state.pop(_USER_SHUTDOWN_FLAG, None)`). Any future refactor
    that adds drive-by mutations would silently drop fields like
    daemon_pid / fsm_state / pending_digest, which main()'s finally
    block depends on for the doctor / next-boot pipeline.
    """
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)
    state_mod.save_state({"user_requested_shutdown": True, "fsm_state": "WAKE"})

    snapshot = {
        "fsm_state": "DREAMING",
        "daemon_pid": 42,
        "pending_digest": {"rem_cycles_completed": 79},
        "user_requested_shutdown": True,
        "fsm_transition_at": "2026-05-01T10:17:54+00:00",
    }
    state = dict(snapshot)
    daemon_mod._clear_user_shutdown_sentinel(state)
    expected = {
        k: v for k, v in snapshot.items() if k != "user_requested_shutdown"
    }
    assert state == expected


# ---------------------------------------------------------------------------
# Test D -- read failure during shutdown is fail-safe (in-memory pop only)
# ---------------------------------------------------------------------------


def test_clear_sentinel_disk_read_failure_is_fail_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If load_state() raises (transient FS error / corrupt file),
    the helper must NOT propagate -- shutdown must always proceed.
    """

    def boom() -> dict:
        raise OSError("simulated transient read error")

    monkeypatch.setattr(daemon_mod, "load_state", boom)

    state: dict = {"fsm_state": "WAKE", "user_requested_shutdown": True}
    daemon_mod._clear_user_shutdown_sentinel(state)
    # In-memory still gets popped even when disk read fails.
    assert "user_requested_shutdown" not in state


# ---------------------------------------------------------------------------
# Test E -- cmd_daemon_stop writes the sentinel BEFORE launchctl (macOS)
# ---------------------------------------------------------------------------


def test_e_cmd_daemon_stop_writes_sentinel_before_launchctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cross-process invariant:
    `iai-mcp daemon stop` writes user_requested_shutdown=True to
    .daemon-state.json BEFORE sending SIGTERM. The daemon's later
    `_clear_user_shutdown_sentinel` then cleans up. The exit code
    no longer branches on the sentinel, but the
    write-before-SIGTERM ordering is still part of the wakeup-
    safe shutdown protocol (a hung CLI write must not delay the
    SIGTERM the user expects).
    """
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    call_log: list[str] = []

    real_save_state = state_mod.save_state

    def tracking_save_state(state: dict) -> None:
        call_log.append(f"save_state:{state.get('user_requested_shutdown')}")
        real_save_state(state)

    monkeypatch.setattr(state_mod, "save_state", tracking_save_state)

    def fake_run(argv, **_kwargs):
        call_log.append(f"subprocess.run:{argv[0]}:{argv[1]}")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    # Hermeticity: the macOS stop reads the lifecycle lockfile + may signal
    # the daemon PID. Stub the read to None so the stop is bootout-only and
    # never reads the real ~/.iai-mcp/.locked or signals a real PID.
    import iai_mcp.lifecycle_lock as lifecycle_lock

    monkeypatch.setattr(lifecycle_lock.LifecycleLock, "read", lambda self: None)

    rc = cli_mod.main(["daemon", "stop"])
    assert rc == 0

    import json as json_mod
    persisted = json_mod.loads(state_path.read_text())
    assert persisted.get("user_requested_shutdown") is True

    assert call_log[0].startswith("save_state:True"), call_log
    assert any(
        entry.startswith("subprocess.run:launchctl") for entry in call_log
    ), call_log
    save_idx = next(
        i for i, e in enumerate(call_log) if e.startswith("save_state:")
    )
    launchctl_idx = next(
        i for i, e in enumerate(call_log)
        if e.startswith("subprocess.run:launchctl")
    )
    assert save_idx < launchctl_idx, call_log


# ---------------------------------------------------------------------------
# Test F -- cmd_daemon_stop writes the sentinel BEFORE systemctl (Linux)
# ---------------------------------------------------------------------------


def test_f_cmd_daemon_stop_writes_sentinel_before_systemctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Linux variant of Test E. Same ordering invariant, different
    process-supervisor command.
    """
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    call_log: list[str] = []

    real_save_state = state_mod.save_state

    def tracking_save_state(state: dict) -> None:
        call_log.append(f"save_state:{state.get('user_requested_shutdown')}")
        real_save_state(state)

    monkeypatch.setattr(state_mod, "save_state", tracking_save_state)

    def fake_run(argv, **_kwargs):
        call_log.append(f"subprocess.run:{argv[0]}")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    rc = cli_mod.main(["daemon", "stop"])
    assert rc == 0

    import json as json_mod
    persisted = json_mod.loads(state_path.read_text())
    assert persisted.get("user_requested_shutdown") is True

    save_idx = next(
        i for i, e in enumerate(call_log) if e.startswith("save_state:")
    )
    systemctl_idx = next(
        i for i, e in enumerate(call_log)
        if e.startswith("subprocess.run:systemctl")
    )
    assert save_idx < systemctl_idx, call_log


# ---------------------------------------------------------------------------
# Test G -- _USER_SHUTDOWN_FLAG constant pinned (cross-process protocol)
# ---------------------------------------------------------------------------


def test_g_user_shutdown_flag_constant_is_stable() -> None:
    """The CLI (separate process) and daemon both reference this
    string literal in different code paths; renaming it would silently
    break the cross-process protocol.
    """
    assert daemon_mod._USER_SHUTDOWN_FLAG == "user_requested_shutdown"
