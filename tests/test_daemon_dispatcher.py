"""End-to-end round-trip tests for the daemon socket dispatcher.

Unlike tests/test_core_bedtime_inject.py (which uses _ThreadedFakeDaemon that
echoes canned OK replies), these tests spin up the REAL serve_control_socket
with the REAL _dispatch_socket_request bound to a REAL state dict. They send
each of the 6 message types as real NDJSON over a real AF_UNIX socket and
assert:
  - correct response shape per message type
  - state mutations actually persisted to ~/.iai-mcp/.daemon-state.json
    (scoped to tmp_path via monkeypatch of daemon_state.STATE_PATH)
  - invalid messages rejected with invalid_message reason code
  - unknown types rejected with unknown_message_type reason code
  - version field present in status response
  - concurrent clients handled without corruption
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_socket_paths(tmp_path, monkeypatch):
    """Redirect SOCKET_PATH + STATE_PATH to tmp_path.

    AF_UNIX on macOS caps socket paths at ~104 bytes; pytest's tmp_path can
    be too long under xdist. Use a short /tmp/iai-<pid>-<n>/ fallback for
    the socket. The state file lives under tmp_path (regular filesystem,
    no length limit).
    """
    from iai_mcp import concurrency, daemon_state

    sock_dir = Path(f"/tmp/iai-disp-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    state_path = tmp_path / ".daemon-state.json"

    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)

    try:
        # First tuple slot retained for back-compat with the `_, sock_path, ...`
        # unpacking used across the test bodies.
        yield None, sock_path, state_path
    finally:
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


async def _send_ndjson(sock_path: Path, message: dict, *, timeout: float = 5.0) -> dict:
    """Connect, send one NDJSON line, read one line back, close."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    if not line:
        raise AssertionError("daemon closed without reply")
    return json.loads(line.decode("utf-8"))


async def _with_real_dispatcher(sock_path: Path, state: dict, coro_fn):
    """Boot real serve_control_socket + real _dispatch_socket_request, run
    `coro_fn(sock_path, state)`, tear down cleanly.
    """
    from iai_mcp.concurrency import serve_control_socket

    shutdown = asyncio.Event()
    server_task = asyncio.create_task(
        serve_control_socket(
            store=None,
            state=state,
            shutdown=shutdown,
            socket_path=sock_path,
        ),
    )
    # Wait for bind.
    for _ in range(250):
        if sock_path.exists():
            break
        await asyncio.sleep(0.01)
    if not sock_path.exists():
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        raise AssertionError("socket never bound")

    try:
        result = await coro_fn(sock_path, state)
    finally:
        shutdown.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Test 1: status returns version + fsm_state + uptime + pending_digest shape
# ---------------------------------------------------------------------------


def test_status_returns_version_and_full_snapshot(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp import __version__ as pkg_version

    state = {
        "fsm_state": "WAKE",
        "daemon_started_at": "2026-04-18T00:00:00+00:00",
        "last_tick_at": "2026-04-18T12:30:00+00:00",
        "quiet_window": [44, 16],
        "pending_digest": {
            "rem_cycles_completed": 2,
            "episodes_processed": 15,
            "schemas_induced_tier0": 3,
            "claude_call_used": True,
            "main_insight_text": "deeply long verbose insight text " * 50,
        },
        "scheduler_paused": False,
    }

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "status"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is True
    # Backwards-compat keys.
    assert resp["state"] == "WAKE"
    assert isinstance(resp["uptime_sec"], (int, float))
    # Additional keys.
    assert resp["version"] == pkg_version
    assert resp["fsm_state"] == "WAKE"
    assert resp["last_tick_at"] == "2026-04-18T12:30:00+00:00"
    assert resp["quiet_window"] == [44, 16]
    assert resp["daemon_started_at"] == "2026-04-18T00:00:00+00:00"
    assert resp["scheduler_paused"] is False
    # pending_digest is truncated to top-level counters (no main_insight_text).
    pd = resp["pending_digest"]
    assert pd["rem_cycles_completed"] == 2
    assert pd["episodes_processed"] == 15
    assert pd["schemas_induced_tier0"] == 3
    assert pd["claude_call_used"] is True
    assert "main_insight_text" not in pd, (
        "truncated digest leaked verbose text over the socket"
    )


# ---------------------------------------------------------------------------
# Test 2: user_initiated_sleep persists state AND respects already_sleeping
# ---------------------------------------------------------------------------


def test_user_initiated_sleep_sets_pending_flag(short_socket_paths):
    _, sock_path, state_path = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "I am going to bed",
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp == {"ok": True, "state": "TRANSITIONING"}

    # State mutation persisted to disk.
    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    req = loaded["user_sleep_request"]
    assert req["pending"] is True
    assert req["reason"] == "I am going to bed"
    assert req["ts"] == "2026-04-18T23:00:00+00:00"


def test_user_initiated_sleep_rejects_when_already_sleeping(short_socket_paths):
    _, sock_path, state_path = short_socket_paths
    state = {"fsm_state": "DREAMING"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "redundant",
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp == {"ok": False, "reason": "already_sleeping"}

    # State was NOT mutated (no user_sleep_request written).
    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    # The dispatcher doesn't touch state in the already_sleeping branch, so
    # the file may not exist (no prior save_state call). Either way: no flag.
    assert "user_sleep_request" not in loaded


# ---------------------------------------------------------------------------
# Test 3: force_wake / force_rem set pending flags + persist
# ---------------------------------------------------------------------------


def test_force_wake_queues_flag(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "DREAMING"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "force_wake", "ts": "2026-04-18T23:45:00+00:00"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "reason": "wake_queued"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["force_wake_request"]["pending"] is True
    assert loaded["force_wake_request"]["ts"] == "2026-04-18T23:45:00+00:00"


def test_force_rem_queues_flag(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "force_rem", "ts": "2026-04-18T10:00:00+00:00"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "reason": "rem_queued"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["force_rem_request"]["pending"] is True
    assert loaded["force_rem_request"]["ts"] == "2026-04-18T10:00:00+00:00"


# ---------------------------------------------------------------------------
# Test 4: pause/resume flip scheduler_paused flag
# ---------------------------------------------------------------------------


def test_pause_then_resume_flips_flag(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        r1 = await _send_ndjson(sock_path, {"type": "pause"})
        r2 = await _send_ndjson(sock_path, {"type": "resume"})
        return r1, r2

    r1, r2 = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert r1 == {"ok": True, "paused": True}
    assert r2 == {"ok": True, "paused": False}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    # After resume, scheduler_paused must be False (the LAST value written).
    assert loaded["scheduler_paused"] is False


def test_pause_persists_True_before_resume(short_socket_paths):
    """After only pause (no resume yet), state["scheduler_paused"] is True."""
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "pause"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "paused": True}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["scheduler_paused"] is True


# ---------------------------------------------------------------------------
# Test 5: unknown type returns structured error
# ---------------------------------------------------------------------------


def test_unknown_message_type_returns_error(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "nuke_from_orbit", "ts": "whatever"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is False
    assert resp["reason"] == "unknown_message_type"
    assert resp["type"] == "nuke_from_orbit"


# ---------------------------------------------------------------------------
# Test 6: invalid messages rejected with ASVS V5 reason code
# ---------------------------------------------------------------------------


def test_invalid_message_missing_ts_on_force_wake(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "force_wake"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"
    assert "ts" in resp["error"]


def test_invalid_message_wrong_type_user_sleep(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "user_initiated_sleep", "reason": 42, "ts": "x"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"
    assert "reason" in resp["error"]


def test_invalid_message_non_string_type(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": 42})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"


def test_invalid_message_pause_wrong_seconds_type(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "pause", "seconds": "forever"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"
    assert "seconds" in resp["error"]


# ---------------------------------------------------------------------------
# Test 7: C2 guard -- dispatcher never transitions FSM directly
# ---------------------------------------------------------------------------


def test_dispatcher_does_not_transition_fsm_directly(short_socket_paths):
    """C2: the socket dispatcher thread never calls daemon.transition().
    user_initiated_sleep sets a pending flag; the FSM stays at WAKE until
    the scheduler tick picks up the flag. Without this invariant, the
    dispatcher and scheduler race on the FSM state.
    """
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "night",
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )
        return state["fsm_state"]

    fsm_after = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    # The dispatcher MUST leave fsm_state at WAKE; only the scheduler
    # transitions it (under the fcntl exclusive lock).
    assert fsm_after == "WAKE"


# ---------------------------------------------------------------------------
# Test 8: reason string clipped to 500 chars (ASVS V5 output hardening)
# ---------------------------------------------------------------------------


def test_user_initiated_sleep_reason_clipped(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    long_reason = "x" * 5000

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": long_reason,
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "state": "TRANSITIONING"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert len(loaded["user_sleep_request"]["reason"]) == 500


# ---------------------------------------------------------------------------
# Test 9: concurrent clients handled without data races
# ---------------------------------------------------------------------------


def test_concurrent_clients_both_succeed(short_socket_paths):
    """Two clients hit the socket in parallel -- the dispatcher must serve
    both without corrupting the state file or double-writing."""
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        # Issue two requests concurrently.
        coro1 = _send_ndjson(
            sock_path,
            {"type": "force_rem", "ts": "2026-04-18T01:00:00+00:00"},
        )
        coro2 = _send_ndjson(sock_path, {"type": "pause"})
        results = await asyncio.gather(coro1, coro2)
        return results

    r1, r2 = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    # Both responses well-formed; dispatcher handled each independently.
    assert r1 == {"ok": True, "reason": "rem_queued"}
    assert r2 == {"ok": True, "paused": True}

    # Both state mutations persisted.
    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["force_rem_request"]["pending"] is True
    assert loaded["scheduler_paused"] is True


# ---------------------------------------------------------------------------
# Test 10: full suite hitting all 6 message types against one daemon
# ---------------------------------------------------------------------------


def test_full_message_type_matrix_end_to_end(short_socket_paths):
    """Single live daemon instance serves all 6 message types sequentially.
    Mirrors what the CLI + MCP wrapper do in production.
    """
    _, sock_path, _ = short_socket_paths
    state = {
        "fsm_state": "WAKE",
        "daemon_started_at": "2026-04-18T00:00:00+00:00",
    }

    async def _runner(sock_path, state):
        out = {}
        out["status"] = await _send_ndjson(sock_path, {"type": "status"})
        out["user_initiated_sleep"] = await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "bedtime",
                "ts": "2026-04-18T23:30:00+00:00",
            },
        )
        out["force_rem"] = await _send_ndjson(
            sock_path,
            {"type": "force_rem", "ts": "2026-04-18T23:31:00+00:00"},
        )
        out["force_wake"] = await _send_ndjson(
            sock_path,
            {"type": "force_wake", "ts": "2026-04-18T23:32:00+00:00"},
        )
        out["pause"] = await _send_ndjson(sock_path, {"type": "pause"})
        out["resume"] = await _send_ndjson(sock_path, {"type": "resume"})
        return out

    results = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert results["status"]["ok"] is True
    assert results["status"]["fsm_state"] == "WAKE"
    assert results["user_initiated_sleep"] == {"ok": True, "state": "TRANSITIONING"}
    assert results["force_rem"] == {"ok": True, "reason": "rem_queued"}
    assert results["force_wake"] == {"ok": True, "reason": "wake_queued"}
    assert results["pause"] == {"ok": True, "paused": True}
    assert results["resume"] == {"ok": True, "paused": False}

    # All mutations land in the ONE state file.
    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["user_sleep_request"]["pending"] is True
    assert loaded["force_rem_request"]["pending"] is True
    assert loaded["force_wake_request"]["pending"] is True
    # scheduler_paused was toggled last via resume -> False.
    assert loaded["scheduler_paused"] is False
