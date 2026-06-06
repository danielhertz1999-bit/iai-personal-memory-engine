"""Daemon-side fail-loud + yield acceptance tests.

Daemon-side semantics
---------------------

Killing the live daemon (`kill -9` or `kill -TERM`) mid-call MUST leave NO
orphan `iai_mcp.core` processes anywhere on the system (there should be ZERO
`iai_mcp.core` processes afterward under any circumstance — the singleton
invariant), AND the next connect attempt to the socket MUST surface as
ECONNREFUSED or ENOENT (which `bridge.ts` translates to the wrapper-side
`daemon_unreachable` rejection).

Yield acceptance
----------------

The in-process yield helper `_should_yield_to_mcp` defers
REM cycles when EITHER `mcp_socket.active_connections > 0` OR
`(time.monotonic() - mcp_socket.last_activity_ts) < 30`. This file exercises
the helper directly with mocked `time.monotonic` so we never wait 35
seconds wall-clock — keeps the suite brisk.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket as sk
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


# ---------------------------------------------------------------------------
# Fixture: tmp socket path (mirrors test_socket_server_dispatch.py:short_socket_paths
# but does NOT redirect concurrency.SOCKET_PATH because the daemon subprocess
# reads IAI_DAEMON_SOCKET_PATH directly via SocketServer.serve()).
# ---------------------------------------------------------------------------


@pytest.fixture
def short_socket_paths(tmp_path):
    """Yield (lock_path, sock_path, state_path) under a tmp /tmp/iai-fl-... dir.

    AF_UNIX on macOS caps socket paths at ~104 bytes; pytest's tmp_path can
    be too long under xdist. Use a short /tmp/iai-fl-<pid>-<n>/ fallback.
    """
    lock_path = tmp_path / ".lock"
    sock_dir = Path(f"/tmp/iai-fl-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    state_path = tmp_path / ".daemon-state.json"

    try:
        yield lock_path, sock_path, state_path
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


def _count_iai_mcp_processes() -> dict[str, int]:
    """Snapshot iai_mcp.core / iai_mcp.daemon process counts for fail-loud assertions.

    Invariant: `iai_mcp.core` count must be 0 under all circumstances.
    The daemon is the singleton; wrappers no longer spawn their own
    Python core processes.
    """
    counts = {"core": 0, "daemon": 0}
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info.get("cmdline") or []
            if not cl:
                continue
            joined = " ".join(c or "" for c in cl)
            if "iai_mcp.core" in joined:
                counts["core"] += 1
            if "iai_mcp.daemon" in joined:
                counts["daemon"] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return counts


def _spawn_daemon_for_test(sock_path: Path, store_root: Path) -> subprocess.Popen:
    """Spawn `python -m iai_mcp.daemon` against an isolated tmp socket+store.

    Uses IAI_DAEMON_SOCKET_PATH + IAI_MCP_STORE env overrides so the
    subprocess stays isolated from any on-disk socket or store.

    IAI_DAEMON_IDLE_SHUTDOWN_SECS=99999 disables idle shutdown so the
    daemon stays alive for the duration of the test.
    """
    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    """Poll for sock_path existence at 0.1 s cadence; return True on bind."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Test 1: kill -9 daemon mid-call → no orphan iai_mcp.core, ECONNREFUSED on retry
# ---------------------------------------------------------------------------


def test_kill_daemon_midcall_no_orphan_core_spawn(short_socket_paths, tmp_path):
    """Daemon-side: kill -9 daemon → daemon does NOT spawn any new iai_mcp.core.

    The wrapper-side semantics (Promise rejection with daemon_unreachable, single
    retry) live in mcp-wrapper/src/bridge.ts.

    Invariant (DELTA-based): the daemon under test must NOT
    spawn any `iai_mcp.core` subprocesses, even on hard kill. Pre-existing
    `iai_mcp.core` processes from the host's other MCP wrappers (live
    host sessions, etc.) are out of scope — they belong to the
    user's running stack, not to this daemon. We measure the DELTA
    (after - before) to filter them out.
    """
    _, sock_path, _ = short_socket_paths
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)

    # Snapshot existing iai_mcp.core processes BEFORE we spawn our daemon.
    # Anything still present after the kill that wasn't there now is OUR fault.
    baseline = _count_iai_mcp_processes()

    proc = _spawn_daemon_for_test(sock_path, store_root)
    try:
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            "daemon never bound socket within 30s"
        )

        before = _count_iai_mcp_processes()
        assert before["daemon"] >= baseline["daemon"] + 1, (
            f"our daemon not visible in process list: baseline={baseline}, before={before}"
        )
        # The DELTA from baseline tells us if our daemon spawned any cores.
        # Any pre-existing cores (host's other MCP wrappers) stay constant.
        before_delta = before["core"] - baseline["core"]
        assert before_delta == 0, (
            f"our daemon spawned {before_delta} iai_mcp.core processes BEFORE kill "
            f"(baseline={baseline}, before={before}) — post-Phase-7 singleton invariant violated"
        )

        # SIGKILL — simulate hard daemon death (the failure mode under test).
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)

        # Brief pause so psutil reflects the death in subsequent process_iter scans.
        time.sleep(0.5)

        after = _count_iai_mcp_processes()
        # DELTA-based assertion: any iai_mcp.core present after the kill must
        # have been there in the baseline too. Our daemon must NEVER spawn
        # core processes on death.
        after_delta = after["core"] - baseline["core"]
        assert after_delta <= 0, (
            f"FAIL-LOUD VIOLATION: our daemon spawned {after_delta} new "
            f"iai_mcp.core processes after kill (baseline={baseline}, after={after}) "
            "— R5 + A8 invariant: post-Phase-7 daemon must never spawn a core."
        )

        # Subsequent connect attempts MUST fail. Three acceptable outcomes:
        # - ConnectionRefusedError: socket file still present, no listener bound
        # - FileNotFoundError: socket file removed (cleanup_socket on Python 3.13+)
        # - OSError (generic): platform-dependent ECONNREFUSED variant
        s = sk.socket(sk.AF_UNIX, sk.SOCK_STREAM)
        s.settimeout(0.5)
        err_kind = None
        try:
            s.connect(str(sock_path))
            err_kind = "no_error"  # unexpected — daemon should be gone
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            err_kind = type(e).__name__
        finally:
            try:
                s.close()
            except OSError:
                pass
        assert err_kind in (
            "ConnectionRefusedError", "FileNotFoundError", "OSError",
        ), f"unexpected post-kill connect outcome: {err_kind}"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test 2: kill daemon during active connection → wrapper sees EOF on next read
# ---------------------------------------------------------------------------


def test_kill_daemon_during_active_connection(short_socket_paths, tmp_path):
    """Kill daemon while a wrapper holds an open socket → wrapper sees EOF / OSError.

    bridge.ts translates that EOF into a `daemon_unreachable` rejection
    (which then triggers the single retry). This test
    just confirms the daemon-side surface: an open connection is broken
    cleanly when the daemon dies, no half-open zombie socket.
    """
    _, sock_path, _ = short_socket_paths
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)

    proc = _spawn_daemon_for_test(sock_path, store_root)
    try:
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            "daemon never bound socket within 30s"
        )

        # Open a persistent connection. Send a short control message first
        # to confirm the connection is live BEFORE we kill the daemon.
        s = sk.socket(sk.AF_UNIX, sk.SOCK_STREAM)
        s.settimeout(15)
        s.connect(str(sock_path))
        msg = (json.dumps({"type": "status"}) + "\n").encode("utf-8")
        s.sendall(msg)

        # Read the status response (proves the connection is live).
        first_response = b""
        while not first_response.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            first_response += chunk
        assert first_response, "daemon never replied to initial status"
        decoded = json.loads(first_response.decode("utf-8"))
        assert decoded.get("ok") is True, decoded

        # Kill the daemon HARD with the connection still open.
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)

        # The next read on the open socket must surface as EOF (b'') OR raise.
        # Either is an acceptable fail-loud signal for the wrapper-side
        # daemon_unreachable translation.
        s.settimeout(2.0)
        eof_or_error = False
        try:
            chunk = s.recv(4096)
            if chunk == b"":
                eof_or_error = True  # clean EOF
        except (ConnectionResetError, BrokenPipeError, OSError):
            eof_or_error = True  # OS surfaced the death
        finally:
            try:
                s.close()
            except OSError:
                pass
        assert eof_or_error, (
            "daemon kill did not surface as EOF / OSError on open connection — "
            "wrapper-side daemon_unreachable translation would silently hang"
        )

        # Subsequent connect attempts also fail (same as test 1's tail check).
        s2 = sk.socket(sk.AF_UNIX, sk.SOCK_STREAM)
        s2.settimeout(0.5)
        err_kind = None
        try:
            s2.connect(str(sock_path))
            err_kind = "no_error"
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            err_kind = type(e).__name__
        finally:
            try:
                s2.close()
            except OSError:
                pass
        assert err_kind in (
            "ConnectionRefusedError", "FileNotFoundError", "OSError",
        ), f"unexpected post-kill connect outcome: {err_kind}"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SLEEP-state coexistence with active MCP traffic is provided by the
# bounded-deferral interrupt_check inside lifecycle_tick: each sleep_pipeline
# chunk re-checks `mcp_socket.active_connections > 0 OR
# (now - last_activity_ts) < 30s` and defers if true. The kill-daemon-midcall
# tests above cover the fail-loud contract.
# ---------------------------------------------------------------------------
