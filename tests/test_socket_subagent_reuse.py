"""Sub-agent socket reuse.

Spawning ephemeral child wrapper processes (the test stand-in
for sub-agents) MUST add zero new `iai_mcp.*` processes when a daemon is
already up. Previously, each spawned wrapper would fork its own
`iai_mcp.core` Python (the per-wrapper architecture has been removed).
Now every wrapper joins the singleton daemon
via the socket-first path in bridge.ts.

The HIGH-4 lock at the top of bridge.ts
(`DAEMON_SOCKET_PATH = process.env.IAI_DAEMON_SOCKET_PATH ?? path.join(
os.homedir(), '.iai-mcp', '.daemon.sock')`) propagates the test's tmp
socket path from this Python test process → spawned `node dist/index.js`
→ bridge.ts at module load. No additional plumbing needed — env vars
inherited through subprocess.Popen `env=` flow naturally to the
TypeScript runtime.

Test isolation: tmp socket dir under /tmp/iai-subagent-<pid>-<id>/ to
avoid collision with user's real daemon. Cleanup matches test-spawned
daemons by IAI_DAEMON_SOCKET_PATH in their env to avoid touching the
production daemon.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


# ---------------------------------------------------------------------------
# Fixture: built wrapper.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    """Build the TS wrapper once per test module; reuse across tests."""
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _count_iai_mcp_processes() -> dict[str, int]:
    """Snapshot iai_mcp.core / iai_mcp.daemon process counts.

    Same shape as tests/test_bridge_socket_first.py and
    tests/test_socket_fail_loud.py. Delta-snapshot strategy: assert
    (after - before) <= 0 to be robust against pre-existing host MCP
    wrappers on the developer machine.
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


def _kill_test_daemons(sock_path: Path) -> None:
    """Kill iai_mcp.daemon processes that hold sock_path open via lsof.

    setproctitle() clobbers the external psutil.Process.environ() view on
    macOS — the daemon's own os.environ is intact but external readers lose
    injected vars after the title is set.  lsof on the specific tmp socket
    path is setproctitle-proof: only the process that HOLDS the file open
    is matched.  This is path-exact and never touches the production daemon,
    which binds ~/.iai-mcp/.daemon.sock — a completely different path.
    The Popen handle terminate() is the primary kill; this helper is a
    defensive sweep for any stragglers.
    """
    target = str(sock_path)
    res = subprocess.run(
        ["lsof", "-U", "-F", "pn"],
        capture_output=True, text=True, check=False,
    )
    current: int | None = None
    pids: set[int] = set()
    for line in res.stdout.splitlines():
        if line.startswith("p"):
            try:
                current = int(line[1:])
            except ValueError:
                current = None
        elif line.startswith("n") and current is not None and line[1:] == target:
            pids.add(current)
    for pid in pids:
        try:
            cl = " ".join(psutil.Process(pid).cmdline())
            if "iai_mcp.daemon" in cl:
                psutil.Process(pid).send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _quick_recall_via_wrapper(
    built_wrapper: Path, env_overrides: dict[str, str], cue: str,
) -> dict:
    """Spawn one wrapper, send initialize + memory_recall, terminate.

    Returns the recall response (result or error). Wraps the full
    sub-agent ephemeral lifecycle in one helper so the test loop body
    stays compact.
    """
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_overrides)
    proc = subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        # MCP initialize handshake.
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "subagent-reuse-test", "version": "0.0"},
            },
        }
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((json.dumps(init) + "\n").encode("utf-8"))
        proc.stdin.flush()
        init_line = proc.stdout.readline()
        if not init_line:
            raise RuntimeError(f"sub-agent wrapper closed stdout before initialize (cue={cue!r})")
        init_resp = json.loads(init_line.decode("utf-8"))
        assert "result" in init_resp, f"initialize failed: {init_resp}"
        # Initialized notification (no id).
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write((json.dumps(note) + "\n").encode("utf-8"))
        proc.stdin.flush()

        # memory_recall via tools/call.
        recall = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory_recall",
                "arguments": {"cue": cue, "budget_tokens": 50},
            },
        }
        proc.stdin.write((json.dumps(recall) + "\n").encode("utf-8"))
        proc.stdin.flush()
        # Wait up to 5s for the response (warm-path sub-agent should be
        # well under this).
        deadline = time.monotonic() + 5.0
        line = b""
        while time.monotonic() < deadline:
            readable, _, _ = select.select([proc.stdout], [], [], 0.5)
            if readable:
                line = proc.stdout.readline()
                break
        if not line:
            raise RuntimeError(f"sub-agent recall timed out (cue={cue!r})")
        return json.loads(line.decode("utf-8"))
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_for_daemon_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    """Poll for sock_path existence at 0.1s cadence; True on bind."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def _spawn_daemon_in_background(
    sock_path: Path, store_dir: Path, idle_secs: int = 120,
) -> subprocess.Popen:
    """Pre-start a daemon manually via `python -m iai_mcp.daemon`.

    Wrappers no longer spawn the daemon themselves
    (the spawn-fallback chain in bridge.ts has been removed);
    in production launchd does the spawn via socket activation, in
    tests we use the manual-run code path (no LISTEN_FDS env
    set), which the daemon supports unchanged for backward
    compat.

    Mirrors the same helper in tests/test_bridge_socket_first.py.
    """
    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = str(idle_secs)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Test.
# ---------------------------------------------------------------------------


def test_subagent_spawns_zero_new_processes(built_wrapper, tmp_path):
    """With daemon already up, spawning 3 ephemeral sub-agent
    wrappers adds zero new iai_mcp.* processes.

    The wrappers connect to the SAME tmp socket the bootstrap wrapper
    used (HIGH-4 lock at bridge.ts module top reads
    IAI_DAEMON_SOCKET_PATH from process.env on each spawn → all three
    sub-agents see the same socket path → all three connect to the
    SAME daemon instance).
    """
    sock_dir = Path(f"/tmp/iai-subagent-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    store_dir = sock_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    assert not sock_path.exists()

    env_overrides = {
        "IAI_DAEMON_SOCKET_PATH": str(sock_path),
        "IAI_MCP_STORE": str(store_dir),
        "IAI_DAEMON_IDLE_SHUTDOWN_SECS": "120",
    }

    # Bootstrap: pre-start a daemon manually. The earlier bootstrap relied
    # on the wrapper spawn-fallback chain in bridge.ts to spawn the daemon
    # as a side-effect of the first _quick_recall_via_wrapper call. That
    # chain has been deleted — wrappers now ONLY connect; if no
    # daemon is up, they throw DaemonUnreachableError. In production
    # launchd handles the spawn via socket activation; in tests we
    # use the manual-run code path (no LISTEN_FDS env set)
    # for backward compat.
    daemon_proc = _spawn_daemon_in_background(sock_path, store_dir)
    try:
        # Wait for the daemon to bind. Cold start is empirically
        # 3-10s on macOS (bge-small load + store open + asyncio
        # start_unix_server).
        assert _wait_for_daemon_socket(sock_path, timeout_sec=30.0), (
            f"daemon did not bind socket {sock_path} within 30s"
        )
        time.sleep(0.3)

        # First wrapper recall — same shape as the pre-7.1 "bootstrap
        # call", but the wrapper now just connects to the already-up
        # daemon instead of spawning it.
        first_resp = _quick_recall_via_wrapper(
            built_wrapper, env_overrides, cue="bootstrap subagent test",
        )
        assert "result" in first_resp or "error" in first_resp, first_resp

        # Snapshot BEFORE spawning sub-agents — the daemon is now up,
        # this is the baseline we must not exceed.
        before = _count_iai_mcp_processes()
        assert before["daemon"] >= 1, (
            f"bootstrap did not leave a running daemon: {before}"
        )

        # Spawn 3 ephemeral sub-agent wrappers serially. Each does
        # init + recall + terminate, exercising the full sub-agent
        # lifecycle. Three is enough to PROVE the reuse property — the
        # assertion is "no new processes appeared", not "all three ran
        # in parallel".
        for i in range(3):
            resp = _quick_recall_via_wrapper(
                built_wrapper, env_overrides, cue=f"subagent recall #{i + 1}",
            )
            assert "result" in resp or "error" in resp, (
                f"sub-agent #{i + 1} response shape unexpected: {resp}"
            )
            # Brief pause between sub-agents — psutil snapshot in the
            # final assertion needs the disconnect from the prior
            # wrapper to settle.
            time.sleep(0.3)

        # Allow a beat for any spawned-but-not-yet-visible processes to
        # surface (defensive against psutil race).
        time.sleep(0.5)

        # CRITICAL ASSERTION: no new iai_mcp.* processes appeared during
        # the 3 sub-agent runs. This is the load-bearing invariant.
        after = _count_iai_mcp_processes()

        # FAIL-LOUD: zero iai_mcp.core spawned by sub-agent wrappers
        # (the post-Phase-7 invariant). Delta against baseline so
        # pre-existing host MCP wrappers don't blow up the assertion.
        core_delta = after["core"] - before["core"]
        assert core_delta <= 0, (
            f"FAIL-LOUD: sub-agent path spawned iai_mcp.core "
            f"(before={before['core']} after={after['core']} delta={core_delta})"
        )

        # Singleton invariant: daemon count is the SAME as before any
        # sub-agent ran. Sub-agents joined the existing daemon; they
        # did NOT spawn parallel daemons.
        daemon_delta = after["daemon"] - before["daemon"]
        assert daemon_delta == 0, (
            f"singleton violated: sub-agent path spawned an extra daemon "
            f"(before={before['daemon']} after={after['daemon']} delta={daemon_delta})"
        )
    finally:
        # Cleanup: SIGTERM the test-started daemon. The Popen handle
        # is the primary stop signal (matches our pid exactly); the
        # _kill_test_daemons env-match sweep is defensive in case the
        # Popen handle terminate() didn't deliver (e.g., if the
        # daemon went into a bedtime/dream cycle that swallowed the
        # signal briefly).
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
        _kill_test_daemons(sock_path)
        time.sleep(0.5)
        try:
            sock_path.unlink()
        except OSError:
            pass
