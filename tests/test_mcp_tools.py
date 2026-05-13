"""End-to-end integration tests for the TypeScript MCP wrapper.

Spawns the built wrapper as a subprocess, sends MCP-shaped JSON-RPC requests,
and verifies the wrapper exposes the 5 Phase-1 tools and round-trips the
autistic-kernel profile defaults (D-12, D-11).

deviation Rule 3 update: pre-7.1 the spawned wrapper would
self-spawn the Python daemon on first connect (the spawn-fallback chain
in bridge.ts that 07.1-04 deleted). Tests in this file relied on either
that fallback OR the user's live production daemon. wrappers
are pure connectors — if no daemon is up, they throw
DaemonUnreachableError and exit non-zero. Tests now pre-start an
isolated tmp daemon (manual `python -m iai_mcp.daemon` per D7.1-09
backward compat) via the `daemon_sock` module fixture and pass the
socket path to the wrapper through IAI_DAEMON_SOCKET_PATH so the test
never touches the user's real ~/.iai-mcp.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _wrapper_ready() -> bool:
    return (WRAPPER / "dist" / "index.js").exists()


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


@pytest.fixture(scope="module")
def daemon_sock() -> "Path":
    """Pre-start an isolated tmp daemon for the wrapper to connect to.

     removed the wrapper-side spawn-fallback;
    wrappers now ONLY connect to an existing daemon socket. In
    production launchd handles daemon spawn via socket activation; in
    tests we use the manual-run code path (no LISTEN_FDS env)
    per D7.1-09 backward compat.

    Module-scoped to amortize the ~3-10s daemon cold-start (bge-small
    embedder load + LanceDB open) across all 3 tests in this file.
    """
    sock_dir = Path(f"/tmp/iai-mcp-tools-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    store_dir = sock_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    # Module-scoped fixture can run before conftest's autouse env patch; the
    # daemon subprocess must always have a deterministic passphrase-derived
    # key path (matches tests/conftest.py _TEST_PASSPHRASE).
    env.setdefault(
        "IAI_MCP_CRYPTO_PASSPHRASE",
        "iai-mcp-test-passphrase-2026-04-30-phase-07.10",
    )
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "300"  # outlive the test module
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    daemon_proc = subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for daemon to bind socket (cold start = 3-10s on macOS).
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if sock_path.exists():
            break
        time.sleep(0.1)
    else:
        try:
            daemon_proc.kill()
        except OSError:
            pass
        raise RuntimeError(f"test daemon did not bind socket {sock_path} within 30s")

    yield sock_path

    # Teardown: stop the test daemon (matched by Popen handle, then
    # defensive env-match sweep).
    try:
        daemon_proc.terminate()
        daemon_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        daemon_proc.kill()
    sock_str = str(sock_path)
    for p in psutil.process_iter(["cmdline", "environ"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            penv = p.info.get("environ") or {}
            if penv.get("IAI_DAEMON_SOCKET_PATH") == sock_str:
                p.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(0.3)
    try:
        sock_path.unlink()
    except OSError:
        pass
    try:
        shutil.rmtree(sock_dir, ignore_errors=True)
    except OSError:
        pass


def _mcp_call(proc: subprocess.Popen, method: str, params: dict, rpc_id: int) -> dict:
    """Send a single MCP JSON-RPC message and read one response line."""
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("wrapper closed stdout before replying")
    return json.loads(line.decode())


def _spawn_wrapper(built_wrapper: Path, daemon_sock: Path | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    # route the wrapper to the test daemon socket (HIGH-4
    # lock at bridge.ts module top reads IAI_DAEMON_SOCKET_PATH from
    # process.env on each spawn).
    if daemon_sock is not None:
        env["IAI_DAEMON_SOCKET_PATH"] = str(daemon_sock)
    # Ensure the python core can find the src/ package by adding it to PYTHONPATH.
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _initialize(proc: subprocess.Popen, rpc_id: int = 1) -> None:
    """Perform the MCP initialize handshake so subsequent tools/* calls are accepted."""
    resp = _mcp_call(
        proc,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "iai-mcp-test", "version": "0.1.0"},
        },
        rpc_id,
    )
    assert "result" in resp, f"initialize failed: {resp}"
    # Send the initialized notification (no id) to complete the handshake.
    assert proc.stdin is not None
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()


def test_wrapper_lists_twelve_tools(built_wrapper: Path, daemon_sock: Path) -> None:
    """Hot surface: 5 Phase-1 + 3 + 3 + 1 = 12 tools."""
    proc = _spawn_wrapper(built_wrapper, daemon_sock)
    try:
        _initialize(proc, 1)
        resp = _mcp_call(proc, "tools/list", {}, 2)
        assert "result" in resp, f"tools/list error: {resp}"
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "memory_recall",
            "memory_reinforce",
            "memory_contradict",
            "memory_consolidate",
            "profile_get_set",
            # additions
            "curiosity_pending",
            "schema_list",
            "events_query",
            # additions
            "memory_recall_structural",
            "topology",
            "camouflaging_status",
            # addition (ambient WRITE-side capture)
            "memory_capture",
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_wrapper_profile_get_returns_live_knobs(built_wrapper: Path, daemon_sock: Path) -> None:
    proc = _spawn_wrapper(built_wrapper, daemon_sock)
    try:
        _initialize(proc, 1)
        resp = _mcp_call(
            proc,
            "tools/call",
            {"name": "profile_get_set", "arguments": {"operation": "get"}},
            2,
        )
        assert "result" in resp, f"tools/call error: {resp}"
        content = resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["live"]["literal_preservation"] == "strong"
        assert payload["live"]["masking_off"] is True
        assert payload["live"]["task_support"] == "cued_recognition"
        assert payload["live"]["scene_construction_scaffold"] is True
        # : 10 autistic-kernel + wake_depth = 11 live (AUTIST-02/08/11/12 removed).
        assert len(payload["live"]) == 11
        assert len(payload["deferred"]) == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_wrapper_memory_consolidate_runs_heavy(built_wrapper: Path, daemon_sock: Path) -> None:
    """memory_consolidate returns real sleep-cycle output
    instead of the stub ({status:queued, phase:placeholder})."""
    proc = _spawn_wrapper(built_wrapper, daemon_sock)
    try:
        _initialize(proc, 1)
        resp = _mcp_call(
            proc,
            "tools/call",
            {"name": "memory_consolidate", "arguments": {}},
            2,
        )
        assert "result" in resp, f"tools/call error: {resp}"
        content = resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["mode"] == "heavy"
        assert payload["tier"] in ("tier0", "tier1")
        assert "summaries_created" in payload
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
