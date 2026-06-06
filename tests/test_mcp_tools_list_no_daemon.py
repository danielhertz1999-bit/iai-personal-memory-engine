"""Regression test for the empty tools/list cache when the daemon is down.

Symptom (pre-fix): when the iai-mcp daemon socket was slow to bind (or
not bound at all), the wrapper's top-level `await bridge.start()` blocked
`server.connect(transport)` past the MCP client's tools/list timeout.
The client cached an empty tool list for the rest of the session ⇒
`mcp__iai-mcp__*` tools never appeared in the registry even though the
server reported "Connected".

Fix (mcp-wrapper/src/index.ts): construct the Server, register
ListToolsRequestSchema + CallToolRequestSchema handlers, assign
`oninitialized`, then `await server.connect(transport)` — BEFORE any
bridge connect attempt. `bridge.start()` is now fired async after
transport is live, and the CallToolRequest handler lazy-awaits it on
first tool invocation. `tools/list` returns from the static
`registry.listHotTools()` registry and is therefore independent of
daemon state.

This test pins that property: spawn the wrapper pointed at a NON-EXISTENT
daemon socket, complete the MCP handshake, request tools/list, and
assert it returns the full 12-tool surface within a tight time budget.

Pre-fix: this test hangs (the wrapper hangs on `connectWithTimeout(5s)`
while server.connect() never runs ⇒ MCP client sees no response on
either initialize or tools/list).
Post-fix: tools/list returns < 2s with all 12 tools.

The test deliberately does NOT use the `daemon_sock` fixture — the whole
point is to prove the wrapper serves tools/list when no daemon is up.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _wrapper_dist() -> Path:
    return WRAPPER / "dist" / "index.js"


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    """Build the TS wrapper (or assume an existing build)."""
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = _wrapper_dist()
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


def _spawn_wrapper_no_daemon(
    built_wrapper: Path,
    nonexistent_sock: Path,
) -> subprocess.Popen:
    """Spawn the wrapper pointed at a socket that does NOT exist.

    The bridge will fail-loud on its first connect attempt (lazy, after
    server.connect ⇒ transport ⇒ tools/list responsiveness). For this
    test we never invoke tools/call, so the daemon-unreachable error
    never surfaces.
    """
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["IAI_DAEMON_SOCKET_PATH"] = str(nonexistent_sock)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _send_rpc(proc: subprocess.Popen, method: str, params: dict, rpc_id: int) -> None:
    """Send a single JSON-RPC line over stdin (no read)."""
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()


def _send_notification(proc: subprocess.Popen, method: str) -> None:
    note = {"jsonrpc": "2.0", "method": method}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()


def _read_response_with_id(
    proc: subprocess.Popen,
    rpc_id: int,
    timeout_sec: float,
) -> dict:
    """Read JSON-RPC response lines from stdout until one matches rpc_id.

    The wrapper may interleave notifications / log lines with responses;
    the test must be tolerant of that. Times out hard at `timeout_sec`
    so a regression hang surfaces as a fast pytest failure rather than
    a stuck CI job.
    """
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        # Use a small per-iteration timeout via os.set_blocking on the fd
        # so this loop honours the deadline tightly. select() doesn't
        # work on the Popen pipe handle directly under all OSes; the
        # safest portable construct is a non-blocking read with a tiny
        # poll interval.
        line = proc.stdout.readline()
        if not line:
            # EOF — wrapper crashed or closed stdout. Surface stderr to
            # help debugging.
            stderr_tail = b""
            try:
                stderr_tail = proc.stderr.read() if proc.stderr else b""
            except Exception:
                pass
            raise RuntimeError(
                f"wrapper closed stdout before responding to id={rpc_id}; "
                f"stderr tail: {stderr_tail.decode(errors='replace')[-2000:]}"
            )
        line_str = line.decode(errors="replace").strip()
        if not line_str:
            continue
        try:
            msg = json.loads(line_str)
        except json.JSONDecodeError:
            # Not JSON (e.g. stray log line) — keep reading.
            continue
        if isinstance(msg, dict) and msg.get("id") == rpc_id:
            return msg
        # Otherwise it's a notification or unrelated response — ignore.
    raise TimeoutError(
        f"wrapper did not respond to id={rpc_id} within {timeout_sec}s"
    )


def test_tools_list_returns_without_daemon(
    built_wrapper: Path,
    tmp_path: Path,
) -> None:
    """The wrapper MUST return the full tools/list surface within 2s
    even when the daemon socket does not exist.

    Pre-fix this test hangs at the wrapper's top-level
    `await bridge.start()` (5s connect timeout) while the MCP transport
    has not been wired up — initialize/tools/list responses never
    arrive within the test's tolerance window.
    Post-fix `server.connect(transport)` happens before any bridge
    connect attempt, so tools/list responds from the static registry
    immediately.
    """
    nonexistent_sock = tmp_path / "iai-mcp-this-socket-does-not-exist.sock"
    assert not nonexistent_sock.exists()

    proc = _spawn_wrapper_no_daemon(built_wrapper, nonexistent_sock)
    try:
        t0 = time.monotonic()
        _send_rpc(
            proc,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "iai-mcp-no-daemon-test", "version": "0.1.0"},
            },
            1,
        )
        # tools/list must NOT depend on the daemon. We tolerate up to
        # 2 seconds total across both calls; in practice both come back
        # in well under 200ms because they're pure in-process work.
        init_resp = _read_response_with_id(proc, 1, timeout_sec=2.0)
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        # The MCP spec requires the client to send the initialized
        # notification before issuing further requests; mirror it so the
        # wrapper's oninitialized handler fires (and silently no-ops on
        # daemon_unreachable).
        _send_notification(proc, "notifications/initialized")

        _send_rpc(proc, "tools/list", {}, 2)
        list_resp = _read_response_with_id(proc, 2, timeout_sec=2.0)
        elapsed = time.monotonic() - t0

        assert "result" in list_resp, f"tools/list error: {list_resp}"
        tools = list_resp["result"]["tools"]
        names = {t["name"] for t in tools}
        expected = {
            "memory_recall",
            "memory_recall_structural",
            "memory_reinforce",
            "memory_contradict",
            "memory_capture",
            "memory_consolidate",
            "profile_get_set",
            "curiosity_pending",
            "schema_list",
            "events_query",
            "topology",
            "camouflaging_status",
            "episodes_recent",
        }
        assert names == expected, (
            f"tools/list returned {len(names)} tools, expected 13: "
            f"missing={expected - names}, extra={names - expected}"
        )
        # Total handshake + tools/list in well under the MCP client's
        # tools/list timeout window. 4s budget gives headroom for slow
        # CI hosts; in practice this completes in ~100-300ms.
        assert elapsed < 4.0, (
            f"tools/list took {elapsed:.2f}s with no daemon — pre-fix "
            f"regression: wrapper is blocking on bridge.start() before "
            f"serving tools/list."
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
