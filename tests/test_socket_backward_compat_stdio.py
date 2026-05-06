"""Plan 07-02 Wave 2 R6 acceptance: stdio path unchanged + parity with socket path.

`python -m iai_mcp.core` is the legacy stdio JSON-RPC entry point used by every
pre-Phase-7 wrapper instance and by ~50 existing tests. R6 mandates zero
behaviour change to that path. D7-08 satisfies it by construction (both
transports import the same core.dispatch); these tests verify that for at
least 5 representative methods the stdio response shape matches the socket
response shape.

The parity tests use independent stores (different IAI_MCP_STORE roots) -- they
assert SHAPE parity, not data parity. Data parity is covered by Wave 6
integration tests.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from .test_socket_server_dispatch import short_socket_paths  # noqa: F401

REPO = Path(__file__).resolve().parent.parent


def _spawn_stdio_core() -> subprocess.Popen:
    """R6: spawn `python -m iai_mcp.core` directly (stdio path); send JSON-RPC over stdin."""
    env = os.environ.copy()
    tmpdir = tempfile.mkdtemp(prefix="iai-mcp-stdio-test-")
    env["IAI_MCP_STORE"] = tmpdir
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.core"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _stdio_call(proc: subprocess.Popen, method: str, params: dict, req_id: int = 1) -> dict:
    """Write one NDJSON line to stdin, read one response line from stdout."""
    envelope = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(envelope) + "\n").encode("utf-8"))
    proc.stdin.flush()
    assert proc.stdout is not None
    # core.main() writes JSON-only on stdout per response; no log lines mixed in
    # (the timezone announcement goes to stderr per src/iai_mcp/core.py:1240).
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("stdio core closed stdout before replying")
    return json.loads(line.decode("utf-8"))


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_stdio_core_still_handles_session_start_payload():
    """R6: pre-Phase-7 stdio entry point unchanged; smoke `session_start_payload`."""
    proc = _spawn_stdio_core()
    try:
        resp = _stdio_call(proc, "session_start_payload", {})
        assert resp["jsonrpc"] == "2.0", resp
        assert resp["id"] == 1, resp
        assert "result" in resp, resp
        # Empty store branch returns the placeholder payload with l0/l1/l2/wake_depth.
        assert "l0" in resp["result"], resp["result"]
        assert "wake_depth" in resp["result"], resp["result"]
    finally:
        _terminate(proc)


@pytest.mark.parametrize("method,params", [
    ("session_start_payload", {}),
    ("memory_recall", {"cue": "test", "budget_tokens": 100}),
    ("profile_get", {"knob": None}),
    ("topology", {}),
    ("schema_list", {}),
])
def test_stdio_and_socket_response_shapes_match(method, params, short_socket_paths):
    """R6 parity: same method via stdio and via socket returns the same top-level keys."""
    from iai_mcp.store import MemoryStore
    from .test_socket_server_dispatch import _send_jsonrpc, _with_socket_server

    _, sock_path, _ = short_socket_paths

    # 1) Socket call (uses the tmp_path-isolated MemoryStore from the fixture).
    async def _runner(sock_path, store):
        return await _send_jsonrpc(sock_path, method, params)
    socket_resp = asyncio.run(
        _with_socket_server(sock_path, MemoryStore(), _runner)
    )

    # 2) Stdio call (separate subprocess, separate store -- only check shape).
    proc = _spawn_stdio_core()
    try:
        stdio_resp = _stdio_call(proc, method, params)
    finally:
        _terminate(proc)

    # Both must be JSON-RPC 2.0.
    assert socket_resp.get("jsonrpc") == "2.0", socket_resp
    assert stdio_resp.get("jsonrpc") == "2.0", stdio_resp

    # Top-level shape parity: result XOR error.
    assert ("result" in socket_resp) == ("result" in stdio_resp), (
        f"shape mismatch for {method}: "
        f"socket={list(socket_resp)} stdio={list(stdio_resp)}"
    )
    assert ("error" in socket_resp) == ("error" in stdio_resp), (
        f"error-key mismatch for {method}: "
        f"socket={list(socket_resp)} stdio={list(stdio_resp)}"
    )

    if "result" in socket_resp:
        socket_keys = (
            set(socket_resp["result"].keys())
            if isinstance(socket_resp["result"], dict) else set()
        )
        stdio_keys = (
            set(stdio_resp["result"].keys())
            if isinstance(stdio_resp["result"], dict) else set()
        )
        assert socket_keys == stdio_keys, (
            f"result keys differ for {method}:\n"
            f"  socket={sorted(socket_keys)}\n"
            f"  stdio ={sorted(stdio_keys)}\n"
            f"  diff(s\\t)={sorted(socket_keys - stdio_keys)}\n"
            f"  diff(t\\s)={sorted(stdio_keys - socket_keys)}"
        )
