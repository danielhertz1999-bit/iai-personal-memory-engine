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
    envelope = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(envelope) + "\n").encode("utf-8"))
    proc.stdin.flush()
    assert proc.stdout is not None
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
    proc = _spawn_stdio_core()
    try:
        resp = _stdio_call(proc, "session_start_payload", {})
        assert resp["jsonrpc"] == "2.0", resp
        assert resp["id"] == 1, resp
        assert "result" in resp, resp
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
    from iai_mcp.store import MemoryStore
    from .test_socket_server_dispatch import _send_jsonrpc, _with_socket_server

    _, sock_path, _ = short_socket_paths

    async def _runner(sock_path, store):
        return await _send_jsonrpc(sock_path, method, params)
    socket_resp = asyncio.run(
        _with_socket_server(sock_path, MemoryStore(), _runner)
    )

    proc = _spawn_stdio_core()
    try:
        stdio_resp = _stdio_call(proc, method, params)
    finally:
        _terminate(proc)

    assert socket_resp.get("jsonrpc") == "2.0", socket_resp
    assert stdio_resp.get("jsonrpc") == "2.0", stdio_resp

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
