"""Wave 2 R1 acceptance: every dispatch method reachable over socket.

Boots SocketServer (NEW per D7-07) against a tmp_path-isolated
~/.iai-mcp/.daemon.sock equivalent and asserts that JSON-RPC 2.0 envelopes
sent over the unix socket return the same response shape as the stdio path
(R1, R6 backward-compat by construction).

Reuses the short_socket_paths fixture pattern from test_daemon_dispatcher.py
(AF_UNIX 104-byte cap mitigation via /tmp/iai-disp-<pid>-<n>/d.sock).
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
    """Redirect LOCK_PATH + SOCKET_PATH + STATE_PATH to tmp_path.

    AF_UNIX on macOS caps socket paths at ~104 bytes; pytest's tmp_path can
    be too long under xdist. Use a short /tmp/iai-<pid>-<n>/ fallback for
    the socket. The state file lives under tmp_path (regular filesystem).

    Per D7-14: also point IAI_MCP_STORE at a tmp dir so MemoryStore()
    constructed inside the test gets an isolated lancedb root.
    """
    from iai_mcp import concurrency, daemon_state

    lock_path = tmp_path / ".lock"
    sock_dir = Path(f"/tmp/iai-srvdisp-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    state_path = tmp_path / ".daemon-state.json"

    monkeypatch.setattr(concurrency, "LOCK_PATH", lock_path)
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    # Per D7-14 isolate the lancedb store under tmp_path.
    store_root = tmp_path / "lancedb_root"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

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


# ---------------------------------------------------------------------------
# JSON-RPC helpers (reused by sibling Wave 2 test files)
# ---------------------------------------------------------------------------


async def _send_jsonrpc(
    sock_path: Path,
    method: str,
    params: dict | None = None,
    req_id: int | str = 1,
    *,
    timeout: float = 10.0,
) -> dict:
    """Per D7-01: send one JSON-RPC 2.0 envelope, read one response line.

    Each call opens a fresh unix-stream connection (matches the per-connection
    multiplexing pattern from D7-02; the daemon gives every client its own
    coroutine).
    """
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        envelope: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            envelope["params"] = params
        writer.write((json.dumps(envelope) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    if not line:
        raise AssertionError(f"daemon closed without reply (method={method})")
    return json.loads(line.decode("utf-8"))


async def _send_raw(sock_path: Path, raw_bytes: bytes, *, timeout: float = 5.0) -> dict:
    """Send arbitrary bytes (used to test parse error path)."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        writer.write(raw_bytes)
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


async def _with_socket_server(sock_path: Path, store, coro_fn):
    """Boot SocketServer + run coro_fn(sock_path, store), tear down cleanly.

    Idle disabled (idle_secs=99999) so the test runner controls shutdown.
    """
    from iai_mcp.socket_server import SocketServer

    srv = SocketServer(store, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

    # Wait for bind (mirrors test_daemon_dispatcher.py:108-117).
    for _ in range(250):
        if sock_path.exists():
            break
        await asyncio.sleep(0.01)
    if not sock_path.exists():
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
        raise AssertionError("socket never bound")

    try:
        result = await coro_fn(sock_path, store)
    finally:
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# R1 acceptance tests
# ---------------------------------------------------------------------------


def test_memory_recall_routed_over_socket(short_socket_paths):
    """R1: memory_recall via socket returns the same shape as stdio path."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "memory_recall",
            {"cue": "test done", "budget_tokens": 400},
            req_id=1,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 1, resp
    assert "result" in resp, resp
    assert "hits" in resp["result"], resp["result"]


def test_session_start_payload_routed(short_socket_paths):
    """R1: session_start_payload via socket returns the assembler's dict."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=2,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 2, resp
    assert "result" in resp, resp
    # Empty store path returns the SessionStartPayload(empty) JSON shape;
    # at minimum the wake_depth and l0/l1 keys must be present.
    result = resp["result"]
    assert "l0" in result and "l1" in result, result
    assert "wake_depth" in result, result


def test_profile_get_routed(short_socket_paths):
    """R1: profile_get via socket returns the 11-knob registry dict.

    Note: tools.ts wraps this as 'profile_get_set' with operation='get'/'set';
    core.dispatch only knows 'profile_get' and 'profile_set' as primitives.
    Tests against the core surface (D7-08: socket_server.py imports core.dispatch).
    """
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "profile_get", {"knob": None}, req_id=3,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 3, resp
    assert "result" in resp, resp
    # profile.profile_get(None, _profile_state) returns the full knob dict;
    # at least one canonical knob name must be present.
    result = resp["result"]
    assert isinstance(result, dict), result
    # profile.profile_get(None, ...) returns
    #   {'live': {<10 AUTIST + wake_depth>}, 'deferred': {}, 'total_knobs': 11}
    # per src/iai_mcp/profile.py (removed AUTIST-02/08/11/12).
    # keeps literal_preservation live.
    assert "live" in result, result
    assert "literal_preservation" in result["live"], result


def test_topology_routed(short_socket_paths):
    """R1: topology via socket returns the sigma snapshot or insufficient_data."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(sock_path, "topology", {}, req_id=4)

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 4, resp
    assert "result" in resp, resp
    result = resp["result"]
    # Empty-store branch returns regime='insufficient_data' with N=0 and
    # sigma=None; non-empty store returns numeric sigma. Both shapes carry
    # a 'regime' key.
    assert "regime" in result or "sigma" in result, result


def test_invalid_jsonrpc_returns_minus_32600(short_socket_paths):
    """R1: malformed envelope (jsonrpc='1.0') returns ERR_INVALID_REQUEST."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        # Bypass _send_jsonrpc helper (which always sends jsonrpc='2.0') by
        # constructing the envelope manually.
        bad = {"jsonrpc": "1.0", "id": 1, "method": "memory_recall"}
        return await _send_raw(
            sock_path, (json.dumps(bad) + "\n").encode("utf-8"),
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert "error" in resp, resp
    assert resp["error"]["code"] == -32600, resp


def test_unknown_method_returns_minus_32601(short_socket_paths):
    """V3-03 fix: unknown method raises UnknownMethodError -> JSON-RPC -32601.

    Pre-Phase-07.13: dispatch returned {"error": f"unknown method {method!r}"}
    inside the result envelope. Post-fix: dispatch raises UnknownMethodError;
    socket_server.handle catches it and emits {error: {code: -32601, ...}}.
    """
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "not_a_real_method", {}, req_id=5,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 5, resp
    assert "error" in resp, resp
    assert "result" not in resp, resp                       # V3-03 tightening
    assert resp["error"]["code"] == -32601, resp
    assert "not_a_real_method" in resp["error"]["message"], resp


def test_missing_required_param_returns_minus_32602(short_socket_paths):
    """V3-04 fix: missing required param (e.g. memory_recall without 'cue')
    raises KeyError inside dispatch -> JSON-RPC -32602 ERR_INVALID_PARAMS.

    Pre-Phase-07.13: KeyError was mapped to -32601 ERR_METHOD_NOT_FOUND
    (wrong code; "method not found" implies the route doesn't exist).
    Post-fix: KeyError maps to -32602 with message 'missing required
    param: <key>'.
    """
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        # memory_recall consumes params["cue"] (required path) at core.py:213/249/273.
        # Sending an empty params dict triggers KeyError on the first cue access.
        return await _send_jsonrpc(
            sock_path, "memory_recall", {}, req_id=6,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 6, resp
    assert "error" in resp, resp
    assert "result" not in resp, resp
    assert resp["error"]["code"] == -32602, resp
    msg = resp["error"]["message"]
    assert "missing required param" in msg, resp
    assert "cue" in msg, resp


def test_id_echoed_unchanged(short_socket_paths):
    """D7-02: response.id matches request.id verbatim across types."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        r1 = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=1,
        )
        r2 = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=999,
        )
        r3 = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id="some-string-id",
        )
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert r1["id"] == 1, r1
    assert r2["id"] == 999, r2
    assert r3["id"] == "some-string-id", r3


def test_unknown_method_does_not_crash_server(short_socket_paths):
    """R1: an unknown method must not crash the server; the next call still works."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        # First call: unknown method — V3-03 fix: must surface as JSON-RPC error.
        r_bad = await _send_jsonrpc(
            sock_path, "definitely_not_a_method", {}, req_id=100,
        )
        # Second call must succeed against the same server.
        r_good = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=101,
        )
        return r_bad, r_good

    r_bad, r_good = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert r_bad["id"] == 100, r_bad
    assert "error" in r_bad, r_bad                          # V3-03 tightening
    assert "result" not in r_bad, r_bad                     # V3-03 tightening
    assert r_good["id"] == 101, r_good
    assert "result" in r_good, r_good


def test_parse_error_returns_minus_32700(short_socket_paths):
    """D7-01: malformed JSON → ERR_PARSE_ERROR with id=None per JSON-RPC 2.0 spec."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_raw(sock_path, b"not valid json\n")

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] is None, resp
    assert "error" in resp, resp
    assert resp["error"]["code"] == -32700, resp


def test_empty_params_defaults_to_empty_dict(short_socket_paths):
    """D7-01: omitted params field → dispatch sees an empty dict, no crash."""
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        # Pass params=None to _send_jsonrpc which omits the params key.
        return await _send_jsonrpc(
            sock_path, "session_start_payload", None, req_id=200,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 200, resp
    assert "result" in resp, resp
