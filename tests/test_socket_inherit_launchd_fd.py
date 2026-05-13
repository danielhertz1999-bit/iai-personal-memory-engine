"""Wave 2 R1 acceptance: LISTEN_FDS inherited-fd protocol.

Verifies `_inherit_launchd_socket()` and `SocketServer.serve(sock=inherited)`
end-to-end without requiring a real launchd LaunchAgent.

Tests A-D: unit-level on `_inherit_launchd_socket()` -- env-var matrix
  (present / pid-mismatch / fds-zero / non-integer) all return None.

Test E: in-process simulation -- pre-bind a real AF_UNIX listener, dup2 it
onto fd 3, set LISTEN_FDS+LISTEN_PID, call _inherit_launchd_socket(),
assert returns a socket whose getsockname() matches the bound path.

Test F: integration -- pre-bind a listener (the launchd analogue), dup2
onto fd 3, set env, run SocketServer.serve() with inherited fd, connect
from same process via asyncio.open_unix_connection, send one bogus
JSON-RPC method, assert response is ERR_METHOD_NOT_FOUND (-32601). This
proves the inherited fd flows through asyncio.start_unix_server AND that
the dispatcher reaches core.dispatch (transport-level success, not just
bind success).

fd 3 hygiene:
  socket.socket(fileno=3) takes ownership -- closing the wrapper closes
  fd 3. Each test that touches fd 3 saves+restores via os.dup/os.dup2 in
  try/finally to avoid leaks across tests (otherwise the next test gets
  a closed stderr or dangling socket on fd 3).

LISTEN_FDS protocol is platform-agnostic (systemd + launchd both honor it),
so these tests run on macOS AND Linux. Skipped only on Windows where
AF_UNIX support is recent + flaky for this pattern.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="AF_UNIX inherited-fd protocol is POSIX-only in this test scope",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _bind_to_fd_3(sock_path: Path) -> Iterator[socket.socket]:
    """Bind an AF_UNIX listener to sock_path, dup2 it onto fd 3.

    Saves whatever fd 3 was (typically nothing or stderr-dup) and restores
    on exit. socket.socket(fileno=3) inside the with-block takes ownership;
    we close any listener we still own and let the restore handle fd 3.
    """
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(128)
    try:
        try:
            saved_fd = os.dup(3)
        except OSError:
            saved_fd = None
        try:
            os.dup2(listener.fileno(), 3)
            yield listener
        finally:
            if saved_fd is not None:
                try:
                    os.dup2(saved_fd, 3)
                finally:
                    os.close(saved_fd)
            else:
                # No prior fd 3 -- close it so the next test starts clean.
                try:
                    os.close(3)
                except OSError:
                    pass
    finally:
        # Listener wrapper may already be closed if a socket.socket(fileno=3)
        # took ownership; suppress.
        try:
            listener.close()
        except OSError:
            pass


def _short_sock_path(suffix: str) -> Path:
    """Short tmp socket path under /tmp/ to dodge macOS AF_UNIX 104-byte cap."""
    sock_dir = Path(f"/tmp/iai-launchd-{os.getpid()}-{suffix}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    return sock_dir / "d.sock"


def _cleanup_sock(sock_path: Path) -> None:
    try:
        if sock_path.exists():
            sock_path.unlink()
    except OSError:
        pass
    try:
        sock_path.parent.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Unit tests A-D: env-var matrix
# ---------------------------------------------------------------------------


def test_inherit_returns_none_when_env_missing(monkeypatch):
    """Test A: no LISTEN_FDS / LISTEN_PID -> None (manual-run path)."""
    from iai_mcp.socket_server import _inherit_launchd_socket

    monkeypatch.delenv("LISTEN_FDS", raising=False)
    monkeypatch.delenv("LISTEN_PID", raising=False)

    assert _inherit_launchd_socket() is None


def test_inherit_returns_none_when_pid_mismatch(monkeypatch):
    """Test B: LISTEN_PID != os.getpid() -> None (env leak from parent)."""
    from iai_mcp.socket_server import _inherit_launchd_socket

    monkeypatch.setenv("LISTEN_FDS", "1")
    # Pick a PID that is almost certainly NOT us. PID 1 (init/launchd itself)
    # is process #1; we are not pid 1. 999999 is also extremely unlikely.
    monkeypatch.setenv("LISTEN_PID", "999999")

    assert _inherit_launchd_socket() is None


def test_inherit_returns_none_when_fds_zero(monkeypatch):
    """Test C: LISTEN_FDS=0 -> None (no fds inherited, despite pid match)."""
    from iai_mcp.socket_server import _inherit_launchd_socket

    monkeypatch.setenv("LISTEN_FDS", "0")
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))

    assert _inherit_launchd_socket() is None


def test_inherit_returns_none_on_non_integer(monkeypatch):
    """Test D: LISTEN_FDS=foo -> None (must NOT raise; caller relies on None)."""
    from iai_mcp.socket_server import _inherit_launchd_socket

    monkeypatch.setenv("LISTEN_FDS", "foo")
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))

    # Must not raise.
    result = _inherit_launchd_socket()
    assert result is None


# ---------------------------------------------------------------------------
# Test E: in-process fd-3 simulation
# ---------------------------------------------------------------------------


def test_inherit_returns_socket_when_env_correct_simulated(monkeypatch):
    """Test E: pre-bind real AF_UNIX listener, dup2 to fd 3, env set -> socket back.

    Asserts the returned socket has the bound path -- proves we got the
    listener, not some other fd.
    """
    from iai_mcp.socket_server import _inherit_launchd_socket

    sock_path = _short_sock_path("e")
    try:
        with _bind_to_fd_3(sock_path):
            monkeypatch.setenv("LISTEN_FDS", "1")
            monkeypatch.setenv("LISTEN_PID", str(os.getpid()))

            inherited = _inherit_launchd_socket()
            assert inherited is not None, "should have returned the inherited socket"
            try:
                # Verify it's the listener we bound (path matches).
                assert inherited.getsockname() == str(sock_path), (
                    f"expected bound path {sock_path}, got {inherited.getsockname()}"
                )
                # Verify non-blocking per protocol.
                assert inherited.getblocking() is False, (
                    "inherited socket must be non-blocking"
                )
            finally:
                # Closing the wrapper closes fd 3 -- _bind_to_fd_3's finally
                # block restores/closes fd 3 for us, but we owned the wrapper
                # so close it explicitly to not leak the asyncio resource.
                try:
                    inherited.close()
                except OSError:
                    pass
    finally:
        _cleanup_sock(sock_path)


# ---------------------------------------------------------------------------
# Test F: integration -- inherited fd flows through serve() to dispatcher
# ---------------------------------------------------------------------------


async def _connect_and_send_jsonrpc(
    sock_path: Path, method: str, *, timeout: float = 5.0,
) -> dict:
    """Open AF_UNIX connection, send one JSON-RPC envelope, read one line."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        envelope = {"jsonrpc": "2.0", "id": 42, "method": method, "params": {}}
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
        raise AssertionError("daemon closed without reply")
    return json.loads(line.decode("utf-8"))


def test_serve_uses_inherited_socket_path(monkeypatch, tmp_path):
    """Test F: serve(inherited fd 3) accepts JSON-RPC; bogus method -> ERR_METHOD_NOT_FOUND.

    End-to-end proof that:
      1. The launchd branch of serve() takes the inherited fd path.
      2. asyncio.start_unix_server(sock=...) accepts the pre-bound listener.
      3. The dispatcher actually serves traffic on that fd (not silently broken).
      4. core.dispatch is reached -- bogus method returns -32601, not a
         transport error.
    """
    # Per D7-14 isolate the lancedb store under tmp_path so MemoryStore()
    # doesn't write to ~/.iai-mcp.
    store_root = tmp_path / "lancedb_root"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    from iai_mcp.socket_server import SocketServer
    from iai_mcp.store import MemoryStore

    sock_path = _short_sock_path("f")

    async def _runner() -> dict:
        return await _connect_and_send_jsonrpc(sock_path, "definitely_not_a_real_method")

    async def _scenario() -> dict:
        store = MemoryStore()
        srv = SocketServer(store, idle_secs=99999)
        # Set env BEFORE serve() runs so the launchd branch is taken.
        os.environ["LISTEN_FDS"] = "1"
        os.environ["LISTEN_PID"] = str(os.getpid())
        try:
            server_task = asyncio.create_task(srv.serve(socket_path=sock_path))
            # Wait briefly for serve() to enter the async-with block (the
            # listener is already bound on fd 3, so this is just letting the
            # event loop advance to the accept-loop).
            await asyncio.sleep(0.2)
            try:
                resp = await asyncio.wait_for(_runner(), timeout=5.0)
            finally:
                srv.shutdown_event.set()
                try:
                    await asyncio.wait_for(server_task, timeout=5)
                except Exception:
                    pass
            return resp
        finally:
            os.environ.pop("LISTEN_FDS", None)
            os.environ.pop("LISTEN_PID", None)

    try:
        with _bind_to_fd_3(sock_path):
            resp = asyncio.run(_scenario())
    finally:
        _cleanup_sock(sock_path)

    # Dispatcher reached -- response is a well-formed JSON-RPC 2.0 envelope
    # with id echoed. Per -02 V3-03 fix, the bogus method now
    # raises UnknownMethodError inside core.dispatch and surfaces as a
    # top-level JSON-RPC error -32601 (no in-band-result fallback).
    # The error shape proves the inherited fd carried the request all the
    # way through asyncio.start_unix_server -> SocketServer.handle ->
    # core.dispatch. See test_socket_server_dispatch.py::
    # test_unknown_method_returns_minus_32601 for the canonical assertion.
    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 42, resp
    assert "error" in resp, resp                          # V3-03 tightening
    assert "result" not in resp, resp                     # V3-03 tightening
    assert resp["error"]["code"] == -32601, resp
    assert "definitely_not_a_real_method" in resp["error"]["message"], resp
