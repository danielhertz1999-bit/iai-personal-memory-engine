"""Tests for iai_mcp.concurrency -- the daemon control socket.

Covers the Unix-socket control plane that survived the process-lifecycle-lock
removal:
  - NDJSON status round-trip via serve_control_socket.
  - Injected dispatcher receives request dicts unchanged.
  - Stale socket cleanup lets the server bind without EADDRINUSE.
  - Socket file mode 0o600.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# fixture: isolate SOCKET_PATH into a short tmp dir
# ---------------------------------------------------------------------------

@pytest.fixture
def socket_path(tmp_path, monkeypatch):
    """Redirect module-level SOCKET_PATH to a short /tmp/iai-<pid>-<n>/ dir.

    AF_UNIX on macOS caps the path at 104 chars; pytest's tmp_path is often
    too long. We place the socket under a short /tmp directory so `bind()`
    succeeds.
    """
    from iai_mcp import concurrency
    sock_dir = Path(f"/tmp/iai-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    try:
        yield sock_path
    finally:
        # Best-effort cleanup so /tmp doesn't accumulate.
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
# socket NDJSON status round-trip
# ---------------------------------------------------------------------------

def test_socket_status_round_trip(socket_path):
    """serve_control_socket answers status with ok=true + state + uptime_sec."""
    from iai_mcp.concurrency import serve_control_socket

    state = {"fsm_state": "WAKE", "daemon_started_at": "2026-04-18T00:00:00+00:00"}

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, state=state, shutdown=shutdown,
                                 socket_path=socket_path)
        )
        # Wait for socket to appear.
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        assert socket_path.exists(), "socket never bound"

        reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
        writer.write(b'{"type":"status"}\n')
        await writer.drain()
        line = await reader.readline()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return json.loads(line)

    resp = asyncio.run(runner())

    assert resp["ok"] is True
    assert resp["state"] == "WAKE"
    # uptime_sec is a non-negative number.
    assert isinstance(resp["uptime_sec"], (int, float))


# ---------------------------------------------------------------------------
# injected dispatcher receives request dicts unchanged
# ---------------------------------------------------------------------------

def test_socket_injected_dispatcher(socket_path):
    """pause/force_rem/tail_logs routed through injected dispatcher unchanged."""
    from iai_mcp.concurrency import serve_control_socket

    received: list[dict] = []

    async def custom_dispatcher(req: dict) -> dict:
        received.append(req)
        return {"ok": True, "seen": req.get("type")}

    requests = [
        {"type": "pause", "seconds": 60},
        {"type": "force_rem"},
        {"type": "tail_logs", "n": 10},
    ]

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(
                store=None, state={}, shutdown=shutdown,
                dispatcher=custom_dispatcher, socket_path=socket_path,
            )
        )
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        assert socket_path.exists()

        responses = []
        for req in requests:
            r, w = await asyncio.open_unix_connection(path=str(socket_path))
            w.write((json.dumps(req) + "\n").encode())
            await w.drain()
            line = await r.readline()
            responses.append(json.loads(line))
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return responses

    responses = asyncio.run(runner())

    assert received == requests, f"dispatcher saw {received!r}"
    for resp, req in zip(responses, requests):
        assert resp == {"ok": True, "seen": req["type"]}


# ---------------------------------------------------------------------------
# stale socket cleanup
# ---------------------------------------------------------------------------

def test_stale_socket_cleanup(socket_path):
    """Pre-existing socket file (SIGKILL-orphaned) is cleaned so bind succeeds."""
    from iai_mcp.concurrency import serve_control_socket

    # Simulate orphaned socket file.
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_text("stale")
    assert socket_path.exists()

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, state={}, shutdown=shutdown,
                                 socket_path=socket_path)
        )
        for _ in range(100):
            if socket_path.exists() and socket_path.stat().st_size == 0:
                # Socket replaces stale file; content is empty binary.
                break
            await asyncio.sleep(0.02)
        # Quick status round-trip to confirm server is live.
        r, w = await asyncio.open_unix_connection(path=str(socket_path))
        w.write(b'{"type":"status"}\n')
        await w.drain()
        line = await r.readline()
        w.close()
        try:
            await w.wait_closed()
        except Exception:
            pass
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return json.loads(line)

    resp = asyncio.run(runner())

    assert resp.get("ok") is True


# ---------------------------------------------------------------------------
# 0o600 permissions on the socket file
# ---------------------------------------------------------------------------

def test_socket_permissions_user_only(socket_path):
    """The control socket must be 0o600 (user-only rw)."""
    from iai_mcp.concurrency import serve_control_socket

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, state={}, shutdown=shutdown,
                                 socket_path=socket_path)
        )
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        # Check socket file mode.
        sock_mode = socket_path.stat().st_mode & 0o777
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return sock_mode

    sock_mode = asyncio.run(runner())
    assert sock_mode == 0o600, f"socket mode is {oct(sock_mode)}, expected 0o600"
