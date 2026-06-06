"""Regression tests for ``iai-mcp topology`` socket-first fix.

cmd_topology previously opened MemoryStore() directly, taking the HippoDB
exclusive fcntl lock. While the live daemon holds that lock the command would
crash with HippoLockHeldError.

The fix probes the AF_UNIX socket first (JSON-RPC topology call); the daemon
answers while keeping its own lock, so there is no contention. Direct-open is
the fallback for the daemon-down case (lock free). A HippoLockHeldError guard
protects the fallback path for the mid-REM edge case where the socket times out
but the daemon still holds the lock.

All tests are fully mocked: no live daemon, no real store, no socket I/O,
no filesystem access.
"""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch


def _args() -> argparse.Namespace:
    return argparse.Namespace()


def _topology_rpc_response() -> dict:
    """Canonical JSON-RPC envelope wrapping a topology result dict."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "N": 9999,
            "C": 0.3812,
            "L": 3.2100,
            "sigma": 2.1,
            "community_count": 5,
            "rich_club_ratio": 0.17,
            "regime": "healthy",
        },
    }


# ---------------------------------------------------------------------------
# (a) Socket path — daemon up
# ---------------------------------------------------------------------------

def test_topology_socket_renders_when_daemon_up():
    """Socket returns a topology envelope -> renders all fields, rc 0."""
    from iai_mcp.cli import cmd_topology

    fake_resp = _topology_rpc_response()

    sentinel = MagicMock(side_effect=AssertionError("MemoryStore must not be called via socket path"))

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=fake_resp),
        patch("iai_mcp.store.MemoryStore", sentinel),
        redirect_stdout(buf),
    ):
        rc = cmd_topology(_args())

    assert rc == 0, f"expected rc=0, got {rc}"
    out = buf.getvalue()
    assert "N: 9999" in out, f"N line missing in: {out!r}"
    assert "regime: healthy" in out, f"regime line missing in: {out!r}"
    assert "C: 0.3812" in out, f"C line missing in: {out!r}"
    assert "communities: 5" in out, f"communities line missing in: {out!r}"
    assert "sigma: 2.1000" in out, f"sigma line missing in: {out!r}"


def test_topology_socket_does_not_open_memorystore():
    """Socket success -> MemoryStore is never instantiated (no lock contention)."""
    from iai_mcp.cli import cmd_topology

    called = []

    class _Sentinel:
        def __init__(self, *a, **k):
            called.append(True)

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=_topology_rpc_response()),
        patch("iai_mcp.store.MemoryStore", _Sentinel),
        redirect_stdout(buf),
    ):
        cmd_topology(_args())

    assert not called, "MemoryStore was instantiated on the socket path — lock contention bug"


# ---------------------------------------------------------------------------
# (b) Direct-open fallback — daemon down (socket returns None)
# ---------------------------------------------------------------------------

def test_topology_fallback_when_socket_none():
    """Socket returns None -> fallback to direct open, renders output, rc 0."""
    from iai_mcp.cli import cmd_topology

    fake_snap = {
        "N": 42,
        "C": 0.5,
        "L": 2.1,
        "sigma": 1.5,
        "community_count": 3,
        "rich_club_ratio": 0.09,
        "regime": "developmental",
    }
    fake_graph = MagicMock()
    fake_store = MagicMock()

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=None),
        patch("iai_mcp.store.MemoryStore", return_value=fake_store),
        patch("iai_mcp.retrieve.build_runtime_graph", return_value=(fake_graph, None, None)),
        patch("iai_mcp.sigma.compute_topology_snapshot", return_value=fake_snap),
        redirect_stdout(buf),
    ):
        rc = cmd_topology(_args())

    assert rc == 0, f"expected rc=0, got {rc}"
    out = buf.getvalue()
    assert "N: 42" in out, f"N line missing in: {out!r}"
    assert "regime: developmental" in out, f"regime line missing in: {out!r}"


# ---------------------------------------------------------------------------
# (c) HippoLockHeldError guard — mid-REM socket timeout
# ---------------------------------------------------------------------------

def test_topology_degrades_on_hippo_lock_held():
    """Socket returns None AND MemoryStore raises HippoLockHeldError -> rc 0, no crash.

    This covers the mid-REM edge case: socket times out (returns None) but the
    daemon is still running and holding the lock. The command must degrade to
    insufficient_data output rather than propagating the exception.
    """
    from iai_mcp.cli import cmd_topology
    from iai_mcp.hippo import HippoLockHeldError

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=None),
        patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("test.lock", "test")),
        redirect_stdout(buf),
    ):
        rc = cmd_topology(_args())

    assert rc == 0, f"expected rc=0 on HippoLockHeldError, got {rc}"
    out = buf.getvalue()
    # All fields should degrade to insufficient_data sentinel
    assert "N: insufficient_data" in out, f"N line missing in degraded output: {out!r}"
    assert "regime: insufficient_data" in out, f"regime line missing in degraded output: {out!r}"
