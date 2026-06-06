"""Regression tests: socket-first read commands honor IAI_MCP_STORE isolation.

When IAI_MCP_STORE points at a custom (non-default) store and
IAI_DAEMON_SOCKET_PATH is NOT set, commands that normally probe the daemon
socket first must NOT attempt a connection to the default daemon socket — they
must fall through directly to their own direct-open path that reads the custom
store. This prevents information disclosure from the live daemon leaking into
tests or operator runs that intend to read an isolated store.

Tests use only tmp_path + monkeypatch.setenv; no real socket or ~/.iai-mcp/.
"""
from __future__ import annotations

import asyncio
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iai_mcp.cli import _send_jsonrpc_request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_stdout(fn) -> tuple[str, int]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn()
    return buf.getvalue(), rc


def _make_connected_asyncmock():
    """Spy that returns a connected reader/writer pair with foreign data.

    If this is ever *awaited*, the caller receives a fake reader that yields a
    JSON-RPC success payload.  The spy also records the call so we can assert
    open_unix_connection was (or was not) called.
    """
    foreign_response = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"events": [], "count": 0}}
    ).encode() + b"\n"

    reader = AsyncMock()
    reader.readline = AsyncMock(return_value=foreign_response)

    writer = AsyncMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    spy = AsyncMock(return_value=(reader, writer))
    return spy


# ---------------------------------------------------------------------------
# _is_custom_store() unit tests
# ---------------------------------------------------------------------------


class TestIsCustomStore:
    """Unit tests for the _is_custom_store() predicate."""

    def test_custom_store_returns_true(self, tmp_path, monkeypatch):
        """IAI_MCP_STORE set to a different dir than DEFAULT_STORAGE_PATH → True."""
        from iai_mcp.cli import _is_custom_store

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        assert _is_custom_store() is True

    def test_unset_store_returns_false(self, monkeypatch):
        """IAI_MCP_STORE not set → False (default store)."""
        from iai_mcp.cli import _is_custom_store

        monkeypatch.delenv("IAI_MCP_STORE", raising=False)
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        assert _is_custom_store() is False

    def test_store_set_to_default_path_returns_false(self, monkeypatch):
        """IAI_MCP_STORE set to DEFAULT_STORAGE_PATH → False (same location)."""
        from iai_mcp.cli import _is_custom_store
        from iai_mcp import store as _store  # read the live attribute, not an

        # import-time binding: _is_custom_store re-imports DEFAULT_STORAGE_PATH at
        # call time, so the redirected (tmp) value must drive both sides here.
        monkeypatch.setenv("IAI_MCP_STORE", str(_store.DEFAULT_STORAGE_PATH))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        assert _is_custom_store() is False

    def test_socket_override_set_does_not_affect_predicate(self, tmp_path, monkeypatch):
        """_is_custom_store ignores IAI_DAEMON_SOCKET_PATH — it only looks at the store."""
        from iai_mcp.cli import _is_custom_store

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", "/tmp/fake.sock")
        # Still True — the store IS custom, regardless of socket override.
        assert _is_custom_store() is True


# ---------------------------------------------------------------------------
# Headline regression: inner-connect spy never called under custom-store-no-socket
# ---------------------------------------------------------------------------


class TestCustomStoreNoSocketSkipsConnect:
    """The headline discriminating regression.

    The inner asyncio.open_unix_connection is patched as a spy.  Under
    custom-store-no-socket, the guard must return None BEFORE any connect
    attempt.  The spy MUST NOT be called.

    This test is RED on unmodified cli.py (the guard does not exist) and GREEN
    after the guard lands.  It discriminates because patching the whole helper
    to return None (or letting connect fail) would be GREEN on *both* fixed and
    unfixed code — the spy-never-called assertion is the only thing that can
    tell the difference.
    """

    def _seed_store(self, tmp_path):
        """Seed the tmp store with one identity event."""
        from iai_mcp.events import write_event
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=tmp_path)
        write_event(
            store,
            kind="s5_invariant_update",
            data={"anchor_id": "x", "new_record_id": "y"},
            severity="info",
            session_id="s1",
        )
        store.close()

    def test_audit_never_probes_default_socket(self, tmp_path, monkeypatch):
        """audit with IAI_MCP_STORE=tmp + no socket override → spy never called."""
        from iai_mcp.cli import main as cli_main

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        self._seed_store(tmp_path)

        spy = _make_connected_asyncmock()
        with patch("asyncio.open_unix_connection", spy):
            out, rc = _capture_stdout(lambda: cli_main(["audit"]))

        # Guard must short-circuit before any connect.
        spy.assert_not_called()
        assert rc == 0
        # Output reflects the tmp store data (seeded event kind appears).
        assert "s5_invariant_update" in out

    def test_trajectory_never_probes_default_socket(self, tmp_path, monkeypatch):
        """trajectory with IAI_MCP_STORE=tmp + no socket override → spy never called."""
        from iai_mcp.cli import main as cli_main

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

        spy = _make_connected_asyncmock()
        with patch("asyncio.open_unix_connection", spy):
            out, rc = _capture_stdout(lambda: cli_main(["trajectory"]))

        spy.assert_not_called()
        assert rc == 0


# ---------------------------------------------------------------------------
# Default store path still probes the daemon socket
# ---------------------------------------------------------------------------


class TestDefaultStoreStillProbes:
    """With IAI_MCP_STORE unset (default store), the spy IS attempted."""

    def test_default_store_probes_default_socket(self, monkeypatch):
        """No IAI_MCP_STORE → helper attempts open_unix_connection, then returns None.

        We call _send_jsonrpc_request directly (not a full command) to avoid
        falling through to a MemoryStore() open against the real ~/.iai-mcp/.
        The helper returns None because the spy raises FileNotFoundError.
        """
        monkeypatch.delenv("IAI_MCP_STORE", raising=False)
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_jsonrpc_request("events_query", {"kind": "llm_health", "limit": 1})

        spy.assert_called_once()
        assert result is None


# ---------------------------------------------------------------------------
# IAI_DAEMON_SOCKET_PATH override still routes to that socket
# ---------------------------------------------------------------------------


class TestSocketOverrideWins:
    """Explicit IAI_DAEMON_SOCKET_PATH wins even with a custom store."""

    def test_socket_override_with_custom_store_still_probes(self, tmp_path, monkeypatch):
        """IAI_DAEMON_SOCKET_PATH set + IAI_MCP_STORE=tmp → probe the override socket."""
        override_sock = "/tmp/iai-test-override.sock"
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", override_sock)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_jsonrpc_request("events_query", {"kind": "llm_health", "limit": 1})

        spy.assert_called_once_with(override_sock)
        assert result is None


# ---------------------------------------------------------------------------
# _send_socket_request honors IAI_DAEMON_SOCKET_PATH
# ---------------------------------------------------------------------------


class TestSendSocketRequestOverride:
    """_send_socket_request routes to IAI_DAEMON_SOCKET_PATH when set."""

    def test_socket_request_uses_env_override(self, monkeypatch):
        """When IAI_DAEMON_SOCKET_PATH is set, _send_socket_request connects there."""
        from iai_mcp.cli import _send_socket_request

        override_sock = "/tmp/iai-control-override.sock"
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", override_sock)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_socket_request({"jsonrpc": "2.0", "id": 1, "method": "status", "params": {}})

        spy.assert_called_once_with(override_sock)
        assert result is None

    def test_socket_request_default_without_override(self, monkeypatch):
        """Without IAI_DAEMON_SOCKET_PATH, _send_socket_request uses the default SOCKET_PATH."""
        from iai_mcp.cli import SOCKET_PATH, _send_socket_request

        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_socket_request({"jsonrpc": "2.0", "id": 1, "method": "status", "params": {}})

        spy.assert_called_once_with(str(SOCKET_PATH))
        assert result is None
