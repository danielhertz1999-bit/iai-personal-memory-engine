"""Shared helpers for tests.

NOT a standard pytest conftest — this file is NOT auto-discovered.
Tests import helpers directly: from tests.conftest_recall import make_tmp_store

The project-wide autouse fixtures in tests/conftest.py (crypto passphrase +
autoflush) already apply to every test here, so store.insert() is synchronous
in tests without any extra setup.

All helpers use tmp_path, never ~/.iai-mcp/ or the live daemon socket.
"""
from __future__ import annotations

from pathlib import Path

from iai_mcp.store import MemoryStore


def make_tmp_store(tmp_path: Path) -> MemoryStore:
    """Construct an isolated MemoryStore rooted at tmp_path.

    Never touches ~/.iai-mcp/ or the live daemon.

    Usage in a test:
        def test_something(tmp_path):
            store = make_tmp_store(tmp_path)
            ...
    """
    store_root = tmp_path / "hippo"
    store_root.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path=store_root)


def set_tmp_env(monkeypatch, tmp_path: Path) -> None:
    """Monkeypatch IAI_MCP_STORE and IAI_DAEMON_SOCKET_PATH to tmp paths.

    Prevents any code path that reads these env vars from touching the
    live store or the live daemon socket.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
