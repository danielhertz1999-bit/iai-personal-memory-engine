"""Tests for crypto_key_watch baseline + rotation detection."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from iai_mcp.crypto_key_watch import (
    check_crypto_key_file_rotation_event,
    sync_crypto_key_watcher_to_disk,
)
from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore


def test_watcher_baseline_then_rotation_emits_event(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    kpath = root / ".crypto.key"
    kpath.write_bytes(secrets.token_bytes(32))
    os.chmod(kpath, 0o600)
    store = MemoryStore(path=root, user_id="default")

    check_crypto_key_file_rotation_event(store)
    ev0 = query_events(store, kind="crypto_key_rotated", limit=10)
    assert len(ev0) == 0

    kpath.write_bytes(secrets.token_bytes(32))
    os.chmod(kpath, 0o600)
    check_crypto_key_file_rotation_event(store)
    ev1 = query_events(store, kind="crypto_key_rotated", limit=10)
    assert len(ev1) == 1

    check_crypto_key_file_rotation_event(store)
    ev2 = query_events(store, kind="crypto_key_rotated", limit=10)
    assert len(ev2) == 1


def test_sync_watcher_without_event(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    kpath = root / ".crypto.key"
    kpath.write_bytes(secrets.token_bytes(32))
    os.chmod(kpath, 0o600)
    store = MemoryStore(path=root, user_id="default")
    sync_crypto_key_watcher_to_disk(store)
    wp = root / ".crypto-key-watcher.json"
    assert wp.is_file()
    data = json.loads(wp.read_text(encoding="utf-8"))
    assert "mtime_ns" in data and "size" in data
