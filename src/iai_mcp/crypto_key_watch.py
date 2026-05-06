"""Boot-time detection of ``.crypto.key`` file rotation for audit events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

WATCHER_REL = ".crypto-key-watcher.json"


def _watcher_path(store: "MemoryStore") -> Path:
    return store.root / WATCHER_REL


def _key_path(store: "MemoryStore") -> Path:
    return store.root / ".crypto.key"


def sync_crypto_key_watcher_to_disk(store: "MemoryStore") -> None:
    """Persist watcher state matching the current key file (no event)."""
    kp = _key_path(store)
    if not kp.is_file():
        return
    st = kp.stat()
    cur = {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}
    wp = _watcher_path(store)
    wp.write_text(json.dumps(cur), encoding="utf-8")
    try:
        os.chmod(wp, 0o600)
    except OSError:
        pass


def check_crypto_key_file_rotation_event(store: "MemoryStore") -> None:
    """Emit ``crypto_key_rotated`` when ``.crypto.key`` mtime/size changed since last persist.

    First run (no watcher file): writes baseline only — no event (cannot
    distinguish "first install" from "rotation" without prior state).
    """
    from iai_mcp.events import write_event

    kp = _key_path(store)
    if not kp.is_file():
        return
    st = kp.stat()
    cur = {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}
    wp = _watcher_path(store)
    prev: dict | None = None
    if wp.is_file():
        try:
            prev = json.loads(wp.read_text(encoding="utf-8"))
        except Exception:
            prev = None
    if prev is None:
        sync_crypto_key_watcher_to_disk(store)
        return
    if prev.get("mtime_ns") == cur["mtime_ns"] and prev.get("size") == cur["size"]:
        return
    try:
        write_event(
            store,
            kind="crypto_key_rotated",
            data={
                "source": "daemon_boot",
                "previous": prev,
                "current": cur,
            },
            severity="info",
        )
    except Exception:
        pass
    sync_crypto_key_watcher_to_disk(store)
