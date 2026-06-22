"""Regression test for _store_is_empty: a count failure must NOT be read as empty.

The daemon tick calls _store_is_empty() and, if true, skips the whole tick
(no idle-check, no drain). A transient count failure -- e.g. the shared sqlite
connection left in an error state by a concurrent heavy reader, raising
HippoIntegrityError/lock errors (all subclass RuntimeError) -- used to be caught
and return True, parking the lifecycle on a store that actually has records.
The fix returns False (unknown != empty) so the tick proceeds.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.daemon import _store_is_empty
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(seed: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface="rec", aaak_index="",
        embedding=(v / np.linalg.norm(v)).tolist(), community_id=None,
        centrality=0.0, detail_level=2, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=[], language="en",
    )


def _make_store(tmp_path: Path, monkeypatch) -> MemoryStore:
    root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    return MemoryStore(path=root)


def test_empty_store_is_empty(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    assert _store_is_empty(store) is True


def test_nonempty_store_is_not_empty(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store.insert(_rec(1))
    assert _store_is_empty(store) is False


def test_count_failure_is_not_treated_as_empty(tmp_path, monkeypatch):
    """The core fix: a RuntimeError during the count must yield False, not True."""
    store = _make_store(tmp_path, monkeypatch)
    store.insert(_rec(2))

    def boom(*_a, **_k):
        raise RuntimeError("connection in error state")

    monkeypatch.setattr(store.db, "open_table", boom)
    assert _store_is_empty(store) is False
