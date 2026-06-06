"""Self-test for the autouse redirect-by-default fixture.

Proves the ``_hermetic_default_paths`` autouse fixture in ``conftest.py``:

- Redirects HOME + IAI_DAEMON_SOCKET_PATH to a per-test tmp ``.iai-mcp`` dir.
- Redirects the four frozen import-time default constants
  (``hippo._DEFAULT_IAI_ROOT``, ``store.DEFAULT_STORAGE_PATH``,
  ``concurrency.SOCKET_PATH``, ``daemon_state.STATE_PATH``) to that tmp dir.
- Does NOT set ``IAI_MCP_STORE`` (which would split ``store.root`` from the
  HippoDB dir on ``path=`` tests).
- Lets a bare ``MemoryStore()`` resolve its root under the tmp default and open
  there (round-trip insert/get), never the operator's real home store.
- Keeps ``MemoryStore(path=...)`` consistent: ``store.root == path`` and the
  HippoDB dir under ``path`` (no divergence).

Test data uses a generic record helper only — no real names, no PII.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest

import iai_mcp.concurrency
import iai_mcp.daemon_state
import iai_mcp.hippo
import iai_mcp.store
from iai_mcp.store import MemoryStore

from _recall_helpers import _deterministic_vec, _make_gold_record


def _under(child: Path, ancestor: Path) -> bool:
    """True if ``child`` is at or under ``ancestor`` (resolved)."""
    child = Path(child).resolve()
    ancestor = Path(ancestor).resolve()
    return child == ancestor or ancestor in child.parents


def test_fixture_does_not_set_iai_mcp_store() -> None:
    """The autouse fixture must NOT set IAI_MCP_STORE (pitfall: splits roots)."""
    assert os.environ.get("IAI_MCP_STORE") is None


def test_home_and_socket_under_tmp() -> None:
    """HOME and IAI_DAEMON_SOCKET_PATH point at a tmp path, not the real home."""
    home = Path(os.environ["HOME"])
    # The real operator home is the import-captured sentinel's parent.
    real_home = iai_mcp.hippo._REAL_IAI_ROOT.parent
    assert home != real_home
    sock = Path(os.environ["IAI_DAEMON_SOCKET_PATH"])
    assert _under(sock, home)


def test_four_default_constants_redirected_to_tmp() -> None:
    """All four frozen defaults resolve under the tmp .iai-mcp dir."""
    real_root = iai_mcp.hippo._REAL_IAI_ROOT
    fake_root = Path(os.environ["HOME"]) / ".iai-mcp"

    assert iai_mcp.hippo._DEFAULT_IAI_ROOT == fake_root
    assert iai_mcp.hippo._DEFAULT_IAI_ROOT != real_root

    assert iai_mcp.store.DEFAULT_STORAGE_PATH == fake_root
    assert iai_mcp.store.DEFAULT_STORAGE_PATH != real_root

    assert _under(iai_mcp.concurrency.SOCKET_PATH, fake_root)
    assert _under(iai_mcp.daemon_state.STATE_PATH, fake_root)


def test_bare_store_resolves_under_tmp_default() -> None:
    """A bare MemoryStore() opens under the tmp default and round-trips a record.

    No path, no IAI_MCP_STORE -> resolves through the redirected default. The
    store root must be the tmp .iai-mcp dir, never the real home store, and a
    tiny insert/get must succeed there (proving the store actually opened in
    tmp, not merely that the constant was redirected).
    """
    fake_root = Path(os.environ["HOME"]) / ".iai-mcp"
    store = MemoryStore()
    assert _under(store.root, fake_root)
    assert store.root != iai_mcp.hippo._REAL_IAI_ROOT

    rec = _make_gold_record(1, _deterministic_vec(1))
    store.insert(rec)
    got = store.get(UUID(int=1))
    assert got is not None
    assert got.id == UUID(int=1)


def test_path_store_is_consistent(tmp_path: Path) -> None:
    """MemoryStore(path=...) keeps store.root == path and the HippoDB dir under it.

    Proves the no-divergence property: because the fixture does NOT set
    IAI_MCP_STORE, an explicit path wins at BOTH the MemoryStore layer and the
    HippoDB resolver layer, so store.root and the HippoDB store root agree.
    """
    store = MemoryStore(path=tmp_path)
    assert store.root == tmp_path
    # HippoDB resolved its own root from the same explicit path.
    assert store.db._store_root == tmp_path
    assert _under(store.db._hippo_dir, tmp_path)
