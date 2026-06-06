"""Meta self-test for the store-root hermeticity backstop.

These tests prove that the call-time backstop in ``iai_mcp.hippo._resolve_root``
catches a deliberately non-hermetic store-root resolution under a test run:

- A regressed fixture that fails to redirect the default store-root constant
  (so it still points at the operator's real home store) is CAUGHT — the
  resolver raises ``RuntimeError`` instead of silently opening the real store.
- The normal redirected default (pointed at a tmp directory by the autouse
  fixture) is allowed — the resolver returns it without raising.
- A bare ``MemoryStore()`` under the same regressed default RAISES, proving the
  guard closes the bare-store vector by construction.

The assertion is always on the RAISE; the real store is never opened.
"""
from __future__ import annotations

import pytest

import iai_mcp.hippo as hippo
from iai_mcp.hippo import HippoDB
from iai_mcp.store import MemoryStore


def test_guard_catches_real_home_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regressed default pointing at the real home store is refused under test.

    Simulates a fixture that redirected HOME but forgot to redirect the frozen
    default constant: the constant still resolves to the operator's real store
    root. The backstop compares against the import-captured sentinel (the real
    root), not the live (HOME-redirected) home, so this regression is caught.
    """
    # Ensure the default branch is reached: env unset, path is None.
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    # Point the default back at the import-captured real root, modelling a
    # fixture that failed to redirect it.
    monkeypatch.setattr(hippo, "_DEFAULT_IAI_ROOT", hippo._REAL_IAI_ROOT, raising=False)

    with pytest.raises(RuntimeError, match="hermeticity guard"):
        hippo._resolve_root()


def test_guard_allows_redirected_tmp_default(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The normal redirected tmp default is allowed (no raise).

    Models the autouse fixture's redirect: the default points at a tmp dir,
    which never equals the import-captured real root, so the guard stays silent.
    """
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    fake_root = tmp_path / ".iai-mcp"
    monkeypatch.setattr(hippo, "_DEFAULT_IAI_ROOT", fake_root, raising=False)

    resolved = hippo._resolve_root()
    assert resolved == fake_root


def test_guard_catches_bare_store_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``MemoryStore()`` under the regressed default RAISES.

    Proves the backstop closes the primary bare-store vector: with the store
    default pointed back at the real root and env unset, constructing a store
    with no explicit path is refused *before any filesystem touch* of the real
    store.

    ``MemoryStore.__init__`` resolves its own root (env > path > default) and
    then passes that root explicitly to ``HippoDB`` — so the resolver-level
    guard alone does not cover this entry point. The store-level guard, placed
    before ``self.root.mkdir(...)``, closes it.
    """
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    # MemoryStore resolves env > path > store.DEFAULT_STORAGE_PATH. Redirect that
    # default to the import-captured real root, modelling a fixture that failed
    # to redirect it. The store guard fires before any real open.
    import iai_mcp.store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_STORAGE_PATH", hippo._REAL_IAI_ROOT, raising=False)

    with pytest.raises(RuntimeError, match="hermeticity guard"):
        MemoryStore()


def test_guard_catches_bare_hippodb_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``HippoDB()`` under the regressed default RAISES.

    Covers the second entry point the resolver-level guard protects: a direct
    ``HippoDB()`` with no path resolves through ``_resolve_root``'s default
    branch (env unset, path None), which is refused before any real open.
    """
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(hippo, "_DEFAULT_IAI_ROOT", hippo._REAL_IAI_ROOT, raising=False)

    with pytest.raises(RuntimeError, match="hermeticity guard"):
        HippoDB()
