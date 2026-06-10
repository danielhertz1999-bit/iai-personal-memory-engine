from __future__ import annotations

import pytest

import iai_mcp.hippo as hippo
from iai_mcp.hippo import HippoDB
from iai_mcp.store import MemoryStore


def test_guard_catches_real_home_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(hippo, "_DEFAULT_IAI_ROOT", hippo._REAL_IAI_ROOT, raising=False)

    with pytest.raises(RuntimeError, match="hermeticity guard"):
        hippo._resolve_root()


def test_guard_allows_redirected_tmp_default(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    fake_root = tmp_path / ".iai-mcp"
    monkeypatch.setattr(hippo, "_DEFAULT_IAI_ROOT", fake_root, raising=False)

    resolved = hippo._resolve_root()
    assert resolved == fake_root


def test_guard_catches_bare_store_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    import iai_mcp.store as store_mod

    monkeypatch.setattr(store_mod, "DEFAULT_STORAGE_PATH", hippo._REAL_IAI_ROOT, raising=False)

    with pytest.raises(RuntimeError, match="hermeticity guard"):
        MemoryStore()


def test_guard_catches_bare_hippodb_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.setattr(hippo, "_DEFAULT_IAI_ROOT", hippo._REAL_IAI_ROOT, raising=False)

    with pytest.raises(RuntimeError, match="hermeticity guard"):
        HippoDB()
