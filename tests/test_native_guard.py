"""Tests for the mandatory native extension startup guard."""
from __future__ import annotations

import sys

import pytest


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


def test_missing_native_fails_loud(monkeypatch):
    """A missing iai_mcp_native raises RuntimeError with the build command."""
    monkeypatch.setitem(sys.modules, "iai_mcp_native", None)
    # Remove any cached submodule entries so the import attempt sees None.
    monkeypatch.setitem(sys.modules, "iai_mcp_native.embed", None)
    monkeypatch.setitem(sys.modules, "iai_mcp_native.graph", None)

    from iai_mcp.native_guard import _require_native

    with pytest.raises(RuntimeError) as exc_info:
        _require_native()

    assert "iai-mcp build-native" in str(exc_info.value)
    assert "maturin develop --release" in str(exc_info.value)


@pytest.mark.skipif(
    not _rust_available(),
    reason="iai_mcp_native wheel not installed on this runner",
)
def test_present_native_passes():
    """When native is importable and functional, _require_native() returns None."""
    from iai_mcp.native_guard import _require_native

    result = _require_native()
    assert result is None
