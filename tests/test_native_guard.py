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
    monkeypatch.setitem(sys.modules, "iai_mcp_native", None)
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
    from iai_mcp.native_guard import _require_native

    result = _require_native()
    assert result is None
