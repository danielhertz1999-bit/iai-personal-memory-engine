"""Regression guard: native encode failure is observable via counter + log.

Both embed() and embed_batch() route through _encode_one, so a native
encode exception increments embed_failure_total and emits a logger.error
breadcrumb before re-raising. This covers all 6 encode call sites at the
store-free embedder boundary.
"""
from __future__ import annotations

import logging

import pytest


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


class _BrokenModel:
    """Stub that mimics iai_mcp_native.embed.Embedder but always raises."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def encode(self, text: str) -> list[float]:
        raise self._exc


@pytest.mark.skipif(
    not _rust_available(),
    reason="iai_mcp_native wheel not installed on this runner",
)
def test_embed_failure_increments_counter_and_logs(monkeypatch, caplog):
    """embed() failure increments embed_failure_total and emits logger.error."""
    import iai_mcp.embed as embed_mod
    from iai_mcp.embed import Embedder

    e = Embedder()
    before = embed_mod.embed_failure_total

    # Replace the native model object with a broken stub.
    e._model = _BrokenModel(RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger="iai_mcp.embed"):
        with pytest.raises(RuntimeError, match="boom"):
            e.embed("x")

    assert embed_mod.embed_failure_total == before + 1, (
        f"counter did not increment: before={before} after={embed_mod.embed_failure_total}"
    )
    assert any("native embed encode failed" in r.message for r in caplog.records), (
        "no logger.error breadcrumb emitted"
    )


@pytest.mark.skipif(
    not _rust_available(),
    reason="iai_mcp_native wheel not installed on this runner",
)
def test_embed_batch_failure_is_observable(monkeypatch, caplog):
    """embed_batch() failure increments embed_failure_total and emits logger.error.

    Proves the per-item batch loop routes through _encode_one (no silent hole).
    """
    import iai_mcp.embed as embed_mod
    from iai_mcp.embed import Embedder

    e = Embedder()
    before = embed_mod.embed_failure_total

    e._model = _BrokenModel(RuntimeError("batch-boom"))

    with caplog.at_level(logging.ERROR, logger="iai_mcp.embed"):
        with pytest.raises(RuntimeError, match="batch-boom"):
            e.embed_batch(["x"])

    assert embed_mod.embed_failure_total == before + 1, (
        f"counter did not increment on batch failure: before={before} after={embed_mod.embed_failure_total}"
    )
    assert any("native embed encode failed" in r.message for r in caplog.records), (
        "no logger.error breadcrumb emitted on batch failure"
    )
