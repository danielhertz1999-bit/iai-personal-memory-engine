"""Tests for the (v) native Rust embedder doctor row.

Two scenarios:
- Healthy native present: row PASSes, detail contains "backend=rust".
- Non-rust backend active: row FAILs even when encode itself would succeed.
"""
from __future__ import annotations

import pytest


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _rust_available(),
    reason="iai_mcp_native wheel not built — run: cd rust/iai_mcp_native && maturin develop --release",
)
def test_check_v_passes_on_healthy_native() -> None:
    """Row PASSes when iai_mcp_native is present and backend==rust."""
    from iai_mcp.doctor import check_v_native_embedder

    result = check_v_native_embedder()
    assert result.passed is True, f"expected PASS, got FAIL: {result.detail}"
    assert "backend=rust" in result.detail, (
        f"expected 'backend=rust' in detail, got: {result.detail!r}"
    )


def test_check_v_fails_on_non_rust_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row FAILs when the Embedder reports a non-rust backend.

    Patches iai_mcp.embed.Embedder with a stub whose instance has
    _backend='pytorch' and embed() returns a valid 384-d list. This proves
    the backend assertion fires even when the encode output would pass.
    """
    import iai_mcp.embed as embed_mod
    from iai_mcp.doctor import check_v_native_embedder

    class _StubEmbedder:
        _backend = "pytorch"

        def embed(self, text: str) -> list[float]:
            return [0.0] * 384

    monkeypatch.setattr(embed_mod, "Embedder", _StubEmbedder)

    result = check_v_native_embedder()
    assert result.passed is False, (
        f"expected FAIL when backend==pytorch, got PASS: {result.detail}"
    )


# --------------------------------------------------------- warm-on-install (H63-6)
#
# The doctor native-embedder check IS the warm-on-install first-touch: it
# constructs Embedder() (paging the bge-small weights into the OS cache) and
# runs one smoke-encode (a full forward-pass touch). Running `iai-mcp doctor`
# once after `maturin develop --release` warms the cache so even the first real
# recall is warm-cache fast. Honest scope: this pages weights once but does NOT
# cover a post-reboot daemon-down first recall (defended by the budget guard on
# the daemon-down construct path).


@pytest.mark.skipif(
    not _rust_available(),
    reason="iai_mcp_native wheel not built — run: cd rust/iai_mcp_native && maturin develop --release",
)
def test_check_v_pages_weights_via_construct_and_smoke_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check constructs Embedder() + smoke-encodes -> the warm-on-install touch.

    Spies on the real native Embedder to PROVE the check pages the weights
    (construct) and runs a full forward-pass touch (smoke-encode), then still
    PASSes. No download is forced: the real cached weights are used (rust ext
    present) — the test is skipped when the native extension is absent.
    """
    import iai_mcp.embed as embed_mod
    from iai_mcp.doctor import check_v_native_embedder

    _RealEmbedder = embed_mod.Embedder
    constructs: list[int] = []
    encodes: list[str] = []

    class _SpyEmbedder(_RealEmbedder):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructs.append(1)  # construct pages the weights

        def embed(self, text, *args, **kwargs):
            encodes.append(text)  # smoke-encode = full forward-pass touch
            return super().embed(text, *args, **kwargs)

    monkeypatch.setattr(embed_mod, "Embedder", _SpyEmbedder)

    result = check_v_native_embedder()

    assert result.passed is True, f"expected PASS, got FAIL: {result.detail}"
    assert constructs, "Embedder() was not constructed (weights not paged)"
    assert encodes, "no smoke-encode ran (no forward-pass touch)"
    assert "backend=rust" in result.detail
