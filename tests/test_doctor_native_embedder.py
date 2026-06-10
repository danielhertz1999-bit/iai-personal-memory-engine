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
    from iai_mcp.doctor import check_v_native_embedder

    result = check_v_native_embedder()
    assert result.passed is True, f"expected PASS, got FAIL: {result.detail}"
    assert "backend=rust" in result.detail, (
        f"expected 'backend=rust' in detail, got: {result.detail!r}"
    )


def test_check_v_fails_on_non_rust_backend(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.mark.skipif(
    not _rust_available(),
    reason="iai_mcp_native wheel not built — run: cd rust/iai_mcp_native && maturin develop --release",
)
def test_check_v_pages_weights_via_construct_and_smoke_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.embed as embed_mod
    from iai_mcp.doctor import check_v_native_embedder

    _RealEmbedder = embed_mod.Embedder
    constructs: list[int] = []
    encodes: list[str] = []

    class _SpyEmbedder(_RealEmbedder):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructs.append(1)

        def embed(self, text, *args, **kwargs):
            encodes.append(text)
            return super().embed(text, *args, **kwargs)

    monkeypatch.setattr(embed_mod, "Embedder", _SpyEmbedder)

    result = check_v_native_embedder()

    assert result.passed is True, f"expected PASS, got FAIL: {result.detail}"
    assert constructs, "Embedder() was not constructed (weights not paged)"
    assert encodes, "no smoke-encode ran (no forward-pass touch)"
    assert "backend=rust" in result.detail
