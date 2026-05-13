"""Tests for the multilingual embedder path in the 3-model registry.

(2026-04-20) flipped the DEFAULT to bge-small-en-v1.5 (384d
English-only). bge-m3 remains selectable via env var or explicit
``Embedder(model_key="bge-m3")`` — these tests pin the key explicitly
so the multilingual coverage keeps running under the new default.

These tests import SentenceTransformer and pull the bge-m3 weights once on
first run (HuggingFace cache is re-used thereafter). If bge-m3 is already
cached by any previous dev session the test runs in seconds.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# ------------------------------------------------------------- bge-m3 opt-in


def test_bge_m3_opt_in_produces_1024d() -> None:
    """Explicit Embedder(model_key="bge-m3") still yields the multilingual
    1024d path after 's default revert."""
    from iai_mcp.embed import Embedder

    e = Embedder(model_key="bge-m3")
    assert e.model_key == "bge-m3"
    assert e.model_name == "BAAI/bge-m3"
    assert e.DIM == 1024


def test_bge_m3_embeds_english() -> None:
    from iai_mcp.embed import Embedder

    e = Embedder(model_key="bge-m3")
    v = e.embed("Hello, how are you?")
    assert len(v) == 1024
    # bge-m3 returns normalised vectors (|v| == 1)
    n = float(np.linalg.norm(np.asarray(v)))
    assert abs(n - 1.0) < 1e-4


def test_bge_m3_embeds_russian() -> None:
    from iai_mcp.embed import Embedder

    e = Embedder(model_key="bge-m3")
    v = e.embed("Привет, как дела?")
    assert len(v) == 1024
    n = float(np.linalg.norm(np.asarray(v)))
    assert abs(n - 1.0) < 1e-4


def test_bge_m3_embeds_japanese() -> None:
    from iai_mcp.embed import Embedder

    e = Embedder(model_key="bge-m3")
    v = e.embed("こんにちは、今日は元気ですか？")
    assert len(v) == 1024
    n = float(np.linalg.norm(np.asarray(v)))
    assert abs(n - 1.0) < 1e-4


def test_bge_m3_cross_language_similarity() -> None:
    """bge-m3 encodes cross-lingual concepts. Pinned explicitly because
    's default is now English-only bge-small."""
    from iai_mcp.embed import Embedder

    e = Embedder(model_key="bge-m3")
    en = np.asarray(e.embed("hello"))
    ru = np.asarray(e.embed("привет"))
    cos = float(en @ ru / (np.linalg.norm(en) * np.linalg.norm(ru)))
    assert cos > 0.5, f"cross-language cosine too low: {cos}"


# ----------------------------------------------------------- env-var selection


def test_embed_model_selectable_via_env(monkeypatch) -> None:
    """IAI_MCP_EMBED_MODEL selects from the 3-model registry."""
    import importlib

    # Clear the process-level cache so re-import exposes the correct default.
    import iai_mcp.embed as embed_mod

    monkeypatch.setenv("IAI_MCP_EMBED_MODEL", "bge-small-en-v1.5")
    importlib.reload(embed_mod)
    e = embed_mod.Embedder()
    assert e.model_key == "bge-small-en-v1.5"
    assert e.DIM == 384

    # Restore default for remaining tests.
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    importlib.reload(embed_mod)


def test_embed_model_explicit_key_overrides_env(monkeypatch) -> None:
    from iai_mcp.embed import Embedder

    monkeypatch.setenv("IAI_MCP_EMBED_MODEL", "bge-m3")
    e = Embedder(model_key="bge-small-en-v1.5")
    # Explicit key wins over env.
    assert e.model_key == "bge-small-en-v1.5"
    assert e.DIM == 384


def test_embed_model_dimension_registered() -> None:
    """Registry reports the correct DIM for every entry."""
    from iai_mcp.embed import MODEL_REGISTRY

    assert MODEL_REGISTRY["bge-m3"]["dim"] == 1024
    assert MODEL_REGISTRY["multilingual-e5-small"]["dim"] == 384
    assert MODEL_REGISTRY["bge-small-en-v1.5"]["dim"] == 384


def test_embed_model_rejects_unknown_key() -> None:
    from iai_mcp.embed import Embedder

    with pytest.raises(ValueError):
        Embedder(model_key="this-model-does-not-exist")


def test_embed_model_rejects_unknown_env(monkeypatch) -> None:
    from iai_mcp.embed import Embedder

    monkeypatch.setenv("IAI_MCP_EMBED_MODEL", "garbage")
    with pytest.raises(ValueError):
        Embedder()


# ------------------------------------------------------- batch + determinism


def test_embed_batch_preserves_order_and_dim() -> None:
    from iai_mcp.embed import Embedder

    e = Embedder(model_key="bge-m3")
    texts = ["one", "два", "三"]
    vecs = e.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 1024 for v in vecs)


def test_embed_deterministic_same_input() -> None:
    from iai_mcp.embed import Embedder

    e = Embedder()
    a = e.embed("deterministic test")
    b = e.embed("deterministic test")
    assert a == b
