"""Tests for iai_mcp.embed -- bge-small-en-v1.5 path (legacy model).

Plan 02-01 made bge-m3 the default. The 3-model registry still exposes
bge-small-en-v1.5 (384d, English-only) for English-only deployments. These
tests exercise the Phase-1 model explicitly via `Embedder(model_key=...)` so
they remain valid regression gates.

Multilingual behaviour is covered by tests/test_embed_multilingual.py.
"""
from __future__ import annotations

import pytest

from iai_mcp.embed import Embedder


def test_embed_returns_384_dim_vector() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    v = emb.embed("hello world")
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)


def test_embed_is_deterministic() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    a = emb.embed("exact same text")
    b = emb.embed("exact same text")
    assert a == b


def test_embed_batch_preserves_order_and_dim() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    texts = ["one", "two", "three"]
    vecs = emb.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)
    # Batch must equal sequential calls (determinism across batching path too).
    assert vecs[0] == emb.embed("one")


def test_embed_empty_string_still_returns_384d() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    v = emb.embed("")
    assert len(v) == 384


def test_embedder_dim_matches_output() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    assert emb.DIM == 384
    v = emb.embed("anything")
    assert len(v) == emb.DIM


def test_bge_small_en_still_registered_for_legacy() -> None:
    """D-02a keeps the model in the registry for English-only deployments."""
    from iai_mcp.embed import MODEL_REGISTRY

    assert "bge-small-en-v1.5" in MODEL_REGISTRY
    assert MODEL_REGISTRY["bge-small-en-v1.5"]["dim"] == 384
