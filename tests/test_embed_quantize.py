
from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.embed import Embedder, QuantizedVector  # noqa: F401


def test_default_env_unset_keeps_fp32_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EMBED_QUANTIZE", raising=False)
    emb = Embedder(model_key="bge-small-en-v1.5")
    v = emb.embed("hello world")
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)
    assert emb._quantize_mode is None


def test_env_int8_enables_embed_quantized_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "int8")
    emb = Embedder(model_key="bge-small-en-v1.5")
    qv = emb.embed_quantized("hello world")
    assert len(qv.values) == 384
    assert all(isinstance(x, int) for x in qv.values)
    assert all(-128 <= x <= 127 for x in qv.values)
    assert isinstance(qv.scale, float)
    assert qv.scale > 0
    assert isinstance(qv.zero_point, int)
    assert qv.dim == 384
    fp32 = emb.embed("hello world")
    assert len(fp32) == 384
    assert all(isinstance(x, float) for x in fp32)


def test_invalid_quantize_value_fails_loud_at_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "foo")
    with pytest.raises(ValueError, match="IAI_MCP_EMBED_QUANTIZE"):
        Embedder(model_key="bge-small-en-v1.5")

    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "INT8")
    with pytest.raises(ValueError, match="IAI_MCP_EMBED_QUANTIZE"):
        Embedder(model_key="bge-small-en-v1.5")

    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "")
    emb = Embedder(model_key="bge-small-en-v1.5")
    assert emb._quantize_mode is None


def test_quantize_dequantize_round_trip_cos_ge_0_99(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "int8")
    emb = Embedder(model_key="bge-small-en-v1.5")
    probe = "the quick brown fox jumps over the lazy dog"
    original = emb.embed(probe)
    qv = emb.embed_quantized(probe)
    recovered = [(int(x) - qv.zero_point) * qv.scale for x in qv.values]
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(recovered, dtype=np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos >= 0.99, f"round-trip cosine {cos:.4f} < 0.99"
