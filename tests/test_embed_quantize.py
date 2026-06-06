"""Tests for opt-in int8 embedding quantization.

Adds the WRITE-side quantization knob
`IAI_MCP_EMBED_QUANTIZE=int8`. Default (env unset) keeps the fp32 path
byte-identical; the int8 surface is exposed via the new
`Embedder.embed_quantized()` method and returns a `QuantizedVector`
dataclass with per-vector min/max calibration metadata.

These tests are written RED-first per the plan: they MUST fail before the
implementation lands in src/iai_mcp/embed.py.

A full LongMemEval-S A/B (recall loss measurement) is OUT of scope and
must run manually post-merge before any default flip from fp32 to int8.
"""

from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.embed import Embedder, QuantizedVector  # noqa: F401


def test_default_env_unset_keeps_fp32_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env preserves fp32 surface byte-identically to pre-task Embedder."""
    monkeypatch.delenv("IAI_MCP_EMBED_QUANTIZE", raising=False)
    emb = Embedder(model_key="bge-small-en-v1.5")
    v = emb.embed("hello world")
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)
    # quantize disabled when env unset
    assert emb._quantize_mode is None


def test_env_int8_enables_embed_quantized_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env=int8 enables embed_quantized() returning int8 values + scale + zero_point.

    fp32 path (.embed) remains untouched — int8 is additive surface.
    """
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "int8")
    emb = Embedder(model_key="bge-small-en-v1.5")
    qv = emb.embed_quantized("hello world")
    # values: 384-length list of signed int8 codes
    assert len(qv.values) == 384
    assert all(isinstance(x, int) for x in qv.values)
    assert all(-128 <= x <= 127 for x in qv.values)
    # metadata
    assert isinstance(qv.scale, float)
    assert qv.scale > 0
    assert isinstance(qv.zero_point, int)
    assert qv.dim == 384
    # fp32 path NOT replaced — embed() still returns 384-length list[float]
    fp32 = emb.embed("hello world")
    assert len(fp32) == 384
    assert all(isinstance(x, float) for x in fp32)


def test_invalid_quantize_value_fails_loud_at_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid IAI_MCP_EMBED_QUANTIZE raises ValueError — no silent fallback.

    Case sensitivity choice (documented in helper docstring): lower-case `int8`
    only; `INT8` is rejected. Empty string is treated as unset (no error).
    """
    # 1. arbitrary garbage value → ValueError mentioning the env var name
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "foo")
    with pytest.raises(ValueError, match="IAI_MCP_EMBED_QUANTIZE"):
        Embedder(model_key="bge-small-en-v1.5")

    # 2. case sensitivity — upper-case is NOT accepted
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "INT8")
    with pytest.raises(ValueError, match="IAI_MCP_EMBED_QUANTIZE"):
        Embedder(model_key="bge-small-en-v1.5")

    # 3. empty string == unset, must not error
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "")
    emb = Embedder(model_key="bge-small-en-v1.5")
    assert emb._quantize_mode is None


def test_quantize_dequantize_round_trip_cos_ge_0_99(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip cosine on a real BGE embedding must be >= 0.99.

    Sanity-check on the per-vector min/max int8 calibration math. A full
    LongMemEval-S A/B (recall loss measurement) is out of scope and pending
    manual run before any default flip.
    """
    monkeypatch.setenv("IAI_MCP_EMBED_QUANTIZE", "int8")
    emb = Embedder(model_key="bge-small-en-v1.5")
    probe = "the quick brown fox jumps over the lazy dog"
    original = emb.embed(probe)
    qv = emb.embed_quantized(probe)
    # Dequantize via the documented inverse:
    # fp32[i] ≈ (int8_code[i] - zero_point) * scale
    recovered = [(int(x) - qv.zero_point) * qv.scale for x in qv.values]
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(recovered, dtype=np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos >= 0.99, f"round-trip cosine {cos:.4f} < 0.99"
