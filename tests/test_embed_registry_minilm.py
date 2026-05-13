"""— Registry invariant tests for the all-MiniLM-L6-v2 additive entry.

Locks (additive-only registry expansion) and (source-freeze-modulo-registry)
from internal architecture spec Verifies that:
- the new MODEL_REGISTRY entry exists with the correct HF id and dimension,
- DEFAULT_MODEL_KEY remains bge-small-en-v1.5 (English-Only Brain lock from
  / holds),
- the 3 pre-existing entries are byte-identical to v3,
- the new entry is functionally usable (loads, produces normalized 384d vectors),
- production zero-arg Embedder() still resolves to the default.
"""
from __future__ import annotations

from iai_mcp.embed import DEFAULT_MODEL_KEY, MODEL_REGISTRY, Embedder


def test_registry_has_minilm_entry() -> None:
    """MODEL_REGISTRY contains the additive all-MiniLM-L6-v2 entry."""
    assert "all-MiniLM-L6-v2" in MODEL_REGISTRY
    spec = MODEL_REGISTRY["all-MiniLM-L6-v2"]
    assert spec["hf"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert spec["dim"] == 384


def test_default_model_key_unchanged() -> None:
    """D-02 + English-Only Brain lock: DEFAULT_MODEL_KEY is still bge-small-en-v1.5."""
    assert DEFAULT_MODEL_KEY == "bge-small-en-v1.5"


def test_registry_has_exactly_four_entries() -> None:
    """D-02 + source-freeze-modulo-registry — exactly 1 additive entry vs v3."""
    expected_keys = {
        "bge-m3",
        "multilingual-e5-small",
        "bge-small-en-v1.5",
        "all-MiniLM-L6-v2",
    }
    assert set(MODEL_REGISTRY.keys()) == expected_keys


def test_existing_entries_byte_identical_to_v3() -> None:
    """the 3 pre-existing entries are unchanged from pre-registered-lme500-v3."""
    assert MODEL_REGISTRY["bge-m3"] == {"hf": "BAAI/bge-m3", "dim": 1024}
    assert MODEL_REGISTRY["multilingual-e5-small"] == {
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
    }
    assert MODEL_REGISTRY["bge-small-en-v1.5"] == {
        "hf": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    }


def test_minilm_embedder_loads_and_produces_normalized_384d() -> None:
    """D-02 functional check: Embedder(model_key='all-MiniLM-L6-v2') is usable."""
    emb = Embedder(model_key="all-MiniLM-L6-v2")
    assert emb.model_key == "all-MiniLM-L6-v2"
    assert emb.DIM == 384
    assert emb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    vec = emb.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384
    # normalized: L2 norm ≈ 1.0 (within float32 tolerance)
    l2 = sum(v * v for v in vec) ** 0.5
    assert abs(l2 - 1.0) < 1e-3, f"vector not normalized: L2={l2}"


def test_default_embedder_still_resolves_to_bge_small() -> None:
    """production zero-arg Embedder() still picks bge-small-en-v1.5."""
    emb = Embedder()
    assert emb.model_key == "bge-small-en-v1.5"
    assert emb.DIM == 384
    assert emb.model_name == "BAAI/bge-small-en-v1.5"
