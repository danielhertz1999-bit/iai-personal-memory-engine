"""Plan 05-08 — revert the Phase-2 deviation and restore the
PROJECT.md original embedder default: ``bge-small-en-v1.5`` (384d
English-only). bge-m3 (1024d multilingual) remains opt-in via the
``IAI_MCP_EMBED_MODEL`` env var or the ``model_key`` kwarg on Embedder.

Phase 9.1 (2026-04-29): MODEL_REGISTRY grew by ONE additive entry
for ``all-MiniLM-L6-v2`` (legacy alternative embedder; bench-only ablation).
DEFAULT_MODEL_KEY remains ``bge-small-en-v1.5``; production callers
unaffected. The "registry retains all original entries" contract here is
relaxed to "registry retains all original entries + at most 1 additive
entry per the source-freeze-modulo-registry invariant".

Covered contracts (9 tests):

    1. DEFAULT_MODEL_KEY is "bge-small-en-v1.5"
    2. Embedder() with no args builds the 384d bge-small embedder
    3. DEFAULT_EMBED_DIM (and legacy EMBED_DIM alias) is 384
    4. MODEL_REGISTRY retains the original 3 entries; D-02
       allows the additive all-MiniLM-L6-v2 entry without breaking the
       English-Only Brain lock
    5. IAI_MCP_EMBED_MODEL=bge-m3 env var still selects bge-m3
    6. embedder_for_store on a 1024d store returns bge-m3 (back-compat)
    7. embedder_for_store on a 384d store returns bge-small-en-v1.5
    8. PROJECT.md line 125 still mentions bge-small-en-v1.5 (constraint)
    9. importing the package does NOT auto-download bge-m3 weights
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch):
    """Every test starts without an IAI_MCP_EMBED_MODEL override."""
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    yield


# --------------------------------------------------------------------------- tests


def test_default_model_key_is_bge_small():
    from iai_mcp.embed import DEFAULT_MODEL_KEY

    assert DEFAULT_MODEL_KEY == "bge-small-en-v1.5"


def test_embedder_defaults_to_384d_small():
    from iai_mcp.embed import Embedder

    assert Embedder.DEFAULT_MODEL_KEY == "bge-small-en-v1.5"
    assert Embedder.DEFAULT_DIM == 384
    assert Embedder.DIM == 384


def test_types_embed_dim_defaults_to_384():
    from iai_mcp.types import DEFAULT_EMBED_DIM, EMBED_DIM

    assert DEFAULT_EMBED_DIM == 384
    assert EMBED_DIM == 384


def test_model_registry_retains_original_three_entries():
    """The 3 original entries must remain unchanged. D-02
    allows additive entries (currently: all-MiniLM-L6-v2) but the original
    contract — bge-m3 / multilingual-e5-small / bge-small-en-v1.5 with their
    canonical dims — is non-negotiable."""
    from iai_mcp.embed import MODEL_REGISTRY

    # Original 3 entries must be present and byte-identical to Plan 05-08.
    assert "bge-m3" in MODEL_REGISTRY
    assert "multilingual-e5-small" in MODEL_REGISTRY
    assert "bge-small-en-v1.5" in MODEL_REGISTRY
    assert MODEL_REGISTRY["bge-m3"] == {"hf": "BAAI/bge-m3", "dim": 1024}
    assert MODEL_REGISTRY["bge-small-en-v1.5"] == {
        "hf": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    }
    assert MODEL_REGISTRY["multilingual-e5-small"] == {
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
    }
    # additive entries are allowed, but the original 3 must
    # never be removed or mutated. Guard explicitly against pruning.
    assert {"bge-m3", "multilingual-e5-small", "bge-small-en-v1.5"}.issubset(
        set(MODEL_REGISTRY)
    )


def test_env_var_still_selects_bge_m3(monkeypatch):
    monkeypatch.setenv("IAI_MCP_EMBED_MODEL", "bge-m3")
    from iai_mcp.embed import _resolve_model_key

    assert _resolve_model_key() == "bge-m3"


def test_embedder_for_store_picks_bge_m3_for_1024d_store():
    """Back-compat: existing 1024d user stores keep working after the
    default flip. The factory routes around the flip transparently."""
    from iai_mcp.embed import embedder_for_store

    store = SimpleNamespace(embed_dim=1024)
    with mock.patch("iai_mcp.embed._get_model") as mock_get:
        mock_get.return_value = mock.MagicMock()
        e = embedder_for_store(store)
    assert e.model_key == "bge-m3"
    assert e.DIM == 1024


def test_embedder_for_store_picks_bge_small_for_384d_store():
    from iai_mcp.embed import embedder_for_store

    store = SimpleNamespace(embed_dim=384)
    with mock.patch("iai_mcp.embed._get_model") as mock_get:
        mock_get.return_value = mock.MagicMock()
        e = embedder_for_store(store)
    assert e.model_key == "bge-small-en-v1.5"
    assert e.DIM == 384


def test_project_md_still_pins_bge_small_constraint():
    """PROJECT.md line 125 was the source of truth all along. This plan
    merely reverts the Phase-2 deviation. Asserting the file content
    here guards against someone silently flipping the spec in the future."""
    p = Path(__file__).resolve().parents[1] / ".planning" / "PROJECT.md"
    if not p.exists():
        pytest.skip(".planning is gitignored; PROJECT.md not present in this checkout")
    content = p.read_text()
    assert "bge-small-en-v1.5" in content
    assert "384d embeddings" in content or "384d" in content


def test_package_import_does_not_auto_download_models():
    """Importing iai_mcp must not trigger a SentenceTransformer download
    for ANY model. The weights pull should happen lazily on first
    Embedder() instantiation, not at import time. Otherwise a fresh
    install spends minutes pulling bge-m3 before the user has even
    decided which model they want."""
    import sys

    # Pretend sentence_transformers is absent so any early reference to
    # SentenceTransformer() would raise. If the import path is clean, this
    # should succeed even without the package loaded.
    with mock.patch.dict(sys.modules):
        # Drop cached iai_mcp modules so the import actually re-runs.
        for name in list(sys.modules):
            if name.startswith("iai_mcp"):
                sys.modules.pop(name, None)
        # Track SentenceTransformer construction attempts.
        from sentence_transformers import SentenceTransformer

        with mock.patch.object(
            SentenceTransformer, "__init__",
            side_effect=AssertionError("model instantiated at import time"),
        ):
            import iai_mcp.embed  # noqa: F401
            import iai_mcp.types  # noqa: F401
