"""English-Only Brain embedder — defaults and registry contracts.

The sole backend is the Rust native extension (iai_mcp_native.embed.Embedder),
which runs bge-small-en-v1.5 (384d, English-only) with no PyTorch fallback.
MODEL_REGISTRY contains exactly one entry. sentence-transformers and torch are
not installed.

Covered contracts (6 tests):

    1. DEFAULT_MODEL_KEY is "bge-small-en-v1.5"
    2. Embedder() with no args builds the 384d bge-small embedder
    3. DEFAULT_EMBED_DIM (and legacy EMBED_DIM alias) is 384
    4. MODEL_REGISTRY has exactly one entry: bge-small-en-v1.5
    5. embedder_for_store on a 384d store returns bge-small-en-v1.5
    6. importing iai_mcp.embed does NOT pull sentence_transformers into sys.modules
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_model_registry_is_english_only_single_entry():
    """MODEL_REGISTRY contains exactly one entry: the English bge-small-en-v1.5
    model. The registry collapse is the English-Only Brain invariant in force:
    bge-m3, multilingual-e5-small, and all-MiniLM-L6-v2 have been removed."""
    from iai_mcp.embed import MODEL_REGISTRY

    assert set(MODEL_REGISTRY.keys()) == {"bge-small-en-v1.5"}
    assert MODEL_REGISTRY["bge-small-en-v1.5"] == {
        "hf": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    }


def test_embedder_for_store_picks_bge_small_for_384d_store():
    from iai_mcp.embed import embedder_for_store

    store = SimpleNamespace(embed_dim=384)
    e = embedder_for_store(store)
    assert e.model_key == "bge-small-en-v1.5"
    assert e.DIM == 384


def test_project_md_still_pins_bge_small_constraint():
    """is the source of truth for the default embedder. Guard
    against silently flipping the spec in the future."""
    p = Path(__file__).resolve().parents[1] / ".planning" / "PROJECT.md"
    if not p.exists():
        pytest.skip(".planning is gitignored; PROJECT.md not present in this checkout")
    content = p.read_text()
    assert "bge-small-en-v1.5" in content
    assert "384d embeddings" in content or "384d" in content


def test_import_embed_does_not_pull_sentence_transformers():
    """Importing iai_mcp.embed must not load sentence_transformers into
    sys.modules. sentence-transformers is not a dependency; any import
    would fail loudly in a clean install."""
    from unittest import mock

    # Use patch.dict so sys.modules is fully restored on exit — including every
    # iai_mcp.* module popped below. A plain pop-and-reload risks leaving
    # module-level caches and singletons stale for the rest of the test session.
    with mock.patch.dict(sys.modules):
        for name in [k for k in sys.modules if k.startswith("iai_mcp")]:
            sys.modules.pop(name, None)
        import iai_mcp.embed  # noqa: F401
        import iai_mcp.types  # noqa: F401
        assert "sentence_transformers" not in sys.modules, (
            "iai_mcp.embed must not pull sentence_transformers into sys.modules; "
            "sentence-transformers is not a dependency"
        )
