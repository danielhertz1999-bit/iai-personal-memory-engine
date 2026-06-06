"""Smoke test for the native iai_mcp_native wheel.

Confirms the three-crate Rust workspace builds into a single.so file that
exposes both `embed` and `graph` Python sub-modules through a single
``iai_mcp_native`` parent module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _native_available() -> bool:
    try:
        import iai_mcp_native  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_embed_submodule_importable() -> None:
    """`from iai_mcp_native import embed` exposes the Embedder class."""
    from iai_mcp_native import embed
    assert embed.Embedder is not None


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_graph_submodule_importable() -> None:
    """`from iai_mcp_native import graph` exposes the answer() callable."""
    from iai_mcp_native import graph
    assert callable(graph.answer)


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_graph_answer_returns_42() -> None:
    """The wiring probe returns the literal 42 — the next plan replaces this."""
    from iai_mcp_native import graph
    assert graph.answer() == 42


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_dotted_imports_register_sys_modules() -> None:
    """`import iai_mcp_native.embed` (and.graph) must register dotted names.

    Without an explicit registration step inside the parent ``#[pymodule]``
    body, PyO3 sub-modules created via ``PyModule::new_bound`` are only
    reachable as attributes (``from iai_mcp_native import embed``) and the
    dotted import (``import iai_mcp_native.embed``) raises
    ``ModuleNotFoundError``. The wrapper crate works around this by setting
    ``sys.modules['iai_mcp_native.embed']`` and
    ``sys.modules['iai_mcp_native.graph']`` at module init.
    """
    import iai_mcp_native  # noqa: F401
    import iai_mcp_native.embed  # noqa: F401
    import iai_mcp_native.graph  # noqa: F401
    assert "iai_mcp_native.embed" in sys.modules
    assert "iai_mcp_native.graph" in sys.modules


@pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)
def test_both_submodules_share_one_so() -> None:
    """The whole point of the three-crate workspace is one.so for two
    sub-modules. We prove that two ways:

    1. The maturin-generated package directory contains exactly one ``.so``
       file (or ``.dylib`` on macOS) — namely the iai_mcp_native extension.
    2. ``id()``-equality holds between the from-import and the dotted-import
       reference, so both code paths return the *same* live module object,
       not two divergent copies.

    PyO3 sub-modules created via ``PyModule::new_bound`` do NOT have a
    ``__file__`` attribute (they live in memory inside the parent module),
    so the simpler ``embed.__file__ == graph.__file__`` invariant can't be
    used — id-equality is the load-bearing check.
    """
    from iai_mcp_native import embed as _embed
    from iai_mcp_native import graph as _graph
    import iai_mcp_native

    # Single.so on disk.
    pkg_dir = Path(iai_mcp_native.__file__).parent
    binaries = [
        f for f in os.listdir(pkg_dir)
        if f.endswith((".so", ".dylib"))
    ]
    assert len(binaries) == 1, (
        f"expected exactly one extension binary in {pkg_dir}, got {binaries}"
    )

    # Single live object identity across all three resolution paths.
    assert id(_embed) == id(iai_mcp_native.embed) == id(sys.modules["iai_mcp_native.embed"])
    assert id(_graph) == id(iai_mcp_native.graph) == id(sys.modules["iai_mcp_native.graph"])
