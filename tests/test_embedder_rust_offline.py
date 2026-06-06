"""Acceptance gate: Accelerate linked, no Metal symbols, offline mode works.

Covers the binary-property half of the Rust embedder build.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


def _rust_lib_path() -> Path:
    import iai_mcp_native
    # The single .so file lives next to __init__.py in the maturin-packaged
    # iai_mcp_native directory. Walk the package directory for a *.so / *.dylib.
    candidate = Path(iai_mcp_native.__file__)
    if candidate.suffix in (".so", ".dylib"):
        return candidate
    for sibling in candidate.parent.glob("iai_mcp_native*"):
        if sibling.suffix in (".so", ".dylib"):
            return sibling
    pytest.skip(f"could not locate native lib next to {candidate}")
    raise RuntimeError("unreachable")


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_accelerate_framework_linked():
    """`otool -L` must list Apple Accelerate.framework in the dylib's link table."""
    lib = _rust_lib_path()
    result = subprocess.run(["otool", "-L", str(lib)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Accelerate.framework" in result.stdout, (
        f"Accelerate.framework not linked in {lib}:\n{result.stdout}"
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_no_metal_symbols():
    """`nm` must show NO Metal / MTL symbols — Metal GPU backend is explicitly disabled.

    Excludes candle-core's `dummy_metal_backend` (intentional trait stubs that
    satisfy the `Device::Metal` enum variant at compile time so the API stays
    consistent — they return errors at runtime, never call any Metal API) and
    `CustomOp::metal_fwd` / `CustomOp::cuda_fwd` trait default impls (always
    present in the candle-core ABI; they too return errors when no GPU is
    compiled in). Neither pulls Metal.framework at link time — verified by the
    paired `test_no_metal_framework_linked` test below.
    """
    lib = _rust_lib_path()
    result = subprocess.run(["nm", str(lib)], capture_output=True, text=True)
    # nm may exit non-zero on stripped binaries but still produces output.
    assert result.stdout, f"nm produced no output for {lib}"
    offenders = [
        line for line in result.stdout.splitlines()
        if re.search(r"metal|MTL", line, flags=re.IGNORECASE)
        and "dummy_metal_backend" not in line
        and "CustomOp" not in line
    ]
    assert not offenders, (
        "Metal symbols leaked into Rust embedder binary:\n"
        + "\n".join(offenders[:10])
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_no_metal_framework_linked():
    """`otool -L` link table must NOT mention Metal.framework or libmetal.

    Strongest possible signal that no Metal GPU code is compiled. Pairs with
    `test_accelerate_framework_linked` — together they prove the binary uses
    Apple Accelerate CPU BLAS only, never the Metal GPU runtime.
    """
    lib = _rust_lib_path()
    result = subprocess.run(["otool", "-L", str(lib)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    bad = [
        line for line in result.stdout.splitlines()
        if "Metal.framework" in line or "libmetal" in line.lower()
    ]
    assert not bad, (
        "Metal framework linked into Rust embedder binary:\n"
        + "\n".join(bad[:10])
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_offline_mode_works_with_warm_cache(monkeypatch):
    """When HF cache is warm, IAI_MCP_EMBED_OFFLINE=1 succeeds without network.

    The native offline branch resolves its cache via HF_HOME (falling back to
    ~/.cache/huggingface when unset). The autouse fixture redirects $HOME to an
    empty tmp dir but sets HF_HOME to the operator's real warm cache, so the
    model is reachable without restoring $HOME. This touches only the read-only
    model cache, never the ~/.iai-mcp store, so store isolation is unaffected.
    """
    monkeypatch.setenv("IAI_MCP_EMBED_OFFLINE", "1")
    from iai_mcp.embed import Embedder
    e = Embedder()
    v = e.embed("hello")
    assert len(v) == 384
