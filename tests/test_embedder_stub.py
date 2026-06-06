"""Acceptance: a `.pyi` stub ships next to the installed
iai_mcp_native wheel and `mypy --strict` accepts the typed Embedder
signature exposed via `iai_mcp_native.embed`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_pyi_stub_ships_in_wheel():
    """`.pyi` stub must ship beside the installed extension module."""
    import iai_mcp_native
    pkg_path = Path(iai_mcp_native.__file__).parent
    candidates = [
        # `maturin develop` mixed-layout: stub next to the compiled extension.
        pkg_path / "__init__.pyi",
        # `maturin build` flat-layout fallback paths.
        pkg_path / "iai_mcp_native.pyi",
        pkg_path.parent / "iai_mcp_native.pyi",
        *list(pkg_path.glob("iai_mcp_native*.pyi")),
        *list(pkg_path.parent.glob("iai_mcp_native*.pyi")),
    ]
    found = [c for c in candidates if c.exists()]
    assert found, (
        f"iai_mcp_native.pyi not found near {pkg_path}; candidates checked: {candidates}"
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_mypy_strict_passes_on_rust_path(tmp_path):
    """Generate a minimal user file using the Rust API; run `mypy --strict`.

    We disable the ``no-untyped-call`` check because pyo3-stub-gen 0.6 emits
    ``def __new__(cls):...`` without a return-type annotation; mypy --strict
    reads any constructor call as untyped under that emission style. The rest
    of the stub (method signatures, return types, parameter types) is fully
    annotated — those are the checks that catch real type drift between Rust
    and Python.
    """
    src = tmp_path / "mypy_target.py"
    src.write_text(
        "from iai_mcp_native import embed\n"
        "e = embed.Embedder()\n"
        "v: list[float] = e.encode('hello')\n"
        "assert len(v) == 384\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--disable-error-code=no-untyped-call",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    # mypy installed? if not, skip — mypy is a dev-time tool
    if "No module named mypy" in result.stderr or result.returncode == 127:
        pytest.skip("mypy not installed in this venv")
    assert result.returncode == 0, (
        f"mypy --strict failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
