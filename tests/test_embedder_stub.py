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
    import iai_mcp_native
    pkg_path = Path(iai_mcp_native.__file__).parent
    candidates = [
        pkg_path / "__init__.pyi",
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
    if "No module named mypy" in result.stderr or result.returncode == 127:
        pytest.skip("mypy not installed in this venv")
    assert result.returncode == 0, (
        f"mypy --strict failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
