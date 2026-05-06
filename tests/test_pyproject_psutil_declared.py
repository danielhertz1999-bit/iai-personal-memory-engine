"""Plan 07.2-01 R5 prep regression fence — psutil MUST be declared in
pyproject.toml [project.dependencies], not just transitively reachable.

Background: CONTEXT.md D7.2-17 claimed psutil was "already a project dep"
but it was only transitive via accelerate in the [compress] extra. A
clean `pip install -e .` produced a venv WITHOUT psutil and Plan 05's
_cpu_watchdog_loop would `import psutil` and fail. Plan 01 added the
explicit declaration. This test prevents accidental removal.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_psutil_declared_in_project_dependencies() -> None:
    """psutil must appear in [project.dependencies] with a version floor."""
    text = PYPROJECT.read_text()
    # Locate the [project] block (we don't want a [project.optional-dependencies]
    # match to satisfy this).
    project_marker = text.find("\n[project]")
    if project_marker < 0:
        project_marker = text.find("[project]") if text.startswith("[project]") else -1
    assert project_marker >= 0, "[project] block not found in pyproject.toml"
    next_section = text.find("\n[", project_marker + len("\n[project]"))
    # next_section may be -1 if [project] is the last block; clip to len.
    section_end = next_section if next_section >= 0 else len(text)
    project_block = text[project_marker:section_end]
    # Permissive match: any psutil line with a version floor.
    assert "psutil" in project_block, (
        "psutil missing from [project] block. Plan 07.2-01 R5 prep added "
        "this declaration so a clean `pip install -e .` reaches psutil "
        "without the [compress] extra. Restore the line."
    )
    # Strong shape check: matches `"psutil>=5.x` or `"psutil >=5.x` etc.
    import re
    match = re.search(r'"\s*psutil\s*>=\s*\d+', project_block)
    assert match, (
        'Expected `"psutil>=X.Y.Z"` style declaration in [project] '
        "dependencies. Plan 07.2-01 chose >=5.9.0 to match the "
        "accelerate transitive-floor and stay broad."
    )
