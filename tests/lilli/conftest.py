"""Lilli test-package conftest: forbidden-import boundary hook.

At pytest collection time, before any test module is imported, every source
file under tests/lilli/ is scanned with a regex for imports of forbidden
infrastructure modules. If a forbidden import is found, pytest raises
UsageError and collection stops.

This enforces the boundary that lilli tests must not reach into daemon,
lifecycle, or coordinator infrastructure.

Permitted cross-tier imports (explicitly allowed despite being outside lilli):
  iai_mcp.events, iai_mcp.store, iai_mcp.types, iai_mcp.hippo,
  iai_mcp.tem, iai_mcp.migrate, iai_mcp.lilli.*

These cross-tier imports are currently retained pending full FSL extraction
in a future phase.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest


# ---------------------------------------------------------------------------
# Forbidden module list — boundary enforcement
# ---------------------------------------------------------------------------

# Infrastructure modules that lilli tests must NOT import.
# Matches any import rooted at these modules (e.g. iai_mcp.daemon.config).
_FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "iai_mcp.daemon",
    "iai_mcp.lifecycle",
    "iai_mcp.lifecycle_state",
    "iai_mcp.lifecycle_event_log",
    "iai_mcp.lifecycle_lock",
    "iai_mcp.s2_coordinator",
    "iai_mcp.socket_server",
    "iai_mcp.daemon_config",
    "iai_mcp.daemon_state",
)

# Regex: matches `from iai_mcp.X` or `import iai_mcp.X` at the start of a
# source line (optional leading whitespace), capturing the full module path
# so the forbidden-prefix check can be applied.
_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(iai_mcp(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
    re.MULTILINE,
)


class ForbiddenImport(NamedTuple):
    """A single forbidden import occurrence in a source file."""

    path: Path
    line_number: int
    module: str
    line_text: str


def scan_for_forbidden_imports(source_path: Path) -> list[ForbiddenImport]:
    """Scan *source_path* for forbidden infrastructure imports.

    Reads the file at *source_path* as text, applies the import regex line
    by line, and returns a (possibly empty) list of ForbiddenImport records
    for any match whose module starts with a forbidden prefix.

    The scan runs on raw source text, before the module is imported, so it
    catches imports that would be skipped at runtime by ``TYPE_CHECKING``
    guards or lazy imports.
    """
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    hits: list[ForbiddenImport] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = _IMPORT_RE.match(line)
        if m is None:
            continue
        module = m.group(1)
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in _FORBIDDEN_MODULE_PREFIXES
        ):
            hits.append(ForbiddenImport(source_path, line_no, module, line.strip()))
    return hits


# ---------------------------------------------------------------------------
# pytest hook — collection-time boundary check
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001 -- required by pytest hook signature
    items: list[pytest.Item],
) -> None:
    """Fail-fast: reject any test whose source file imports a forbidden module.

    Iterates over collected items, reads each test module's source file via
    :func:`scan_for_forbidden_imports`, and raises:class:`pytest.UsageError`
    immediately on the first violation found. The hook reads from disk
    (``item.module.__file__``), not from already-imported module objects, so
    it catches imports that were conditionally skipped during import.
    """
    # Only apply the boundary to modules collected under this package
    # (tests/lilli). The hook fires for ALL collected items in the session
    # (parent conftest items too), so we must filter.
    this_dir = Path(__file__).resolve().parent

    seen: set[Path] = set()
    for item in items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        source_path = Path(module_file).resolve()
        # Only check files inside this package directory
        try:
            source_path.relative_to(this_dir)
        except ValueError:
            continue
        if source_path in seen:
            continue
        seen.add(source_path)

        violations = scan_for_forbidden_imports(source_path)
        if violations:
            v = violations[0]
            raise pytest.UsageError(
                f"Boundary violation: forbidden import in {v.path.relative_to(this_dir.parent)}:"
                f"{v.line_number}: '{v.module}'\n"
                f"  {v.line_text}\n"
                f"Lilli tests must not import daemon/lifecycle/coordinator infrastructure."
            )
