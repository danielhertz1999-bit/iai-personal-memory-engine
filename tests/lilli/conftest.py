from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

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

_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(iai_mcp(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
    re.MULTILINE,
)

class ForbiddenImport(NamedTuple):

    path: Path
    line_number: int
    module: str
    line_text: str

def scan_for_forbidden_imports(source_path: Path) -> list[ForbiddenImport]:
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

def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001 -- required by pytest hook signature
    items: list[pytest.Item],
) -> None:
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
