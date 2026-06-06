"""SpatialTagger — pure Wing/Room/Drawer heuristic for records.

Derive a place/grid spatial scaffold (`wing`, `room`, `drawer`) from a
record's source_path string, so downstream navigation-style queries can
address records by location rather than only by content.

Pure module: imports only `pathlib` and `typing`. NO imports of
`iai_mcp.daemon`, `iai_mcp.store`, `iai_mcp.events` — keep it testable in
isolation and free of import cycles.

Example::

    >>> from iai_mcp.spatial_tagger import tag
    >>> tag(None, "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py")
    ('IAI-MCP', 'iai_mcp', 'store')
"""

# pure-module placement; no daemon/store imports at module top level.

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

# tier-2 fallback allowlist. When no `Desktop/` marker is found in
# the source_path, the first path component appearing in this set is used
# as the wing. Operator extension via an env var is intentionally out of
# scope for — keep this frozen for now.
DEFAULT_WINGS: frozenset[str] = frozenset(
    {
        "IAI-MCP",
        "iai-mcp",
        "Documents",
        "Projects",
        "src",
        "code",
        "repos",
    }
)

# tier-1 marker. Lifted to a module constant so a future Windows or
# non-macOS extension can override it without rewriting the function body.
DESKTOP_MARKER: str = "Desktop"


def tag(
    record: Any,
    source_path: str | None,
    default_wing: str = "general",
) -> tuple[str | None, str | None, str | None]:
    """Return ``(wing, room, drawer)`` derived from ``source_path``.

    All three may be ``None`` when no signal is available (e.g., when
    ``source_path`` is ``None`` or empty). ``record`` is currently unused
    but reserved for future heuristics that inspect record fields (e.g.,
    topic tags driving drawer derivation in +); keep the
    parameter to avoid a cross-call breaking change.

    Heuristic per (three ordered tiers for wing):

    1. **Desktop marker** — if any path component equals ``Desktop``, the
       next component is the wing.
    2. **Default-wings allowlist** — first component in:data:`DEFAULT_WINGS`.
    3. **Default fallback** — ``default_wing`` (env-configured;
       ``"general"`` by default).

    ``room`` is the immediate parent directory's name; ``drawer`` is the
    filename stem (no extension). Both fall back to ``None`` when absent.
    """
    #: no source_path -> all three default to None. DO NOT substitute
    # default_wing in this branch — absence of signal is meaningful.
    if source_path is None or not source_path.strip():
        return (None, None, None)

    path = PurePosixPath(source_path)
    components = path.parts

    # Wing — three tiers, in order.
    wing: str | None = None

    # Tier 1: Desktop marker.
    for i, part in enumerate(components):
        if part == DESKTOP_MARKER and i + 1 < len(components):
            wing = components[i + 1]
            break

    # Tier 2: default-wings allowlist.
    if wing is None:
        for part in components:
            if part in DEFAULT_WINGS:
                wing = part
                break

    # Tier 3: env-configured default.
    if wing is None:
        wing = default_wing

    # Room — immediate parent directory name.
    parent_name = path.parent.name
    room: str | None = parent_name if parent_name else None

    # Drawer — basename stem (filename without extension).
    stem = path.stem
    drawer: str | None = stem if stem else None

    return (wing, room, drawer)


class SpatialTagger:
    """Thin class facade exposing:func:`tag` as a classmethod.

    The canonical API is the module-level:func:`tag` function — pure,
    no state. This class is a convenience wrapper so callers that prefer
    the ``SpatialTagger.tag(...)`` namespace style work without a
    second implementation.
    """

    @classmethod
    def tag(
        cls,
        record: Any,
        source_path: str | None,
        default_wing: str = "general",
    ) -> tuple[str | None, str | None, str | None]:
        """Delegate to the module-level:func:`tag` function."""
        return tag(record, source_path, default_wing=default_wing)
