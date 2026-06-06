"""Back-compat shim — pattern separation has moved to iai_mcp.lilli.ops.separation.

Re-exports the public API so callers that have not yet migrated keep working.
"""
from __future__ import annotations

from iai_mcp.lilli.ops.separation import (
    OrthogonalizationResult,
    detect_hubness,
    orthogonalize_for_routing,
)

__all__ = [
    "OrthogonalizationResult",
    "detect_hubness",
    "orthogonalize_for_routing",
]
