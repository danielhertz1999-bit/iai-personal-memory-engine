"""Back-compat shim -- Hebbian structure ops have moved to iai_mcp.lilli.ops.hebbian."""
from __future__ import annotations

from iai_mcp.lilli.ops.hebbian import (
    STRUCTURAL_SIMILARITY_THRESHOLD,
    structural_similarity,
    strengthen_structure_edge,
    co_retrieval_trigger,
    monitor_similarity_window,
)

__all__ = [
    "STRUCTURAL_SIMILARITY_THRESHOLD",
    "structural_similarity",
    "strengthen_structure_edge",
    "co_retrieval_trigger",
    "monitor_similarity_window",
]
