"""Back-compat shim — reconsolidation has moved to iai_mcp.lilli.ops.reconsolidation."""
from __future__ import annotations

from iai_mcp.lilli.ops.reconsolidation import (
    LABILE_WINDOW_SEC,
    MAX_RECONSOLIDATION_DEPTH,
    STABILITY_BOOST_ON_RECALL,
    STABILITY_PENALTY_ON_CONTRADICTION,
    LabileEntry,
    ReconsolidationBuffer,
    compute_stability_update,
)

__all__ = [
    "LABILE_WINDOW_SEC",
    "MAX_RECONSOLIDATION_DEPTH",
    "STABILITY_BOOST_ON_RECALL",
    "STABILITY_PENALTY_ON_CONTRADICTION",
    "LabileEntry",
    "ReconsolidationBuffer",
    "compute_stability_update",
]
