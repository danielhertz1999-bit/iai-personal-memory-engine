"""Back-compat shim — camouflaging detector + double_empathy invariant
have moved to iai_mcp.lilli.profile.invariants.
"""
from __future__ import annotations

from iai_mcp.lilli.profile.invariants import (
    detect_camouflaging,
    record_user_formality,
    relax_register,
    run_weekly_pass,
    DOUBLE_EMPATHY_PASSIVE_INVARIANT,
)

__all__ = [
    "detect_camouflaging",
    "record_user_formality",
    "relax_register",
    "run_weekly_pass",
    "DOUBLE_EMPATHY_PASSIVE_INVARIANT",
]
