"""Back-compat shim -- profile knob registry has moved to iai_mcp.lilli.profile.knobs.

Re-exports the full public API. The 11-knob sealed invariant lives in
iai_mcp.lilli.profile.knobs (10 AUTIST + 1 wake_depth).
"""
from __future__ import annotations

from iai_mcp.lilli.profile.knobs import (
    KnobSpec,
    PROFILE_KNOBS,
    PHASE_1_LIVE,
    PHASE_2_DEFERRED,
    PHASE_3_DEFERRED,
    SIGNAL_WEIGHT,
    PROFILE_SENTINEL_UUID_STR,
    default_state,
    _validate,
    profile_get,
    profile_set,
    bayesian_update,
    profile_modulation_for_record,
)

__all__ = [
    "KnobSpec",
    "PROFILE_KNOBS",
    "PHASE_1_LIVE",
    "PHASE_2_DEFERRED",
    "PHASE_3_DEFERRED",
    "SIGNAL_WEIGHT",
    "PROFILE_SENTINEL_UUID_STR",
    "default_state",
    "_validate",
    "profile_get",
    "profile_set",
    "bayesian_update",
    "profile_modulation_for_record",
]
