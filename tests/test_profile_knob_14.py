"""Plan 03-03 — 14th autistic-kernel profile knob FLIP verification.

The 14th autistic-kernel knob `camouflaging_relaxation` is FLIPPED from phase=3
(deferred) to phase=1 (live). subsequently appends the 15th
sealed operator-facing knob `wake_depth`, so PHASE_1_LIVE=15 post-Phase-5.

This test locks in:

- Static: PHASE_1_LIVE=15 (14 autistic + wake_depth), PHASE_3_DEFERRED=0, DEFERRED_KNOBS=0.
- Runtime: profile_get returns 15 live knobs; profile_set accepts + validates range
  for camouflaging_relaxation.
- Import-time: `import iai_mcp.core` must succeed (core.py assertion must hold).
"""
from __future__ import annotations

import pytest

from iai_mcp.profile import (
    PHASE_1_LIVE,
    PHASE_2_DEFERRED,
    PHASE_3_DEFERRED,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)


# ------------------------------------------------------------- static FLIP state
def test_phase_1_live_is_14():
    """Plan 07.12-02: 10 autistic-kernel + wake_depth = 11 live.

    Test name kept for git stability. Verifies camouflaging_relaxation
    remained live after flipped it.
    """
    assert len(PHASE_1_LIVE) == 11
    assert "camouflaging_relaxation" in PHASE_1_LIVE


def test_phase_3_deferred_is_empty():
    assert len(PHASE_3_DEFERRED) == 0
    assert "camouflaging_relaxation" not in PHASE_3_DEFERRED


def test_phase_2_deferred_is_empty():
    assert len(PHASE_2_DEFERRED) == 0


def test_knob_spec_phase_is_1():
    spec = PROFILE_KNOBS["camouflaging_relaxation"]
    # Positional arg mapped to `phase` field — check the dataclass attribute.
    assert spec.phase == 1
    assert spec.requirement_id == "AUTIST-13"
    # Description reflects the FLIP (no stale label).
    assert "Phase 3" not in spec.description


def test_core_import_succeeds_with_deferred_knobs_zero():
    """core.py has a module-level assertion that must hold post-FLIP."""
    import iai_mcp.core as core
    assert len(core.DEFERRED_KNOBS) == 0


# ------------------------------------------------------------- runtime semantics
def test_profile_get_returns_14():
    """Plan 07.12-02: 11 total (10 autistic-kernel + wake_depth MCP-12)."""
    state = default_state()
    r = profile_get(None, state)
    assert r["total_knobs"] == 11
    assert len(r["live"]) == 11
    assert len(r["deferred"]) == 0


def test_profile_get_camouflaging_returns_live_value():
    state = default_state()
    r = profile_get("camouflaging_relaxation", state)
    assert r["knob"] == "camouflaging_relaxation"
    assert r["value"] == 0.0  # D-AUTIST13 default


def test_profile_set_camouflaging_accepts_in_range():
    state = default_state()
    r = profile_set("camouflaging_relaxation", 0.3, state)
    assert r["status"] == "ok"
    assert state["camouflaging_relaxation"] == 0.3


def test_profile_set_camouflaging_rejects_out_of_range():
    state = default_state()
    r = profile_set("camouflaging_relaxation", 1.5, state)
    assert r["status"] == "error"


def test_profile_set_camouflaging_rejects_negative():
    state = default_state()
    r = profile_set("camouflaging_relaxation", -0.1, state)
    assert r["status"] == "error"


def test_default_state_includes_camouflaging_relaxation():
    state = default_state()
    assert "camouflaging_relaxation" in state
    assert state["camouflaging_relaxation"] == 0.0
