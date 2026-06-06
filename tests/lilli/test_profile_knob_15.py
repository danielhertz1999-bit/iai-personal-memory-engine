"""RED-state test scaffold. Tasks 2-5 turn these GREEN.

Covers MCP-12 /: 15th profile knob `wake_depth` (enum minimal|standard|deep,
default=minimal, sealed) registered in KNOB_REGISTRY, set via profile_get_set.
"""
from __future__ import annotations

from iai_mcp.profile import (
    PHASE_1_LIVE,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)


def test_registry_has_15_knobs():
    """11 sealed entries (10 AUTIST + wake_depth MCP-12).

    Test/file name kept for git history stability — was '15' post-MCP-12, now 11 after removed AUTIST-02/08/11/12.
    """
    assert len(PROFILE_KNOBS) == 11


def test_wake_depth_knob_exists():
    assert "wake_depth" in PROFILE_KNOBS


def test_wake_depth_knob_shape():
    """enum:minimal|standard|deep, default=minimal, MCP-12."""
    spec = PROFILE_KNOBS["wake_depth"]
    # value_schema shape
    assert spec.value_schema == "enum:minimal|standard|deep", spec.value_schema
    # default
    assert spec.default == "minimal"
    # phase = live in (counts toward PHASE_1_LIVE)
    assert spec.phase == 1
    # requirement_id
    assert spec.requirement_id == "MCP-12"


def test_wake_depth_in_phase_1_live():
    """wake_depth is live, not deferred."""
    assert "wake_depth" in PHASE_1_LIVE


def test_wake_depth_default_applies():
    """default_state returns wake_depth='minimal' when not set elsewhere."""
    state = default_state()
    assert state.get("wake_depth") == "minimal"


def test_wake_depth_set_valid():
    """profile_set('wake_depth', 'standard', state) succeeds."""
    state = default_state()
    r = profile_set("wake_depth", "standard", state)
    assert r["status"] == "ok"
    assert state["wake_depth"] == "standard"
    # And 'deep' too
    r2 = profile_set("wake_depth", "deep", state)
    assert r2["status"] == "ok"


def test_wake_depth_set_invalid_rejected():
    """profile_set rejects values outside the enum."""
    state = default_state()
    r = profile_set("wake_depth", "weird", state)
    assert r["status"] == "error"


def test_profile_get_wake_depth_returns_value():
    state = default_state()
    r = profile_get("wake_depth", state)
    assert r["knob"] == "wake_depth"
    assert r["value"] == "minimal"
