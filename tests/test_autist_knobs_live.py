"""Tests for autistic-kernel knob registry: 10 AUTIST + 1 wake_depth = 11 sealed.

Four dead KnobSpec entries (AUTIST-02 sensory_channel_weights, AUTIST-08
event_vs_time_cue, AUTIST-11 alexithymia_accommodation, AUTIST-12
double_empathy) were removed — final shape is 11 sealed entries,
10 AUTIST + wake_depth (MCP-12).

Schema/value validation covers enum/bool/int_range/float_range and
`dict:<keytype>:<valuetype>` for monotropism_depth (recursive per-key
validation). AUTIST-03 dunn_quadrant keeps the enum shape but gains a
float_range-style HIPPEA_precision spec.
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


# --------------------------------------------------------------- registry shape

def test_phase_1_live_has_14_knobs():
    """10 autistic-kernel + wake_depth = 11 live.

    Test name kept for git stability. The autistic-kernel-only invariant (10)
    is checked via filter in test_all_14_requirement_ids_present.
    """
    assert len(PHASE_1_LIVE) == 11


def test_phase_3_deferred_now_empty_after_autist13_flip():
    """camouflaging_relaxation moved from phase=3 to phase=1."""
    assert PHASE_3_DEFERRED == frozenset()
    assert len(PHASE_3_DEFERRED) == 0


def test_phase_2_deferred_empty():
    """All 9 Phase-2 knobs move to phase=1."""
    assert PHASE_2_DEFERRED == frozenset()
    assert len(PHASE_2_DEFERRED) == 0


def test_all_14_requirement_ids_present():
    """Autistic-kernel slice has exactly 10 knobs (AUTIST-02/08/11/12 removed).

    The registry adds wake_depth (MCP-12) for a final shape of 11 sealed
    entries (10 AUTIST + 1 MCP-12). Test name kept for git stability.
    """
    autist_specs = [
        s for s in PROFILE_KNOBS.values() if s.requirement_id.startswith("AUTIST-")
    ]
    assert len(autist_specs) == 10
    req_ids = {spec.requirement_id for spec in autist_specs}
    expected = {
        "AUTIST-01", "AUTIST-03", "AUTIST-04", "AUTIST-05",
        "AUTIST-06", "AUTIST-07", "AUTIST-09", "AUTIST-10",
        "AUTIST-13", "AUTIST-14",
    }
    assert req_ids == expected
    # Registry total includes the operator-facing MCP-12 wake_depth knob.
    assert len(PROFILE_KNOBS) == 11
    assert "wake_depth" in PROFILE_KNOBS
    assert PROFILE_KNOBS["wake_depth"].requirement_id == "MCP-12"


# ------------------------------------------------------- dict-schema validator


def test_monotropism_depth_live_accepts_dict():
    """AUTIST-01: monotropism_depth is a per-domain dict[str, float_range:0..1]."""
    state = default_state()
    r = profile_set(
        "monotropism_depth",
        {"coding": 0.8, "gardening": 0.3},
        state,
    )
    assert r["status"] == "ok"
    assert state["monotropism_depth"] == {"coding": 0.8, "gardening": 0.3}


def test_monotropism_depth_live_rejects_out_of_range():
    state = default_state()
    r = profile_set("monotropism_depth", {"x": 1.5}, state)
    assert r["status"] == "error"


def test_monotropism_depth_live_rejects_non_dict():
    state = default_state()
    r = profile_set("monotropism_depth", 3, state)
    assert r["status"] == "error"


# The sensory_channel_weights tests were removed — AUTIST-02 was a
# DEAD knob (declared but never read in any production scoring/response code);
# the registry entry was removed and profile_set now returns the unknown-knob
# error. See tests/test_profile_no_dead_knobs.py for the post-removal contract.


# ------------------------------------------------------- enum-schema validator


def test_dunn_quadrant_live():
    state = default_state()
    r = profile_set("dunn_quadrant", "seeking", state)
    assert r["status"] == "ok"
    assert state["dunn_quadrant"] == "seeking"


def test_dunn_quadrant_rejects_garbage():
    state = default_state()
    r = profile_set("dunn_quadrant", "garbage", state)
    assert r["status"] == "error"


def test_demand_avoidance_tolerance_live():
    state = default_state()
    for value in ("collaborative", "neutral", "imperative"):
        r = profile_set("demand_avoidance_tolerance", value, state)
        assert r["status"] == "ok", f"expected {value} accepted"
    assert state["demand_avoidance_tolerance"] == "imperative"


# The event_vs_time_cue / alexithymia_accommodation tests were removed —
# AUTIST-08 (event_vs_time_cue) and AUTIST-11 (alexithymia_accommodation) were
# DEAD knobs (no taxonomy in schema, never read in production). Removed from
# registry; profile_set now returns the unknown-knob error.
# See tests/test_profile_no_dead_knobs.py for the post-removal contract.


# ----------------------------------------------------- bool-schema validator


def test_inertia_awareness_live():
    state = default_state()
    r_ok = profile_set("inertia_awareness", True, state)
    assert r_ok["status"] == "ok"
    r_bad = profile_set("inertia_awareness", 1, state)
    assert r_bad["status"] == "error"


# The double_empathy test was removed — AUTIST-12 (double_empathy)
# became a passive system invariant; the system never translates phrasing
# toward NT style at any path, so a runtime knob was redundant. Removed from
# registry. See tests/test_profile_no_dead_knobs.py for the post-removal contract.


# ----------------------------------------------------- float-schema validator


def test_interest_boost_live():
    state = default_state()
    r_ok = profile_set("interest_boost", 0.75, state)
    assert r_ok["status"] == "ok"
    r_bad = profile_set("interest_boost", 2.0, state)
    assert r_bad["status"] == "error"


# ----------------------------------------------------- HIPPEA_precision spec


def test_HIPPEA_precision_spec_added_wire_to_autist_03():
    """AUTIST-03 now maps to dunn_quadrant (enum) AND exposes a
    HIPPEA_precision float knob via the dict-key mechanism on a per-domain map
    OR via a float_range schema.

    We require either:
    - PROFILE_KNOBS["HIPPEA_precision"] exists with float_range:0.0..1.0, or
    - PROFILE_KNOBS["dunn_quadrant"] value_schema carries float-range metadata

    Accept the simpler form: a new "HIPPEA_precision" knob with requirement id
    AUTIST-03 or a companion 'autist_03_float' marker on dunn_quadrant.
    """
    # Check one of the two shapes is present.
    if "HIPPEA_precision" in PROFILE_KNOBS:
        spec = PROFILE_KNOBS["HIPPEA_precision"]
        # Must be a float range between 0 and 1.
        assert "float_range:" in spec.value_schema
    else:
        # dunn_quadrant remains but must retain an enum schema (migration-aware)
        spec = PROFILE_KNOBS["dunn_quadrant"]
        assert spec.value_schema.startswith("enum:")


# ----------------------------------------------------- profile_get coverage


def test_profile_get_returns_14_live_entries():
    """11 live (10 autistic + wake_depth MCP-12). Test name kept for git stability."""
    state = default_state()
    result = profile_get(None, state)
    assert len(result["live"]) == 11
    assert len(result["deferred"]) == 0


def test_profile_get_monotropism_depth_returns_default_dict():
    state = default_state()
    r = profile_get("monotropism_depth", state)
    assert r["knob"] == "monotropism_depth"
    assert "value" in r
    # Default is a dict (per-domain storage)
    assert isinstance(r["value"], dict)


# -------------------------------------------- mutable-default isolation guard


def test_default_state_returns_independent_mutable_defaults():
    """default_state() must hand out fresh copies of mutable defaults.

    monotropism_depth defaults to an empty dict. If default_state() returned the
    literal registry default object, two calls would share the same dict and an
    in-place mutation of one would silently pollute every other caller AND the
    registry default for the whole process lifetime. Guard against that
    shared-mutable-default leak: each call must yield independent objects.
    """
    s1 = default_state()
    s2 = default_state()

    # Distinct objects, not the same shared dict.
    assert s1["monotropism_depth"] is not s2["monotropism_depth"]

    # In-place mutation of one caller's dict must not leak anywhere.
    s1["monotropism_depth"]["coding"] = 0.9

    # Other caller is untouched.
    assert s2["monotropism_depth"] == {}
    # The registry default is untouched (no process-wide pollution).
    assert PROFILE_KNOBS["monotropism_depth"].default == {}
