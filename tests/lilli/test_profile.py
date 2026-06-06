"""Tests for the 11-knob profile registry (+ + MCP-12 + removals).

 (AUTIST-13) flipped the 14th autistic-kernel knob camouflaging_relaxation
from phase=3 (deferred) to phase=1 (live). (MCP-12) appends the
15th sealed knob `wake_depth` (operator-facing, default="minimal"). All 15
knobs now live; PHASE_3_DEFERRED empty.

Scope:
- Registry shape: 15 total, 15 live, 0 deferred, 0 deferred.
- defaults on autistic-kernel live knobs (AUTIST-01..14).
- Every autistic-kernel knob carries an AUTIST-* requirement_id; wake_depth
  carries MCP-12.
- profile_get(None) shape: live/deferred/total_knobs.
- profile_get on each branch (live / unknown).
- profile_set success + schema validation (enum, bool, int_range, float_range).
- profile_set unknown-knob error.
- profile_set rejects out-of-enum, wrong-type, out-of-range values.
"""
from __future__ import annotations

from iai_mcp.profile import (
    PHASE_1_LIVE,
    PHASE_2_DEFERRED,
    PHASE_3_DEFERRED,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)


# ------------------------------------------------------------- registry shape


def test_profile_has_exactly_14_knobs():
    """11 knobs total (10 autistic-kernel + wake_depth).

    Test name kept for git stability (was 14 pre-MCP-12, 15 post-MCP-12, 11
    after removed AUTIST-02/08/11/12).
    """
    assert len(PROFILE_KNOBS) == 11


def test_phase_1_live_has_exactly_fourteen():
    """11 live knobs (10 autistic-kernel + wake_depth MCP-12)."""
    assert len(PHASE_1_LIVE) == 11
    # original four must remain
    assert "literal_preservation" in PHASE_1_LIVE
    assert "masking_off" in PHASE_1_LIVE
    assert "task_support" in PHASE_1_LIVE
    assert "scene_construction_scaffold" in PHASE_1_LIVE
    # additions (still live; double_empathy removed)
    assert "monotropism_depth" in PHASE_1_LIVE
    assert "dunn_quadrant" in PHASE_1_LIVE
    # FLIP: AUTIST-13 camouflaging_relaxation
    assert "camouflaging_relaxation" in PHASE_1_LIVE
    # APPEND: the operator-facing knob (MCP-12)
    assert "wake_depth" in PHASE_1_LIVE


def test_phase_2_deferred_is_empty():
    """PHASE_2_DEFERRED is empty (all 9 flipped to phase=1)."""
    assert PHASE_2_DEFERRED == frozenset()
    assert len(PHASE_2_DEFERRED) == 0


def test_phase_3_deferred_is_empty_after_autist13_flip():
    """FLIP: AUTIST-13 camouflaging_relaxation flipped to live; nothing deferred."""
    assert PHASE_3_DEFERRED == frozenset()
    assert len(PHASE_3_DEFERRED) == 0


def test_every_knob_has_autist_requirement_id():
    """10 autistic-kernel knobs carry AUTIST-*; wake_depth carries MCP-12."""
    for name, spec in PROFILE_KNOBS.items():
        if name == "wake_depth":
            assert spec.requirement_id == "MCP-12", (
                f"wake_depth must carry MCP-12 requirement_id, got {spec.requirement_id}"
            )
            continue
        assert spec.requirement_id.startswith("AUTIST-"), (
            f"knob {name} missing AUTIST-* requirement_id"
        )


def test_live_knob_defaults_match_d11():
    """specifies autistic-kernel defaults on the 4 live knobs."""
    state = default_state()
    assert state["literal_preservation"] == "strong"
    assert state["masking_off"] is True
    assert state["task_support"] == "cued_recognition"
    assert state["scene_construction_scaffold"] is True


def test_default_state_excludes_deferred_knobs():
    """default_state() returns only the live knobs; deferred keys must be absent."""
    state = default_state()
    assert set(state.keys()) == PHASE_1_LIVE
    #: 11 live knobs (10 autistic-kernel + wake_depth MCP-12).
    assert len(state) == 11


# -------------------------------------------------------------- profile_get


def test_profile_get_none_returns_total_14():
    """11 live + 0 deferred = 11 total (10 autistic-kernel + wake_depth).

    Test name kept for git stability (was 14 pre-MCP-12, 15 post-MCP-12, 11
    after removed AUTIST-02/08/11/12).
    """
    state = default_state()
    result = profile_get(None, state)
    assert result["total_knobs"] == 11
    assert len(result["live"]) == 11
    assert len(result["deferred"]) == 0


def test_profile_get_none_live_values_match_d11():
    state = default_state()
    result = profile_get(None, state)
    assert result["live"]["literal_preservation"] == "strong"
    assert result["live"]["masking_off"] is True
    assert result["live"]["task_support"] == "cued_recognition"
    assert result["live"]["scene_construction_scaffold"] is True


def test_profile_get_none_deferred_entries_have_requirement_id():
    state = default_state()
    result = profile_get(None, state)
    for name, entry in result["deferred"].items():
        assert entry["status"] == "not-yet-implemented"
        assert entry["phase"] in (2, 3)
        assert entry["requirement_id"].startswith("AUTIST-")
        assert "description" in entry


def test_profile_get_live_specific_knob():
    state = default_state()
    r = profile_get("literal_preservation", state)
    assert r == {"knob": "literal_preservation", "value": "strong"}


def test_profile_get_monotropism_depth_now_live():
    """monotropism_depth is live -> returns {knob, value}."""
    state = default_state()
    r = profile_get("monotropism_depth", state)
    assert r["knob"] == "monotropism_depth"
    assert "value" in r
    # Default is an empty per-domain dict.
    assert r["value"] == {}


def test_profile_get_camouflaging_now_live_after_autist13_flip():
    """FLIP: camouflaging_relaxation is live; profile_get returns value."""
    state = default_state()
    r = profile_get("camouflaging_relaxation", state)
    assert r["knob"] == "camouflaging_relaxation"
    assert "value" in r
    assert r["value"] == 0.0  # D-AUTIST13 default


def test_profile_get_unknown_knob():
    state = default_state()
    r = profile_get("does_not_exist", state)
    assert r == {"knob": "does_not_exist", "status": "unknown"}


# -------------------------------------------------------------- profile_set


def test_profile_set_live_enum_success():
    """Live enum knob: set within the allowed set -> ok + state mutated."""
    state = default_state()
    r = profile_set("literal_preservation", "loose", state)
    assert r["status"] == "ok"
    assert r["value"] == "loose"
    assert profile_get("literal_preservation", state)["value"] == "loose"


def test_profile_set_live_enum_rejects_bogus_value():
    state = default_state()
    r = profile_set("literal_preservation", "bogus", state)
    assert r["status"] == "error"
    # State must not have been mutated.
    assert state["literal_preservation"] == "strong"


def test_profile_set_live_bool_rejects_non_bool():
    """bool schema must not accept int 1 / string "true" etc."""
    state = default_state()
    r = profile_set("masking_off", 1, state)
    assert r["status"] == "error"
    assert state["masking_off"] is True


def test_profile_set_live_bool_accepts_true():
    state = default_state()
    r = profile_set("masking_off", False, state)
    assert r["status"] == "ok"
    assert state["masking_off"] is False


def test_profile_set_monotropism_depth_rejects_non_dict():
    """monotropism_depth now a dict schema; int values rejected."""
    state = default_state()
    r = profile_set("monotropism_depth", 3, state)
    assert r["status"] == "error"
    # Schema validator rejects ints for dict schema.
    assert "dict" in r["reason"].lower()


def test_profile_set_camouflaging_now_accepts_value_after_autist13_flip():
    """FLIP: AUTIST-13 camouflaging_relaxation is live; profile_set succeeds."""
    state = default_state()
    r = profile_set("camouflaging_relaxation", 0.5, state)
    assert r["status"] == "ok"
    assert state["camouflaging_relaxation"] == 0.5


def test_profile_set_camouflaging_rejects_out_of_range():
    """live schema is float_range:0.0..1.0; out-of-range rejected."""
    state = default_state()
    r = profile_set("camouflaging_relaxation", 1.5, state)
    assert r["status"] == "error"


def test_profile_set_unknown_knob_returns_unknown_reason():
    state = default_state()
    r = profile_set("does_not_exist", 1, state)
    assert r["status"] == "error"
    assert r["reason"] == "unknown knob"


def test_profile_set_task_support_enum_accepts_blank_recall():
    """task_support="cued_recognition" is the default; enum allows toggle."""
    state = default_state()
    r = profile_set("task_support", "blank_recall", state)
    assert r["status"] == "ok"
    assert state["task_support"] == "blank_recall"


def test_profile_set_scene_construction_scaffold_rejects_string():
    state = default_state()
    r = profile_set("scene_construction_scaffold", "yes", state)
    assert r["status"] == "error"
