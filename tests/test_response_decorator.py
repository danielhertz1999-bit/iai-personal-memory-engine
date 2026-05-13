"""Tests for the apply_profile decorator ( / D5-04).

Covers server-side apply_profile decorator that transforms the response dict
per the 11 profile knobs — per-knob silent-fail, knob names never cross the
MCP wire. removed tests for the deleted orphan helpers
_apply_verbosity_level and _apply_surface_language (see comments below for
the post-removal contract location).
"""
from __future__ import annotations

import pytest


def test_apply_profile_is_noop_on_default_state():
    """Default profile state → only the _knobs_applied telemetry
    block is added; no other surprising field additions to response.

    -03 (CONTEXT ): apply_profile now emits a
    response['_knobs_applied'] audit-trail block on every call. This is the
    one and only top-level field apply_profile is allowed to add. Pre-07.12-03
    the contract was "no additions"; post-07.12-03 the contract is "exactly
    one addition: _knobs_applied (a dict)".
    """
    from iai_mcp import profile
    from iai_mcp.response_decorator import apply_profile

    state = profile.default_state()
    # wake_depth default must exist post MCP-12.
    state.setdefault("wake_depth", "minimal")
    resp = {"hits": [{"record_id": "r1", "literal_surface": "x"}], "anti_hits": []}
    before_keys = set(resp.keys())
    out = apply_profile(dict(resp), state)
    added = set(out.keys()) - before_keys
    assert added == {"_knobs_applied"}, (
        f"apply_profile added unexpected keys on default state: {added}; "
        f"expected exactly {{'_knobs_applied'}} per "
    )
    assert isinstance(out["_knobs_applied"], dict), out["_knobs_applied"]


# removed test_verbosity_level_drops_fields — the
# _apply_verbosity_level orphan helper read a non-sealed-knob field
# (`verbosity_level` is NOT in PROFILE_KNOBS) and was deleted alongside the
# 4 dead-knob helpers. See tests/test_profile_no_dead_knobs.py for the
# orphan-absence assertions.


def test_formality_relaxation_applied_to_surface_text():
    """camouflaging_relaxation high → surface_text should be transformed.

    Concrete transform: when camouflaging_relaxation > 0.5, any all-lowercase
    surface_text remains untouched; but apply_profile must not raise. This is a
    contract test — the exact transform can evolve, but silent-fail is mandatory.
    """
    from iai_mcp import profile
    from iai_mcp.response_decorator import apply_profile

    state = profile.default_state()
    state["camouflaging_relaxation"] = 0.8
    resp = {
        "hits": [{"record_id": "r1", "literal_surface": "Good morning Sir."}],
        "anti_hits": [],
    }
    # Must not raise.
    apply_profile(resp, state)
    # Response structure intact.
    assert "hits" in resp and len(resp["hits"]) == 1


# removed test_surface_language_transform_noop_on_english — the
# _apply_surface_language orphan helper read a non-sealed-knob field
# (`surface_language` is NOT in PROFILE_KNOBS) and was deleted alongside the
# 4 dead-knob helpers. See tests/test_profile_no_dead_knobs.py for the
# orphan-absence assertions.


def test_monotropic_focus_narrows_hits():
    """monotropism_depth high → apply_profile must not crash; narrowing optional."""
    from iai_mcp import profile
    from iai_mcp.response_decorator import apply_profile

    state = profile.default_state()
    state["monotropism_depth"] = {"coding": 0.9}
    resp = {
        "hits": [
            {"record_id": "r1", "literal_surface": "x", "community_id": "A"},
            {"record_id": "r2", "literal_surface": "y", "community_id": "A"},
            {"record_id": "r3", "literal_surface": "z", "community_id": "B"},
        ],
        "anti_hits": [],
    }
    # Must not raise. Narrowing behaviour is policy choice — no hard assertion on
    # final count (the helper may choose to leave hits unchanged if domain tag
    # absent on hits).
    apply_profile(resp, state)
    assert "hits" in resp


def test_malformed_knob_silent_fail():
    """Malformed profile state → apply_profile does NOT raise."""
    from iai_mcp.response_decorator import apply_profile

    bad_state = {"verbosity_level": object(), "surface_language": 42}
    resp = {"hits": [], "anti_hits": []}
    # Must not raise.
    apply_profile(resp, bad_state)


def test_pre_existing_keys_untouched_on_exception():
    """If a helper raises, pre-existing response keys are preserved."""
    from iai_mcp import response_decorator

    # Monkey-patch one helper to raise, via attribute override.
    resp = {"hits": [], "anti_hits": [], "budget_used": 42}

    def _boom(*a, **k):
        raise RuntimeError("synthetic helper failure")

    # Override an internal helper if present — we only require apply_profile
    # to swallow any helper's exception.
    # : switched probe target from the deleted _apply_verbosity_level
    # orphan to _apply_dunn_quadrant (a still-live helper).
    original = None
    helper_name = "_apply_dunn_quadrant"
    if hasattr(response_decorator, helper_name):
        original = getattr(response_decorator, helper_name)
        setattr(response_decorator, helper_name, _boom)
    try:
        response_decorator.apply_profile(resp, {"dunn_quadrant": "seeking"})
    finally:
        if original is not None:
            setattr(response_decorator, helper_name, original)
    assert resp["budget_used"] == 42
