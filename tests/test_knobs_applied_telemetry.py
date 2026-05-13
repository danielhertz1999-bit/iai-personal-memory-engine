"""-03: assert _knobs_applied audit-trail block on recall.

Closes (RE-ASSERTED per CONTEXT D-08).

CONTEXT contract:
  (a) Calling the production recall path (core.dispatch — NOT apply_profile
      standalone) with default profile produces a response with
      _knobs_applied listing 11 entries (8 helper + 2 upstream-gains +
      1 wake_depth seed).
  (b) Setting dunn_quadrant to a non-default value produces a
      _knobs_applied entry whose provenance contains 'profile.py' —
      proves upstream-gains accumulator is wired all the way to response.
  (c) The accumulator value is deterministic.

BLOCKER 3 fix (CONTEXT D-04, 2026-04-30): the production-path test exercises
core.dispatch (or end-to-end MCP), NOT apply_profile standalone — to prove
the upstream-gains accumulator is wired through pipeline.recall_for_response
to the response. A passing apply_profile-only test would be a false GREEN
(V2-07 anti-pattern recurring inside the phase chartered to eliminate it).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.profile import default_state, profile_modulation_for_record
from iai_mcp.response_decorator import HELPER_TO_KNOB_ID, apply_profile
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------------------
# Synthetic helpers (apply_profile unit tests)
# --------------------------------------------------------------------------


def _hit(literal: str = "h", suggestions: list[str] | None = None) -> dict:
    """Build a synthetic hit dict matching _hit_to_json shape (core.py:712-719)."""
    return {
        "record_id": "00000000-0000-0000-0000-000000000001",
        "score": 0.5,
        "reason": "test",
        "literal_surface": literal,
        "adjacent_suggestions": suggestions or [],
    }


def _resp(hits: list[dict], **extra) -> dict:
    base: dict = {"hits": hits}
    base.update(extra)
    return base


# ---- Unit: apply_profile dispatch-loop telemetry ---------------------------


def test_knobs_applied_present_after_apply_profile() -> None:
    """CONTEXT every recall response carries _knobs_applied."""
    response = _resp([_hit()])
    profile = default_state()
    apply_profile(response, profile)
    assert "_knobs_applied" in response, response
    assert isinstance(response["_knobs_applied"], dict), response["_knobs_applied"]


def test_knobs_applied_provenance_shape() -> None:
    """Each value is '<file>:<symbol>' or '<file>:<symbol>:<extra>' (no-op marker).

    All file components end in '.py'; all entries have at least file:symbol.
    """
    response = _resp([_hit()])
    apply_profile(response, default_state())
    assert response["_knobs_applied"], "expected at least one helper entry"
    for knob_id, provenance in response["_knobs_applied"].items():
        assert isinstance(provenance, str), (knob_id, provenance)
        assert provenance, (knob_id, provenance)
        parts = provenance.split(":")
        assert len(parts) >= 2, (knob_id, provenance)
        assert parts[0].endswith(".py"), (knob_id, provenance)


def test_knobs_applied_deterministic() -> None:
    """CONTEXT test (c): same call → same _knobs_applied dict."""
    response_1 = _resp([_hit()])
    response_2 = _resp([_hit()])
    profile = default_state()
    apply_profile(response_1, profile)
    apply_profile(response_2, profile)
    assert response_1["_knobs_applied"] == response_2["_knobs_applied"]


def test_knobs_applied_preserves_upstream_seeded_entries() -> None:
    """apply_profile MUST extend, never overwrite — preserves entries
    seeded by core.dispatch (BLOCKER 3 binding). The dispatch loop only
    adds entries; pre-existing entries (AUTIST-03, AUTIST-09, MCP-12) stay.
    """
    response = _resp(
        [_hit()],
        _knobs_applied={
            "AUTIST-03": "profile.py:profile_modulation_for_record:dunn_quadrant=seeking",
            "AUTIST-09": "profile.py:profile_modulation_for_record:interest_boost",
            "MCP-12": "session.py:assemble_session_start:wake_depth=minimal",
        },
    )
    profile = default_state()
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-03" in ka
    assert "profile.py" in ka["AUTIST-03"]
    assert "AUTIST-09" in ka
    assert "profile.py" in ka["AUTIST-09"]
    assert "MCP-12" in ka
    assert "session.py" in ka["MCP-12"]


def test_knobs_applied_no_op_markers_for_pda_neutral() -> None:
    """PDA-tolerance with mode=neutral records a no-op marker."""
    response = _resp([_hit()])
    profile = default_state()
    profile["demand_avoidance_tolerance"] = "neutral"
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-05" in ka
    assert "no-op" in ka["AUTIST-05"], ka["AUTIST-05"]
    assert "neutral" in ka["AUTIST-05"], ka["AUTIST-05"]


def test_knobs_applied_no_op_markers_for_inertia_off() -> None:
    """inertia_awareness with knob=False records a no-op marker."""
    response = _resp([_hit()])
    profile = default_state()
    # default inertia_awareness is False per profile.py KnobSpec.
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-10" in ka
    assert "no-op" in ka["AUTIST-10"], ka["AUTIST-10"]


def test_knobs_applied_no_op_marker_for_scene_construction_off() -> None:
    """scene_construction_scaffold=False records a no-op marker."""
    response = _resp([_hit()])
    profile = default_state()
    profile["scene_construction_scaffold"] = False
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-14" in ka
    assert "no-op" in ka["AUTIST-14"], ka["AUTIST-14"]


# ---- HELPER_TO_KNOB_ID exhaustiveness + no-fabrication ---------------------


def test_helper_to_knob_id_has_11_verified_entries() -> None:
    """contract: HELPER_TO_KNOB_ID has exactly 11 verified
    entries — 8 helper-keyed (the wired AUTIST helpers) + 2 upstream-gains
    (dunn_quadrant, interest_boost) + 1 session-start (wake_depth).

    NO entries for removed knobs (AUTIST-02 sensory_channel_weights,
    event_vs_time_cue, alexithymia_accommodation,
    double_empathy) — those were deleted in Wave 1 .
    Re-introducing them here = silent regression.
    """
    assert len(HELPER_TO_KNOB_ID) == 11, (
        f"HELPER_TO_KNOB_ID must have exactly 11 verified entries "
        f"(8 helper + 2 upstream-gains + 1 wake_depth seed), "
        f"got {len(HELPER_TO_KNOB_ID)}: {HELPER_TO_KNOB_ID}"
    )
    knob_ids = set(HELPER_TO_KNOB_ID.values())
    # 10 AUTIST + 1 = 11 unique knob IDs.
    assert len(knob_ids) == 11, knob_ids
    # No removed knobs.
    for removed in ("AUTIST-02", "AUTIST-08", "AUTIST-11", "AUTIST-12"):
        assert removed not in knob_ids, (
            f"{removed} was removed in ; do not re-add"
        )
    # Required knob IDs are present.
    expected_autist = {f"AUTIST-{i:02d}" for i in (1, 3, 4, 5, 6, 7, 9, 10, 13, 14)}
    assert expected_autist.issubset(knob_ids), (expected_autist - knob_ids)
    assert "MCP-12" in knob_ids


# ---- Profile gains accumulator (Action 4a contract) -----------------------


def test_profile_modulation_records_into_accumulator() -> None:
    """profile_modulation_for_record(record, state, knobs_applied=acc) writes
    / / provenance strings into acc when the
    corresponding gain branch fires. Provenance MUST contain 'profile.py'
    (proves upstream-gains accumulator is wired in profile.py, not stubbed
    elsewhere — BLOCKER 3 fix).
    """
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["domain:coding"],
    )
    state = default_state()
    state["monotropism_depth"] = {"coding": 0.5}
    state["interest_boost"] = 0.3
    state["dunn_quadrant"] = "seeking"

    accumulator: dict[str, str] = {}
    gains = profile_modulation_for_record(rec, state, knobs_applied=accumulator)
    assert "monotropism_depth" in gains  # behaviour unchanged
    assert "AUTIST-01" in accumulator, accumulator
    assert "AUTIST-09" in accumulator, accumulator
    assert "AUTIST-03" in accumulator, accumulator
    # BLOCKER 3 binding: provenance MUST anchor in profile.py.
    assert "profile.py" in accumulator["AUTIST-01"], accumulator["AUTIST-01"]
    assert "profile.py" in accumulator["AUTIST-03"], accumulator["AUTIST-03"]
    assert "profile.py" in accumulator["AUTIST-09"], accumulator["AUTIST-09"]


def test_profile_modulation_back_compat_without_kwarg() -> None:
    """profile_modulation_for_record without knobs_applied still returns gains —
    back-compat preserved for callers that don't pass the kwarg.
    """
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["domain:coding"],
    )
    state = default_state()
    state["interest_boost"] = 0.3
    # No kwarg — must not raise, must return gains as before.
    gains = profile_modulation_for_record(rec, state)
    assert "interest_boost" in gains


# ---- Integration: production core.dispatch path (BLOCKER 3 binary gate) ---


def _seed_one_record(store, text: str = "reference content") -> None:
    """Canonical seed pattern from tests/test_first_turn_recall.py:18-41."""
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["domain:coding"],
    )
    store.insert(rec)


def _call_production_dispatch_path(tmp_path, monkeypatch) -> dict:
    """Exercise the PRODUCTION recall path end-to-end via core.dispatch.

    Per BLOCKER 3: this MUST hit core.dispatch with a non-empty store so the
    recall_for_response branch (line 227) runs and the upstream-gains
    accumulator fires. An empty store would route to retrieve.recall (line
    194) which does NOT enter profile_modulation_for_record.

    The fixture sets profile_state values that exercise the upstream gains
    so / / entries are recorded with
    profile.py provenance:
      - dunn_quadrant="seeking" → fires
      - interest_boost=0.5 → fires
      - monotropism_depth has no matching tag → not from profile,
        but the apply_profile dispatch loop still records it from the
        helper.
    """
    from iai_mcp import core
    from iai_mcp.store import MemoryStore

    # Save module-level state so we don't leak into other tests.
    saved_profile = dict(core._profile_state)
    pending = {"sknobs": True}

    def _load_state():
        return {"first_turn_pending": dict(pending)}

    def _save_state(state):
        fresh = state.get("first_turn_pending", {})
        pending.clear()
        pending.update(fresh)

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _load_state)
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", _save_state)

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store, "reference content for knobs telemetry test")

    try:
        core._profile_state["dunn_quadrant"] = "seeking"
        core._profile_state["interest_boost"] = 0.5
        core._profile_state["monotropism_depth"] = {"coding": 0.5}

        params = {
            "cue": "reference content for knobs telemetry test",
            "session_id": "sknobs",
            "cue_embedding": [0.1] * EMBED_DIM,
        }
        response = core.dispatch(store, "memory_recall", params)
    finally:
        core._profile_state.clear()
        core._profile_state.update(saved_profile)
    return response


def test_knobs_applied_via_production_dispatch_path(tmp_path, monkeypatch) -> None:
    """BLOCKER 3 acceptance criterion: production recall path (core.dispatch)
    populates _knobs_applied with 11 entries, including / AUTIST-09
    with provenance pointing into profile.py and with provenance
    pointing into session.py.

    A passing apply_profile-only test would be a false GREEN — the
    upstream-gains accumulator could be stubbed and we would never know.
    This test exercises the production wiring end-to-end.
    """
    response = _call_production_dispatch_path(tmp_path, monkeypatch)

    assert "_knobs_applied" in response, sorted(response.keys())
    ka = response["_knobs_applied"]
    assert isinstance(ka, dict), ka

    # 11 entries: 8 helper-keyed + 2 upstream-gains + 1 wake_depth seed.
    # Default-state recall fires every helper (helpers always record their
    # entry; no-op markers preserve presence). Fixture sets seeking/0.5 so
    # the upstream-gains entries fire too.
    assert len(ka) == 11, ka

    # BLOCKER 3 binary acceptance — the upstream-gains entries MUST be
    # present and anchored in profile.py.
    for required in ("AUTIST-03", "AUTIST-09", "MCP-12"):
        assert required in ka, (required, sorted(ka.keys()))
    assert "profile.py" in ka["AUTIST-03"], ka["AUTIST-03"]
    assert "profile.py" in ka["AUTIST-09"], ka["AUTIST-09"]
    assert "session.py" in ka["MCP-12"], ka["MCP-12"]

    # Removed-knob keys MUST NOT appear (deleted them).
    for removed in ("AUTIST-02", "AUTIST-08", "AUTIST-11", "AUTIST-12"):
        assert removed not in ka, (removed, ka)

    # The 10 AUTIST knob IDs that should be present.
    for autist in (
        "AUTIST-01", "AUTIST-03", "AUTIST-04", "AUTIST-05",
        "AUTIST-06", "AUTIST-07", "AUTIST-09", "AUTIST-10",
        "AUTIST-13", "AUTIST-14",
    ):
        assert autist in ka, (autist, sorted(ka.keys()))
