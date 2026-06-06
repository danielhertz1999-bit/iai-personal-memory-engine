"""Assert dead knobs and orphan helpers are removed.

AUTIST-02 / AUTIST-08 / AUTIST-11 / AUTIST-12 were declared in
profile.PROFILE_KNOBS but never read in production scoring or response code.
They are removed rather than inventing taxonomy that doesn't exist (sensory
channels / event-vs-time anchors / somatic-vs-labelled tags) or promoting a
passive design invariant (double_empathy) to a runtime knob.

Two orphan helpers (_apply_verbosity_level, _apply_surface_language) read
profile fields that are NOT in the KnobSpec registry. Both helpers are deleted
and removed from the dispatch tuple — they were legacy noise burning ~5 µs/call.

After:
- registry holds 11 knobs (10 AUTIST + 1 MCP-12 wake_depth)
- profile_set on each removed knob returns the unknown-knob error
- apply_profile dispatch tuple no longer references either orphan helper
"""

import inspect

from iai_mcp import profile, response_decorator
from iai_mcp.profile import PROFILE_KNOBS, default_state, profile_set


def test_registry_has_11_knobs() -> None:
    """Registry shrinks 15 → 11."""
    assert len(PROFILE_KNOBS) == 11, (
        f"Expected 11 knobs (10 AUTIST + wake_depth), "
        f"got {len(PROFILE_KNOBS)}: {sorted(PROFILE_KNOBS.keys())}"
    )
    autist_specs = [
        s for s in PROFILE_KNOBS.values() if s.requirement_id.startswith("AUTIST-")
    ]
    assert len(autist_specs) == 10
    assert "wake_depth" in PROFILE_KNOBS
    # Removed knobs absent from registry.
    assert "sensory_channel_weights" not in PROFILE_KNOBS
    assert "event_vs_time_cue" not in PROFILE_KNOBS
    assert "alexithymia_accommodation" not in PROFILE_KNOBS
    assert "double_empathy" not in PROFILE_KNOBS


def test_profile_set_rejects_sensory_channel_weights() -> None:
    """AUTIST-02 removed — profile_set must reject."""
    state = default_state()
    result = profile_set("sensory_channel_weights", {"vision": 0.5}, state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result


def test_profile_set_rejects_event_vs_time_cue() -> None:
    """AUTIST-08 removed — profile_set must reject.

    No event-vs-time anchor taxonomy exists in the schema; no
    `_apply_event_vs_time_cue` helper exists in response_decorator.py.
    Documented as a deferred future capability in the project convention.
    """
    state = default_state()
    result = profile_set("event_vs_time_cue", "time", state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result


def test_profile_set_rejects_alexithymia_accommodation() -> None:
    """AUTIST-11 removed — profile_set must reject."""
    state = default_state()
    result = profile_set("alexithymia_accommodation", "labeled", state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result


def test_profile_set_rejects_double_empathy() -> None:
    """AUTIST-12 removed — promoted to passive invariant.

    See the project convention "Architectural Invariants — Pinned" section for the
    project-level invariant that replaces the runtime-mutable knob.
    """
    state = default_state()
    result = profile_set("double_empathy", False, state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result


def test_orphan_helpers_absent_from_dispatch_tuple() -> None:
    """deletes _apply_verbosity_level and _apply_surface_language.

    These two helpers read non-sealed-knob fields (`verbosity_level`,
    `surface_language`) — they're legacy that burned CPU silently
    on every dispatch. After this plan they are gone.

    The check inspects the source of apply_profile to ensure the deleted
    function names are not referenced (no module-level definition AND not
    invoked in the dispatch tuple body).
    """
    # Definition-level check: the orphan helpers must NOT exist as attrs
    # of the response_decorator module.
    assert not hasattr(response_decorator, "_apply_verbosity_level"), (
        "_apply_verbosity_level should be deleted (orphan)"
    )
    assert not hasattr(response_decorator, "_apply_surface_language"), (
        "_apply_surface_language should be deleted (orphan)"
    )
    # Source-level check: the dispatch loop in apply_profile must not
    # reference either name.
    src = inspect.getsource(response_decorator.apply_profile)
    assert "_apply_verbosity_level" not in src, src
    assert "_apply_surface_language" not in src, src
