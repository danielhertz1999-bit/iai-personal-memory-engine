"""AUTIST-13 Task 3 — E2E register-relaxation smoke test.

Simulates an 8-week rising-formality trajectory and runs run_weekly_pass on
expanding 5-point windows. Verifies that:
- camouflaging_detected + register_relaxed events accumulate across passes.
- The 14th knob `camouflaging_relaxation` moves up from 0.0.
- A control (flat 0.5 trajectory) produces NO detections and leaves the knob untouched.

Real longitudinal validation (post-session-~30) is deferred to a later phase
(see 03-03- §Deferred Items). This test covers the synthetic E2E
per D-AUTIST13-04.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore


def _seed_weekly_scores(store, values: list[float]) -> None:
    """Write N `formality_score_weekly` events, one per simulated week."""
    base = datetime.now(timezone.utc) - timedelta(days=7 * len(values))
    for i, v in enumerate(values):
        write_event(
            store,
            kind="formality_score_weekly",
            data={
                "score": float(v),
                "lang": "en",
                "week_iso": (base + timedelta(days=7 * i)).isoformat(),
                "samples": 5,
            },
            severity="info",
        )


def test_e2e_rising_trajectory_accumulates_detections(tmp_path):
    """8-week rising trajectory crosses threshold and relaxes knob across passes.

    Expectation:
    - At least 2 camouflaging_detected events across the pass sequence.
    - At least 2 register_relaxed events.
    - Final knob value >= 0.2 after 4 passes (4 * DEFAULT_DELTA=0.1 = 0.4 in theory;
      allow a lower floor since trajectory dips in early passes shouldn't fire).
    """
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)

    # Simulate pass 1..4 by seeding incrementally longer rising trajectories.
    # detect_camouflaging reads the last `window_size=5` events, so each pass
    # sees the last 5 of the accumulated sequence.
    full_trajectory = [0.4, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95]
    # Pass 1: first 5 points (rising ~0.4 -> 0.82; mean ~0.634)
    # Pass 2: points 2-6 (rising; mean ~0.73)
    # Pass 3: points 3-7 (rising; mean ~0.80)
    # Pass 4: points 4-8 (rising; mean ~0.86)
    for pass_idx in range(4):
        # Fresh event stream per pass: window seeds 5 points.
        window = full_trajectory[pass_idx : pass_idx + 5]
        # Clear events table between passes so detect_camouflaging sees exactly
        # this window (not the accumulated previous passes).
        # We use a fresh store per pass to keep the simulation clean.
        pass_store = MemoryStore(path=tmp_path / f"pass-{pass_idx}")
        _seed_weekly_scores(pass_store, window)
        run_weekly_pass(pass_store)

    # Aggregate events across all pass stores.
    total_detected = 0
    total_relaxed = 0
    for pass_idx in range(4):
        pass_store = MemoryStore(path=tmp_path / f"pass-{pass_idx}")
        total_detected += len(query_events(pass_store, kind="camouflaging_detected", limit=10))
        total_relaxed += len(query_events(pass_store, kind="register_relaxed", limit=10))

    assert total_detected >= 2, f"expected >= 2 detections, got {total_detected}"
    assert total_relaxed >= 2, f"expected >= 2 relaxations, got {total_relaxed}"

    # Knob accumulated state across passes (all relax_register calls share _profile_state).
    assert core._profile_state["camouflaging_relaxation"] >= 0.2, (
        f"expected knob >= 0.2, got {core._profile_state['camouflaging_relaxation']}"
    )


def test_e2e_flat_control_no_detection_no_relax(tmp_path):
    """Flat 0.5 trajectory -> no detections, no events, knob unchanged."""
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.5] * 8)
    # Four passes on same store (events accumulate but flat -> never detected).
    for _ in range(4):
        run_weekly_pass(store)

    detected = query_events(store, kind="camouflaging_detected", limit=10)
    relaxed = query_events(store, kind="register_relaxed", limit=10)
    assert detected == []
    assert relaxed == []
    assert core._profile_state["camouflaging_relaxation"] == 0.0


def test_e2e_single_pass_bumps_knob_from_zero(tmp_path):
    """Single pass with detected trajectory -> knob > 0 + exactly one event pair."""
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.4, 0.55, 0.65, 0.75, 0.85])
    run_weekly_pass(store)

    assert core._profile_state["camouflaging_relaxation"] > 0.0

    # Should have exactly 1 of each.
    detected = query_events(store, kind="camouflaging_detected", limit=5)
    relaxed = query_events(store, kind="register_relaxed", limit=5)
    assert len(detected) == 1
    assert len(relaxed) == 1


def test_constitutional_guard_no_user_masking_code_paths():
    """Import the camouflaging + formality modules and confirm no user-state
    modeling symbols or names are exposed.

    This is a lightweight guard: forbidden identifiers must not appear in the
    modules' public API or in any emitted event kind.
    """
    import iai_mcp.camouflaging as cm
    import iai_mcp.formality as fm

    forbidden = {"user_masking_score", "is_masking", "infer_masking", "user_internal"}
    for mod in (cm, fm):
        names = set(dir(mod))
        assert not (names & forbidden), (
            f"forbidden identifier in {mod.__name__}: {names & forbidden}"
        )
