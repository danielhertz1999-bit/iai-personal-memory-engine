from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore

def _seed_weekly_scores(store, values: list[float]) -> None:
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
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)

    full_trajectory = [0.4, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95]
    for pass_idx in range(4):
        window = full_trajectory[pass_idx : pass_idx + 5]
        pass_store = MemoryStore(path=tmp_path / f"pass-{pass_idx}")
        _seed_weekly_scores(pass_store, window)
        run_weekly_pass(pass_store)

    total_detected = 0
    total_relaxed = 0
    for pass_idx in range(4):
        pass_store = MemoryStore(path=tmp_path / f"pass-{pass_idx}")
        total_detected += len(query_events(pass_store, kind="camouflaging_detected", limit=10))
        total_relaxed += len(query_events(pass_store, kind="register_relaxed", limit=10))

    assert total_detected >= 2, f"expected >= 2 detections, got {total_detected}"
    assert total_relaxed >= 2, f"expected >= 2 relaxations, got {total_relaxed}"

    assert core._profile_state["camouflaging_relaxation"] >= 0.2, (
        f"expected knob >= 0.2, got {core._profile_state['camouflaging_relaxation']}"
    )

def test_e2e_flat_control_no_detection_no_relax(tmp_path):
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.5] * 8)
    for _ in range(4):
        run_weekly_pass(store)

    detected = query_events(store, kind="camouflaging_detected", limit=10)
    relaxed = query_events(store, kind="register_relaxed", limit=10)
    assert detected == []
    assert relaxed == []
    assert core._profile_state["camouflaging_relaxation"] == 0.0

def test_e2e_single_pass_bumps_knob_from_zero(tmp_path):
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.4, 0.55, 0.65, 0.75, 0.85])
    run_weekly_pass(store)

    assert core._profile_state["camouflaging_relaxation"] > 0.0

    detected = query_events(store, kind="camouflaging_detected", limit=5)
    relaxed = query_events(store, kind="register_relaxed", limit=5)
    assert len(detected) == 1
    assert len(relaxed) == 1

def test_no_user_masking_code_paths():
    import iai_mcp.camouflaging as cm
    import iai_mcp.formality as fm

    forbidden = {"user_masking_score", "is_masking", "infer_masking", "user_internal"}
    for mod in (cm, fm):
        names = set(dir(mod))
        assert not (names & forbidden), (
            f"forbidden identifier in {mod.__name__}: {names & forbidden}"
        )
