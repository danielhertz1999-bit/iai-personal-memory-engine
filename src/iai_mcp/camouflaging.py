"""Plan 03-03 — camouflaging detector + register relaxer (ecological self-regulation).

Constitutional anchor:
- Observes the user's SURFACE formality over a weekly sliding 5-point window.
- On a sustained over-formal trajectory, adjusts OUR register (the 14th profile
  knob `camouflaging_relaxation`). NEVER pushes the user to change. NEVER models
  user internal-state (Cook 2021 / Raymaker 2020 — masking is out-of-scope).
- Chapman 2021 ecological self-regulation framing: the system relaxes ITS OWN
  response register so the user does not have to match ours.

Detection (D-AUTIST13-03): sliding 5-point weekly window. Trigger condition:
linear-regression slope > 0.05/week AND current mean > 0.6. Both must hold.

Event kinds emitted (new in Phase 3):
- `formality_score_weekly` — weekly aggregate of the user's formality scores.
- `camouflaging_detected` — the detector fired (over-formal trajectory confirmed).
- `register_relaxed` — OUR `camouflaging_relaxation` knob was bumped UP (toward
  informal register in OUR responses).

Knob semantics: `camouflaging_relaxation` in [0, 1]. Higher = more relaxed OUR register.
relax_register INCREMENTS the knob (pushing OUR output toward informal) when the user
is observed to be over-formal. The user is never modified or nudged.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from iai_mcp.events import query_events, write_event
from iai_mcp.formality import formality_score
from iai_mcp.profile import profile_get, profile_set


# ------------------------------------------------------------------- constants
DEFAULT_WINDOW_SIZE: int = 5        # D-AUTIST13-03 sliding 5-point window
DEFAULT_CADENCE_DAYS: int = 7       # weekly
TRIGGER_SLOPE: float = 0.05         # formality delta per week floor
TRIGGER_MEAN: float = 0.6           # absolute formality floor
DEFAULT_DELTA: float = 0.1          # knob step per relaxation


# ------------------------------------------------------------------- detector
def detect_camouflaging(
    store,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    cadence_days: int = DEFAULT_CADENCE_DAYS,
) -> dict:
    """Sliding 5-point weekly window detector (D-AUTIST13-03).

    Reads the last `window_size` `formality_score_weekly` events, computes the
    linear-regression slope (numpy.polyfit deg=1), and the current mean. Detected
    iff slope > TRIGGER_SLOPE AND mean > TRIGGER_MEAN (both required).

    Args:
        store: open MemoryStore.
        window_size: number of weekly points to consider (default 5).
        cadence_days: cadence label (default 7 = weekly); not used arithmetically
            but stored in event metadata by callers.

    Returns:
        {detected: bool, trajectory_slope: float, current_mean: float, sample_count: int}.
    """
    events = query_events(store, kind="formality_score_weekly", limit=window_size)
    # Events are newest-first; we want chronological order for slope.
    events = list(reversed(events))
    sample_count = len(events)

    if sample_count < window_size:
        return {
            "detected": False,
            "trajectory_slope": 0.0,
            "current_mean": 0.0,
            "sample_count": sample_count,
        }

    scores = np.asarray(
        [float(e["data"].get("score", 0.0)) for e in events], dtype=np.float64
    )
    xs = np.arange(len(scores), dtype=np.float64)
    slope, _intercept = np.polyfit(xs, scores, 1)
    current_mean = float(scores.mean())

    detected = bool(slope > TRIGGER_SLOPE and current_mean > TRIGGER_MEAN)

    return {
        "detected": detected,
        "trajectory_slope": float(slope),
        "current_mean": current_mean,
        "sample_count": sample_count,
    }


# ------------------------------------------------------------------- relaxer
def relax_register(store, *, delta: float = DEFAULT_DELTA) -> None:
    """Bump profile.camouflaging_relaxation by delta (capped at 1.0).

    Writes go through `profile.profile_set(..., store=store)` so the existing
    `profile_updated` event also fires alongside `register_relaxed`. This is the
    ONE pathway the system uses to relax its own register in response to a
    detected over-formal user trajectory (D-AUTIST13-02).
    """
    import iai_mcp.core as core

    current = core._profile_state.get("camouflaging_relaxation", 0.0)
    new_value = min(1.0, max(0.0, current + delta))

    # Only call profile_set if the value actually changes; otherwise profile_set
    # will silently no-op and NOT emit profile_updated (correct behaviour).
    if new_value != current:
        profile_set(
            "camouflaging_relaxation",
            new_value,
            core._profile_state,
            store=store,
        )

    write_event(
        store,
        kind="register_relaxed",
        data={
            "from": float(current),
            "to": float(new_value),
            "delta": float(delta),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info",
    )


# ------------------------------------------------------------------- recorder
def record_user_formality(store, text: str, lang: str) -> None:
    """Compute formality on USER surface text and emit a formality_score_weekly event.

    Called on every user turn. Constitutional guard: the scorer sees ONLY the
    user's surface output; no inferred state is computed or persisted.
    """
    score = formality_score(text, lang)
    now = datetime.now(timezone.utc)
    # Simple per-turn emit; aggregation is done at query time in detect_camouflaging
    # (taking last window_size). Per-week aggregation via week_iso tag for audit.
    week_iso = f"{now.year}-W{now.isocalendar()[1]:02d}"
    write_event(
        store,
        kind="formality_score_weekly",
        data={
            "score": float(score),
            "lang": lang,
            "week_iso": week_iso,
            "samples": 1,
            "timestamp": now.isoformat(),
        },
        severity="info",
    )


# ------------------------------------------------------------------- weekly pass
def run_weekly_pass(store) -> dict:
    """Convenience entry: detect_camouflaging; if detected, emit
    `camouflaging_detected` event AND call relax_register.

    Returns the detection result dict (same shape as detect_camouflaging).
    """
    result = detect_camouflaging(store)
    if result["detected"]:
        write_event(
            store,
            kind="camouflaging_detected",
            data={
                "slope": result["trajectory_slope"],
                "mean": result["current_mean"],
                "window_size": DEFAULT_WINDOW_SIZE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
        )
        relax_register(store)
    return result
