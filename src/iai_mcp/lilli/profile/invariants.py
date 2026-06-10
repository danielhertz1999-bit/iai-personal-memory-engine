from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from iai_mcp.events import query_events, write_event
from iai_mcp.formality import formality_score
from iai_mcp.profile import profile_get, profile_set  # noqa: F401  (profile_get retained for callers)


DOUBLE_EMPATHY_PASSIVE_INVARIANT: bool = True


DEFAULT_WINDOW_SIZE: int = 5
DEFAULT_CADENCE_DAYS: int = 7
TRIGGER_SLOPE: float = 0.05
TRIGGER_MEAN: float = 0.6
DEFAULT_DELTA: float = 0.1


def detect_camouflaging(
    store,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    cadence_days: int = DEFAULT_CADENCE_DAYS,
) -> dict:
    events = query_events(store, kind="formality_score_weekly", limit=window_size)
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


def relax_register(store, *, delta: float = DEFAULT_DELTA) -> None:
    import iai_mcp.core as core

    current = core._profile_state.get("camouflaging_relaxation", 0.0)
    new_value = min(1.0, max(0.0, current + delta))

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


def record_user_formality(store, text: str, lang: str) -> None:
    score = formality_score(text, lang)
    now = datetime.now(timezone.utc)
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


def run_weekly_pass(store) -> dict:
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
