"""Trajectory metrics M1..M6.

Every session_exit writes one `trajectory_metric` event per metric. The CLI
aggregator reads these events via aggregate_trajectory.

Metrics (all computed in session-local scope):
- M1: clarifying questions per session (decreasing over time)
- M2: retrieval precision@5 (growing)
- M3: tokens per session (decreasing)
- M4: profile-vector variance (decreasing -> converged by session ~30)
- M5: curiosity question frequency (entropy dropping)
- M6: context-repeat rate (> 90% by session ~20)

Scope: event emission + basic aggregation, plus the CLI aggregator and
synthetic-corpus benchmark.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from iai_mcp.events import flush_event_buffer, query_events, write_event
from iai_mcp.store import MemoryStore


METRIC_NAMES: list[str] = ["m1", "m2", "m3", "m4", "m5", "m6"]


# ---------------------------------------------------------------- emit


def record_session_metrics(
    store: MemoryStore,
    session_id: str,
    metrics: dict[str, float],
) -> None:
    """Emit one `trajectory_metric` event per valid metric key in `metrics`.

    Keys outside METRIC_NAMES are ignored silently -- this is a public API;
    strict validation would force every test harness to chase whitespace in
    metric names.
    """
    for m, v in metrics.items():
        if m not in METRIC_NAMES:
            continue
        try:
            value = float(v)
        except (TypeError, ValueError):
            continue
        write_event(
            store,
            kind="trajectory_metric",
            data={"metric": m, "value": value},
            severity="info",
            session_id=session_id,
        )


def aggregate_trajectory(
    store: MemoryStore,
    since: datetime | None = None,
) -> dict[str, list[tuple[datetime, float]]]:
    """CLI support: group all trajectory_metric events by metric.

    Returns {"m1": [(ts, value),...],..., "m6": [...]}.
    """
    events = query_events(
        store, kind="trajectory_metric", since=since, limit=10000,
    )
    out: dict[str, list[tuple[datetime, float]]] = {m: [] for m in METRIC_NAMES}
    for e in events:
        m = e["data"].get("metric")
        v = e["data"].get("value")
        if m in METRIC_NAMES and v is not None:
            try:
                out[m].append((e["ts"], float(v)))
            except (TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------- individual signals


def compute_m1_clarifying_questions_per_session(
    store: MemoryStore,
    session_id: str,
) -> float:
    """M1: count of curiosity_question events for a session."""
    events = query_events(store, kind="curiosity_question", limit=1000)
    count = sum(1 for e in events if e.get("session_id") == session_id)
    return float(count)


def compute_m3_token_budget(
    store: MemoryStore,
    session_id: str,
) -> float:
    """M3: mean of session_start_tokens events for this session."""
    events = query_events(store, kind="session_start_tokens", limit=100)
    session_events = [e for e in events if e.get("session_id") == session_id]
    if not session_events:
        return 0.0
    total = 0.0
    for e in session_events:
        try:
            total += float(e["data"].get("tokens", 0))
        except (TypeError, ValueError):
            continue
    return total / len(session_events)


def compute_m5_curiosity_frequency(
    store: MemoryStore,
    session_id: str,
) -> float:
    """M5: sum of curiosity_silent_log + curiosity_question events per session."""
    silent = query_events(store, kind="curiosity_silent_log", limit=1000)
    questions = query_events(store, kind="curiosity_question", limit=1000)
    total = 0
    for ev_list in (silent, questions):
        total += sum(1 for e in ev_list if e.get("session_id") == session_id)
    return float(total)


def compute_session_metrics_snapshot(
    store: MemoryStore,
    session_id: str,
) -> dict[str, float]:
    """Produce a partial snapshot of M1..M6 from the current event stream.

    M1/M3/M5 are computable directly from the event stream. M2/M4/M6 are
    LIVE (read retrieval_used / profile_updated / session_started events
    emitted by retrieve.py / profile.py / session.py respectively).
    """
    return {
        "m1": compute_m1_clarifying_questions_per_session(store, session_id),
        "m2": m2_precision_at_5_live(store),
        "m3": compute_m3_token_budget(store, session_id),
        "m4": m4_profile_variance_live(store),
        "m5": compute_m5_curiosity_frequency(store, session_id),
        "m6": m6_context_repeat_rate_live(store),
    }


# -------------------------------------------------- M2/M4/M6 LIVE


# Backward-compat synthetic constants (baseline; bench compares
# live vs synthetic to prove the promotion is real -- see test_trajectory_live_smoke.py).
M2_SYNTHETIC_CONSTANT: float = 0.0
M4_SYNTHETIC_CONSTANT: float = 0.0
M6_SYNTHETIC_CONSTANT: float = 0.0


def m2_precision_at_5_synthetic() -> float:
    """Legacy synthetic placeholder. Kept for trajectory bench comparison."""
    return M2_SYNTHETIC_CONSTANT


def m4_profile_variance_synthetic() -> float:
    """Legacy synthetic placeholder. Kept for trajectory bench comparison."""
    return M4_SYNTHETIC_CONSTANT


def m6_context_repeat_rate_synthetic() -> float:
    """Legacy synthetic placeholder. Kept for trajectory bench comparison."""
    return M6_SYNTHETIC_CONSTANT


def m2_precision_at_5_live(
    store: MemoryStore,
    *,
    window: int = 100,
) -> float:
    """M2 LIVE: precision@5 over the last ``window`` retrieval_used events.

    Each ``retrieval_used`` event carries ``hit_ids`` (list of UUID strings) and
    optionally a ``ground_truth`` list. When ground_truth is present, count
    hits in the top-5 that intersect ground_truth and divide by 5. When absent,
    fall back to the **hit-presence rate** -- (# events with at least one hit)
    / (# events) -- which is a coarse but honest proxy and never returns the
    synthetic 0.0 when the system is actually retrieving.

    The fallback path is what makes the live value differ from the synthetic
    constant in production -- the metric stops being a flat zero the moment
    retrieve.recall starts returning hits.

    retrieval_used events may be deferred in the in-memory write buffer (to
    avoid lock contention on concurrent socket recalls). Flush before reading
    so the metric always reflects all retrievals that have happened, including
    any that are still buffered.
    """
    flush_event_buffer(store)
    events = query_events(store, kind="retrieval_used", limit=window)
    if not events:
        return 0.0

    precisions: list[float] = []
    for ev in events:
        data = ev.get("data") or {}
        hits = data.get("hit_ids") or []
        ground_truth = set(data.get("ground_truth") or [])
        top5 = list(hits)[:5]
        if ground_truth:
            tp = sum(1 for h in top5 if h in ground_truth)
            precisions.append(tp / 5.0)
        else:
            # Fallback: hit-presence at top-5 (1.0 if any hit, else 0.0).
            precisions.append(1.0 if top5 else 0.0)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def m4_profile_variance_live(
    store: MemoryStore,
    *,
    n_updates: int = 20,
) -> float:
    """M4 LIVE: variance over the last N profile_updated events per knob.

    Aggregates the most recent ``n_updates`` ``profile_updated`` events,
    groups by knob, computes per-knob variance over the new values (only for
    numeric knobs -- bool/enum knobs are skipped), and returns the mean
    variance across knobs.

    Returns 0.0 when no events exist (back-compat with the synthetic baseline).
    """
    events = query_events(store, kind="profile_updated", limit=n_updates * 5)
    if not events:
        return 0.0

    per_knob: dict[str, list[float]] = {}
    for ev in events[:n_updates]:
        data = ev.get("data") or {}
        knob = data.get("knob")
        new_val = data.get("new")
        if knob is None or new_val is None:
            continue
        # Skip bool/enum knobs explicitly: bool is a subclass of int, so
        # float(True/False) succeeds; we want only int/float values.
        if isinstance(new_val, bool) or not isinstance(new_val, (int, float)):
            continue
        per_knob.setdefault(str(knob), []).append(float(new_val))

    if not per_knob:
        return 0.0

    variances: list[float] = []
    for _knob, vals in per_knob.items():
        if len(vals) < 2:
            variances.append(0.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        variances.append(var)
    if not variances:
        return 0.0
    return sum(variances) / len(variances)


def m6_context_repeat_rate_live(
    store: MemoryStore,
    *,
    window_days: int = 30,
) -> float:
    """M6 LIVE: context-repeat-rate over the last ``window_days`` of session_started.

    Reads ``kind='session_started'`` events with ``data.session_state_hash``,
    counts unique vs total hashes, and returns the *repeat rate*:

        repeat_rate = (total - unique) / total

    A value near 0.0 means every session looked novel; near 1.0 means heavy
    context reuse (which is the continuity ideal at session ~20+).
    """
    from datetime import datetime, timedelta, timezone
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    events = query_events(
        store, kind="session_started", since=since, limit=10000,
    )
    if not events:
        return 0.0

    hashes: list[str] = []
    for ev in events:
        data = ev.get("data") or {}
        hsh = data.get("session_state_hash")
        if hsh:
            hashes.append(str(hsh))
    if not hashes:
        return 0.0
    total = len(hashes)
    unique = len(set(hashes))
    return (total - unique) / total


def m2(store: MemoryStore) -> float:
    """Public M2 entry point (always live)."""
    return m2_precision_at_5_live(store)


def m4(store: MemoryStore) -> float:
    """Public M4 entry point (always live)."""
    return m4_profile_variance_live(store)


def m6(store: MemoryStore) -> float:
    """Public M6 entry point (always live)."""
    return m6_context_repeat_rate_live(store)
