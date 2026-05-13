"""-- activity-learned quiet-window scheduler .

Learn the user's quiet window from their own `session_started` event history.
48 buckets of 30-min granularity over a 7-day rolling window. Find the longest
contiguous span where bucket activity < threshold. Min 3h, max 8h. Bootstrap
when <7 days of data: trigger on 2h MCP idle. Re-learn every 24h.

Constitutional guard:
- learned from events, NOT clock-based.
- global-product mandate -- no Western 9-5 assumption, no baked-in
  local-time default. Respects nocturnal / shift / time-zone-mobile users.
- C3: no LLM code, no paid-API env var reference in this module.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore

# Bucket sizing.
BUCKET_COUNT = 48          # 30-min * 48 = 24h
BUCKET_MINUTES = 30

# Window bounds.
MIN_WINDOW_HOURS = 3       # discard spans shorter than 3h
MAX_WINDOW_HOURS = 8       # human sleep ceiling

# Learning / bootstrap parameters.
MIN_DAYS_FOR_LEARN = 7
BOOTSTRAP_IDLE_HOURS = 2   # fallback trigger when <7d data

# Scheduler cadence gates (used by daemon; exported for caller convenience).
WIND_DOWN_GATE_MINUTES_BEFORE = 30   # dual-gate: within 30min of quiet start
DIGEST_SHOW_THRESHOLD_HOURS = 18     # morning digest gating (re-exported by daemon_state)


def learn_quiet_window(
    store: MemoryStore,
    now: datetime,
    tz: ZoneInfo,
) -> Optional[tuple[int, int]]:
    """Learn the user's quiet window from 7-day session_started history.

    Returns (start_bucket, duration_buckets) in LOCAL time, or None if
    insufficient data / no contiguous quiet span (caller falls back to the
    bootstrap idle rule).

    start_bucket: 0..BUCKET_COUNT-1 index into 30-min-bucket local-time day.
    duration_buckets: number of 30-min buckets in the quiet span (3h=6, 8h=16).
    """
    since = now - timedelta(days=MIN_DAYS_FOR_LEARN)
    events = query_events(store, kind="session_started", since=since, limit=10000)
    if not events:
        return None

    # Count sessions per 30-min local-time bucket + track unique days seen.
    counts = [0] * BUCKET_COUNT
    days_seen: set[tuple[int, int, int]] = set()
    for e in events:
        ts = e["ts"]
        # Pandas may surface a Timestamp -- coerce to aware datetime.
        if not isinstance(ts, datetime):
            try:
                ts = ts.to_pydatetime()
            except Exception:
                continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        try:
            ts_local = ts.astimezone(tz)
        except Exception:
            # DST edge: astimezone is robust on stdlib, but guard anyway.
            continue
        bucket = (ts_local.hour * 60 + ts_local.minute) // BUCKET_MINUTES
        if 0 <= bucket < BUCKET_COUNT:
            counts[bucket] += 1
        days_seen.add((ts_local.year, ts_local.month, ts_local.day))

    if len(days_seen) < MIN_DAYS_FOR_LEARN:
        return None  # bootstrap path -- caller uses 2h-idle.

    # Low-activity threshold = 20% of peak.
    peak = max(counts)
    if peak == 0:
        return None
    threshold = max(1, int(peak * 0.2))

    # Longest contiguous circular span of sub-threshold buckets.
    # Double-array walk to handle wrap-around across local midnight.
    doubled = counts + counts
    best_start, best_len = 0, 0
    cur_start, cur_len = None, 0
    for i, c in enumerate(doubled):
        if c < threshold:
            if cur_start is None:
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_start, cur_len = None, 0

    min_buckets = MIN_WINDOW_HOURS * (60 // BUCKET_MINUTES)   # 6
    max_buckets = MAX_WINDOW_HOURS * (60 // BUCKET_MINUTES)   # 16
    if best_len < min_buckets:
        # 24/7 user with no contiguous quiet span -> fallback to idle-only.
        return None
    duration = min(best_len, max_buckets)
    # Don't allow a span longer than a full day after wrap.
    if duration > BUCKET_COUNT:
        duration = BUCKET_COUNT
    return (best_start % BUCKET_COUNT, duration)


def should_relearn(last_learned_at: Optional[datetime], now: datetime) -> bool:
    """Re-learn cadence: 24h since last learn ( 24h adaptation)."""
    if last_learned_at is None:
        return True
    if last_learned_at.tzinfo is None:
        last_learned_at = last_learned_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - last_learned_at) >= timedelta(hours=24)


def should_bootstrap_trigger(last_session_ts: Optional[datetime], now: datetime) -> bool:
    """Bootstrap idle trigger: daemon fires when no MCP session for 2h.

    Used when `learn_quiet_window` returns None (insufficient data or 24/7
    user). Also used by the daemon as the always-on idle rule in addition to
    the learned quiet window.
    """
    if last_session_ts is None:
        return True
    if last_session_ts.tzinfo is None:
        last_session_ts = last_session_ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - last_session_ts) >= timedelta(hours=BOOTSTRAP_IDLE_HOURS)
