"""Tests for iai_mcp.quiet_window -- Task 2.

Covers 8 behaviours:
1. Western 9-5 user -> quiet window in 22:00-06:00 range.
2. Nocturnal autistic user -> quiet window in 14:00-20:00 range.
3. Shift worker rotating weekly -> returns some valid tuple OR None, no crash.
4. New user (<7d data) -> returns None; caller bootstraps.
5. 24/7 user with no quiet span -> returns None.
6. DST transition -> does not crash; returns tuple or None.
7. should_relearn 24h cadence.
8. should_bootstrap_trigger 2h-idle.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest


def _fresh_store(tmp_path, monkeypatch):
    """Isolated MemoryStore under tmp_path via IAI_MCP_STORE env override."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")  # light schema, no real embeds
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_sessions(
    store,
    *,
    local_tz: ZoneInfo,
    day_start_local: datetime,
    hours: list[float],
    days: int = 7,
    sessions_per_hour: int = 3,
) -> None:
    """Emit synthetic `session_started` events at the given local-time hours
    across `days` consecutive days, `sessions_per_hour` per hour.
    """
    from iai_mcp.events import write_event
    for d in range(days):
        for h in hours:
            for s in range(sessions_per_hour):
                # local time -> UTC
                local_dt = day_start_local + timedelta(days=d, hours=h, minutes=5 * s)
                if local_dt.tzinfo is None:
                    local_dt = local_dt.replace(tzinfo=local_tz)
                utc_dt = local_dt.astimezone(timezone.utc)
                # Patch write_event's automatic ts by using raw table add:
                # write_event uses datetime.now(timezone.utc), so we cannot
                # control ts directly. Instead, directly insert into the
                # events table with the synthetic ts.
                _insert_event_direct(store, kind="session_started", ts=utc_dt, data={"n": s})


def _insert_event_direct(store, *, kind: str, ts: datetime, data: dict) -> None:
    """Bypass write_event so we can control `ts` deterministically."""
    import json
    from uuid import uuid4

    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import EVENTS_TABLE

    event_id = str(uuid4())
    data_plain = json.dumps(data)
    ad = event_id.encode("ascii")
    # store._key() lazy-loads the encryption key.
    data_ct = encrypt_field(data_plain, store._key(), associated_data=ad)
    row = {
        "id": event_id,
        "kind": kind,
        "severity": "",
        "domain": "",
        "ts": ts,
        "data_json": data_ct,
        "session_id": "-",
        "source_ids_json": json.dumps([]),
    }
    store.db.open_table(EVENTS_TABLE).add([row])


# ---------------------------------------------------------------------------
# Test 1: Western 9-5 -> quiet ~ 22:00-06:00
# ---------------------------------------------------------------------------

def test_western_9_to_5_user(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import (
        BUCKET_COUNT,
        BUCKET_MINUTES,
        learn_quiet_window,
    )

    tz = ZoneInfo("America/New_York")
    store = _fresh_store(tmp_path, monkeypatch)

    # 9-5 user: active 09:00-18:00 on 7 consecutive local-time days.
    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        days=7,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=7, hours=8)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is not None, "should detect quiet window for 9-5 user"
    start_bucket, duration = result

    # Start bucket should map to evening/night (17:30-02:00 local).
    # Activity spans 09:00-17:10 (last event at 17:10), so the quiet window
    # typically starts by 17:30. Accept any start in the 17:30-02:00 band.
    start_hour = (start_bucket * BUCKET_MINUTES) // 60
    start_minute = (start_bucket * BUCKET_MINUTES) % 60
    in_evening = start_hour >= 17 and (start_hour > 17 or start_minute >= 30)
    in_early_morning = start_hour <= 2
    assert in_evening or in_early_morning, (
        f"expected quiet start in 17:30-02:00 evening/night band, "
        f"got {start_hour}:{start_minute:02d} (bucket={start_bucket})"
    )
    # Duration in 3-8h range.
    assert 6 <= duration <= 16, f"duration out of range: {duration}"


# ---------------------------------------------------------------------------
# Test 2: Nocturnal autistic -> quiet ~ 14:00-20:00
# ---------------------------------------------------------------------------

def test_nocturnal_autistic_user(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import BUCKET_MINUTES, learn_quiet_window

    tz = ZoneInfo("Europe/Moscow")
    store = _fresh_store(tmp_path, monkeypatch)

    # Nocturnal: active 22:00 through 04:00 (next day), sleeping during
    # afternoon. Split around midnight: 22, 23 same day; 0, 1, 2, 3 next day.
    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[22, 23, 24, 25, 26, 27, 28],  # 22, 23, 0, 1, 2, 3, 4 local
        days=7,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=7, hours=12)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is not None, "should detect quiet window for nocturnal user"
    start_bucket, duration = result
    start_hour = (start_bucket * BUCKET_MINUTES) // 60
    # Expect quiet roughly in the daytime band (04:30-21:00): last activity ends
    # around 04:10 local, so the first empty bucket is 04:30.
    # The key invariant: NOT overlapping with the 22:00-04:00 active window.
    assert 4 <= start_hour <= 21, (
        f"nocturnal: expected quiet start in 04-21 band, got {start_hour}:00"
    )
    # And the window must not cover the 22:00-04:00 active region: check end
    # bucket wraps back before 22:00 local.
    end_bucket = (start_bucket + duration) % 48
    end_hour = (end_bucket * BUCKET_MINUTES) // 60
    assert end_hour <= 22, (
        f"nocturnal window ends at {end_hour}:00, overlaps active 22-04 band"
    )
    assert 6 <= duration <= 16


# ---------------------------------------------------------------------------
# Test 3: Shift worker (alternating day/night every 2 days)
# ---------------------------------------------------------------------------

def test_shift_worker_alternating(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("UTC")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    # Days 0, 1: day shift (active 08-16).
    # Days 2, 3: night shift (active 20-04).
    # Days 4, 5: day shift. Day 6: night shift.
    for d in range(7):
        if d in (0, 1, 4, 5):
            hours = [8, 9, 10, 11, 12, 13, 14, 15]
        else:
            hours = [20, 21, 22, 23, 24, 25, 26, 27]
        _seed_sessions(
            store,
            local_tz=tz,
            day_start_local=day_start + timedelta(days=d),
            hours=hours,
            days=1,
            sessions_per_hour=2,
        )

    now = (day_start + timedelta(days=7)).astimezone(timezone.utc)
    # Must not crash; result is either a valid tuple or None.
    result = learn_quiet_window(store, now, tz)
    if result is not None:
        start_bucket, duration = result
        assert 0 <= start_bucket < 48
        assert 6 <= duration <= 16


# ---------------------------------------------------------------------------
# Test 4: New user (<7d) -> None (bootstrap)
# ---------------------------------------------------------------------------

def test_new_user_insufficient_days(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("UTC")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[9, 10, 11, 12, 13],
        days=2,  # < MIN_DAYS_FOR_LEARN
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=2, hours=14)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is None, "should return None when <7d data"


# ---------------------------------------------------------------------------
# Test 5: 24/7 user with no contiguous quiet window
# ---------------------------------------------------------------------------

def test_24_7_user_no_quiet_span(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("UTC")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    # Active every hour of every day (no dip below threshold).
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=list(range(24)),
        days=7,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=7)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    # Completely uniform -> peak==every_bucket -> threshold=0.2*peak.
    # All buckets equal -> none < threshold -> best_len=0 < min_buckets=6 -> None.
    assert result is None, "24/7 uniform user should return None"


# ---------------------------------------------------------------------------
# Test 6: DST spring-forward doesn't crash
# ---------------------------------------------------------------------------

def test_dst_spring_forward_no_crash(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("America/New_York")
    store = _fresh_store(tmp_path, monkeypatch)

    # Seed 7 days that span DST start (US: 2026-03-08 at 02:00 jumps to 03:00).
    day_start = datetime(2026, 3, 5, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[9, 10, 12, 14, 17, 20],
        days=7,
        sessions_per_hour=2,
    )

    now = (day_start + timedelta(days=7)).astimezone(timezone.utc)
    # Must not crash.
    result = learn_quiet_window(store, now, tz)
    if result is not None:
        start_bucket, duration = result
        assert 0 <= start_bucket < 48
        assert 6 <= duration <= 16


# ---------------------------------------------------------------------------
# Test 7: should_relearn 24h cadence
# ---------------------------------------------------------------------------

def test_should_relearn_24h_cadence():
    from iai_mcp.quiet_window import should_relearn

    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    # Never learned -> True.
    assert should_relearn(None, now) is True
    # 25h ago -> True.
    assert should_relearn(now - timedelta(hours=25), now) is True
    # Exactly 24h -> True (>= threshold).
    assert should_relearn(now - timedelta(hours=24), now) is True
    # 12h ago -> False.
    assert should_relearn(now - timedelta(hours=12), now) is False


# ---------------------------------------------------------------------------
# Test 8: should_bootstrap_trigger 2h-idle
# ---------------------------------------------------------------------------

def test_should_bootstrap_trigger_2h_idle():
    from iai_mcp.quiet_window import should_bootstrap_trigger

    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    # No last session -> True (first-run idle).
    assert should_bootstrap_trigger(None, now) is True
    # 3h idle -> True.
    assert should_bootstrap_trigger(now - timedelta(hours=3), now) is True
    # 2h idle (== threshold) -> True.
    assert should_bootstrap_trigger(now - timedelta(hours=2), now) is True
    # 1h idle -> False.
    assert should_bootstrap_trigger(now - timedelta(hours=1), now) is False
