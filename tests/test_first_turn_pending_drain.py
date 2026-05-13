"""-02 R3 unit tests for prune_first_turn_pending pure helper.

Distinct from tests/test_daemon_state.py::test_prune_* which covers the
24h-default `prune_stale_first_turn`. This file covers the new 1h-default
`prune_first_turn_pending` (tuple return + dropped session_ids list).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.daemon_state import (
    FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
    prune_first_turn_pending,
)

NOW = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def test_default_ttl_is_3600_seconds() -> None:
    """D7.2-08: default TTL is 3600s (1h)."""
    assert FIRST_TURN_PENDING_TTL_SEC_DEFAULT == 3600.0


def test_keeps_fresh_evicts_stale_returns_dropped_ids() -> None:
    """Mixed input: some entries < ttl_sec, some > ttl_sec."""
    fresh_ts = (NOW - timedelta(seconds=1800)).isoformat()  # 30min — keep
    stale_ts = (NOW - timedelta(seconds=7200)).isoformat()  # 2h — evict
    state = {
        "first_turn_pending": {
            "sess-fresh": fresh_ts,
            "sess-stale": stale_ts,
        },
    }

    new_state, dropped = prune_first_turn_pending(state, now=NOW, ttl_sec=3600.0)

    assert new_state["first_turn_pending"] == {"sess-fresh": fresh_ts}
    assert dropped == ["sess-stale"]


def test_legacy_bool_entries_evict_with_no_timestamp() -> None:
    """D7.2-07 contract: non-string values treated as stale."""
    state = {
        "first_turn_pending": {"sess-1": True, "sess-2": False, "sess-3": None},
    }

    new_state, dropped = prune_first_turn_pending(state, now=NOW)

    assert new_state["first_turn_pending"] == {}
    assert sorted(dropped) == ["sess-1", "sess-2", "sess-3"]


def test_malformed_iso_string_evicts() -> None:
    """Defensive: corrupt ISO strings evict rather than crash."""
    state = {
        "first_turn_pending": {
            "sess-bad": "not-an-iso-string-2026-99-99",
            "sess-good": (NOW - timedelta(seconds=60)).isoformat(),
        },
    }

    new_state, dropped = prune_first_turn_pending(state, now=NOW)

    assert "sess-bad" in dropped
    assert "sess-good" in new_state["first_turn_pending"]


def test_naive_timestamps_treated_as_utc() -> None:
    """Naive ISO strings (no tzinfo) get assumed UTC at parse time."""
    # A naive ISO string for "2 hours ago" — must evict at 1h TTL.
    naive_stale = (NOW - timedelta(seconds=7200)).replace(tzinfo=None).isoformat()
    state = {"first_turn_pending": {"sess-naive": naive_stale}}

    new_state, dropped = prune_first_turn_pending(state, now=NOW, ttl_sec=3600.0)

    assert dropped == ["sess-naive"]
    assert new_state["first_turn_pending"] == {}


def test_empty_or_missing_pending_returns_no_drops() -> None:
    """Idempotent on empty/missing first_turn_pending key."""
    # Missing key.
    new_state, dropped = prune_first_turn_pending({}, now=NOW)
    assert new_state == {"first_turn_pending": {}} or new_state == {}
    # Implementation contract: when the key is missing, return state
    # unchanged (we set "first_turn_pending" only when there was a dict
    # to prune). Both shapes are acceptable; the important property is
    # `dropped == []`.
    assert dropped == []

    # Present-but-empty dict.
    new_state2, dropped2 = prune_first_turn_pending(
        {"first_turn_pending": {}}, now=NOW,
    )
    assert dropped2 == []
    assert new_state2["first_turn_pending"] == {}

    # Present-but-None.
    new_state3, dropped3 = prune_first_turn_pending(
        {"first_turn_pending": None}, now=NOW,
    )
    assert dropped3 == []


def test_does_not_mutate_state_outside_first_turn_pending() -> None:
    """Pure function discipline: only first_turn_pending should change."""
    unrelated = {"unrelated_key": "unrelated_value", "fsm_state": "WAKE"}
    state = dict(unrelated)
    state["first_turn_pending"] = {
        "sess-stale": (NOW - timedelta(hours=2)).isoformat(),
    }

    new_state, _ = prune_first_turn_pending(state, now=NOW)

    for k, v in unrelated.items():
        assert new_state.get(k) == v
