from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.daemon_state import (
    FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
    prune_first_turn_pending,
)

NOW = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def test_default_ttl_is_3600_seconds() -> None:
    assert FIRST_TURN_PENDING_TTL_SEC_DEFAULT == 3600.0


def test_keeps_fresh_evicts_stale_returns_dropped_ids() -> None:
    fresh_ts = (NOW - timedelta(seconds=1800)).isoformat()
    stale_ts = (NOW - timedelta(seconds=7200)).isoformat()
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
    state = {
        "first_turn_pending": {"sess-1": True, "sess-2": False, "sess-3": None},
    }

    new_state, dropped = prune_first_turn_pending(state, now=NOW)

    assert new_state["first_turn_pending"] == {}
    assert sorted(dropped) == ["sess-1", "sess-2", "sess-3"]


def test_malformed_iso_string_evicts() -> None:
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
    naive_stale = (NOW - timedelta(seconds=7200)).replace(tzinfo=None).isoformat()
    state = {"first_turn_pending": {"sess-naive": naive_stale}}

    new_state, dropped = prune_first_turn_pending(state, now=NOW, ttl_sec=3600.0)

    assert dropped == ["sess-naive"]
    assert new_state["first_turn_pending"] == {}


def test_empty_or_missing_pending_returns_no_drops() -> None:
    new_state, dropped = prune_first_turn_pending({}, now=NOW)
    assert new_state == {"first_turn_pending": {}} or new_state == {}
    assert dropped == []

    new_state2, dropped2 = prune_first_turn_pending(
        {"first_turn_pending": {}}, now=NOW,
    )
    assert dropped2 == []
    assert new_state2["first_turn_pending"] == {}

    new_state3, dropped3 = prune_first_turn_pending(
        {"first_turn_pending": None}, now=NOW,
    )
    assert dropped3 == []


def test_does_not_mutate_state_outside_first_turn_pending() -> None:
    unrelated = {"unrelated_key": "unrelated_value", "fsm_state": "WAKE"}
    state = dict(unrelated)
    state["first_turn_pending"] = {
        "sess-stale": (NOW - timedelta(hours=2)).isoformat(),
    }

    new_state, _ = prune_first_turn_pending(state, now=NOW)

    for k, v in unrelated.items():
        assert new_state.get(k) == v
