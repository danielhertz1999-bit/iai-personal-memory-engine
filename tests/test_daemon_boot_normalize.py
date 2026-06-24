"""Boot recovery: a crash mid-SLEEP leaves lifecycle_state.json incoherent
(current_state=SLEEP, sleep_cycle_progress=None). The daemon must reset that one
case to a clean WAKE at boot (and drop the stale crisis flag) instead of resuming
a sleep cycle that never progresses. Guards _normalize_boot_lifecycle_state.
"""
from __future__ import annotations

from iai_mcp.daemon import _normalize_boot_lifecycle_state


def test_stale_sleep_without_progress_is_reset_to_wake():
    raw = {
        "current_state": "SLEEP",
        "sleep_cycle_progress": None,
        "crisis_mode": True,
        "crisis_mode_since_ts": "2026-06-21T11:20:25+00:00",
        "since_ts": "2026-06-21T10:47:08+00:00",
    }
    out, changed = _normalize_boot_lifecycle_state(raw)
    assert changed is True
    assert out["current_state"] == "WAKE"
    assert out["crisis_mode"] is False
    assert out["crisis_mode_since_ts"] is None
    # other fields preserved
    assert out["since_ts"] == raw["since_ts"]
    # input not mutated
    assert raw["current_state"] == "SLEEP"


def test_sleep_with_active_progress_is_left_untouched():
    # A real in-flight cycle carries a progress dict -> must NOT be reset.
    raw = {
        "current_state": "SLEEP",
        "sleep_cycle_progress": {"step": "CLUSTER_SUMMARY", "attempt": 1},
        "crisis_mode": False,
    }
    out, changed = _normalize_boot_lifecycle_state(raw)
    assert changed is False
    assert out is raw


def test_wake_state_is_left_untouched():
    raw = {"current_state": "WAKE", "sleep_cycle_progress": None, "crisis_mode": True}
    out, changed = _normalize_boot_lifecycle_state(raw)
    assert changed is False
    assert out is raw


def test_drowsy_state_is_left_untouched():
    raw = {"current_state": "DROWSY", "sleep_cycle_progress": None}
    out, changed = _normalize_boot_lifecycle_state(raw)
    assert changed is False
