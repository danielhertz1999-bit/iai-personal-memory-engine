"""Tests for iai_mcp.daemon_state -- Task 2.

Covers:
1. save_state atomically persists and load_state round-trips.
2. File mode is 0o600.
3. save_state is atomic under simulated mid-write failure (temp file unlinked).
4. get_pending_digest returns + clears digest when > threshold elapsed.
5. get_pending_digest returns None when <18h since last shown.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def isolated_state_path(tmp_path, monkeypatch):
    """Redirect STATE_PATH to tmp_path for test isolation."""
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


# ---------------------------------------------------------------------------
# Test 1 + 2: roundtrip + 0o600
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip_with_0600_mode(isolated_state_path):
    from iai_mcp.daemon_state import load_state, save_state

    # Fresh load -> {}.
    assert load_state() == {}

    state = {
        "fsm_state": "WAKE",
        "daemon_started_at": "2026-04-18T00:00:00+00:00",
        "pending_digest": {"cycles": 4, "insight": "test"},
    }
    save_state(state)

    # File exists, mode is 0o600.
    assert isolated_state_path.exists()
    mode = isolated_state_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    # load returns identical dict.
    loaded = load_state()
    assert loaded == state


# ---------------------------------------------------------------------------
# Test 3: atomic write via tempfile + os.replace
# ---------------------------------------------------------------------------

def test_save_state_atomic_rename_preserves_old_on_failure(isolated_state_path, monkeypatch):
    """If os.replace raises, the target file must remain untouched and the
    temp file must be cleaned up."""
    from iai_mcp.daemon_state import load_state, save_state

    # Seed a known-good file.
    original = {"fsm_state": "WAKE", "version": 1}
    save_state(original)
    assert load_state() == original

    # Patch os.replace to raise on the next call so the atomic swap fails.
    import iai_mcp.daemon_state as ds
    real_replace = os.replace

    def _boom(src, dst):
        raise OSError("simulated swap failure")

    monkeypatch.setattr(ds.os, "replace", _boom)

    with pytest.raises(OSError):
        save_state({"fsm_state": "SLEEP", "version": 2})

    # Original file preserved (atomic rename never happened).
    loaded = load_state()
    assert loaded == original

    # Temp file cleaned up -- no leftover .tmp files in the directory.
    leftovers = list(isolated_state_path.parent.glob(".daemon-state.*.tmp"))
    assert leftovers == [], f"temp files not cleaned: {leftovers}"


# ---------------------------------------------------------------------------
# Test 4: pending digest returned after threshold window
# ---------------------------------------------------------------------------

def test_pending_digest_returned_after_18h(isolated_state_path):
    from iai_mcp.daemon_state import (
        DIGEST_SHOW_THRESHOLD_HOURS,
        get_pending_digest,
        load_state,
        save_state,
    )
    assert DIGEST_SHOW_THRESHOLD_HOURS == 18

    now = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    last_shown = now - timedelta(hours=20)
    state = {
        "last_digest_shown_at": last_shown.isoformat(),
        "pending_digest": {"cycles": 4, "insight": "after-threshold"},
    }
    save_state(state)

    digest = get_pending_digest(state, now)
    assert digest == {"cycles": 4, "insight": "after-threshold"}

    # State mutated and persisted: pending_digest cleared, last_digest_shown_at bumped.
    assert "pending_digest" not in state
    assert state["last_digest_shown_at"] == now.isoformat()

    # Persisted to disk.
    on_disk = load_state()
    assert "pending_digest" not in on_disk
    assert on_disk["last_digest_shown_at"] == now.isoformat()


# ---------------------------------------------------------------------------
# Test 5: digest withheld when <18h since last shown
# ---------------------------------------------------------------------------

def test_pending_digest_withheld_before_18h(isolated_state_path):
    from iai_mcp.daemon_state import get_pending_digest

    now = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    last_shown = now - timedelta(hours=4)
    state = {
        "last_digest_shown_at": last_shown.isoformat(),
        "pending_digest": {"cycles": 4, "insight": "too-early"},
    }
    digest = get_pending_digest(state, now)
    assert digest is None

    # State preserved (digest still pending for later).
    assert state["pending_digest"] == {"cycles": 4, "insight": "too-early"}
    assert state["last_digest_shown_at"] == last_shown.isoformat()


# ---------------------------------------------------------------------------
# Extra: no digest when state has no pending_digest
# ---------------------------------------------------------------------------

def test_pending_digest_none_when_not_set(isolated_state_path):
    from iai_mcp.daemon_state import get_pending_digest

    now = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    state: dict = {}
    assert get_pending_digest(state, now) is None


# ---------------------------------------------------------------------------
# prune_stale_first_turn: evicts legacy bool + aged ISO entries
# ---------------------------------------------------------------------------

def test_prune_evicts_legacy_bool_first_turn_pending():
    """Legacy {sid: True} entries evict on first prune — they have no
    recoverable timestamp so we cannot age them sensibly."""
    from iai_mcp.daemon_state import prune_stale_first_turn

    state = {"first_turn_pending": {"sess-1": True, "sess-2": False, "sess-3": True}}
    removed = prune_stale_first_turn(state)

    assert removed == 3
    assert state["first_turn_pending"] == {}


def test_prune_keeps_fresh_iso_entries_and_evicts_aged():
    """ISO timestamps within TTL survive; older than TTL get evicted."""
    from iai_mcp.daemon_state import prune_stale_first_turn

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=1)).isoformat()
    stale = (now - timedelta(hours=48)).isoformat()
    state = {"first_turn_pending": {"fresh": fresh, "stale": stale}}

    removed = prune_stale_first_turn(state, now=now, ttl_hours=24)

    assert removed == 1
    assert "fresh" in state["first_turn_pending"]
    assert "stale" not in state["first_turn_pending"]


def test_prune_caps_max_entries_keeps_newest():
    """Secondary cap: keep newest max_entries entries by timestamp."""
    from iai_mcp.daemon_state import prune_stale_first_turn

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    pending = {f"sess-{i}": (now - timedelta(minutes=i)).isoformat() for i in range(10)}
    state = {"first_turn_pending": pending}

    removed = prune_stale_first_turn(state, now=now, ttl_hours=24, max_entries=3)

    assert removed == 7
    kept = state["first_turn_pending"]
    assert len(kept) == 3
    # Newest three minutes (0, 1, 2) survive.
    assert set(kept.keys()) == {"sess-0", "sess-1", "sess-2"}


def test_prune_handles_empty_and_missing_pending():
    """Idempotent on empty / missing first_turn_pending."""
    from iai_mcp.daemon_state import prune_stale_first_turn

    assert prune_stale_first_turn({}) == 0
    assert prune_stale_first_turn({"first_turn_pending": {}}) == 0
    assert prune_stale_first_turn({"first_turn_pending": None}) == 0
