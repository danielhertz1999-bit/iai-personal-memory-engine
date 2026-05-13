"""Task 1.1 -- lifecycle_state typed schema tests.

Covers the round-trip, atomic-replace crash safety, and schema-validation
self-heal behaviour of `lifecycle_state.{load_state,save_state}`. Mirrors
the test layout of `test_daemon_state.py` (-01) since the
persistence pattern is identical.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp.lifecycle_state import (
    LIFECYCLE_STATE_PATH,
    LifecycleState,
    LifecycleStateRecord,
    default_state,
    load_state,
    save_state,
)


# ---------------------------------------------------------------------------
# default_state shape
# ---------------------------------------------------------------------------

def test_default_state_is_wake_with_shadow_run_disabled():
    """shadow_run flipped to False by
    default. HIBERNATION transitions now actually exit the daemon.
    """
    record = default_state()
    assert record["current_state"] == "WAKE"
    assert record["shadow_run"] is False
    assert record["wrapper_event_seq"] == 0
    assert record["sleep_cycle_progress"] is None
    assert record["quarantine"] is None
    # Timestamps parse as UTC ISO-8601.
    parsed = datetime.fromisoformat(record["since_ts"])
    assert parsed.tzinfo is not None


def test_default_state_uses_lifecycle_state_enum_value():
    """Defensive: future enum renames must not desync the default."""
    assert default_state()["current_state"] == LifecycleState.WAKE.value


# ---------------------------------------------------------------------------
# load_state self-heal
# ---------------------------------------------------------------------------

def test_load_state_returns_default_when_file_absent(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    assert not target.exists()
    record = load_state(target)
    assert record["current_state"] == "WAKE"
    # default_state did NOT write to disk; load is read-only.
    assert not target.exists()


def test_load_state_returns_default_on_malformed_json(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    target.write_text("{not valid json at all")
    record = load_state(target)
    assert record["current_state"] == "WAKE"
    # Malformed file is left in place (no auto-delete) so the operator
    # can inspect it; save_state will overwrite on the next persist.
    assert target.exists()


def test_load_state_returns_default_on_invalid_schema(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    target.write_text(json.dumps({"current_state": "INVALID"}))
    record = load_state(target)
    assert record["current_state"] == "WAKE"


def test_load_state_returns_default_on_wrong_state_value(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    target.write_text(json.dumps({
        "current_state": "AWAKE",  # not a LifecycleState member
        "since_ts": "2026-05-02T00:00:00+00:00",
        "last_activity_ts": "2026-05-02T00:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }))
    record = load_state(target)
    assert record["current_state"] == "WAKE"


# ---------------------------------------------------------------------------
# save_state round trip
# ---------------------------------------------------------------------------

def test_save_then_load_roundtrip(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    original: LifecycleStateRecord = {
        "current_state": "DROWSY",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:14:30+00:00",
        "wrapper_event_seq": 42,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    save_state(original, target)
    assert target.exists()
    loaded = load_state(target)
    assert loaded == original


def test_save_state_with_progress_and_quarantine(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    record: LifecycleStateRecord = {
        "current_state": "SLEEP",
        "since_ts": "2026-05-02T03:00:00+00:00",
        "last_activity_ts": "2026-05-02T03:00:00+00:00",
        "wrapper_event_seq": 7,
        "sleep_cycle_progress": {
            "last_completed_step": 3,
            "attempt": 1,
            "last_error": None,
            "started_at": "2026-05-02T03:00:00+00:00",
        },
        "quarantine": {
            "until_ts": "2026-05-03T03:00:00+00:00",
            "reason": "sleep step 4 failed 3x",
            "since_ts": "2026-05-02T03:00:00+00:00",
        },
        "shadow_run": False,
    }
    save_state(record, target)
    loaded = load_state(target)
    assert loaded == record


def test_save_state_creates_parent_dir(tmp_path):
    target = tmp_path / "deep" / "nested" / "lifecycle_state.json"
    record = default_state()
    save_state(record, target)
    assert target.exists()


def test_save_state_chmod_user_only(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    save_state(default_state(), target)
    mode = os.stat(target).st_mode & 0o777
    assert mode == 0o600


def test_save_state_rejects_invalid_record(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    bad = {
        "current_state": "NOT_A_STATE",
        "since_ts": "2026-05-02T00:00:00+00:00",
        "last_activity_ts": "2026-05-02T00:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    with pytest.raises(ValueError):
        save_state(bad, target)  # type: ignore[arg-type]
    # File never created on validation failure.
    assert not target.exists()


def test_save_state_rejects_negative_seq(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    bad = {
        "current_state": "WAKE",
        "since_ts": "2026-05-02T00:00:00+00:00",
        "last_activity_ts": "2026-05-02T00:00:00+00:00",
        "wrapper_event_seq": -1,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    with pytest.raises(ValueError):
        save_state(bad, target)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Atomic replace: simulated crash mid-write leaves the OLD file intact
# ---------------------------------------------------------------------------

def test_atomic_replace_old_file_survives_temp_orphan(tmp_path, monkeypatch):
    """If os.replace is interrupted (simulated by raising), the old file
    must still be intact and readable. Tempfile must be cleaned up.
    """
    target = tmp_path / "lifecycle_state.json"
    # Seed an existing valid record.
    initial = default_state()
    initial["wrapper_event_seq"] = 99
    save_state(initial, target)

    # Force os.replace to fail mid-write.
    real_replace = os.replace

    def boom(src, dst):  # noqa: ARG001
        raise RuntimeError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)

    new_record = default_state()
    new_record["wrapper_event_seq"] = 555
    with pytest.raises(RuntimeError, match="simulated crash"):
        save_state(new_record, target)

    # Restore os.replace so subsequent ops in this test can use it normally.
    monkeypatch.setattr(os, "replace", real_replace)

    # Old file content unchanged.
    loaded = load_state(target)
    assert loaded["wrapper_event_seq"] == 99

    # Temp file orphan was cleaned up.
    leftover = list(tmp_path.glob(".lifecycle_state.*.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# default path constant points at ~/.iai-mcp/lifecycle_state.json
# ---------------------------------------------------------------------------

def test_default_path_is_under_iai_mcp_home():
    assert LIFECYCLE_STATE_PATH.name == "lifecycle_state.json"
    assert LIFECYCLE_STATE_PATH.parent.name == ".iai-mcp"
    # Sanity: path is anchored under the user's home, not /tmp or /var.
    assert str(LIFECYCLE_STATE_PATH).startswith(str(Path.home()))
