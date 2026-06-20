from __future__ import annotations

import sys

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


def test_default_state_is_wake_with_shadow_run_disabled():
    record = default_state()
    assert record["current_state"] == "WAKE"
    assert record["shadow_run"] is False
    assert record["wrapper_event_seq"] == 0
    assert record["sleep_cycle_progress"] is None
    assert record["quarantine"] is None
    parsed = datetime.fromisoformat(record["since_ts"])
    assert parsed.tzinfo is not None


def test_default_state_uses_lifecycle_state_enum_value():
    assert default_state()["current_state"] == LifecycleState.WAKE.value


def test_load_state_returns_default_when_file_absent(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    assert not target.exists()
    record = load_state(target)
    assert record["current_state"] == "WAKE"
    assert not target.exists()


def test_load_state_returns_default_on_malformed_json(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    target.write_text("{not valid json at all")
    record = load_state(target)
    assert record["current_state"] == "WAKE"
    assert target.exists()


def test_load_state_returns_default_on_invalid_schema(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    target.write_text(json.dumps({"current_state": "INVALID"}))
    record = load_state(target)
    assert record["current_state"] == "WAKE"


def test_load_state_returns_default_on_wrong_state_value(tmp_path):
    target = tmp_path / "lifecycle_state.json"
    target.write_text(json.dumps({
        "current_state": "AWAKE",
        "since_ts": "2026-05-02T00:00:00+00:00",
        "last_activity_ts": "2026-05-02T00:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }))
    record = load_state(target)
    assert record["current_state"] == "WAKE"


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
    if sys.platform != "win32":
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


def test_atomic_replace_old_file_survives_temp_orphan(tmp_path, monkeypatch):
    target = tmp_path / "lifecycle_state.json"
    initial = default_state()
    initial["wrapper_event_seq"] = 99
    save_state(initial, target)

    real_replace = os.replace

    def boom(src, dst):  # noqa: ARG001
        raise RuntimeError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)

    new_record = default_state()
    new_record["wrapper_event_seq"] = 555
    with pytest.raises(RuntimeError, match="simulated crash"):
        save_state(new_record, target)

    monkeypatch.setattr(os, "replace", real_replace)

    loaded = load_state(target)
    assert loaded["wrapper_event_seq"] == 99

    leftover = list(tmp_path.glob(".lifecycle_state.*.tmp"))
    assert leftover == []


def test_default_path_is_under_iai_mcp_home():
    import iai_mcp.lifecycle_state as _ls

    path = _ls.LIFECYCLE_STATE_PATH
    assert path.name == "lifecycle_state.json"
    assert path.parent.name == ".iai-mcp"
    assert path.is_relative_to(Path.home())
