"""`iai-mcp lifecycle status` CLI tests.

Covers status output for each of the 4 states, default WAKE when the
file is absent, and the formatted lines for sleep_cycle_progress and
quarantine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest

from iai_mcp.lifecycle_state import (
    LifecycleState,
    LifecycleStateRecord,
    save_state,
)


# ---------------------------------------------------------------------------
# Helper -- patch LIFECYCLE_STATE_PATH to a tmp file for each test
# ---------------------------------------------------------------------------

def _run_status(tmp_path, monkeypatch, capsys, record: LifecycleStateRecord | None):
    """Patch the module-level path constant, optionally seed a record,
    invoke the subcommand directly, return captured stdout.
    """
    target = tmp_path / "lifecycle_state.json"
    monkeypatch.setattr(
        "iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH",
        target,
    )
    if record is not None:
        save_state(record, target)

    # Late import of cmd_lifecycle_status so the monkeypatch above
    # applies before the function reads LIFECYCLE_STATE_PATH.
    from iai_mcp.cli import cmd_lifecycle_status

    args = argparse.Namespace()
    rc = cmd_lifecycle_status(args)
    out = capsys.readouterr().out
    return rc, out


# ---------------------------------------------------------------------------
# Status output for each of the 4 states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", list(LifecycleState))
def test_status_prints_state_label(tmp_path, monkeypatch, capsys, state):
    record: LifecycleStateRecord = {
        "current_state": state.value,
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:11:30+00:00",
        "wrapper_event_seq": 42,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert f"state: {state.value}" in out


# ---------------------------------------------------------------------------
# Absent file -> default WAKE
# ---------------------------------------------------------------------------

def test_status_returns_default_wake_when_file_absent(tmp_path, monkeypatch, capsys):
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record=None)
    assert rc == 0
    assert "state: WAKE" in out


# ---------------------------------------------------------------------------
# Wrapper-event seq + last_activity rendered
# ---------------------------------------------------------------------------

def test_status_renders_seq_and_last_activity(tmp_path, monkeypatch, capsys):
    record: LifecycleStateRecord = {
        "current_state": "WAKE",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:11:30+00:00",
        "wrapper_event_seq": 137,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "wrapper_event_seq: 137" in out
    assert "last_activity: 2026-05-02T15:11:30+00:00" in out


# ---------------------------------------------------------------------------
# sleep_cycle_progress rendering
# ---------------------------------------------------------------------------

def test_status_progress_none_says_none(tmp_path, monkeypatch, capsys):
    record: LifecycleStateRecord = {
        "current_state": "WAKE",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "sleep_cycle_progress: none" in out


def test_status_progress_active_renders_step_attempt(tmp_path, monkeypatch, capsys):
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
        "quarantine": None,
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "step=3" in out
    assert "attempt=1" in out
    assert "last_error=none" in out


# ---------------------------------------------------------------------------
# Quarantine rendering
# ---------------------------------------------------------------------------

def test_status_quarantine_none_says_none(tmp_path, monkeypatch, capsys):
    record: LifecycleStateRecord = {
        "current_state": "WAKE",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "quarantine: none" in out


def test_status_quarantine_active_renders_until_and_reason(tmp_path, monkeypatch, capsys):
    record: LifecycleStateRecord = {
        "current_state": "SLEEP",
        "since_ts": "2026-05-02T03:00:00+00:00",
        "last_activity_ts": "2026-05-02T03:00:00+00:00",
        "wrapper_event_seq": 1,
        "sleep_cycle_progress": None,
        "quarantine": {
            "until_ts": "2026-05-03T03:00:00+00:00",
            "reason": "sleep step 4 failed 3x",
            "since_ts": "2026-05-02T03:00:00+00:00",
        },
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "until=2026-05-03T03:00:00+00:00" in out
    assert "reason=sleep step 4 failed 3x" in out
    assert "since=2026-05-02T03:00:00+00:00" in out


# ---------------------------------------------------------------------------
# shadow_run flag rendering
# ---------------------------------------------------------------------------

def test_status_shadow_run_true_mentions_legacy_watchdog(tmp_path, monkeypatch, capsys):
    record: LifecycleStateRecord = {
        "current_state": "WAKE",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "shadow_run: true" in out
    assert "Phase 10.6" in out


def test_status_shadow_run_false(tmp_path, monkeypatch, capsys):
    record: LifecycleStateRecord = {
        "current_state": "WAKE",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
    }
    rc, out = _run_status(tmp_path, monkeypatch, capsys, record)
    assert rc == 0
    assert "shadow_run: false" in out


# ---------------------------------------------------------------------------
# Helper formatter sanity
# ---------------------------------------------------------------------------

def test_format_relative_minutes(tmp_path, monkeypatch):
    from iai_mcp.cli import _format_relative

    now = datetime(2026, 5, 2, 15, 12, 0, tzinfo=timezone.utc)
    out = _format_relative("2026-05-02T15:00:00+00:00", now=now)
    assert out == "12 minutes"


def test_format_relative_hours():
    from iai_mcp.cli import _format_relative

    now = datetime(2026, 5, 2, 15, 12, 0, tzinfo=timezone.utc)
    out = _format_relative("2026-05-02T13:12:00+00:00", now=now)
    assert out == "2 hours"


def test_format_relative_days():
    from iai_mcp.cli import _format_relative

    now = datetime(2026, 5, 5, 0, 0, 0, tzinfo=timezone.utc)
    out = _format_relative("2026-05-02T00:00:00+00:00", now=now)
    assert out == "3 days"


def test_format_relative_singular_minute():
    from iai_mcp.cli import _format_relative

    now = datetime(2026, 5, 2, 15, 1, 0, tzinfo=timezone.utc)
    out = _format_relative("2026-05-02T15:00:00+00:00", now=now)
    assert out == "1 minute"


def test_format_relative_handles_garbage():
    from iai_mcp.cli import _format_relative

    assert _format_relative("not-a-timestamp") == "unknown"


# ---------------------------------------------------------------------------
# End-to-end: invoke via main([...])
# ---------------------------------------------------------------------------

def test_cli_main_lifecycle_status_via_main(tmp_path, monkeypatch, capsys):
    target = tmp_path / "lifecycle_state.json"
    monkeypatch.setattr(
        "iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH",
        target,
    )
    record: LifecycleStateRecord = {
        "current_state": "DROWSY",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:11:30+00:00",
        "wrapper_event_seq": 42,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": True,
    }
    save_state(record, target)

    from iai_mcp.cli import main

    rc = main(["lifecycle", "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "state: DROWSY" in out


# ---------------------------------------------------------------------------
# Lifecycle force-unlock subcommand
# ---------------------------------------------------------------------------


def test_force_unlock_with_yes_flag(tmp_path, monkeypatch, capsys):
    """``--yes`` skips the prompt and clears a present lockfile."""
    import json as _json

    from iai_mcp.cli import cmd_lifecycle_force_unlock

    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        _json.dumps(
            {
                "pid": 4242,
                "hostname": "stale-host.local",
                "started_at": "2026-04-29T08:00:00+00:00",
                "schema_version": 1,
            }
        )
    )

    args = argparse.Namespace(yes=True, lock_path=lock_path)
    rc = cmd_lifecycle_force_unlock(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "pid=4242" in out
    assert "stale-host.local" in out
    assert "Lockfile removed." in out
    assert not lock_path.exists()


def test_force_unlock_without_yes_prompts_no_aborts(
    tmp_path, monkeypatch, capsys,
):
    """No ``--yes`` flag: prompt is read, "n" aborts with rc=1, file kept."""
    import json as _json

    from iai_mcp.cli import cmd_lifecycle_force_unlock

    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        _json.dumps(
            {
                "pid": 4242,
                "hostname": "stale-host.local",
                "started_at": "2026-04-29T08:00:00+00:00",
                "schema_version": 1,
            }
        )
    )

    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    args = argparse.Namespace(yes=False, lock_path=lock_path)
    rc = cmd_lifecycle_force_unlock(args)
    captured = capsys.readouterr()
    assert rc == 1
    assert "cancelled" in captured.err.lower()
    assert lock_path.exists()


def test_force_unlock_without_yes_prompts_y_succeeds(
    tmp_path, monkeypatch, capsys,
):
    """Prompt receives "y" -> file cleared, rc=0."""
    import json as _json

    from iai_mcp.cli import cmd_lifecycle_force_unlock

    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        _json.dumps(
            {
                "pid": 4242,
                "hostname": "stale-host.local",
                "started_at": "2026-04-29T08:00:00+00:00",
                "schema_version": 1,
            }
        )
    )

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    args = argparse.Namespace(yes=False, lock_path=lock_path)
    rc = cmd_lifecycle_force_unlock(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Lockfile removed." in out
    assert not lock_path.exists()


def test_force_unlock_when_no_lockfile(tmp_path, capsys):
    """Absent lockfile -> rc=0 with "nothing to unlock" message."""
    from iai_mcp.cli import cmd_lifecycle_force_unlock

    lock_path = tmp_path / ".locked"  # never created
    args = argparse.Namespace(yes=True, lock_path=lock_path)
    rc = cmd_lifecycle_force_unlock(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to unlock" in out.lower()


def test_cli_main_lifecycle_force_unlock_via_main(
    tmp_path, monkeypatch, capsys,
):
    """End-to-end: invoke via ``iai-mcp lifecycle force-unlock --yes``.

    Production path uses ``DEFAULT_LOCK_PATH``; we monkey-patch it so
    the test does not touch ``~/.iai-mcp/.locked``.
    """
    import json as _json

    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        _json.dumps(
            {
                "pid": 9999,
                "hostname": "foreign-host.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": 1,
            }
        )
    )

    monkeypatch.setattr(
        "iai_mcp.lifecycle_lock.DEFAULT_LOCK_PATH",
        lock_path,
    )

    from iai_mcp.cli import main

    rc = main(["lifecycle", "force-unlock", "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Lockfile removed." in out
    assert not lock_path.exists()
