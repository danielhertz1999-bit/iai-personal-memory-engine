"""Phase 10.4 — regression tests for doctor rows (m) and (n).

Tests cover:
- (m) heartbeat scanner row with fresh wrappers + empty wrappers dir.
- (n) HID idle source row in the macOS-tools-available case + the
  fallback case where ``ioreg`` is missing (cross-OS portability).

The CONTEXT 10.4 specification requires:
- Row (m): PASS if wrappers dir readable; display "n=X fresh, Y stale,
  Z orphan".
- Row (n): PASS if ``available_signals`` includes ``"HIDIdleTime"``;
  WARN otherwise; display includes HID idle seconds + pmset state.

All subprocess interactions in this file are mocked so the suite is
deterministic and runs on non-macOS hosts as well (real ioreg / pmset
calls would make the suite host-dependent).
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp.idle_detector import IdleStatus


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def wrappers_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``IAI_MCP_STORE`` -> tmp_path; ensure ``<root>/wrappers/`` exists.

    The doctor row (m) resolves the wrappers dir from ``IAI_MCP_STORE``
    (test isolation pattern carried from check_i). Returns the wrappers
    subdirectory so tests can drop heartbeat fixtures directly.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    wdir = tmp_path / "wrappers"
    wdir.mkdir(parents=True)
    return wdir


def _write_fresh_heartbeat(wrappers_dir: Path, pid: int, uuid: str) -> Path:
    """Drop a heartbeat file with a current PID and now() timestamp.

    Uses ``os.getpid()`` by default so ``_is_pid_alive`` returns True
    deterministically — caller can override with a known-dead PID.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path = wrappers_dir / f"heartbeat-{pid}-{uuid}.json"
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "uuid": uuid,
                "started_at": now,
                "last_refresh": now,
                "wrapper_version": "1.0.0",
                "schema_version": 1,
            }
        )
    )
    return path


# ---------------------------------------------------------------- row (m)


def test_doctor_row_m_heartbeat_scanner_with_fresh_wrappers(
    wrappers_dir: Path,
) -> None:
    """Row (m) PASS with display showing the fresh count when wrappers exist."""
    own_pid = os.getpid()
    _write_fresh_heartbeat(wrappers_dir, own_pid, "uuid-aaa")
    _write_fresh_heartbeat(wrappers_dir, own_pid, "uuid-bbb")

    from iai_mcp.doctor import check_m_heartbeat_scanner

    result = check_m_heartbeat_scanner()
    assert result.status == "PASS"
    assert result.passed is True
    assert "n=2 fresh" in result.detail
    assert "0 stale" in result.detail
    assert "0 orphan" in result.detail


def test_doctor_row_m_heartbeat_scanner_empty(wrappers_dir: Path) -> None:
    """Row (m) PASS with display 'n=0 fresh' when wrappers dir is empty."""
    from iai_mcp.doctor import check_m_heartbeat_scanner

    result = check_m_heartbeat_scanner()
    assert result.status == "PASS"
    assert result.passed is True
    assert "n=0 fresh" in result.detail


def test_doctor_row_m_heartbeat_scanner_dir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row (m) PASS with 'not present yet' when wrappers dir absent.

    This is the steady-state on a fresh install before any wrapper has
    refreshed — must NOT report FAIL (the daemon is healthy, the dir
    just hasn't been created yet).
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    # Note: do NOT mkdir wrappers/ — that's the absent-state we're testing.
    from iai_mcp.doctor import check_m_heartbeat_scanner

    result = check_m_heartbeat_scanner()
    assert result.status == "PASS"
    assert result.passed is True
    assert "not present yet" in result.detail


# ---------------------------------------------------------------- row (n)


def test_doctor_row_n_hid_idle_source_macos() -> None:
    """Row (n) PASS when IdleDetector reports HIDIdleTime available.

    Patches ``IdleDetector.status`` to return a synthetic ``IdleStatus``
    with both signals available — avoids real ioreg/pmset calls so the
    test is deterministic on non-macOS CI hosts as well.
    """
    fake_status = IdleStatus(
        hid_idle_sec=612,
        pmset_recent_sleep=False,
        available_signals=["HIDIdleTime", "pmset"],
    )

    with patch(
        "iai_mcp.idle_detector.IdleDetector.status",
        return_value=fake_status,
    ):
        from iai_mcp.doctor import check_n_hid_idle_source

        result = check_n_hid_idle_source()

    assert result.status == "PASS"
    assert result.passed is True
    assert "HIDIdleTime: 612s" in result.detail
    assert "pmset: clean" in result.detail
    assert "HIDIdleTime" in result.detail


def test_doctor_row_n_hid_idle_source_missing() -> None:
    """Row (n) WARN when no hardware signals are available.

    Patches ``IdleDetector.status`` to return an empty signal list —
    simulates ioreg + pmset both missing (non-macOS host or broken
    install). Must report WARN and ``passed=True`` (advisory; does NOT
    flip the doctor exit code, mirroring check_i WARN).
    """
    fake_status = IdleStatus(
        hid_idle_sec=None,
        pmset_recent_sleep=False,
        available_signals=[],
    )

    with patch(
        "iai_mcp.idle_detector.IdleDetector.status",
        return_value=fake_status,
    ):
        from iai_mcp.doctor import check_n_hid_idle_source

        result = check_n_hid_idle_source()

    assert result.status == "WARN"
    # WARN must NOT flip the gate — passed stays True per CheckResult contract.
    assert result.passed is True
    assert "HIDIdleTime: unavailable" in result.detail
    assert "available: none" in result.detail
    assert "fall back to heartbeat-idle only" in result.detail


# ---------------------------------------------------------------- run_diagnosis wire-in


def test_run_diagnosis_includes_rows_m_and_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 10.4 wire-in: run_diagnosis() now includes rows (m) and (n)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    m_rows = [r for r in results if "(m)" in r.name]
    n_rows = [r for r in results if "(n)" in r.name]
    assert len(m_rows) == 1, f"expected exactly one (m) row, got {names}"
    assert len(n_rows) == 1, f"expected exactly one (n) row, got {names}"
    # (m) must come before (n) in the checklist sequence.
    assert names.index(m_rows[0].name) < names.index(n_rows[0].name)


# ----------------- Plan 10.6-01 Task 1.3: rows (j), (k), (l) ------


@pytest.fixture
def lifecycle_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """``IAI_MCP_STORE`` -> tmp_path; lets doctor's resolver point to tmp."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path


def test_doctor_row_j_lifecycle_state_default_when_absent(
    lifecycle_state_root: Path,
) -> None:
    """Row (j) PASS reporting WAKE when no lifecycle_state.json exists."""
    from iai_mcp.doctor import check_j_lifecycle_current_state

    result = check_j_lifecycle_current_state()
    assert result.status == "PASS"
    assert result.passed is True
    assert "WAKE" in result.detail
    # shadow_run default for default_state() is True; this test does not
    # care about its value, only that the row formats it.
    assert "shadow_run=" in result.detail


def test_doctor_row_j_lifecycle_state_reports_drowsy(
    lifecycle_state_root: Path,
) -> None:
    """Row (j) reports the recorded state when lifecycle_state.json present."""
    from iai_mcp.lifecycle_state import save_state

    record = {
        "current_state": "DROWSY",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:00:00+00:00",
        "wrapper_event_seq": 7,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_j_lifecycle_current_state

    result = check_j_lifecycle_current_state()
    assert result.status == "PASS"
    assert "DROWSY" in result.detail
    assert "shadow_run=false" in result.detail


def test_doctor_row_k_lifecycle_history_24h_no_log(
    lifecycle_state_root: Path,
) -> None:
    """Row (k) PASS with 'no event log yet' when log dir absent."""
    from iai_mcp.doctor import check_k_lifecycle_history_24h

    result = check_k_lifecycle_history_24h()
    assert result.status == "PASS"
    assert "no event log" in result.detail


def test_doctor_row_k_lifecycle_history_24h_zero_transitions(
    lifecycle_state_root: Path,
) -> None:
    """Row (k) PASS with '0 transitions' when log dir empty."""
    (lifecycle_state_root / "logs").mkdir()
    from iai_mcp.doctor import check_k_lifecycle_history_24h

    result = check_k_lifecycle_history_24h()
    assert result.status == "PASS"
    assert "0 transitions" in result.detail


def test_doctor_row_k_lifecycle_history_24h_counts_transitions(
    lifecycle_state_root: Path,
) -> None:
    """Row (k) sums state_transition events from today's JSONL file."""
    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    log = LifecycleEventLog(log_dir=lifecycle_state_root / "logs")
    # Three transitions: WAKE->DROWSY, DROWSY->WAKE, DROWSY->SLEEP.
    log.append(
        {"event": "state_transition", "from": "WAKE", "to": "DROWSY",
         "trigger": "idle_5min"}
    )
    log.append(
        {"event": "state_transition", "from": "DROWSY", "to": "WAKE",
         "trigger": "heartbeat_refresh"}
    )
    log.append(
        {"event": "state_transition", "from": "DROWSY", "to": "SLEEP",
         "trigger": "idle_30min"}
    )
    # Non-transition event must NOT be counted.
    log.append({"event": "wrapper_event", "kind": "boot"})

    from iai_mcp.doctor import check_k_lifecycle_history_24h

    result = check_k_lifecycle_history_24h()
    assert result.status == "PASS"
    assert "3 transitions" in result.detail
    # Bucket summary names destinations.
    assert "DROWSY=" in result.detail
    assert "WAKE=" in result.detail
    assert "SLEEP=" in result.detail


def test_doctor_row_l_quarantine_none_passes(
    lifecycle_state_root: Path,
) -> None:
    """Row (l) PASS when no quarantine record present."""
    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "PASS"
    assert "no quarantine" in result.detail


def test_doctor_row_l_quarantine_active_short_warns(
    lifecycle_state_root: Path,
) -> None:
    """Row (l) WARN for an active quarantine younger than 12 hours."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import save_state

    now = _dt.now(_tz.utc)
    since = (now - _td(hours=2)).isoformat()
    until = (now + _td(hours=22)).isoformat()
    record = {
        "current_state": "WAKE",
        "since_ts": now.isoformat(),
        "last_activity_ts": now.isoformat(),
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": {
            "since_ts": since,
            "until_ts": until,
            "reason": "sleep step 3 (DREAM_DECAY) failed 3x",
        },
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "WARN"
    assert result.passed is True  # WARN advisory only
    assert "quarantined" in result.detail
    assert "DREAM_DECAY" in result.detail


def test_doctor_row_l_quarantine_active_long_fails(
    lifecycle_state_root: Path,
) -> None:
    """Row (l) FAIL for a quarantine 12+ hours old."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import save_state

    now = _dt.now(_tz.utc)
    since = (now - _td(hours=14)).isoformat()  # 14h ago
    until = (now + _td(hours=10)).isoformat()
    record = {
        "current_state": "WAKE",
        "since_ts": now.isoformat(),
        "last_activity_ts": now.isoformat(),
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": {
            "since_ts": since,
            "until_ts": until,
            "reason": "sleep step 4 (OPTIMIZE_LANCE) failed 3x",
        },
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "FAIL"
    assert result.passed is False  # FAIL flips the exit code
    assert "reset-quarantine" in result.detail


def test_doctor_row_l_quarantine_expired_passes(
    lifecycle_state_root: Path,
) -> None:
    """Row (l) PASS for a quarantine whose until_ts is already in the past."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import save_state

    now = _dt.now(_tz.utc)
    since = (now - _td(hours=25)).isoformat()
    until = (now - _td(hours=1)).isoformat()  # already expired
    record = {
        "current_state": "WAKE",
        "since_ts": now.isoformat(),
        "last_activity_ts": now.isoformat(),
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": {
            "since_ts": since,
            "until_ts": until,
            "reason": "sleep step 5 (COMPACT_RECORDS) failed 3x",
        },
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "PASS"
    assert "expired" in result.detail


def test_run_diagnosis_includes_rows_j_k_l_in_order(
    lifecycle_state_root: Path,
) -> None:
    """Phase 10.6 wire-in: run_diagnosis returns 14 rows in correct order."""
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    # Expect 14 rows: a..i (9), j/k/l (3), m/n (2).
    assert len(results) == 14, f"expected 14 rows, got {len(results)}: {names}"

    # The new rows are present...
    j_idx = next(i for i, r in enumerate(results) if "(j)" in r.name)
    k_idx = next(i for i, r in enumerate(results) if "(k)" in r.name)
    l_idx = next(i for i, r in enumerate(results) if "(l)" in r.name)
    m_idx = next(i for i, r in enumerate(results) if "(m)" in r.name)

    # ...and ordered j < k < l < m so the lifecycle block is contiguous.
    assert j_idx < k_idx < l_idx < m_idx, (
        f"row order broken: j={j_idx} k={k_idx} l={l_idx} m={m_idx}"
    )
