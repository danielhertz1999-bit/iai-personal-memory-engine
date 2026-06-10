from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.heartbeat_scanner import (
    DEFAULT_STALE_THRESHOLD_SEC,
    HeartbeatScanner,
    HeartbeatStatus,
    _is_pid_alive,
)


@pytest.fixture
def wrappers_dir(tmp_path: Path) -> Path:
    wdir = tmp_path / "wrappers"
    wdir.mkdir()
    return wdir


def _write_heartbeat(
    wrappers_dir: Path,
    pid: int,
    uuid: str,
    last_refresh: datetime,
) -> Path:
    path = wrappers_dir / f"heartbeat-{pid}-{uuid}.json"
    payload = {
        "pid": pid,
        "uuid": uuid,
        "started_at": last_refresh.isoformat().replace("+00:00", "Z"),
        "last_refresh": last_refresh.isoformat().replace("+00:00", "Z"),
        "wrapper_version": "1.0.0",
        "schema_version": 1,
    }
    path.write_text(json.dumps(payload))
    return path


_DEAD_PID = 99999


def test_dead_pid_fixture_is_actually_dead() -> None:
    assert _is_pid_alive(_DEAD_PID) is False


def test_scan_empty_dir_returns_empty(wrappers_dir: Path) -> None:
    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert entries == []
    assert scanner.fresh_count() == 0
    assert scanner.is_active() is False


def test_scan_single_fresh_heartbeat(wrappers_dir: Path) -> None:
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    _write_heartbeat(wrappers_dir, own_pid, "uuid-aaa", now)

    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.pid == own_pid
    assert entry.uuid == "uuid-aaa"
    assert entry.status is HeartbeatStatus.FRESH
    assert scanner.is_active() is True


def test_scan_stale_heartbeat(wrappers_dir: Path) -> None:
    own_pid = os.getpid()
    stale_ts = datetime.now(timezone.utc) - timedelta(
        seconds=DEFAULT_STALE_THRESHOLD_SEC + 10
    )
    _write_heartbeat(wrappers_dir, own_pid, "uuid-bbb", stale_ts)

    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert len(entries) == 1
    assert entries[0].status is HeartbeatStatus.STALE
    assert scanner.fresh_count() == 0
    assert scanner.is_active() is False


def test_scan_orphan_heartbeat(wrappers_dir: Path) -> None:
    now = datetime.now(timezone.utc)
    _write_heartbeat(wrappers_dir, _DEAD_PID, "uuid-ccc", now)

    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert len(entries) == 1
    assert entries[0].status is HeartbeatStatus.ORPHAN
    assert scanner.fresh_count() == 0


def test_scan_5_simultaneous_wrappers(wrappers_dir: Path) -> None:
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    for i in range(5):
        _write_heartbeat(wrappers_dir, own_pid, f"uuid-{i}", now)

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.fresh_count() == 5
    assert scanner.is_active() is True


def test_cleanup_stale_orphans_deletes_files(wrappers_dir: Path) -> None:
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    stale_ts = now - timedelta(seconds=DEFAULT_STALE_THRESHOLD_SEC + 10)

    fresh_path = _write_heartbeat(wrappers_dir, own_pid, "uuid-fresh", now)
    stale_path1 = _write_heartbeat(wrappers_dir, own_pid, "uuid-s1", stale_ts)
    stale_path2 = _write_heartbeat(wrappers_dir, own_pid, "uuid-s2", stale_ts)
    orphan_path = _write_heartbeat(wrappers_dir, _DEAD_PID, "uuid-orphan", now)

    scanner = HeartbeatScanner(wrappers_dir)
    deleted = scanner.cleanup_stale_orphans()
    assert deleted == 3

    assert fresh_path.exists()
    assert not stale_path1.exists()
    assert not stale_path2.exists()
    assert not orphan_path.exists()

    remaining = scanner.scan()
    assert len(remaining) == 1
    assert remaining[0].uuid == "uuid-fresh"


def test_heartbeat_idle_30min_with_recent_fresh_returns_false(
    wrappers_dir: Path,
) -> None:
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    _write_heartbeat(wrappers_dir, own_pid, "uuid-fresh", now)

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.heartbeat_idle_30min() is False


def test_heartbeat_idle_30min_no_fresh_returns_true(wrappers_dir: Path) -> None:
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    stale_ts = now - timedelta(seconds=DEFAULT_STALE_THRESHOLD_SEC + 10)
    _write_heartbeat(wrappers_dir, own_pid, "uuid-s", stale_ts)
    _write_heartbeat(wrappers_dir, _DEAD_PID, "uuid-o", now)

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.heartbeat_idle_30min() is True


def test_concurrent_scan_safe(wrappers_dir: Path) -> None:
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    write_count = 50
    written: list[Path] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for i in range(write_count):
                if stop.is_set():
                    return
                p = _write_heartbeat(
                    wrappers_dir, own_pid, f"uuid-cc-{i}", now
                )
                written.append(p)
        except BaseException as exc:  # noqa: BLE001 — surface in test
            errors.append(exc)

    scanner = HeartbeatScanner(wrappers_dir)
    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(20):
            scanner.scan()
            time.sleep(0.001)
    finally:
        stop.set()
        t.join(timeout=5)

    assert errors == [], f"writer raised: {errors!r}"
    final = scanner.scan()
    assert len(final) == len(written), (
        f"final scan count {len(final)} != written count {len(written)}"
    )
    assert all(e.status is HeartbeatStatus.FRESH for e in final)


def test_torn_write_falls_back_to_mtime(wrappers_dir: Path) -> None:
    path = wrappers_dir / f"heartbeat-{os.getpid()}-uuid-torn.json"
    path.write_text("{")

    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert len(entries) == 1
    assert entries[0].status is HeartbeatStatus.FRESH
    assert entries[0].pid == os.getpid()
