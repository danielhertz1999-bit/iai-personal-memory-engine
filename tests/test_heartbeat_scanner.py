"""Comprehensive tests for ``HeartbeatScanner``.

Covers the 9-test matrix:
- Empty dir scan returns [].
- Single fresh heartbeat is FRESH (PID = current process, just-now refresh).
- Stale heartbeat (last_refresh older than M) is STALE even if PID alive.
- Orphan heartbeat (PID dead, fresh refresh) is ORPHAN.
- Five simultaneous fresh heartbeats: ``fresh_count`` == 5; ``is_active`` True.
- ``cleanup_stale_orphans`` deletes 3 of 4, leaves the fresh one.
- ``heartbeat_idle_30min`` False when at least one fresh exists.
- ``heartbeat_idle_30min`` True when only stale + orphan remain.
- Concurrent scan tolerates a writer adding a heartbeat mid-scan.

Tests use ``os.getpid()`` for live-PID fixtures (deterministic) and a
known-dead PID 99999 for orphan fixtures (verified dead at session start
by the implementation's ``_is_pid_alive``).
"""
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


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def wrappers_dir(tmp_path: Path) -> Path:
    """Empty wrappers directory under a fresh tmp_path."""
    wdir = tmp_path / "wrappers"
    wdir.mkdir()
    return wdir


def _write_heartbeat(
    wrappers_dir: Path,
    pid: int,
    uuid: str,
    last_refresh: datetime,
) -> Path:
    """Write a heartbeat file with the given pid/uuid/last_refresh.

    Returns the file path so tests can assert presence/absence after
    ``cleanup_stale_orphans``.
    """
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


# Known-dead PID — verified by ``_is_pid_alive`` in the test below.
# 99999 is above macOS's PID ceiling (typically <99998) so it is a stable
# choice for orphan fixtures. The verification test runs first to fail
# loudly if this assumption is wrong on a future host.
_DEAD_PID = 99999


# ---------------------------------------------------------------- sanity


def test_dead_pid_fixture_is_actually_dead() -> None:
    """Sanity: confirm PID 99999 is dead before relying on it in fixtures.

    If a future host happens to allocate PID 99999, the orphan-status
    fixture would silently degrade into a FRESH classification. This
    test fails loudly so we notice the collision.
    """
    assert _is_pid_alive(_DEAD_PID) is False


# ---------------------------------------------------------------- scan / classify


def test_scan_empty_dir_returns_empty(wrappers_dir: Path) -> None:
    """Empty wrappers dir yields an empty entries list."""
    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert entries == []
    assert scanner.fresh_count() == 0
    assert scanner.is_active() is False


def test_scan_single_fresh_heartbeat(wrappers_dir: Path) -> None:
    """Heartbeat with current PID + just-now refresh classifies FRESH."""
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
    """last_refresh older than threshold is STALE even if PID alive."""
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
    """Fresh refresh + dead PID classifies ORPHAN."""
    now = datetime.now(timezone.utc)
    _write_heartbeat(wrappers_dir, _DEAD_PID, "uuid-ccc", now)

    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert len(entries) == 1
    assert entries[0].status is HeartbeatStatus.ORPHAN
    assert scanner.fresh_count() == 0


def test_scan_5_simultaneous_wrappers(wrappers_dir: Path) -> None:
    """Five fresh heartbeats: fresh_count == 5; is_active True."""
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    for i in range(5):
        _write_heartbeat(wrappers_dir, own_pid, f"uuid-{i}", now)

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.fresh_count() == 5
    assert scanner.is_active() is True


# ---------------------------------------------------------------- cleanup


def test_cleanup_stale_orphans_deletes_files(wrappers_dir: Path) -> None:
    """2 stale + 1 orphan + 1 fresh; cleanup returns 3; fresh remains."""
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

    # Only the fresh file should still be on disk.
    assert fresh_path.exists()
    assert not stale_path1.exists()
    assert not stale_path2.exists()
    assert not orphan_path.exists()

    # Subsequent scan reflects the cleanup.
    remaining = scanner.scan()
    assert len(remaining) == 1
    assert remaining[0].uuid == "uuid-fresh"


# ---------------------------------------------------------------- heartbeat_idle_30min


def test_heartbeat_idle_30min_with_recent_fresh_returns_false(
    wrappers_dir: Path,
) -> None:
    """A single fresh heartbeat suppresses the idle predicate."""
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    _write_heartbeat(wrappers_dir, own_pid, "uuid-fresh", now)

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.heartbeat_idle_30min() is False


def test_heartbeat_idle_30min_no_fresh_returns_true(wrappers_dir: Path) -> None:
    """Only stale + orphan entries: predicate returns True (no live wrapper)."""
    own_pid = os.getpid()
    now = datetime.now(timezone.utc)
    stale_ts = now - timedelta(seconds=DEFAULT_STALE_THRESHOLD_SEC + 10)
    _write_heartbeat(wrappers_dir, own_pid, "uuid-s", stale_ts)
    _write_heartbeat(wrappers_dir, _DEAD_PID, "uuid-o", now)

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.heartbeat_idle_30min() is True


# ---------------------------------------------------------------- concurrency


def test_concurrent_scan_safe(wrappers_dir: Path) -> None:
    """A scan running concurrently with a writer must not raise.

    Spawns a background writer that drops new heartbeat files in tight
    succession while the main thread runs ``scan()`` repeatedly. The
    contract is "no exception" — final fresh count after the writer
    finishes equals the number of files actually written.
    """
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
        # Spin scans while the writer adds files. The race we are testing
        # is "scanner glob includes a file that vanishes" or "writer
        # half-writes JSON" — both must be tolerated silently.
        for _ in range(20):
            scanner.scan()  # must not raise
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


# ---------------------------------------------------------------- corruption tolerance


def test_torn_write_falls_back_to_mtime(wrappers_dir: Path) -> None:
    """Half-written JSON falls back to filename + mtime parse.

    Drops a file containing only the opening brace ``{`` (simulating a
    crash mid-write). The scanner must still classify the file by its
    filesystem mtime + filename PID rather than dropping the entry.
    """
    path = wrappers_dir / f"heartbeat-{os.getpid()}-uuid-torn.json"
    path.write_text("{")  # invalid JSON

    scanner = HeartbeatScanner(wrappers_dir)
    entries = scanner.scan()
    assert len(entries) == 1
    # Mtime is "now" by default so this should be FRESH (alive PID).
    assert entries[0].status is HeartbeatStatus.FRESH
    assert entries[0].pid == os.getpid()
