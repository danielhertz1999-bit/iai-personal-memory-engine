"""psutil cmdline cross-check in `_is_pid_alive`.

Locks the recycled-PID mitigation: a stale ``~/.iai-mcp/.locked`` file whose
PID has been recycled by an unrelated process (shell, browser, etc.) must
NOT false-positive as a live daemon. The helper now requires the substring
``iai_mcp.daemon`` to appear in ``psutil.Process(pid).cmdline()`` before
treating the PID as live -- mirroring the pattern already in
``src/iai_mcp/doctor.py``.

Four cases:
  1. Recycled PID, cmdline NOT containing 'iai_mcp.daemon' -> stale (acquire succeeds).
  2. PID with daemon-like cmdline -> live (acquire raises LifecycleLockConflict).
  3. psutil.NoSuchProcess between os.kill and Process() -> treat as stale.
  4. psutil import fails -> fall back to os.kill-only semantics + debug log.

Tests use ``os.getpid()`` as the on-disk PID so the real ``os.kill(pid, 0)``
inside ``_is_pid_alive`` always succeeds; only the psutil cmdline layer is
faked, exercising the new code path. The lock_path is always under
``tmp_path`` so production state is never touched.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import psutil as _real_psutil
import pytest

from iai_mcp.lifecycle_lock import (
    LifecycleLock,
    LifecycleLockConflict,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_lockfile(lock_path: Path, *, pid: int, hostname: str) -> None:
    """Pre-populate ``.locked`` with the given PID + hostname."""
    lock_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "hostname": hostname,
                "started_at": "2026-05-17T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )


def _install_fake_psutil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cmdline_value: list[str] | None = None,
    cmdline_exc: Exception | None = None,
) -> MagicMock:
    """Install a fake ``psutil`` module in ``sys.modules``.

    Pattern mirrors ``tests/test_cli_maintenance_compact_records.py``.

    The fake module exposes:
      - ``Process(pid)`` returning a mock whose ``.cmdline()`` returns
        ``cmdline_value`` OR raises ``cmdline_exc``.
      - The real ``psutil.NoSuchProcess`` / ``AccessDenied`` / ``ZombieProcess``
        classes (MagicMock-auto-attrs are NOT catchable by ``except``).
    """
    fake_proc = MagicMock()
    if cmdline_exc is not None:
        fake_proc.cmdline.side_effect = cmdline_exc
    else:
        fake_proc.cmdline.return_value = cmdline_value or []
    fake_psutil = MagicMock()
    fake_psutil.Process.return_value = fake_proc
    # Attach REAL exception classes so production-side
    # ``except (psutil.NoSuchProcess,...)`` actually catches them.
    fake_psutil.NoSuchProcess = _real_psutil.NoSuchProcess
    fake_psutil.AccessDenied = _real_psutil.AccessDenied
    fake_psutil.ZombieProcess = _real_psutil.ZombieProcess
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    return fake_psutil


# ---------------------------------------------------------------------------
# Test 1: recycled PID, cmdline is `["bash"]` -> stale (acquire succeeds)
# ---------------------------------------------------------------------------


def test_acquire_treats_recycled_pid_with_non_daemon_cmdline_as_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a live PID whose cmdline is `bash` must be treated as stale.

    Without the psutil cross-check, today's ``_is_pid_alive`` returns True
    for ANY PID the kernel accepts -- including a PID belonging to the test
    process itself, which is NOT a daemon. ``acquire()`` would wrongly raise
    ``LifecycleLockConflict``. The fix: require ``iai_mcp.daemon`` substring.
    """
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()  # always alive, never iai_mcp.daemon during pytest
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    _install_fake_psutil(monkeypatch, cmdline_value=["bash"])

    lock = LifecycleLock(lock_path)
    # Must NOT raise: cmdline does not contain 'iai_mcp.daemon' -> stale.
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == real_pid, (
        f"cue=acquire-overwrote-stale lock pid={payload['pid']} expected={real_pid}"
    )


# ---------------------------------------------------------------------------
# Test 2: PID with daemon cmdline -> live (raises conflict)
# ---------------------------------------------------------------------------


def test_acquire_treats_daemon_cmdline_as_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live PID whose cmdline contains 'iai_mcp.daemon' is a real daemon.

    No regression on the conflict path. Marker assertion ``Process.called``
    pins that the new psutil path actually ran (without it, today's code
    would PASS this test trivially via the os.kill-only path -- ambiguous RED).
    """
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    fake_psutil = _install_fake_psutil(
        monkeypatch, cmdline_value=["python3", "-m", "iai_mcp.daemon"]
    )

    lock = LifecycleLock(lock_path)
    with pytest.raises(LifecycleLockConflict) as exc_info:
        lock.acquire()
    assert exc_info.value.existing is not None, "cue=conflict-carries-existing-payload"
    assert exc_info.value.existing["pid"] == real_pid

    # Marker assertion: the new code MUST have consulted psutil. Without
    # this, today's `_is_pid_alive` -- which never imports psutil --
    # would pass via the os.kill-only path, making RED ambiguous.
    assert fake_psutil.Process.called, (
        "cue=psutil-not-consulted; helper still on os.kill-only path"
    )


# ---------------------------------------------------------------------------
# Test 3: psutil.NoSuchProcess between os.kill and Process() -> stale
# ---------------------------------------------------------------------------


def test_acquire_treats_psutil_nosuchprocess_as_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race window: PID alive at os.kill, vanished by psutil.Process().

    Mapping: NoSuchProcess -> 'not our daemon' -> lockfile is overwriteable.
    Same disposition for AccessDenied / ZombieProcess (covered by the
    tuple-narrowed except in the production helper).
    """
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    _install_fake_psutil(
        monkeypatch,
        cmdline_exc=_real_psutil.NoSuchProcess(real_pid),
    )

    lock = LifecycleLock(lock_path)
    # Must NOT raise: NoSuchProcess means PID vanished -> stale lock.
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == real_pid, (
        f"cue=acquire-overwrote-on-NoSuchProcess pid={payload['pid']} expected={real_pid}"
    )


# ---------------------------------------------------------------------------
# Test 4: psutil ImportError -> fall back to os.kill-only + debug log
# ---------------------------------------------------------------------------


def test_acquire_falls_back_to_os_kill_when_psutil_unimportable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Defensive: if ``import psutil`` fails, fall back to old semantics.

    psutil is a hard dep today (pyproject.toml:23), but the fallback is
    future-proofing. Behavior: behaves like today's helper -- live PID
    raises conflict. A DEBUG log entry must be emitted so the operator
    can diagnose the missing dep. Marker assertion: the log entry pins
    that the fallback branch ran (without it, today's code passes via
    the os.kill-only path -- ambiguous RED).
    """
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")

    # Force `import psutil` to raise ImportError. setting sys.modules
    # value to None makes Python raise ImportError on a fresh import
    # attempt -- exactly the failure mode the helper must handle.
    monkeypatch.setitem(sys.modules, "psutil", None)

    caplog.set_level(logging.DEBUG, logger="iai_mcp.lifecycle_lock")

    lock = LifecycleLock(lock_path)
    # Fallback semantics: PID alive (real os.getpid()) -> conflict, as today.
    with pytest.raises(LifecycleLockConflict):
        lock.acquire()

    # Marker assertion: the fallback branch MUST have logged the debug entry.
    # Without this, today's helper (which never tries to import psutil)
    # PASSES this test trivially -- ambiguous RED. Catching the log proves
    # the new try/except ImportError branch actually executed.
    fallback_logged = any(
        "psutil unavailable" in rec.getMessage()
        for rec in caplog.records
        if rec.name == "iai_mcp.lifecycle_lock"
    )
    assert fallback_logged, (
        "cue=psutil-import-fallback-branch-not-executed; "
        f"records={[r.getMessage() for r in caplog.records]}"
    )
