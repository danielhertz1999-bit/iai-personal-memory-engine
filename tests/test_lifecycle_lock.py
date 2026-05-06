"""Phase 10.6 Plan 10.6-01 Task 1.1 -- LifecycleLock unit tests.

Locks the single-machine assumption:

- ``acquire()`` succeeds in a clean state.
- ``acquire()`` over a dead-PID lockfile succeeds (takeover).
- ``acquire()`` over a live-PID same-host lockfile raises
  ``LifecycleLockConflict`` (the production conflict path).
- ``acquire()`` over a foreign-hostname lockfile succeeds with no
  error (cross-host iCloud / NFS sync takeover).
- ``release()`` deletes the lockfile and is idempotent.
- ``force_unlock()`` returns the prior payload so the CLI can show
  PID / hostname / started_at in its diagnostic output.

Tests use ``tmp_path`` and an explicit ``lock_path`` argument so the
production ``~/.iai-mcp/.locked`` file is never touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from iai_mcp.lifecycle_lock import (
    LifecycleLock,
    LifecycleLockConflict,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# A. Clean state -> acquire writes fresh
# ---------------------------------------------------------------------------


def test_acquire_in_clean_state(tmp_path: Path) -> None:
    """No lockfile present -> ``acquire`` writes a complete payload."""
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)

    lock.acquire()

    assert lock_path.exists()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert isinstance(payload["hostname"], str) and payload["hostname"]
    assert isinstance(payload["started_at"], str) and payload["started_at"]
    assert payload["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# B. Existing lockfile, dead PID, same host -> takeover succeeds
# ---------------------------------------------------------------------------


def test_acquire_when_existing_lock_dead_pid_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale lockfile from a crashed daemon must not block boot."""
    lock_path = tmp_path / ".locked"
    # Pre-populate with a "dead" PID. Use 1 (init) and patch the
    # liveness check to report it dead -- using 1 directly is risky
    # because it IS alive on every Unix host. Patching the helper is
    # the deterministic isolation pattern.
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999_999,  # implausible PID; further isolated by patch
                "hostname": "Some-Other-Mac.local",  # different from runtime
                "started_at": "2026-04-30T15:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    # Force same hostname so the takeover hits the dead-PID branch
    # (foreign hostname would also take over, but for different reasons).
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "Some-Other-Mac.local")
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: False)

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == "Some-Other-Mac.local"


# ---------------------------------------------------------------------------
# C. Existing lockfile, live PID, same host -> conflict raised
# ---------------------------------------------------------------------------


def test_acquire_when_existing_lock_live_pid_same_host_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live daemon on the same host blocks a second boot attempt."""
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 12_345,
                "hostname": "test-host.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: True)

    lock = LifecycleLock(lock_path)
    with pytest.raises(LifecycleLockConflict) as exc_info:
        lock.acquire()

    # The exception carries the existing payload so the caller can
    # print PID + started_at without a second disk read.
    assert exc_info.value.existing is not None
    assert exc_info.value.existing["pid"] == 12_345
    assert exc_info.value.existing["hostname"] == "test-host.local"
    # Lockfile content unchanged: conflict must NOT clobber the
    # existing payload (otherwise we lose forensic data).
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == 12_345


# ---------------------------------------------------------------------------
# D. Existing lockfile, foreign hostname -> silent takeover
# ---------------------------------------------------------------------------


def test_acquire_when_existing_lock_different_hostname_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon on a different host (iCloud / NFS sync scenario) is
    treated as "not relevant" and the local boot wins.

    Rationale: the original host's daemon cannot share Unix-socket
    state with us over a sync filesystem, so two daemons on two hosts
    sharing one ``~/.iai-mcp/`` is already broken; the only safe
    behaviour is "new host wins" so the user can use the second
    machine without manual cleanup.
    """
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 12_345,
                "hostname": "Other-Mac.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    # Local hostname differs from the on-disk one.
    monkeypatch.setattr(ll, "_current_hostname", lambda: "This-Mac.local")
    # Even if the foreign PID happens to be live (recycled on this host),
    # the hostname mismatch alone must trigger takeover.
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: True)

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == "This-Mac.local"


# ---------------------------------------------------------------------------
# E. release() deletes the file; idempotent
# ---------------------------------------------------------------------------


def test_release_deletes_file(tmp_path: Path) -> None:
    """``release`` removes the lockfile; calling twice is not an error."""
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    lock.acquire()
    assert lock_path.exists()

    lock.release()
    assert not lock_path.exists()

    # Idempotent.
    lock.release()
    assert not lock_path.exists()


# ---------------------------------------------------------------------------
# F. is_held_by_self()
# ---------------------------------------------------------------------------


def test_is_held_by_self_true_after_acquire(tmp_path: Path) -> None:
    """After ``acquire`` the helper returns True for this process."""
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    assert lock.is_held_by_self() is False  # nothing on disk yet

    lock.acquire()
    assert lock.is_held_by_self() is True


def test_is_held_by_self_false_when_pid_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the on-disk PID is a different process, helper returns False."""
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid() + 1,  # not us
                "hostname": "test-host.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")

    lock = LifecycleLock(lock_path)
    assert lock.is_held_by_self() is False


# ---------------------------------------------------------------------------
# G. force_unlock returns prior content
# ---------------------------------------------------------------------------


def test_force_unlock_returns_previous_content(tmp_path: Path) -> None:
    """``force_unlock`` deletes the file and returns the prior payload.

    Used by ``iai-mcp lifecycle force-unlock`` to surface PID +
    hostname + started_at in the diagnostic output.
    """
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "hostname": "stale-host.local",
                "started_at": "2026-04-29T08:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )

    lock = LifecycleLock(lock_path)
    previous = lock.force_unlock()

    assert previous is not None
    assert previous["pid"] == 4242
    assert previous["hostname"] == "stale-host.local"
    assert not lock_path.exists()


def test_force_unlock_when_no_lockfile(tmp_path: Path) -> None:
    """``force_unlock`` returns None when no lockfile exists; no error."""
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    assert lock.force_unlock() is None
    assert not lock_path.exists()


# ---------------------------------------------------------------------------
# H. Corrupt JSON is treated as "no lock" rather than raising
# ---------------------------------------------------------------------------


def test_acquire_overwrites_corrupt_lockfile(tmp_path: Path) -> None:
    """Operator hand-edit producing invalid JSON must not block boot."""
    lock_path = tmp_path / ".locked"
    lock_path.write_text("not-valid-json{{{")

    lock = LifecycleLock(lock_path)
    lock.acquire()  # should succeed, overwriting the garbage

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# I. read() returns None for missing / corrupt files
# ---------------------------------------------------------------------------


def test_read_returns_none_for_missing_file(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    assert lock.read() is None


def test_read_returns_none_for_corrupt_json(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text("garbage---")
    lock = LifecycleLock(lock_path)
    assert lock.read() is None


def test_read_returns_none_for_invalid_schema(tmp_path: Path) -> None:
    """Missing required field -> read returns None (treated as absent)."""
    lock_path = tmp_path / ".locked"
    # Missing 'started_at'.
    lock_path.write_text(
        json.dumps({"pid": 1, "hostname": "h", "schema_version": 1})
    )
    lock = LifecycleLock(lock_path)
    assert lock.read() is None


# ---------------------------------------------------------------------------
# J. File mode is 0o600 (consistent with project state-file convention)
# ---------------------------------------------------------------------------


def test_acquire_writes_mode_0600(tmp_path: Path) -> None:
    """The lockfile must be user-readable only (T-04-07 mitigation)."""
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    lock.acquire()

    mode = lock_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected mode 0o600, got 0o{mode:o}"
