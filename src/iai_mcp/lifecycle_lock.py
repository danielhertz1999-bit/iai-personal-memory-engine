"""Single-machine ``~/.iai-mcp/.locked`` lockfile.

The daemon writes ``~/.iai-mcp/.locked`` on startup with PID + hostname +
started_at. A second daemon attempt on the same host raises
``LifecycleLockConflict``; a daemon on a different host (e.g. via
iCloud / NFS sync of ``~/.iai-mcp``) detects the foreign hostname and
takes over with a warning.

The ``.locked`` lockfile is a higher-level, human-readable singleton marker
for the lifecycle state machine; it does NOT use ``fcntl.flock`` because
single-machine is the assumption and the JSON content (PID + hostname) is the
diagnostic surface that ``iai-mcp lifecycle force-unlock`` consumes.

Design constraints:

- stdlib only -- ``os``, ``socket``, ``json``, ``pathlib``, ``datetime``.
- POSIX-atomic write via ``tempfile.mkstemp`` + ``os.replace`` (same
  pattern as ``daemon_state.save_state`` / ``lifecycle_state.save_state``).
- 0o600 file mode -- consistent with the rest of the project's state files.
- Hostname recorded so iCloud / NFS sync of ``~/.iai-mcp`` does NOT
  produce a deadlock when the user moves to a second machine.
- PID-liveness check uses ``os.kill(pid, 0)`` (same trick as
  ``heartbeat_scanner._is_pid_alive``).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

# Module logger for the psutil-fallback debug entry inside _is_pid_alive.
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

def _default_lock_path() -> Path:
    """Resolve the default lockfile path, honoring ``IAI_MCP_STORE``.

    Tests + multi-tenant deployments override the iai-mcp data root via
    the ``IAI_MCP_STORE`` env var.
    Falling back to ``~/.iai-mcp`` keeps the production default
    untouched.
    """
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / ".locked"


# Production lock-file path. Re-resolved via the helper so monkey-
# patching ``IAI_MCP_STORE`` in tests redirects the production
# default automatically. Tests can also pass an explicit ``lock_path``
# argument to ``LifecycleLock``.
DEFAULT_LOCK_PATH: Path = _default_lock_path()

#: Schema version persisted alongside the payload so a future bump can
#: be detected at takeover time.
SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LifecycleLockConflict(RuntimeError):
    """Raised when ``acquire()`` finds a live daemon on the same host.

    The exception carries the existing lockfile content as a dict so the
    caller (daemon main, ``iai-mcp lifecycle force-unlock``) can surface
    PID / started_at to the operator without a second disk read.
    """

    def __init__(self, message: str, existing: "LockPayload | None" = None) -> None:
        super().__init__(message)
        self.existing = existing


# ---------------------------------------------------------------------------
# Typed payload schema
# ---------------------------------------------------------------------------


class LockPayload(TypedDict):
    """On-disk schema for ``.locked``."""

    pid: int
    hostname: str
    started_at: str   # ISO-8601 UTC
    schema_version: int


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp -- single point so tests can patch."""
    return datetime.now(timezone.utc).isoformat()


def _current_hostname() -> str:
    """Return ``socket.gethostname()``; central so tests can monkey-patch."""
    return socket.gethostname()


def _is_pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` is a live ``iai_mcp.daemon`` process.

    Two-stage check to guard against PID-recycle:

    1. ``os.kill(pid, 0)`` -- kernel-level liveness. Raises
       ``ProcessLookupError`` when the PID has been reaped;
       ``PermissionError`` (EPERM) means the process exists but is owned
       by another UID. EPERM is treated as alive (we cannot inspect the
       cmdline of a foreign-UID process; the lockfile is treated as live
       defensively rather than overwritten).
    2. ``psutil.Process(pid).cmdline()`` must contain the substring
       ``iai_mcp.daemon`` -- rules out a recycled PID belonging to an
       unrelated process (shell, browser, etc.). Mirrors the discipline
       in ``doctor``.

    Returns False when:
      - ``pid`` is non-positive.
      - ``os.kill(pid, 0)`` raises ``ProcessLookupError``.
      - ``psutil.Process(pid)`` / ``.cmdline()`` raises one of
        ``NoSuchProcess`` / ``AccessDenied`` / ``ZombieProcess`` (the PID
        vanished between ``os.kill`` and ``Process``, or psutil cannot
        read it -- treat as not-our-daemon so the lockfile is overwriteable).
      - ``cmdline`` does NOT contain ``iai_mcp.daemon``.

    Defensive future-proofing: if ``import psutil`` itself fails (the
    library is a hard dep today per ``pyproject.toml:23``, so this is
    not load-bearing), falls back to the previous ``os.kill``-only
    semantics with a debug log entry.
    """
    if pid <= 0:
        return False

    # Stage 1: kernel-level liveness via os.kill(pid, 0).
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM = PID exists but owned by another UID. We cannot inspect
        # the cmdline of a foreign-UID process; treat as live defensively.
        return True

    # Stage 2: psutil cmdline cross-check -- reject recycled PIDs whose
    # cmdline does not contain 'iai_mcp.daemon'.
    try:
        import psutil
    except ImportError:
        log.debug(
            "lifecycle_lock: psutil unavailable; falling back to "
            "os.kill-only liveness for pid=%d",
            pid,
        )
        return True

    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline() or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # PID was alive at os.kill but vanished / unreadable by the time
        # psutil opened it (race window OR foreign-UID process). Treat
        # as not-our-daemon: the lockfile is stale and may be force-released.
        return False
    except Exception:  # noqa: BLE001 -- defensive against psutil backend quirks
        log.debug(
            "lifecycle_lock: psutil.Process(%d).cmdline() raised "
            "unexpectedly; assuming live",
            pid,
            exc_info=True,
        )
        return True

    return "iai_mcp.daemon" in cmdline


def _validate_payload(raw: object) -> LockPayload:
    """Reject malformed JSON; return a typed copy on success.

    Schema check kept light -- enough to catch operator hand-edits and
    out-of-band writes from a stale schema version. We do NOT require
    ``schema_version`` to equal ``SCHEMA_VERSION``; a higher schema is
    treated as forward-compatible (the daemon refuses to overwrite it
    only if PID is alive on same host -- the conflict path).
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"lockfile payload must be a JSON object, got {type(raw).__name__}"
        )
    pid = raw.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"lockfile.pid must be a positive int, got {pid!r}")
    hostname = raw.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError(
            f"lockfile.hostname must be a non-empty string, got {hostname!r}"
        )
    started_at = raw.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError(
            f"lockfile.started_at must be a non-empty string, got {started_at!r}"
        )
    sv = raw.get("schema_version")
    if not isinstance(sv, int) or sv <= 0:
        raise ValueError(
            f"lockfile.schema_version must be a positive int, got {sv!r}"
        )
    return {
        "pid": pid,
        "hostname": hostname,
        "started_at": started_at,
        "schema_version": sv,
    }


# ---------------------------------------------------------------------------
# LifecycleLock
# ---------------------------------------------------------------------------


class LifecycleLock:
    """Single-machine lockfile for the lifecycle state machine.

    Construction is cheap; no I/O happens until ``acquire()`` is called.
    Tests instantiate with an explicit ``lock_path`` under ``tmp_path``
    so production state is never touched.
    """

    def __init__(self, lock_path: Path | None = None) -> None:
        # Resolve at construction time (not import time) so a test
        # that monkey-patches IAI_MCP_STORE before instantiating sees
        # the redirected path. Production callers pass no argument
        # and get the canonical ~/.iai-mcp/.locked.
        self._lock_path = (
            lock_path if lock_path is not None else _default_lock_path()
        )

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        """Filesystem location of the ``.locked`` file."""
        return self._lock_path

    def read(self) -> LockPayload | None:
        """Return the on-disk payload, or ``None`` if absent / corrupt.

        Corrupt-file behaviour is "no lock" rather than raising: an
        operator hand-edit that produces invalid JSON should not block
        a fresh daemon boot. ``acquire()`` will then overwrite the file.
        """
        if not self._lock_path.exists():
            return None
        try:
            raw = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return _validate_payload(raw)
        except ValueError:
            return None

    def is_held_by_self(self) -> bool:
        """True iff the on-disk lockfile names this process + this host.

        Used by the daemon to short-circuit a redundant ``acquire()``
        on a fast restart where the file was never released (e.g. a
        crash that bypassed the ``finally`` cleanup -- in that case
        the PID will not match either, so this returns False and
        ``acquire()`` does the dead-PID takeover).
        """
        payload = self.read()
        if payload is None:
            return False
        return (
            payload["pid"] == os.getpid()
            and payload["hostname"] == _current_hostname()
        )

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Write the lockfile, claiming the singleton slot for this process.

        Decision tree:

        1. No lockfile present -> write fresh.
        2. Lockfile present, corrupt JSON -> overwrite (treat as absent).
        3. Lockfile present, foreign hostname -> overwrite + log a warning
           (cross-host scenario via iCloud / NFS sync; daemon on the new
           host wins because the original host's daemon cannot reach
           this filesystem).
        4. Lockfile present, same hostname, dead PID -> overwrite (the
           previous daemon crashed before releasing).
        5. Lockfile present, same hostname, live PID -> ``raise
           LifecycleLockConflict`` (a real concurrent boot attempt).

        Atomic write via ``tempfile.mkstemp`` + ``os.replace`` -- same
        pattern as ``lifecycle_state.save_state`` / ``daemon_state.save_state``.
        """
        existing = self.read()
        if existing is not None:
            # Live PID on same host -> conflict.
            if existing["hostname"] == _current_hostname() and _is_pid_alive(
                existing["pid"]
            ):
                raise LifecycleLockConflict(
                    f"daemon already running: pid={existing['pid']} "
                    f"hostname={existing['hostname']} "
                    f"started_at={existing['started_at']}",
                    existing=existing,
                )
            # Dead PID OR foreign hostname -> takeover (no error). The
            # foreign-hostname branch corresponds to the cross-host
            # iCloud / NFS sync scenario; we silently overwrite because
            # the only viable remediation is "the new host wins"
            # (the original host's daemon cannot share state with us
            # over a sync filesystem, by definition).

        payload: LockPayload = {
            "pid": os.getpid(),
            "hostname": _current_hostname(),
            "started_at": _utc_now_iso(),
            "schema_version": SCHEMA_VERSION,
        }

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".locked.",
            suffix=".tmp",
            dir=str(self._lock_path.parent),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._lock_path)
        except (OSError, TypeError, ValueError):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def release(self) -> None:
        """Delete the lockfile. Idempotent -- absent file is not an error.

        Called from the daemon's graceful-shutdown ``finally`` block. A
        crash before this point leaves the file intact; the next
        ``acquire()`` will detect the dead PID and overwrite.
        """
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            return

    def force_unlock(self) -> LockPayload | None:
        """Delete the lockfile unconditionally; return the prior content.

        Operator-facing helper used by ``iai-mcp lifecycle force-unlock``
        when a daemon crashed before ``release()`` and the dead-PID
        takeover did not catch the case (e.g. the operator wants to
        clear a foreign-hostname lock without booting a daemon first).

        Returns the parsed prior payload (or ``None`` if absent /
        corrupt) so the caller can print PID / hostname / started_at
        in the diagnostic output.
        """
        previous = self.read()
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
        return previous
