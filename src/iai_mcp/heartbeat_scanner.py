"""Daemon-side heartbeat scanner (per-wrapper, PID-scoped).

Reads ``~/.iai-mcp/wrappers/heartbeat-<pid>-<uuid>.json`` files written by
each MCP wrapper instance, validates freshness (``now - last_refresh <= M``)
AND PID liveness (``os.kill(pid, 0)``), and aggregates presence so the daemon's
state machine can decide WAKE vs BEDTIME.

Constraints:
- Idle CPU near zero — scanner runs on lifecycle TICK (every 30s), not faster.
- Scanner code is reentrant: ``scan()`` MUST be safe to call concurrently with
  a wrapper writing a heartbeat file (atomic rename pattern + JSON-parse-fail
  fallback to file mtime).
- No new third-party dependencies — stdlib only.
- macOS-first; Linux subset works the same; Windows is unsupported.

Heartbeat file schema (written by wrapper, read here)::

    {
      "pid": 12345,
      "uuid": "01HZQ...",
      "started_at": "2026-05-02T15:00:00Z",
      "last_refresh": "2026-05-02T15:14:30Z",
      "wrapper_version": "1.0.0",
      "schema_version": 1
    }

Status semantics:
- FRESH: ``last_refresh`` within ``M`` seconds AND PID alive.
- STALE: ``last_refresh`` older than ``M`` seconds (regardless of PID).
- ORPHAN: PID is dead (``ProcessLookupError`` from ``kill(pid, 0)``) and the
          file's freshness window has not yet expired. Treated as not-active.

A file that fails JSON parse falls back to its filesystem mtime so a torn
half-written write does not silently mask presence.

"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


# Module-level constants -------------------------------------------------------

#: Default refresh staleness threshold (seconds). A heartbeat older than this
#: is STALE regardless of PID liveness. The wrapper SHOULD refresh every
#: ``REFRESH_INTERVAL_SEC`` — three missed refreshes (~90 s)
#: trip staleness.
DEFAULT_STALE_THRESHOLD_SEC = 90

#: Window for the "no fresh activity in last 30 minutes" predicate consumed
#: by the L6 ``IdleDetector.sleep_eligible`` rule.
IDLE_WINDOW_SEC = 30 * 60

#: Filename glob used to enumerate heartbeat files. Matches the
#: ``heartbeat-<pid>-<uuid>.json`` convention.
_HEARTBEAT_GLOB = "heartbeat-*.json"


class HeartbeatStatus(Enum):
    """Tri-state classification of a single heartbeat file."""

    FRESH = "fresh"
    STALE = "stale"
    ORPHAN = "orphan"


@dataclass
class HeartbeatEntry:
    """One scanned heartbeat file with its derived status.

    Attributes:
        path: Absolute path of the heartbeat file on disk.
        pid: Wrapper PID parsed from the file's payload.
        uuid: Wrapper UUID parsed from the file's payload (used as a stable
            tie-breaker when the same PID is reused after wrapper restart).
        last_refresh: Timezone-aware UTC datetime parsed from
            ``last_refresh``; falls back to file mtime if JSON parse fails.
        status: One of ``HeartbeatStatus.{FRESH, STALE, ORPHAN}``.
    """

    path: Path
    pid: int
    uuid: str
    last_refresh: datetime
    status: HeartbeatStatus


# PID liveness ----------------------------------------------------------------


def _is_pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` exists in the kernel's process table.

    Uses the ``kill(pid, 0)`` POSIX trick — sends no signal but raises
    ``ProcessLookupError`` (ESRCH) when the PID has been reaped. A
    ``PermissionError`` (EPERM) means the process exists but the current
    user cannot signal it — for liveness purposes we count that as alive.
    A negative or zero ``pid`` is treated as dead (those values would map
    to ``kill(self_pgrp, 0)`` semantics which is not what we want).
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# Atomic-read-with-mtime-fallback helper --------------------------------------


def _parse_heartbeat_file(path: Path) -> tuple[int, str, datetime] | None:
    """Best-effort parse of a single heartbeat file.

    Returns ``(pid, uuid, last_refresh_utc)`` on success or ``None`` if the
    file disappeared mid-read (race with wrapper rotation) or its content
    cannot be coerced into the minimum schema.

    A JSON-parse failure falls back to the file's mtime so that a torn
    write produced by a wrapper crash mid-rename is treated as STALE-on-
    age rather than silently dropped, satisfying the "reentrant + safe under
    concurrent writers" requirement.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Torn write — fall back to filename PID + filesystem mtime so we
        # at least get a STALE classification rather than dropping the file.
        return _fallback_parse_from_filename(path)

    pid = payload.get("pid")
    uuid_str = payload.get("uuid", "")
    last_refresh_raw = payload.get("last_refresh")

    if not isinstance(pid, int) or not isinstance(uuid_str, str):
        return _fallback_parse_from_filename(path)
    if not isinstance(last_refresh_raw, str):
        return _fallback_parse_from_filename(path)

    try:
        # ``2026-05-02T15:14:30Z`` — Python 3.11+ accepts the trailing Z;
        # for safety we normalize to ``+00:00`` for older 3.10 compatibility.
        normalized = last_refresh_raw.replace("Z", "+00:00")
        last_refresh = datetime.fromisoformat(normalized)
    except ValueError:
        return _fallback_parse_from_filename(path)

    if last_refresh.tzinfo is None:
        last_refresh = last_refresh.replace(tzinfo=timezone.utc)
    else:
        last_refresh = last_refresh.astimezone(timezone.utc)

    return pid, uuid_str, last_refresh


def _fallback_parse_from_filename(path: Path) -> tuple[int, str, datetime] | None:
    """Recover ``(pid, uuid, mtime_utc)`` from filename + filesystem stat.

    Filename convention: ``heartbeat-<pid>-<uuid>.json``. We split on ``-``
    once for ``heartbeat`` and once for the PID, joining the remainder as
    the UUID (UUIDs may contain dashes).
    """
    name = path.stem  # heartbeat-<pid>-<uuid>
    parts = name.split("-", 2)
    if len(parts) != 3 or parts[0] != "heartbeat":
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    uuid_str = parts[2]
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    return pid, uuid_str, datetime.fromtimestamp(mtime, tz=timezone.utc)


# HeartbeatScanner -------------------------------------------------------------


class HeartbeatScanner:
    """Aggregates per-wrapper heartbeat files into a daemon-side presence signal.

     standalone module — wires this into the daemon
    main-loop TICK to dispatch HEARTBEAT_REFRESH / IDLE state events.
    """

    def __init__(
        self,
        wrappers_dir: Path,
        stale_threshold_sec: int = DEFAULT_STALE_THRESHOLD_SEC,
    ) -> None:
        self._wrappers_dir = wrappers_dir
        self._stale_threshold_sec = stale_threshold_sec
        self._last_scan: list[HeartbeatEntry] = []

    # ----- Scan / classify -----------------------------------------------

    def scan(self) -> list[HeartbeatEntry]:
        """Read all heartbeat files, classify each, and return entries.

        Reentrant: tolerates concurrent writes by ignoring files that vanish
        mid-read and falling back to mtime when JSON is half-written.

        Empty / missing wrappers dir → empty list (the daemon hasn't seen
        any wrappers yet, which is a valid steady state on a fresh install).
        """
        entries: list[HeartbeatEntry] = []
        if not self._wrappers_dir.exists():
            self._last_scan = entries
            return entries

        try:
            candidates = list(self._wrappers_dir.glob(_HEARTBEAT_GLOB))
        except OSError:
            self._last_scan = entries
            return entries

        now = datetime.now(timezone.utc)
        for path in candidates:
            parsed = _parse_heartbeat_file(path)
            if parsed is None:
                # File vanished mid-glob (cleanup race) — skip silently.
                continue
            pid, uuid_str, last_refresh = parsed

            age_sec = (now - last_refresh).total_seconds()
            is_alive = _is_pid_alive(pid)

            if age_sec > self._stale_threshold_sec:
                # Stale wins over orphan — the file is too old to trust
                # regardless of whether its PID happens to still be live.
                status = HeartbeatStatus.STALE
            elif not is_alive:
                status = HeartbeatStatus.ORPHAN
            else:
                status = HeartbeatStatus.FRESH

            entries.append(
                HeartbeatEntry(
                    path=path,
                    pid=pid,
                    uuid=uuid_str,
                    last_refresh=last_refresh,
                    status=status,
                )
            )

        self._last_scan = entries
        return entries

    # ----- Aggregations consumed by the state machine --------------------

    def fresh_count(self) -> int:
        """Number of heartbeats classified as FRESH on the most recent scan.

        Re-runs ``scan()`` so callers don't have to remember to invoke it
        first; the cost is one filesystem walk per call which is negligible
        at TICK cadence (every 30 s).
        """
        return sum(1 for e in self.scan() if e.status is HeartbeatStatus.FRESH)

    def is_active(self) -> bool:
        """True iff at least one wrapper is currently FRESH.

        This is the primary signal the state machine uses to dispatch
        HEARTBEAT_REFRESH (→ WAKE) vs. begin the IDLE-eligibility check.
        """
        return self.fresh_count() >= 1

    def heartbeat_idle_30min(self) -> bool:
        """True iff no FRESH heartbeats existed in the last 30 minutes.

        Consumed by ``IdleDetector.sleep_eligible`` as one of the three
        disjuncts that gate L6 sleep. "No FRESH in window" is implemented
        as: scan now, and if zero entries are FRESH, the window is empty.
        STALE / ORPHAN entries imply the wrapper has not refreshed for at
        least the staleness threshold (90 s by default), so a single scan
        suffices — we don't keep a history buffer in this module.
        """
        # Fresh count == 0 means no wrapper is currently active. Combined
        # with the 30-min wall-clock window enforced by the daemon's TICK
        # rhythm and the L6 idle predicate's hardware backstop (HIDIdleTime
        # ≥ 1800 s), this gives the same observable behavior as a separate
        # 30-minute history without keeping in-memory state.
        return self.fresh_count() == 0

    # ----- Cleanup -------------------------------------------------------

    def cleanup_stale_orphans(self) -> int:
        """Delete heartbeat files classified STALE or ORPHAN. Returns count deleted.

        Best-effort: a delete that races with another process unlinking the
        same file (``FileNotFoundError``) is counted as a successful
        cleanup; any other ``OSError`` is swallowed so a single problematic
        file cannot break the rest of the cleanup pass.
        """
        deleted = 0
        for entry in self.scan():
            if entry.status is HeartbeatStatus.FRESH:
                continue
            try:
                entry.path.unlink()
                deleted += 1
            except FileNotFoundError:
                # Already unlinked (concurrent wrapper rotation / sibling
                # daemon scan). Count as cleaned — the file is gone.
                deleted += 1
            except OSError:
                # Permission / FS error on a single file: skip it, keep
                # going. The doctor row will surface persistent
                # cleanup failures via "n=X stale" delta on next run.
                continue
        return deleted
