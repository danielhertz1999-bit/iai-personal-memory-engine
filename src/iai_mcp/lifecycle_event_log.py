"""-- JSONL event log for lifecycle state machine validation.

The lifecycle state machine needs an append-only event log to validate
transitions in shadow-run mode and to provide a post-mortem trail when
something misbehaves. The log is the empirical ground truth for "did the
machine compute the right state at the right moment", separate from the
live `lifecycle_state.json` snapshot.

Format: JSONL (one JSON record per line), file per UTC date, kept under
`~/.iai-mcp/logs/lifecycle-events-YYYY-MM-DD.jsonl`. Daily rotation
keyed off the UTC date of the appended event so writes near local
midnight do not silently fragment across two files in unpredictable
timezones. 30-day retention with gzip compression for older files
matches the retention spec.

Atomic line writes: each `append` opens the file with `O_APPEND |
O_CREAT` and uses `fcntl.flock(LOCK_EX)` to serialise concurrent writers
across processes. POSIX guarantees `O_APPEND` writes <= PIPE_BUF bytes
are atomic on local filesystems; the explicit lock keeps us safe past
that threshold (a single JSONL line for our event shapes is well under
PIPE_BUF=512, but the lock costs ~microseconds and saves us debugging
on the day a payload grows).
"""
from __future__ import annotations

import errno
import fcntl
import gzip
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Default location. Overridable via constructor `log_dir` for tests.
DEFAULT_LOG_DIR: Path = Path.home() / ".iai-mcp" / "logs"

# Event kinds emitted by the state machine and helpers; treat as the
# closed set for now — adding a kind requires updating downstream
# consumers (panel R7 validation script in a future phase).
KNOWN_EVENT_KINDS: frozenset[str] = frozenset(
    {
        "state_transition",
        "wrapper_event",
        "shadow_run_warning",
        "sleep_step_started",
        "sleep_step_completed",
        "quarantine_entered",
        "quarantine_lifted",
    }
)


def _utc_now() -> datetime:
    """Single point of `datetime.now(UTC)` -- patchable in tests."""
    return datetime.now(timezone.utc)


def _utc_date_string(dt: datetime | None = None) -> str:
    """Return the UTC date as `YYYY-MM-DD` for filename derivation."""
    moment = dt if dt is not None else _utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


class LifecycleEventLog:
    """Append-only JSONL event log with daily rotation + retention.

    Public surface:
        append(event)             -- write one event line, lock + fsync.
        rotate_old_files(...)     -- gzip files older than retention.
        current_file()            -- return path to today's log file.

    Thread/process safety: a per-call `fcntl.flock` on the destination
    file makes concurrent writers (daemon, hooks) safe. The lock is
    released as soon as the bytes hit disk; we do NOT keep a long-lived
    handle, so the file can rotate / be archived between calls without
    leaving a stale fd open.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir if log_dir is not None else DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path derivation
    # ------------------------------------------------------------------

    def file_for_date(self, date_str: str) -> Path:
        """Return the JSONL path for the given `YYYY-MM-DD` date string."""
        return self._log_dir / f"lifecycle-events-{date_str}.jsonl"

    def current_file(self, now: datetime | None = None) -> Path:
        """Return the path that `append` would write to right now."""
        return self.file_for_date(_utc_date_string(now))

    # ------------------------------------------------------------------
    # Appender
    # ------------------------------------------------------------------

    def append(self, event: dict[str, Any], now: datetime | None = None) -> None:
        """Append one event as a JSONL line; auto-rotate by UTC date.

        Adds `ts` (current UTC ISO-8601) if the caller did not pass one.
        Verifies `event["event"]` is a non-empty string but does NOT
        gate on `KNOWN_EVENT_KINDS` — adding a new kind should not
        require a code change to the log writer.

        Concurrency: held lock via `fcntl.flock(LOCK_EX)`. Crash mid
        write: the partial line is on disk because we are O_APPEND
        without buffering, but `fsync` keeps the *prior* lines
        durable. Readers MUST tolerate a truncated final line (trim
        or skip on JSON decode error).
        """
        if not isinstance(event, dict):
            raise TypeError(
                f"event must be a dict, got {type(event).__name__}"
            )
        kind = event.get("event")
        if not isinstance(kind, str) or not kind:
            raise ValueError("event['event'] must be a non-empty string")

        moment = now if now is not None else _utc_now()
        if "ts" not in event:
            # Mutate a shallow copy so the caller's dict stays clean.
            event = {"ts": moment.astimezone(timezone.utc).isoformat(), **event}

        line = json.dumps(event, separators=(",", ":")) + "\n"
        target = self.current_file(moment)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Open with O_APPEND so seeks land at EOF even under concurrent
        # write; flock for cross-process serialisation.
        fd = os.open(
            str(target),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # ------------------------------------------------------------------
    # Retention / rotation
    # ------------------------------------------------------------------

    def rotate_old_files(
        self,
        retention_days: int = 30,
        now: datetime | None = None,
    ) -> int:
        """Gzip log files whose UTC date is older than `retention_days`.

        Already-gzipped files (`*.jsonl.gz`) are left alone. Returns
        the number of files newly compressed in this call. Files older
        than `retention_days` that are *also* already gzipped are kept
        forever in this phase — the spec asks for compression after
        the window, not deletion. (Deletion is a future-phase decision.)
        """
        moment = now if now is not None else _utc_now()
        cutoff_date = (moment - timedelta(days=retention_days)).date()

        compressed = 0
        for path in self._log_dir.glob("lifecycle-events-*.jsonl"):
            stem = path.stem  # lifecycle-events-YYYY-MM-DD
            try:
                date_part = stem.rsplit("-", 3)[-3:]  # ['YYYY','MM','DD']
                file_date = datetime.strptime(
                    "-".join(date_part), "%Y-%m-%d"
                ).date()
            except (ValueError, IndexError):
                # Unrecognised filename — skip rather than guess.
                continue
            if file_date > cutoff_date:
                continue

            gz_path = path.with_suffix(".jsonl.gz")
            if gz_path.exists():
                # Idempotent: already compressed in a prior run.
                continue
            try:
                with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                # Match prior chmod to keep the tarball user-only.
                os.chmod(gz_path, 0o600)
                # Remove the plaintext only after the gzip is durable.
                os.unlink(path)
                compressed += 1
            except OSError as exc:
                # Best-effort: a single broken file should not stop
                # the next iterations.
                if exc.errno in (errno.EACCES, errno.EPERM):
                    continue
                # Unknown OSError — let the caller see it.
                raise
        return compressed

    # ------------------------------------------------------------------
    # Read helpers (non-essential but useful for tests + CLI)
    # ------------------------------------------------------------------

    def read_all(self, date_str: str | None = None) -> list[dict[str, Any]]:
        """Read all events from the file for `date_str` (or today).

        Skips truncated final lines silently — only fully-decoded JSON
        records are returned. Returns [] if the file does not exist.
        """
        target = self.file_for_date(
            date_str if date_str is not None else _utc_date_string()
        )
        if not target.exists():
            return []
        out: list[dict[str, Any]] = []
        with target.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
