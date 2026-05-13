"""-- persistent capture queue with atomic append + idempotent ingest.

The capture queue is the durable buffer that makes the L1 hibernation contract
viable. Wrapper writes to ``~/.iai-mcp/pending/`` whenever the daemon socket
is unreachable (Hibernation, mid-restart, crashed). On the next Wake transition
the daemon drains the queue via ``ingest_pending(handler)`` -- the handler
plugs into the existing ``iai_mcp.capture`` path so the verbatim contract
 is preserved end-to-end.

Storage layout under ``~/.iai-mcp/pending/``::

    pending-<ulid>.json   -- one queued record (committed file)
    pending-<ulid>.json.tmp -- transient temp file before atomic rename
    pending-<ulid>.lock   -- present only during in-flight ingest of <ulid>
    .overflow-audit.log   -- JSONL append-only log of dropped-oldest events

Hard guarantees:

- **Atomic append**: writes go to ``.tmp`` then ``os.replace`` to final name
  (POSIX atomic rename). A crash mid-write leaves a stray ``.tmp`` but never
  a half-written final file. ``pending_count`` and ``list_pending`` ignore
  ``.tmp``.
- **Idempotent ingest**: each pending file is claimed via ``fcntl.flock`` on
  the matching ``.lock`` file. Lock contention => skip (another worker has
  it). Handler success => delete pending + lock atomically. Handler raises
  => leave both intact for next-call retry.
- **Bounded queue**: ``append`` triggers ``prune_oldest`` once
  ``pending_count > max_size``. Drops the oldest ``max_size - 9_900`` files
  in one batch (amortised I/O) and writes one JSONL line per drop to the
  audit log.
- **Verbatim round-trip**: the JSON payload uses ``ensure_ascii=False`` so
  ``record["surface"]`` round-trips byte-identically including UTF-8 BMP +
  astral characters and combining marks.
- **No new deps**: stdlib only -- ``os, pathlib, json, uuid, fcntl, secrets,
  time, datetime, threading, errno``.

ULID derivation: 48-bit millisecond unix timestamp (big-endian) + 80 bits of
``secrets.token_bytes`` randomness, encoded with Crockford base32 per the
ulid spec (https://github.com/ulid/spec). The result is 26 characters,
lexicographically sortable by time, and collision-resistant for thousands of
appends per millisecond. Implemented inline -- the project deliberately
avoids a ``python-ulid`` dependency.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------------

DEFAULT_QUEUE_DIR: Path = Path.home() / ".iai-mcp" / "pending"
"""Production location for the persistent queue."""

DEFAULT_MAX_SIZE: int = 10_000
"""Default ceiling before ``prune_oldest`` kicks in."""

# Drop ~100 oldest at once when overflowing so the I/O cost is amortised
# across many subsequent appends rather than paid on every single overflow.
_PRUNE_BATCH_HEADROOM: int = 100

SCHEMA_VERSION: int = 1
"""Bumped only when the on-disk pending-<ulid>.json layout changes."""

_AUDIT_LOG_NAME: str = ".overflow-audit.log"

# Crockford base32 alphabet (no I, L, O, U) per ulid spec.
_CROCKFORD: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CaptureQueueError(Exception):
    """Base class for all capture-queue errors."""


class CaptureQueueSchemaError(CaptureQueueError):
    """Raised when a pending file declares a ``schema_version`` we don't grok."""


class CaptureQueueLocked(CaptureQueueError):
    """Raised when an in-flight ingest cannot acquire the per-record lock.

    Currently only used internally; ``ingest_pending`` swallows lock contention
    and treats the file as "claimed by another worker" rather than raising.
    """


# ---------------------------------------------------------------------------
# ULID generator (stdlib-only, time-sortable)
# ---------------------------------------------------------------------------

# Monotonic-ish guard: if two ULIDs would land in the same millisecond, bump
# the timestamp by 1ms so lexicographic sort matches insertion order. The
# bump is bounded -- once wall clock advances past the bumped value the
# guard resets. Threadsafe via a module-level lock.
_ulid_lock = threading.Lock()
_last_ms: int = 0


def _now_ms() -> int:
    """Current wall-clock time in unix milliseconds (UTC)."""
    return int(time.time() * 1000)


def _b32_encode(data: bytes, length: int) -> str:
    """Crockford base32 encode ``data`` to exactly ``length`` characters.

    ``data`` is treated as an unsigned big-endian integer. Result is
    zero-padded on the left if the integer would naturally render to
    fewer characters. Caller is responsible for sizing ``length``
    correctly: 10 chars for the 48-bit timestamp prefix, 16 chars for
    the 80-bit randomness suffix.
    """
    n = int.from_bytes(data, "big")
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def generate_ulid() -> str:
    """Return a fresh 26-character Crockford-base32 ULID.

    The first 10 chars encode the millisecond unix timestamp; the next 16
    encode 80 bits of random data. Lexicographic sort = chronological sort
    (with millisecond resolution; finer ordering within a millisecond is
    not guaranteed by ULID itself but the monotonic guard below preserves
    insertion order in practice).
    """
    global _last_ms
    with _ulid_lock:
        ms = _now_ms()
        if ms <= _last_ms:
            ms = _last_ms + 1
        _last_ms = ms

    ts_bytes = ms.to_bytes(6, "big")  # 48 bits
    rand_bytes = secrets.token_bytes(10)  # 80 bits
    return _b32_encode(ts_bytes, 10) + _b32_encode(rand_bytes, 16)


# ---------------------------------------------------------------------------
# CaptureQueue
# ---------------------------------------------------------------------------


class CaptureQueue:
    """Persistent on-disk FIFO buffer for ``memory_capture`` records.

    See module docstring for storage layout and guarantees.
    """

    def __init__(
        self,
        queue_dir: Path | None = None,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._queue_dir = (
            Path(queue_dir) if queue_dir is not None else DEFAULT_QUEUE_DIR
        )
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._audit_log = self._queue_dir / _AUDIT_LOG_NAME

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def queue_dir(self) -> Path:
        """Filesystem location of the queue directory."""
        return self._queue_dir

    @property
    def max_size(self) -> int:
        """Maximum number of pending records before overflow pruning kicks in."""
        return self._max_size

    @property
    def audit_log_path(self) -> Path:
        """Path to ``.overflow-audit.log`` (may not exist if no overflows happened)."""
        return self._audit_log

    def pending_count(self) -> int:
        """Return number of committed pending files (ignores ``.tmp`` and ``.lock``)."""
        return sum(1 for _ in self._iter_pending_files())

    def list_pending(self) -> list[Path]:
        """Return committed pending files sorted by ULID (oldest first)."""
        return sorted(self._iter_pending_files(), key=lambda p: p.name)

    def _iter_pending_files(self):
        """Yield every ``pending-<ulid>.json`` (no ``.tmp``, no ``.lock``)."""
        for entry in self._queue_dir.iterdir():
            name = entry.name
            if (
                entry.is_file()
                and name.startswith("pending-")
                and name.endswith(".json")
                and not name.endswith(".json.tmp")
            ):
                yield entry

    # ------------------------------------------------------------------
    # Append (atomic temp + rename)
    # ------------------------------------------------------------------

    def append(self, record: dict) -> str:
        """Append a record to the queue. Returns the assigned ULID.

        Atomic: writes ``pending-<ulid>.json.tmp`` then ``os.replace`` to
        ``pending-<ulid>.json``. A crash between write and rename leaves a
        stray ``.tmp`` (cleaned up by future ``prune_oldest`` if it ever
        looks at the directory listing -- but ``pending_count`` already
        ignores it). Triggers ``prune_oldest`` once the post-append count
        exceeds ``max_size``.
        """
        if not isinstance(record, dict):
            raise TypeError(f"record must be a dict, got {type(record).__name__}")

        ulid = generate_ulid()
        appended_at = datetime.now(timezone.utc).isoformat()
        envelope: dict = {
            "ulid": ulid,
            "appended_at": appended_at,
            "record": record,
            "schema_version": SCHEMA_VERSION,
        }

        final_path = self._queue_dir / f"pending-{ulid}.json"
        tmp_path = self._queue_dir / f"pending-{ulid}.json.tmp"

        # Open with O_CREAT|O_EXCL|O_WRONLY so a colliding ULID is detected
        # rather than silently overwriting (collision => generate_ulid bug).
        # 0o600 keeps records user-only on disk.
        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            payload = json.dumps(
                envelope,
                ensure_ascii=False,  # verbatim Unicode round-trip
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(fd, payload)
            os.fsync(fd)
        except Exception:
            # On any failure between open and rename, drop the temp file so
            # we don't accumulate orphans. If the unlink itself fails (very
            # unlikely on a file we just created) re-raise the original.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)

        # POSIX-atomic rename: visible-or-not, never half-visible.
        os.replace(tmp_path, final_path)

        # Overflow check happens AFTER the rename so the new record is
        # never the one we drop -- prune_oldest by definition drops the
        # oldest, not the newest.
        if self.pending_count() > self._max_size:
            target = max(0, self._max_size - _PRUNE_BATCH_HEADROOM)
            self.prune_oldest(target_size=target)

        return ulid

    # ------------------------------------------------------------------
    # Ingest (idempotent, lock-claimed)
    # ------------------------------------------------------------------

    def ingest_pending(self, handler: Callable[[dict], None]) -> int:
        """Drain pending records via ``handler``. Returns count successfully ingested.

        For each pending file (oldest first):

        1. ``open`` ``pending-<ulid>.lock`` (creating if needed).
        2. ``fcntl.flock(LOCK_EX | LOCK_NB)`` -- if already locked, skip.
        3. Read + JSON-decode ``pending-<ulid>.json``; raise
           ``CaptureQueueSchemaError`` on schema mismatch.
        4. Call ``handler(record)`` where ``record`` is the inner dict
           (not the envelope).
        5. On success: ``unlink`` the pending file FIRST (so a crash
           between unlink calls cannot resurrect a deleted record), then
           release the lock and unlink the lock file.
        6. On handler exception: release the lock fd but leave the lock
           file AND the pending file on disk. Future calls retry.

        Schema errors propagate to the caller after closing fds for the
        offending file -- we do NOT swallow them, because a schema bump
        is a deploy-time event the caller needs to see.
        """
        if not callable(handler):
            raise TypeError("handler must be callable")

        ingested = 0
        for pending_path in self.list_pending():
            ulid = self._ulid_from_path(pending_path)
            lock_path = self._queue_dir / f"pending-{ulid}.lock"

            # Open (or create) the lock file. 0o600 to keep it user-only.
            try:
                lock_fd = os.open(
                    str(lock_path),
                    os.O_WRONLY | os.O_CREAT,
                    0o600,
                )
            except OSError:
                # Cannot even create the lock -- skip this record. Leave
                # the pending file in place so a future retry can pick
                # it up once the disk situation clears.
                continue

            try:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    # EWOULDBLOCK / EAGAIN => another worker has the lock.
                    # Anything else: surface it; we don't expect it here.
                    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                        continue
                    raise

                # Lock acquired. The pending file may have been deleted
                # between list_pending and now (rare race with another
                # worker that claimed-and-finished), so re-check.
                if not pending_path.exists():
                    continue

                envelope = self._read_envelope(pending_path)
                # Schema check -- raise loud so deploys notice.
                version = envelope.get("schema_version")
                if version != SCHEMA_VERSION:
                    raise CaptureQueueSchemaError(
                        f"unsupported schema_version={version!r} in "
                        f"{pending_path.name}; expected {SCHEMA_VERSION}",
                    )

                record = envelope["record"]
                # Handler runs OUTSIDE any try/except below: if it raises,
                # we explicitly leave the pending file + lock file on disk
                # for the next call to retry.
                handler(record)

                # Handler returned cleanly: delete pending FIRST to make
                # the success durable; lock cleanup is best-effort.
                try:
                    os.unlink(pending_path)
                except FileNotFoundError:
                    # Already gone -- another worker raced us. Treat as
                    # success since the record is no longer pending.
                    pass
                ingested += 1
            finally:
                # Always release + unlink the lock fd. If the handler
                # raised, the bare ``finally`` runs before the exception
                # propagates, so the lock fd never leaks.
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
                # Only unlink the lock file if we ALSO unlinked the pending
                # file (i.e. a clean handler success). On handler exception
                # we want the lock file to remain so a follow-up
                # ``ingest_pending`` can detect mid-flight crash state.
                if not pending_path.exists():
                    try:
                        os.unlink(lock_path)
                    except FileNotFoundError:
                        pass

        return ingested

    # ------------------------------------------------------------------
    # Overflow pruning
    # ------------------------------------------------------------------

    def prune_oldest(self, target_size: int | None = None) -> int:
        """Drop oldest pending files until count <= ``target_size``.

        ``target_size`` defaults to ``max_size`` -- in normal overflow flow
        ``append`` passes ``max_size - 100`` so the next 99 appends amortise
        the I/O cost. Each dropped file produces one JSONL line in
        ``.overflow-audit.log``.
        """
        if target_size is None:
            target_size = self._max_size
        if target_size < 0:
            raise ValueError(f"target_size must be >= 0, got {target_size}")

        oldest_first = self.list_pending()
        excess = len(oldest_first) - target_size
        if excess <= 0:
            return 0

        queue_size_before = len(oldest_first)
        dropped = 0
        for pending_path in oldest_first[:excess]:
            ulid = self._ulid_from_path(pending_path)
            try:
                envelope = self._read_envelope(pending_path)
                appended_at = envelope.get("appended_at", "")
            except (FileNotFoundError, json.JSONDecodeError, CaptureQueueError):
                # Read failure is non-fatal for pruning: we still drop the
                # file and log "unknown" appended_at to audit.
                appended_at = ""

            try:
                os.unlink(pending_path)
            except FileNotFoundError:
                # Someone else raced us (concurrent prune?) -- skip
                # without auditing since we didn't actually drop it.
                continue

            self._audit_drop(
                dropped_ulid=ulid,
                appended_at=appended_at,
                queue_size_before_prune=queue_size_before,
            )
            dropped += 1
        return dropped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ulid_from_path(path: Path) -> str:
        """Extract the ULID from a ``pending-<ulid>.json`` filename."""
        # ``stem`` for ``pending-XYZ.json`` is ``pending-XYZ``.
        return path.stem[len("pending-"):]

    @staticmethod
    def _read_envelope(path: Path) -> dict:
        """Read + JSON-decode a pending file. Raises ``json.JSONDecodeError``
        or ``FileNotFoundError`` on read failure; caller decides handling."""
        with path.open("rb") as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8"))

    def _audit_drop(
        self,
        *,
        dropped_ulid: str,
        appended_at: str,
        queue_size_before_prune: int,
    ) -> None:
        """Append one JSONL line to ``.overflow-audit.log``.

        Uses ``O_APPEND`` + ``flock`` for cross-process safety, mirroring
        ``LifecycleEventLog.append``. Failures are swallowed: the audit
        log is observability, not authoritative state -- a failed audit
        write must not abort the prune.
        """
        line = (
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "dropped_ulid": dropped_ulid,
                    "appended_at": appended_at,
                    "reason": "queue_overflow",
                    "queue_size_before_prune": queue_size_before_prune,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        try:
            fd = os.open(
                str(self._audit_log),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
        except OSError:
            return
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)


__all__ = [
    "CaptureQueue",
    "CaptureQueueError",
    "CaptureQueueLocked",
    "CaptureQueueSchemaError",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_QUEUE_DIR",
    "SCHEMA_VERSION",
    "generate_ulid",
]
