"""Consolidation-intent flag helpers for the scoped Hippo lock model (REQ-4).

Intent flag: ``hippo/.consolidation-pending``

These helpers implement the yield protocol that prevents LOCK_EX starvation
when continuous LOCK_SH client holders are present.

Protocol (mirrored from RESEARCH Finding 2):
  1. Consolidator sets the intent flag (``set_consolidation_intent``).
  2. New LOCK_SH clients see the flag, back off, and do NOT acquire a new
     LOCK_SH (``acquire_client_shared_nb`` returns False when flag is set).
  3. Existing LOCK_SH holders see the flag on post-acquire recheck
     (``check_consolidation_intent``) and release promptly.
  4. Consolidator polls LOCK_EX|LOCK_NB until all LOCK_SH drain.
  5. Consolidator clears the flag after the consolidation window
     (``clear_consolidation_intent``).

These helpers operate on the ``hippo/.lock`` PATH argument (not hippo_dir) so
callers can work with the raw lock file path without constructing a HippoDB.

Note: ``acquire_client_shared_nb`` is a SINGLE non-blocking attempt — the
caller is responsible for the retry loop. This design allows callers to
interleave their own logic (post-acquire recheck, timeout tracking) without
nesting loops.

Thread-safety: all helpers are pure filesystem operations (exists / creat /
unlink) plus a single non-blocking fcntl call. No process-wide locks held.
"""
from __future__ import annotations

import errno
import fcntl
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent-flag helpers (operate on the *lock path*, not hippo_dir)
# ---------------------------------------------------------------------------


def _intent_path(lock_path: Path) -> Path:
    """Return the consolidation-intent flag path adjacent to the lock file."""
    return lock_path.parent / ".consolidation-pending"


def set_consolidation_intent(lock_path: Path) -> None:
    """Atomically touch hippo/.consolidation-pending.

    Idempotent: if the file already exists (concurrent call or left from a
    previous run) the call succeeds silently (FileExistsError swallowed).
    """
    flag = _intent_path(lock_path)
    flag.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        pass  # idempotent


def clear_consolidation_intent(lock_path: Path) -> None:
    """Unlink hippo/.consolidation-pending.

    Idempotent: if the file is already absent (concurrent clear) the call
    succeeds silently (FileNotFoundError swallowed).
    """
    flag = _intent_path(lock_path)
    try:
        flag.unlink()
    except FileNotFoundError:
        pass  # idempotent


def check_consolidation_intent(lock_path: Path) -> bool:
    """Return True if hippo/.consolidation-pending exists.

    Cheap existence check: used by clients as the precheck BEFORE attempting
    LOCK_SH|LOCK_NB AND as the post-acquire recheck after acquiring LOCK_SH
    (H1 TOCTOU fix: release if the flag was set between precheck and acquire).
    """
    return _intent_path(lock_path).exists()


# backward-compat alias used in some test and daemon call sites
is_consolidation_pending = check_consolidation_intent


def cleanup_stale_consolidation_intent(lock_path: Path) -> None:
    """Remove a stale hippo/.consolidation-pending left by a crashed daemon.

    Called at daemon boot before acquiring any locks. A stale flag would
    permanently block all LOCK_SH clients — cleaning it at boot restores
    normal operation. Logs a warning so the event is observable.
    """
    flag = _intent_path(lock_path)
    try:
        flag.unlink()
        logger.warning(
            "cleanup_stale_intent: removed stale %s — previous daemon may have crashed",
            flag,
        )
    except FileNotFoundError:
        pass  # normal: no stale flag


# ---------------------------------------------------------------------------
# Client-side LOCK_SH non-blocking acquire (single attempt)
# ---------------------------------------------------------------------------


def acquire_client_shared_nb(fd: int, lock_path: Path) -> bool:
    """Single non-blocking LOCK_SH attempt with intent-flag precheck.

    Contract:
    - Checks hippo/.consolidation-pending BEFORE the flock call.
      If set, returns False immediately (intent-honoring precheck).
    - Attempts fcntl.flock(fd, LOCK_SH | LOCK_NB).
      If it raises EAGAIN/EWOULDBLOCK (EX holder active), returns False.
    - If LOCK_SH is acquired, caller is responsible for the post-acquire
      recheck (H1 TOCTOU fix):
        1. ``check_consolidation_intent(lock_path)``
           → if True, release (``fcntl.flock(fd, LOCK_UN)``) and retry.
        2. Doing work under the lock.
        3. Releasing: ``fcntl.flock(fd, LOCK_UN)``.

    The post-acquire recheck is the caller's responsibility so the caller
    can observe the recheck firing (important for correctness testing).

    This is a SINGLE non-blocking attempt — the caller drives the retry loop
    with its own timing/budget logic so it can stay strictly below the 1.5 s SLO.

    Returns True if LOCK_SH was acquired, False otherwise.
    Raises OSError on unexpected flock errors (not EAGAIN/EWOULDBLOCK).
    """
    # Precheck: if consolidation is pending, do NOT acquire a new SH holder.
    if check_consolidation_intent(lock_path):
        return False

    # The flock syscall releases the GIL, creating the genuine TOCTOU window
    # where the intent flag can be set between the precheck (above) and a
    # successful lock acquisition. After this returns True, the caller MUST
    # do a post-acquire recheck (check_consolidation_intent → if True, release).
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return False
        raise


__all__ = [
    "set_consolidation_intent",
    "clear_consolidation_intent",
    "check_consolidation_intent",
    "is_consolidation_pending",
    "cleanup_stale_consolidation_intent",
    "acquire_client_shared_nb",
]
