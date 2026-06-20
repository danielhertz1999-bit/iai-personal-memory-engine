"""Platform-agnostic file locking shim.

On POSIX: thin wrapper around fcntl.flock.
On Windows: msvcrt.locking with errno normalisation so callers checking
errno.EWOULDBLOCK / errno.EAGAIN on non-blocking failures work unchanged.
"""
from __future__ import annotations

import os
import platform

if platform.system() == "Windows":
    import errno as _errno
    import msvcrt as _msvcrt

    LOCK_SH = 1
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def flock(fd: int, operation: int) -> None:
        if not isinstance(fd, int):
            fd = fd.fileno()
        # msvcrt.locking locks bytes from the current file position.
        # Do NOT seek here: callers that need a specific offset already seek
        # themselves, and fcntl.flock (POSIX) never moves the offset — matching
        # that behaviour avoids surprising callers that read/write after locking.
        if operation & LOCK_UN:
            try:
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 2**30)
            except OSError:
                pass
        elif operation & (LOCK_EX | LOCK_SH):
            if operation & LOCK_NB:
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 2**30)
                except OSError:
                    raise OSError(
                        _errno.EWOULDBLOCK, "resource temporarily unavailable"
                    )
            else:
                # LK_LOCK retries for ~10 s then raises OSError.
                _msvcrt.locking(fd, _msvcrt.LK_LOCK, 2**30)

else:
    import fcntl as _fcntl

    LOCK_SH = _fcntl.LOCK_SH
    LOCK_EX = _fcntl.LOCK_EX
    LOCK_NB = _fcntl.LOCK_NB
    LOCK_UN = _fcntl.LOCK_UN

    def flock(fd: int, operation: int) -> None:
        _fcntl.flock(fd, operation)


__all__ = ["flock", "LOCK_EX", "LOCK_NB", "LOCK_SH", "LOCK_UN"]
