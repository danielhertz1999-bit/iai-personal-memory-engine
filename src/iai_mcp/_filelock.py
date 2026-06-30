"""Platform-agnostic file locking shim.

On POSIX: thin wrapper around fcntl.flock.
On Windows: Win32 ``LockFileEx`` / ``UnlockFileEx`` via ctypes, with errno
normalisation so callers checking errno.EWOULDBLOCK / errno.EAGAIN on
non-blocking failures work unchanged. ``LockFileEx`` locks a byte range on the
file handle itself (independent of the file position), so — unlike the previous
``msvcrt.locking`` backend — there is no need to seek to 0 or save/restore the
file offset.

Shared locks ARE truly shared on Windows. ``LockFileEx`` supports both shared
(reader) and exclusive (writer) byte-range locks, so LOCK_SH lets multiple
concurrent readers in, matching fcntl.flock semantics. (The older msvcrt
backend could only take exclusive locks, so LOCK_SH silently behaved as
LOCK_EX — that throughput limitation is now gone.)

Remaining divergence — lock *conversion* is not atomic on Windows.
fcntl.flock converts a held lock in place: calling LOCK_SH on an fd that
already holds LOCK_EX (or LOCK_EX on one holding LOCK_SH) atomically swaps the
mode without ever releasing. ``LockFileEx`` has no in-place conversion — the
old range must be unlocked and a new one re-locked. The two conversion call
sites in ``hippo/_db.py`` (``downgrade_to_shared`` and ``escalate_to_exclusive``)
are handled by detecting an already-held lock on the fd and routing through an
Unlock-then-Lock. This opens a brief race window where another waiter could
acquire the lock between the unlock and the re-lock. Both call sites already
tolerate this: ``escalate_to_exclusive`` retries against a deadline, and
``downgrade_to_shared`` returns on OSError. This is strictly better than the
old behaviour (where SH was silently exclusive); a fully race-free port would
still need those call sites reworked to a conversion-free protocol.
"""
from __future__ import annotations

import os
import platform

if platform.system() == "Windows":
    import ctypes as _ctypes
    import errno as _errno
    import msvcrt as _msvcrt
    import time as _time
    from ctypes import wintypes as _wintypes

    # msvcrt is used only to translate a CRT fd into the OS file HANDLE that
    # LockFileEx/UnlockFileEx operate on.
    _msvcrt_get_osfhandle = _msvcrt.get_osfhandle

    LOCK_SH = 1
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    # Lock the maximum possible byte range so every caller contends on the
    # same region regardless of file size. fcntl.flock locks the whole file;
    # this 64-bit-wide range is the LockFileEx equivalent.
    _LOCK_LOW = 0xFFFFFFFF
    _LOCK_HIGH = 0xFFFFFFFF

    # dwFlags for LockFileEx.
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001

    # Win32 error codes returned via GetLastError on a failed non-blocking lock.
    _ERROR_LOCK_VIOLATION = 33
    _ERROR_IO_PENDING = 997

    # Poll interval when emulating POSIX's block-until-acquired behaviour for
    # the conversion path (see below).
    _BLOCK_POLL_SECONDS = 0.05

    _kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)

    class _OVERLAPPED(_ctypes.Structure):
        _fields_ = [
            ("Internal", _ctypes.c_void_p),
            ("InternalHigh", _ctypes.c_void_p),
            ("Offset", _wintypes.DWORD),
            ("OffsetHigh", _wintypes.DWORD),
            ("hEvent", _wintypes.HANDLE),
        ]

    _LockFileEx = _kernel32.LockFileEx
    _LockFileEx.restype = _wintypes.BOOL
    _LockFileEx.argtypes = [
        _wintypes.HANDLE,   # hFile
        _wintypes.DWORD,    # dwFlags
        _wintypes.DWORD,    # dwReserved
        _wintypes.DWORD,    # nNumberOfBytesToLockLow
        _wintypes.DWORD,    # nNumberOfBytesToLockHigh
        _ctypes.POINTER(_OVERLAPPED),
    ]

    _UnlockFileEx = _kernel32.UnlockFileEx
    _UnlockFileEx.restype = _wintypes.BOOL
    _UnlockFileEx.argtypes = [
        _wintypes.HANDLE,   # hFile
        _wintypes.DWORD,    # dwReserved
        _wintypes.DWORD,    # nNumberOfBytesToUnlockLow
        _wintypes.DWORD,    # nNumberOfBytesToUnlockHigh
        _ctypes.POINTER(_OVERLAPPED),
    ]

    # Per-fd record of whether THIS process currently holds a lock on the fd,
    # so a second lock call (a conversion request) can be routed through an
    # unlock first. fcntl.flock does the conversion in-kernel; we have to track
    # it ourselves because LockFileEx would otherwise deadlock against our own
    # held lock. Keyed by os.open fd; cleared on LOCK_UN.
    _HELD: dict[int, bool] = {}

    def _handle_for(fd: int) -> int:
        return _msvcrt_get_osfhandle(fd)

    def _new_overlapped() -> _OVERLAPPED:
        ov = _OVERLAPPED()
        ov.Offset = 0
        ov.OffsetHigh = 0
        ov.hEvent = 0
        return ov

    def _do_unlock(handle: int) -> None:
        ov = _new_overlapped()
        _UnlockFileEx(handle, 0, _LOCK_LOW, _LOCK_HIGH, _ctypes.byref(ov))

    def _do_lock(handle: int, flags: int) -> bool:
        ov = _new_overlapped()
        ok = _LockFileEx(
            handle, flags, 0, _LOCK_LOW, _LOCK_HIGH, _ctypes.byref(ov)
        )
        return bool(ok)

    def flock(fd: int, operation: int) -> None:
        if not isinstance(fd, int):
            fd = fd.fileno()
        handle = _handle_for(fd)

        if operation & LOCK_UN:
            _do_unlock(handle)
            _HELD.pop(fd, None)
            return

        if not (operation & (LOCK_EX | LOCK_SH)):
            return

        flags = 0
        if operation & LOCK_EX:
            flags |= _LOCKFILE_EXCLUSIVE_LOCK
        if operation & LOCK_NB:
            flags |= _LOCKFILE_FAIL_IMMEDIATELY

        # Conversion: this fd already holds a lock and is now asking for a
        # different mode. LockFileEx cannot convert in place and would block
        # forever against our own lock, so release first, then re-acquire.
        # This is the documented non-atomic window (see module docstring).
        if _HELD.get(fd):
            _do_unlock(handle)
            _HELD.pop(fd, None)

        if operation & LOCK_NB:
            if _do_lock(handle, flags):
                _HELD[fd] = True
                return
            raise OSError(
                _errno.EWOULDBLOCK, "resource temporarily unavailable"
            )

        # Blocking acquire. LockFileEx without FAIL_IMMEDIATELY blocks until the
        # lock is granted, matching POSIX flock. (We still loop defensively in
        # case a spurious failure is reported, polling like the old backend.)
        while True:
            if _do_lock(handle, flags):
                _HELD[fd] = True
                return
            _time.sleep(_BLOCK_POLL_SECONDS)

else:
    import fcntl as _fcntl

    LOCK_SH = _fcntl.LOCK_SH
    LOCK_EX = _fcntl.LOCK_EX
    LOCK_NB = _fcntl.LOCK_NB
    LOCK_UN = _fcntl.LOCK_UN

    def flock(fd: int, operation: int) -> None:
        _fcntl.flock(fd, operation)


__all__ = ["flock", "LOCK_EX", "LOCK_NB", "LOCK_SH", "LOCK_UN"]
