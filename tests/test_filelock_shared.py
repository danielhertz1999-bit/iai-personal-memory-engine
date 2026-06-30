"""Cross-platform behavioural tests for the iai_mcp._filelock shim.

These assert true POSIX-flock semantics: concurrent shared locks coexist,
an exclusive lock excludes both readers and other writers, and a
non-blocking failure raises OSError with EWOULDBLOCK/EAGAIN. They pass on
POSIX (fcntl) and — critically — encode the contract the Windows LockFileEx
backend must honour. The old msvcrt backend, which served LOCK_SH as an
exclusive lock, would have failed ``test_two_shared_locks_coexist``.
"""
from __future__ import annotations

import errno
import os

import pytest

from iai_mcp._filelock import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock


def _open(path: str) -> int:
    return os.open(path, os.O_CREAT | os.O_RDWR, 0o600)


def test_two_shared_locks_coexist(tmp_path) -> None:
    """Two independent fds may both hold LOCK_SH at the same time."""
    p = str(tmp_path / "lockfile")
    a = _open(p)
    b = _open(p)
    try:
        flock(a, LOCK_SH | LOCK_NB)
        # The second shared acquire must NOT raise — readers share.
        flock(b, LOCK_SH | LOCK_NB)
        flock(a, LOCK_UN)
        flock(b, LOCK_UN)
    finally:
        os.close(a)
        os.close(b)


def test_exclusive_excludes_shared(tmp_path) -> None:
    """A held LOCK_EX blocks a non-blocking LOCK_SH from another fd."""
    p = str(tmp_path / "lockfile")
    a = _open(p)
    b = _open(p)
    try:
        flock(a, LOCK_EX | LOCK_NB)
        with pytest.raises(OSError) as exc:
            flock(b, LOCK_SH | LOCK_NB)
        assert exc.value.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
        flock(a, LOCK_UN)
    finally:
        os.close(a)
        os.close(b)


def test_shared_excludes_exclusive(tmp_path) -> None:
    """A held LOCK_SH blocks a non-blocking LOCK_EX from another fd."""
    p = str(tmp_path / "lockfile")
    a = _open(p)
    b = _open(p)
    try:
        flock(a, LOCK_SH | LOCK_NB)
        with pytest.raises(OSError) as exc:
            flock(b, LOCK_EX | LOCK_NB)
        assert exc.value.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
        flock(a, LOCK_UN)
    finally:
        os.close(a)
        os.close(b)


def test_unlock_releases(tmp_path) -> None:
    """After LOCK_UN another fd can take an exclusive lock."""
    p = str(tmp_path / "lockfile")
    a = _open(p)
    b = _open(p)
    try:
        flock(a, LOCK_EX | LOCK_NB)
        flock(a, LOCK_UN)
        # Now b should acquire exclusively without raising.
        flock(b, LOCK_EX | LOCK_NB)
        flock(b, LOCK_UN)
    finally:
        os.close(a)
        os.close(b)


def test_downgrade_in_place(tmp_path) -> None:
    """EXCLUSIVE -> SHARED conversion on the same fd, then a second reader.

    Mirrors hippo/_db.py downgrade_to_shared(): take LOCK_EX, convert to
    LOCK_SH on the same fd, and verify another fd can then also read-share.
    """
    p = str(tmp_path / "lockfile")
    a = _open(p)
    b = _open(p)
    try:
        flock(a, LOCK_EX | LOCK_NB)
        # Convert in place to shared (no explicit unlock between).
        flock(a, LOCK_SH)
        # A second reader must now be admitted.
        flock(b, LOCK_SH | LOCK_NB)
        flock(a, LOCK_UN)
        flock(b, LOCK_UN)
    finally:
        os.close(a)
        os.close(b)


def test_escalate_in_place(tmp_path) -> None:
    """SHARED -> EXCLUSIVE conversion on the same fd when sole holder.

    Mirrors hippo/_db.py escalate_to_exclusive(): take LOCK_SH, then convert
    to LOCK_EX on the same fd. With no other holders this must succeed, and a
    subsequent reader from another fd must then be excluded.
    """
    p = str(tmp_path / "lockfile")
    a = _open(p)
    b = _open(p)
    try:
        flock(a, LOCK_SH | LOCK_NB)
        flock(a, LOCK_EX | LOCK_NB)
        with pytest.raises(OSError) as exc:
            flock(b, LOCK_SH | LOCK_NB)
        assert exc.value.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
        flock(a, LOCK_UN)
    finally:
        os.close(a)
        os.close(b)
