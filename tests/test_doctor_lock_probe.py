"""Positive tests for the doctor (c) store-lock health probe.

Self-hermetic: each test sets its own ``IAI_MCP_STORE`` to a tmp dir, so the
probe resolves the store lock under tmp and never touches a real store. The
probe is read-only and must report HEALTHY whether the lock is absent, idle
(acquirable), or held (a consolidation process / active recall holds it).
"""
from __future__ import annotations

import fcntl

import pytest

from iai_mcp.doctor import check_c_lock_healthy


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Point IAI_MCP_STORE at a tmp dir and create its hippo subdir."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo_dir = tmp_path / "hippo"
    hippo_dir.mkdir(parents=True, exist_ok=True)
    return hippo_dir


def test_absent_lock_is_healthy(tmp_store):
    """No .lock file (fresh install / store never opened) -> PASS."""
    lock_path = tmp_store / ".lock"
    assert not lock_path.exists()

    result = check_c_lock_healthy()

    assert result.passed is True
    assert result.name == "(c) lock file healthy"
    assert "not yet initialized" in result.detail


def test_idle_acquirable_lock_is_healthy(tmp_store):
    """An existing, unlocked .lock file -> PASS (store idle)."""
    lock_path = tmp_store / ".lock"
    lock_path.write_bytes(b"")
    assert lock_path.exists()

    result = check_c_lock_healthy()

    assert result.passed is True
    assert result.name == "(c) lock file healthy"
    assert "acquirable" in result.detail


def test_held_lock_is_healthy(tmp_store):
    """A .lock file held EXCLUSIVE by another fd -> PASS (EWOULDBLOCK branch).

    Simulates the consolidation process holding the store lock: the probe's
    non-blocking shared acquire fast-fails with EWOULDBLOCK, which the check
    classifies as a normal, healthy held state.
    """
    lock_path = tmp_store / ".lock"
    lock_path.write_bytes(b"")

    held_fd = None
    try:
        with open(lock_path, "r") as held:
            held_fd = held.fileno()
            fcntl.flock(held_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            result = check_c_lock_healthy()

            assert result.passed is True
            assert result.name == "(c) lock file healthy"
            assert "held" in result.detail
            # Release while the fd is still open so teardown is clean.
            fcntl.flock(held_fd, fcntl.LOCK_UN)
    finally:
        # The `with` block already closed the fd; nothing else to release.
        pass


def test_probe_does_not_create_lock_file(tmp_store):
    """The probe opens O_RDONLY (no O_CREAT) — an absent lock stays absent."""
    lock_path = tmp_store / ".lock"
    assert not lock_path.exists()

    check_c_lock_healthy()

    assert not lock_path.exists(), "probe must not fabricate a lock file"
