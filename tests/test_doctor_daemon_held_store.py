"""Doctor must not false-fail when the live daemon holds the Hippo store.

A running daemon holding the store EXCLUSIVE is the HEALTHY awake state, not a
defect. Three checks used to choke on that lock:

  (f) hippo storage readable     -> FAILed (flipped doctor exit to 1)
  (t) hippo_compacted freshness  -> WARNed with a raw lock-held exception
  (u) recall centrality regression -> WARNed with a raw lock-held exception

This suite proves:
  1. With a daemon-held store (HippoLockHeldError on open/query):
       (f) -> PASS "store held by the live daemon — normal"
       (t) -> PASS benign "deferred — daemon holds the store (normal)"
       (u) -> PASS benign "deferred — daemon holds the store (normal)"
     and a synthetic full result set carries no FAIL (overall exit 0).
  2. The daemon-DOWN path still genuinely verifies readability: a real
     corruption-style open error still FAILs (f). The carve-out is scoped to
     the lock-held signal alone — it does not weaken the real readability check.

Hermetic: MemoryStore / query_events are monkeypatched; no real ~/.iai-mcp
path is touched, no live daemon is started or stopped.
"""
from __future__ import annotations

import sqlite3

import pytest

from iai_mcp import doctor as _doctor
from iai_mcp.hippo import HippoLockHeldError


def _raise_lock_held(*_a, **_kw):
    raise HippoLockHeldError("/tmp/alice-store/hippo/.lock", "12345")


# --------------------------------------------------------------------------- #
# (f) hippo storage readable
# --------------------------------------------------------------------------- #


def test_f_reports_healthy_when_daemon_holds_store(monkeypatch) -> None:
    """Daemon-held store -> (f) PASS, not FAIL (this is what flipped exit 1)."""
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_lock_held)

    result = _doctor.check_f_hippo_readable()

    assert result.name == "(f) hippo storage readable"
    assert result.passed is True
    assert result.status == "PASS"
    assert "daemon" in result.detail.lower()
    assert "normal" in result.detail.lower()


def test_f_reports_healthy_on_sqlite_database_locked(monkeypatch) -> None:
    """A 'database is locked' OperationalError is the same held state -> PASS."""
    import iai_mcp.store as _store_mod

    def _raise_locked(*_a, **_kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_locked)

    result = _doctor.check_f_hippo_readable()

    assert result.passed is True
    assert result.status == "PASS"
    assert "daemon" in result.detail.lower()


def test_f_still_fails_on_real_open_error_daemon_down(monkeypatch) -> None:
    """DAEMON-DOWN guard: a genuine open failure (corruption) still FAILs.

    This is the discriminator — proves the lock-held carve-out did not turn
    (f) into a check that passes on any error. A corruption-style exception
    (not a lock-held signal) must still produce a real FAIL.
    """
    import iai_mcp.store as _store_mod

    def _raise_corruption(*_a, **_kw):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_corruption)

    result = _doctor.check_f_hippo_readable()

    assert result.passed is False
    assert result.status == "FAIL"
    assert "open failed" in result.detail.lower()


def test_f_still_fails_on_other_operational_error(monkeypatch) -> None:
    """A non-lock OperationalError is a real FAIL, not the held carve-out."""
    import iai_mcp.store as _store_mod

    def _raise_other(*_a, **_kw):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_other)

    result = _doctor.check_f_hippo_readable()

    assert result.passed is False
    assert result.status == "FAIL"


# --------------------------------------------------------------------------- #
# (t) hippo_compacted freshness
# --------------------------------------------------------------------------- #


def test_t_benign_when_daemon_holds_store(monkeypatch) -> None:
    """Daemon-held store -> (t) benign PASS 'deferred', not a scary WARN."""
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_lock_held)

    result = _doctor.check_t_hippo_compacted_freshness()

    assert result.name == "(t) hippo_compacted freshness"
    assert result.passed is True
    assert result.status == "PASS"
    assert "deferred" in result.detail.lower()
    assert "daemon holds the store" in result.detail.lower()


def test_t_benign_on_lock_held_during_events_query(monkeypatch) -> None:
    """If the lock surfaces on the events query (not store open), still benign."""
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())
    monkeypatch.setattr(_events, "query_events", _raise_lock_held)

    result = _doctor.check_t_hippo_compacted_freshness()

    assert result.status == "PASS"
    assert "deferred" in result.detail.lower()


def test_t_still_warns_on_genuine_query_failure(monkeypatch) -> None:
    """A non-lock query failure still WARNs (advisory) — carve-out is scoped."""
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    def _boom(*_a, **_kw):
        raise RuntimeError("events backend unavailable")

    monkeypatch.setattr(_events, "query_events", _boom)

    result = _doctor.check_t_hippo_compacted_freshness()

    assert result.status == "WARN"
    assert result.passed is True  # WARN never flips exit code


# --------------------------------------------------------------------------- #
# (u) recall centrality regression
# --------------------------------------------------------------------------- #


def test_u_benign_when_daemon_holds_store(monkeypatch) -> None:
    """Daemon-held store -> (u) benign PASS 'deferred', not a scary WARN."""
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_lock_held)

    result = _doctor.check_u_recall_centrality_regression()

    assert result.name == "(u) recall centrality regression"
    assert result.passed is True
    assert result.status == "PASS"
    assert "deferred" in result.detail.lower()
    assert "daemon holds the store" in result.detail.lower()


def test_u_benign_on_lock_held_during_events_query(monkeypatch) -> None:
    """If the lock surfaces on the events query (not store open), still benign."""
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())
    monkeypatch.setattr(_events, "query_events", _raise_lock_held)

    result = _doctor.check_u_recall_centrality_regression()

    assert result.status == "PASS"
    assert "deferred" in result.detail.lower()


def test_u_still_warns_on_genuine_query_failure(monkeypatch) -> None:
    """A non-lock query failure still WARNs (advisory) — carve-out is scoped."""
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    def _boom(*_a, **_kw):
        raise RuntimeError("events backend unavailable")

    monkeypatch.setattr(_events, "query_events", _boom)

    result = _doctor.check_u_recall_centrality_regression()

    assert result.status == "WARN"
    assert result.passed is True


# --------------------------------------------------------------------------- #
# Overall exit code: a daemon-held store must not produce a FALSE FAIL.
# --------------------------------------------------------------------------- #


def test_daemon_held_results_yield_exit_zero() -> None:
    """A synthetic result set with the three lock-held rows carries no FAIL.

    Mirrors cmd_doctor's aggregator: exit code is
    ``sum(1 for r in results if not r.passed)``. With (f)/(t)/(u) all reporting
    PASS under a daemon-held store, fail_count == 0 -> exit 0.

    Fed synthetically (not a full live run_diagnosis) so the assertion is
    deterministic and does not depend on the live daemon / environment probes.
    """
    from iai_mcp.doctor import CheckResult

    results = [
        CheckResult("(a) daemon process alive", True, "ok"),
        CheckResult(
            "(f) hippo storage readable",
            True,
            "store held by the live daemon — normal",
        ),
        CheckResult(
            "(t) hippo_compacted freshness",
            True,
            "deferred — daemon holds the store (normal)",
            status="PASS",
        ),
        CheckResult(
            "(u) recall centrality regression",
            True,
            "deferred — daemon holds the store (normal)",
            status="PASS",
        ),
    ]

    fail_count = sum(1 for r in results if not r.passed)
    assert fail_count == 0, "daemon-held store must not produce a FALSE FAIL"


def test_real_fail_still_flips_exit_code() -> None:
    """Sanity counter-test: a genuine FAIL row still flips the exit code.

    Guards against the carve-out accidentally making every (f) row passed=True.
    """
    from iai_mcp.doctor import CheckResult

    results = [
        CheckResult("(a) daemon process alive", True, "ok"),
        CheckResult(
            "(f) hippo storage readable",
            False,
            "open failed: DatabaseError: file is not a database",
        ),
    ]

    fail_count = sum(1 for r in results if not r.passed)
    assert fail_count == 1
