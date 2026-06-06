"""Regression tests for Hippo-specific doctor rows (f), (i), (r), (s), (t).

Tests:
- (f) check_f_hippo_readable: PASS on fresh store, FAIL on missing/broken store.
- (i) check_i_hippo_db_size: reports size in MB; WARN > 500 MB threshold.
- (r) check_r_hippo_hnsw_loadable: loadable index PASSes; corrupt/absent WAR/FAILs.
- (s) check_s_hippo_schema_version: fresh store PASSes; mismatch WARNs.
- (t) check_t_hippo_compacted_freshness: recent event PASSes; absent WARNs.
- total count: run_diagnosis() returns 22 rows.
- identity_audit: no lance_storage_optimized reference remains.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Row (f): check_f_hippo_readable
# ---------------------------------------------------------------------------


def test_row_f_hippo_readable_clean_store(tmp_path, monkeypatch):
    """Fresh store environment -> row (f) PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import check_f_hippo_readable, run_diagnosis

    # Probe check directly; monkeypatch MemoryStore to succeed.
    from iai_mcp import doctor as _doctor

    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore",
        lambda: None,  # open succeeds (returns None, no exception)
    )
    result = check_f_hippo_readable()
    assert result.status == "PASS"
    assert result.passed is True
    assert "hippo storage readable" in result.name
    assert "Hippo storage opens without error" in result.detail


def test_row_f_hippo_readable_missing_file_fail(monkeypatch):
    """MemoryStore() raises -> row (f) FAIL."""
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore",
        _raise_runtime_error,
    )
    from iai_mcp.doctor import check_f_hippo_readable

    result = check_f_hippo_readable()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "open failed" in result.detail


def _raise_runtime_error():
    raise RuntimeError("simulated open failure")


# ---------------------------------------------------------------------------
# Row (i): check_i_hippo_db_size
# ---------------------------------------------------------------------------


def test_row_i_hippo_db_size_reported(tmp_path, monkeypatch):
    """Present file -> returns size in MB integer format in detail."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db = hippo / "brain.sqlite3"
    db.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MB

    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "PASS"
    assert "MB" in result.detail
    assert "healthy" in result.detail


class _FakePathWithSize:
    """Fake Path-like object that reports a fixed file size via.stat()."""

    def __init__(self, size_bytes: int, *, exists: bool = True, raise_stat: bool = False):
        self._size = size_bytes
        self._exists = exists
        self._raise_stat = raise_stat

    def exists(self) -> bool:
        return self._exists

    def stat(self):
        if self._raise_stat:
            raise OSError("permission denied")

        class _R:
            pass

        r = _R()
        r.st_size = self._size  # type: ignore[attr-defined]
        return r

    def __truediv__(self, other):
        return self  # records.hnsw resolution falls through here too

    def __str__(self):
        return "/fake/brain.sqlite3"


def test_row_i_hippo_db_size_warn_over_500mb(tmp_path, monkeypatch):
    """File >= 500 MB -> WARN with compact-hippo hint."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathWithSize(600 * 1024 * 1024),
    )
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "WARN"
    assert result.passed is True
    assert "compact-hippo" in result.detail


def test_row_i_hippo_db_size_fail_at_2048mb(tmp_path, monkeypatch):
    """File at 2048 MB -> FAIL with immediate compaction instruction."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathWithSize(2048 * 1024 * 1024),
    )
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "run compaction immediately" in result.detail


def test_row_i_hippo_db_size_warn_on_stat_oserror(tmp_path, monkeypatch):
    """OSError on stat -> WARN (probe failure is advisory)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathWithSize(0, raise_stat=True),
    )
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "WARN"
    assert result.passed is True
    assert "stat failed" in result.detail


# ---------------------------------------------------------------------------
# Row (r): check_r_hippo_hnsw_loadable
# ---------------------------------------------------------------------------


def test_row_r_hnsw_loadable_absent_warn(tmp_path, monkeypatch):
    """records.hnsw absent -> WARN (HippoDB will rebuild)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    # hippo/ dir but no records.hnsw
    (tmp_path / "hippo").mkdir()

    from iai_mcp.doctor import check_r_hippo_hnsw_loadable

    result = check_r_hippo_hnsw_loadable()
    assert result.status == "WARN"
    assert result.passed is True
    assert "absent" in result.detail


def test_row_r_hnsw_zero_bytes_fail(tmp_path, monkeypatch):
    """records.hnsw is zero bytes -> FAIL with corrupt message."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    hnsw = hippo / "records.hnsw"
    hnsw.write_bytes(b"")  # zero bytes

    from iai_mcp.doctor import check_r_hippo_hnsw_loadable

    result = check_r_hippo_hnsw_loadable()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "zero bytes" in result.detail


def test_row_r_hnsw_corrupted_fail(tmp_path, monkeypatch):
    """records.hnsw present but hnswlib.load_index raises -> FAIL with actionable message."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    hnsw = hippo / "records.hnsw"
    hnsw.write_bytes(b"this is not a valid hnsw index")

    from iai_mcp.doctor import check_r_hippo_hnsw_loadable

    result = check_r_hippo_hnsw_loadable()
    # Corrupt index -> load_index raises -> FAIL
    assert result.status == "FAIL"
    assert result.passed is False
    assert "rebuild" in result.detail


# ---------------------------------------------------------------------------
# Row (s): check_s_hippo_schema_version
# ---------------------------------------------------------------------------


def test_row_s_schema_version_match(tmp_path, monkeypatch):
    """Fresh store with correct schema_version -> PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db_path = hippo / "brain.sqlite3"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE _hippo_meta (key TEXT, value TEXT)")
    conn.execute("INSERT INTO _hippo_meta VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()

    from iai_mcp.doctor import check_s_hippo_schema_version

    result = check_s_hippo_schema_version()
    assert result.status == "PASS"
    assert result.passed is True
    assert "schema_version=1" in result.detail


def test_row_s_schema_drift_warn(tmp_path, monkeypatch):
    """Manually write schema_version=99 -> WARN with expected version in detail."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db_path = hippo / "brain.sqlite3"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE _hippo_meta (key TEXT, value TEXT)")
    conn.execute("INSERT INTO _hippo_meta VALUES ('schema_version', '99')")
    conn.commit()
    conn.close()

    from iai_mcp.doctor import check_s_hippo_schema_version

    result = check_s_hippo_schema_version()
    assert result.status == "WARN"
    assert result.passed is True
    assert "schema_version=99" in result.detail


def test_row_s_db_absent_pass(tmp_path, monkeypatch):
    """No brain.sqlite3 -> PASS (fresh install)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    from iai_mcp.doctor import check_s_hippo_schema_version

    result = check_s_hippo_schema_version()
    assert result.status == "PASS"
    assert "absent" in result.detail


# ---------------------------------------------------------------------------
# Row (t): check_t_hippo_compacted_freshness
# ---------------------------------------------------------------------------


def test_row_t_hippo_compaction_fresh_pass(tmp_path, monkeypatch):
    """Recent hippo_compacted event -> PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from datetime import datetime, timezone

    recent_ts = datetime.now(timezone.utc).isoformat()
    fake_event = {"kind": "hippo_compacted", "ts": recent_ts}

    # Patch the modules that check_t imports locally.
    import iai_mcp.store as _store
    import iai_mcp.events as _events

    class _FakeStore:
        pass

    monkeypatch.setattr(_store, "MemoryStore", _FakeStore)
    monkeypatch.setattr(
        _events, "query_events",
        lambda store, kind=None, limit=1: [fake_event],
    )

    from iai_mcp.doctor import check_t_hippo_compacted_freshness

    result = check_t_hippo_compacted_freshness()
    assert result.status == "PASS"
    assert result.passed is True
    assert "hippo_compacted" in result.name


def test_row_t_hippo_compaction_stale_warn(tmp_path, monkeypatch):
    """No hippo_compacted event -> WARN."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    import iai_mcp.store as _store
    import iai_mcp.events as _events

    class _FakeStore:
        pass

    monkeypatch.setattr(_store, "MemoryStore", _FakeStore)
    monkeypatch.setattr(
        _events, "query_events",
        lambda store, kind=None, limit=1: [],
    )

    from iai_mcp.doctor import check_t_hippo_compacted_freshness

    result = check_t_hippo_compacted_freshness()
    assert result.status == "WARN"
    assert result.passed is True
    assert "no hippo_compacted event" in result.detail


# ---------------------------------------------------------------------------
# Total row count: 21
# ---------------------------------------------------------------------------


def test_doctor_total_count_22(tmp_path, monkeypatch):
    """run_diagnosis() returns exactly 24 rows (a..v + (w) permanent-failed + (z) AVX2)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    assert len(results) == 24, (
        f"expected 24 rows; got {len(results)}: {[r.name for r in results]}"
    )


# ---------------------------------------------------------------------------
# No lance_storage_optimized reference in identity_audit
# ---------------------------------------------------------------------------


def test_no_lance_storage_optimized_in_identity_audit():
    """identity_audit.py must not reference lance_storage_optimized."""
    import inspect
    from iai_mcp import identity_audit

    src = inspect.getsource(identity_audit)
    assert "lance_storage_optimized" not in src, (
        "identity_audit.py still contains 'lance_storage_optimized'; "
        "it must use 'hippo_compacted' instead."
    )
    assert "optimize_lance_storage" not in src, (
        "identity_audit.py still imports optimize_lance_storage; "
        "it must use optimize_hippo_storage."
    )
