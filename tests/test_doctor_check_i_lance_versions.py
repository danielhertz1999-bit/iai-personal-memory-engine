"""Regression tests for doctor row (i) hippo db size.

PASS: size < 500 MB. WARN: 500 MB.. 2048 MB. FAIL: >= 2048 MB.

The check reads the brain.sqlite3 stat via IAI_MCP_STORE/hippo/brain.sqlite3
(env-var first, ~/.iai-mcp fallback). Tests redirect IAI_MCP_STORE at a
tmp_path to avoid touching the user's real store.

Status mapping is asserted both via direct call and via run_diagnosis().
The wire-in test uses name-based lookup rather than positional / count
assertions so future doctor-row additions do not break this regression test.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# Direct check_i tests
# ----------------------------------------------------------------------


def test_pass_when_db_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No brain.sqlite3 -> PASS (fresh install)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "PASS"
    assert result.passed is True
    assert "not present" in result.detail


def test_pass_small_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Small file (< 500 MB) -> PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db = hippo / "brain.sqlite3"
    db.write_bytes(b"x" * 1024)  # 1 KB
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "PASS"
    assert result.passed is True
    assert "MB" in result.detail
    assert "healthy" in result.detail


class _FakePathSize:
    """Fake path-like object for doctoring stat responses."""

    def __init__(self, size_bytes: int, *, raise_stat: bool = False):
        self._size = size_bytes
        self._raise_stat = raise_stat

    def exists(self) -> bool:
        return True

    def stat(self):
        if self._raise_stat:
            raise OSError("permission denied")

        class _R:
            pass

        r = _R()
        r.st_size = self._size  # type: ignore[attr-defined]
        return r

    def __truediv__(self, other):
        return self

    def __str__(self):
        return "/fake/brain.sqlite3"


def test_warn_at_500mb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """File at 500 MB boundary -> WARN with compact-hippo hint."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathSize(500 * 1024 * 1024),
    )

    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "WARN"
    assert result.passed is True
    assert "compact-hippo" in result.detail


def test_fail_at_2048mb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """File at 2048 MB -> FAIL with immediate compaction instruction."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathSize(2048 * 1024 * 1024),
    )

    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "run compaction immediately" in result.detail


def test_warn_on_stat_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError on stat -> WARN (probe failure is advisory)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathSize(0, raise_stat=True),
    )

    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "WARN"
    assert result.passed is True
    assert "stat failed" in result.detail


# ----------------------------------------------------------------------
# run_diagnosis wire-in
# ----------------------------------------------------------------------


def test_run_diagnosis_includes_hippo_db_size_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_diagnosis() includes row (i) hippo db size."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    matching = [r for r in results if r.name == "(i) hippo db size"]
    assert len(matching) == 1, (
        f"expected exactly one '(i) hippo db size' row; "
        f"got {len(matching)} from {[r.name for r in results]}"
    )


def test_run_diagnosis_hippo_row_pass_on_clean_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With IAI_MCP_STORE pointing at a fresh tmp dir, (i) reports PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    matching = [r for r in results if r.name == "(i) hippo db size"]
    assert len(matching) == 1
    assert matching[0].status == "PASS"
    assert matching[0].passed is True
