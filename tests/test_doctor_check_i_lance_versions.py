"""[Wave2-Option-C] regression test for doctor row (i).

PASS: <=500 manifests. WARN: 501..2000. FAIL: >2000.

The check reads ``IAI_MCP_STORE/lancedb/records.lance/_versions/*.manifest``
(env-var first, ``~/.iai-mcp`` fallback). Tests redirect ``IAI_MCP_STORE``
at a tmp_path to avoid touching the user's real store.

Status mapping is asserted both via direct call and via ``run_diagnosis()``.
The wire-in test below uses name-based lookup rather than positional / count
assertions so future doctor-row additions (e.g. added rows m, n)
do not break this regression test.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def fake_versions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """IAI_MCP_STORE -> tmp_path, with records.lance/_versions/ pre-created.

    The check resolves ``IAI_MCP_STORE/lancedb/records.lance/_versions``;
    fixture creates the directory tree so seeding manifest files is direct.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    versions = tmp_path / "lancedb" / "records.lance" / "_versions"
    versions.mkdir(parents=True)
    return versions


def _seed(versions_dir: Path, count: int) -> None:
    """Create ``count`` distinct fake manifest files."""
    for i in range(count):
        (versions_dir / f"{i:020d}.manifest").write_bytes(b"x" * 10)


# ----------------------------------------------------------------------
# Direct check_i tests
# ----------------------------------------------------------------------


def test_pass_at_500(fake_versions_dir: Path) -> None:
    """500 manifests -> PASS (boundary inclusive)."""
    _seed(fake_versions_dir, 500)
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "PASS"
    assert result.passed is True
    assert "500" in result.detail


def test_pass_at_low_count(fake_versions_dir: Path) -> None:
    """100 manifests -> PASS (typical post-compaction state)."""
    _seed(fake_versions_dir, 100)
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "PASS"
    assert result.passed is True
    assert "100" in result.detail


def test_warn_at_1500(fake_versions_dir: Path) -> None:
    """1500 manifests -> WARN with compact-records hint; still passes the gate."""
    _seed(fake_versions_dir, 1500)
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "WARN"
    # WARN must NOT flip the exit code -- advisory only.
    assert result.passed is True
    assert "compact-records" in result.detail


def test_warn_boundary_at_2000(fake_versions_dir: Path) -> None:
    """2000 manifests -> WARN (boundary inclusive)."""
    _seed(fake_versions_dir, 2000)
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "WARN"
    assert result.passed is True


def test_fail_at_2500(fake_versions_dir: Path) -> None:
    """2500 manifests -> FAIL with daemon-stop recovery instructions."""
    _seed(fake_versions_dir, 2500)
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "daemon stop" in result.detail
    assert "compact-records" in result.detail


def test_fail_boundary_at_2001(fake_versions_dir: Path) -> None:
    """2001 manifests -> FAIL (boundary just over)."""
    _seed(fake_versions_dir, 2001)
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "FAIL"
    assert result.passed is False


def test_pass_when_dir_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No records.lance/_versions/ directory -> PASS (fresh install)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import check_i_lance_versions_count

    result = check_i_lance_versions_count()
    assert result.status == "PASS"
    assert result.passed is True
    assert "not present" in result.detail


# ----------------------------------------------------------------------
# run_diagnosis wire-in: row (i) is present and PASS on a clean store.
# Tests use name-based lookup rather than positional indexing so future
# row additions (added m + n) do not regress this check.
# ----------------------------------------------------------------------


def test_run_diagnosis_includes_lance_versions_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wire-in: run_diagnosis includes row (i) lance versions."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    matching = [
        r for r in results
        if "(i)" in r.name and "lance" in r.name.lower()
    ]
    assert len(matching) == 1, (
        f"expected exactly one (i) lance versions row in run_diagnosis(); "
        f"got {len(matching)} from {[r.name for r in results]}"
    )


def test_run_diagnosis_lance_row_pass_on_clean_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With IAI_MCP_STORE pointing at a fresh tmp dir, (i) reports PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    matching = [
        r for r in results
        if "(i)" in r.name and "lance" in r.name.lower()
    ]
    assert len(matching) == 1
    assert matching[0].status == "PASS"
    assert matching[0].passed is True
