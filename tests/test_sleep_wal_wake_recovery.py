"""WAL pending-entries detection at daemon startup."""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Standard project test isolation. Without this fixture
    the test will fail on the construction host because the OS keyring is
    unavailable."""
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


# --------------------------------------------------------------------------- tests


def test_wal_pending_entries_behavioral(tmp_path):
    """SleepWAL.pending_entries() returns [] when WAL file does not exist,
    and returns at least 1 entry after a begin() call with no commit.

    This exercises the mechanism the startup recovery relies on: a begin()
    with no subsequent commit() or rollback() leaves a pending entry that
    the startup recovery block will detect."""
    from iai_mcp.sleep_wal import SleepWAL

    wal = SleepWAL(path=tmp_path / ".sleep-wal.jsonl")

    # No WAL file yet — must return empty list
    assert wal.pending_entries() == []

    # Write a pending entry (no commit → remains pending)
    wal.begin(operation="optimize_drop", target_ids=["test-id"])

    # At least one entry must be pending
    assert len(wal.pending_entries()) >= 1


def test_grep_production_callsite_exists():
    """Regression guard: daemon.py MUST contain the
    'sleep_wal_pending_recovered' event emit string.

    If this string is removed from daemon.py, the WAL startup recovery
    has been de-wired and this test will fail, catching the regression."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    daemon_src = repo_root / "src" / "iai_mcp" / "daemon.py"
    if not daemon_src.exists():
        pytest.skip("daemon.py source not on disk (sdist/wheel install)")
    text = daemon_src.read_text()
    assert "sleep_wal_pending_recovered" in text, (
        "Regression: 'sleep_wal_pending_recovered' event emit not found in "
        "daemon.py. WAL startup recovery may have been removed."
    )
