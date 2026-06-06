"""Regression test documenting schema migration idempotency.

All migration blocks in Store.__init__ check column existence before calling add_columns().
This test fails if any migration block loses its existence guard.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Standard project test isolation for OS keyring.

    Without this fixture the test will fail on the construction host because
    the OS keyring is unavailable."""
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


# --------------------------------------------------------------------------- test


def test_schema_migration_already_idempotent(tmp_path):
    """Opening the same MemoryStore twice at the same path must produce identical
    column sets with no error and no duplicate-add.

    The migration blocks in Store.__init__ guard each add_columns() call with a
    column-existence check (``if "col" not in tbl.schema.names``). This test
    documents that invariant so future column additions cannot accidentally omit
    the guard and regress the behaviour.
    """
    from pathlib import Path  # noqa: F401 (imported for clarity; tmp_path is already a Path)

    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    # --- first open: tables are created, migration columns are added ---
    store1 = MemoryStore(path=tmp_path)
    tbl1 = store1.db.open_table(RECORDS_TABLE)
    cols_after_first_open = set(tbl1.schema.names)

    # --- second open: same directory, migration blocks must be no-ops ---
    store2 = MemoryStore(path=tmp_path)
    tbl2 = store2.db.open_table(RECORDS_TABLE)
    cols_after_second_open = set(tbl2.schema.names)

    # Column set must be identical — no new columns, none dropped
    assert cols_after_first_open == cols_after_second_open, (
        f"Column sets diverged between first and second open.\n"
        f"  First open : {sorted(cols_after_first_open)}\n"
        f"  Second open: {sorted(cols_after_second_open)}\n"
        f"  Added      : {sorted(cols_after_second_open - cols_after_first_open)}\n"
        f"  Removed    : {sorted(cols_after_first_open - cols_after_second_open)}"
    )

    # All six migration columns must be present after both opens
    migration_columns = (
        "tombstoned_at",  # migration block 1
        "schema_bypass",  # migration block 2 (part 1)
        "labile_until",   # migration block 2 (part 2)
        "wing",           # migration block 3 (part 1)
        "room",           # migration block 3 (part 2)
        "drawer",         # migration block 3 (part 3)
    )
    for col in migration_columns:
        assert col in cols_after_second_open, (
            f"Expected column {col!r} missing after second open"
        )
