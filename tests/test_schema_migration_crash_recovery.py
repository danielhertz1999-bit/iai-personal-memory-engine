"""Schema migration crash-recovery injection test.

Verifies that a partial migration interrupted by RuntimeError does not corrupt
the store and that the next open succeeds self-consistently. Injection: drop
the ``wing`` column (one of the reconciled V5 columns) from the on-disk schema
to simulate a pre-migration state, then monkeypatch ``HippoTable.add_columns``
to raise RuntimeError during the second MemoryStore open. The third open uses
the real ``add_columns`` to verify idempotent recovery: the full migration
column set is restored and baseline columns are intact.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Standard project test isolation — verbatim from
    tests/test_migrate_reembed_crash_safe.py. Without this fixture
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


# --------------------------------------------------------------------------- test


def test_schema_migration_crash_recovery(tmp_path, monkeypatch):
    """Simulates a partial schema migration crash and verifies store recovery.

    Steps:
    1. Open a clean store so the full V5 schema is on disk.
    2. Drop ``wing`` from the on-disk records table to simulate a pre-migration
       state. Monkeypatch ``HippoTable.add_columns`` to raise RuntimeError on
       the second open, so the reconcile path fails mid-migration.
    3. Third open uses the real ``add_columns`` (patch removed): idempotent
       reconcile re-adds the dropped column; assert full migration column set
       present and baseline columns intact.

    Non-vacuity: the injection fires on the second open only (call_count >= 1);
    without the patch the second open would complete silently and
    second_open_raised would be False.
    """
    from iai_mcp.hippo import HippoTable
    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    # ------------------------------------------------------------------ Step 1
    # Establish a clean fully-migrated store (all V5 columns present on disk).
    store1 = MemoryStore(path=tmp_path)
    tbl1 = store1.db.open_table(RECORDS_TABLE)
    assert "wing" in tbl1.schema.names, (
        "Precondition failed: 'wing' not in clean store schema"
    )

    # ------------------------------------------------------------------ Step 2
    # Simulate a pre-migration state: drop 'wing' from the on-disk schema.
    # 'wing' is a V5 spatial column handled by _reconcile_columns, so the
    # second MemoryStore open will detect it as missing and call add_columns.
    tbl1.drop_columns(["wing"])
    names_pre = store1.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" not in names_pre, "drop_columns must remove 'wing' from disk schema"

    # Inject a crash: monkeypatch HippoTable.add_columns to raise RuntimeError.
    # Use monkeypatch.context() so only this patch is reverted on context exit
    # while the autouse _isolated_keyring patches remain active for the third open.
    crash_mock = MagicMock(side_effect=RuntimeError("injected add_columns failure"))

    second_open_raised = False
    with monkeypatch.context() as m:
        m.setattr(HippoTable, "add_columns", crash_mock)
        # Second open: _reconcile_columns detects 'wing' missing, calls
        # add_columns (crash_mock), which raises. _reconcile_columns aggregates
        # the failure and re-raises as RuntimeError (FAIL-LOUD contract).
        try:
            MemoryStore(path=tmp_path)
        except RuntimeError:
            second_open_raised = True
    # Context exit restores HippoTable.add_columns to the real method.

    # ------------------------------------------------------------------ Step 3
    # Third open: real add_columns, idempotent reconcile re-adds 'wing'.
    store3 = MemoryStore(path=tmp_path)
    tbl3 = store3.db.open_table(RECORDS_TABLE)
    cols3 = set(tbl3.schema.names)

    # Baseline schema must be intact after crash-and-recovery.
    assert "id" in cols3, (
        "id column missing after crash-recovery open — store corrupted"
    )
    assert "tier" in cols3, (
        "tier column missing after crash-recovery open — store corrupted"
    )

    # Non-vacuity: the injection MUST have fired during the second open.
    assert crash_mock.call_count >= 1, (
        "add_columns was never called — injection is vacuous "
        "(column absence did not reach the add_columns branch)"
    )

    # Full migration column-set must be present after idempotent third-open.
    MIGRATION_COLS = {
        "tombstoned_at",
        "schema_bypass",
        "labile_until",
        "wing",
        "room",
        "drawer",
    }
    missing = MIGRATION_COLS - cols3
    assert not missing, (
        f"Migration columns missing post-crash recovery: {missing}"
    )

    # The second open must have raised (FAIL-LOUD contract).
    assert second_open_raised, (
        "Second open did not raise — the store absorbed the injected error. "
        "This is unexpected given the FAIL-LOUD migration contract."
    )
