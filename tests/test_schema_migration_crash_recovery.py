
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
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

def test_schema_migration_crash_recovery(tmp_path, monkeypatch):
    from iai_mcp.hippo import HippoTable
    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    store1 = MemoryStore(path=tmp_path)
    tbl1 = store1.db.open_table(RECORDS_TABLE)
    assert "wing" in tbl1.schema.names, (
        "Precondition failed: 'wing' not in clean store schema"
    )

    tbl1.drop_columns(["wing"])
    names_pre = store1.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" not in names_pre, "drop_columns must remove 'wing' from disk schema"

    crash_mock = MagicMock(side_effect=RuntimeError("injected add_columns failure"))

    second_open_raised = False
    with monkeypatch.context() as m:
        m.setattr(HippoTable, "add_columns", crash_mock)
        try:
            MemoryStore(path=tmp_path)
        except RuntimeError:
            second_open_raised = True

    store3 = MemoryStore(path=tmp_path)
    tbl3 = store3.db.open_table(RECORDS_TABLE)
    cols3 = set(tbl3.schema.names)

    assert "id" in cols3, (
        "id column missing after crash-recovery open — store corrupted"
    )
    assert "tier" in cols3, (
        "tier column missing after crash-recovery open — store corrupted"
    )

    assert crash_mock.call_count >= 1, (
        "add_columns was never called — injection is vacuous "
        "(column absence did not reach the add_columns branch)"
    )

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

    assert second_open_raised, (
        "Second open did not raise — the store absorbed the injected error. "
        "This is unexpected given the FAIL-LOUD migration contract."
    )
