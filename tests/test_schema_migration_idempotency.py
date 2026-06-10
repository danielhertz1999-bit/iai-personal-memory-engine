
from __future__ import annotations

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

def test_schema_migration_already_idempotent(tmp_path):
    from pathlib import Path  # noqa: F401 (imported for clarity; tmp_path is already a Path)

    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    store1 = MemoryStore(path=tmp_path)
    tbl1 = store1.db.open_table(RECORDS_TABLE)
    cols_after_first_open = set(tbl1.schema.names)

    store2 = MemoryStore(path=tmp_path)
    tbl2 = store2.db.open_table(RECORDS_TABLE)
    cols_after_second_open = set(tbl2.schema.names)

    assert cols_after_first_open == cols_after_second_open, (
        f"Column sets diverged between first and second open.\n"
        f"  First open : {sorted(cols_after_first_open)}\n"
        f"  Second open: {sorted(cols_after_second_open)}\n"
        f"  Added      : {sorted(cols_after_second_open - cols_after_first_open)}\n"
        f"  Removed    : {sorted(cols_after_first_open - cols_after_second_open)}"
    )

    migration_columns = (
        "tombstoned_at",
        "schema_bypass",
        "labile_until",
        "wing",
        "room",
        "drawer",
    )
    for col in migration_columns:
        assert col in cols_after_second_open, (
            f"Expected column {col!r} missing after second open"
        )
