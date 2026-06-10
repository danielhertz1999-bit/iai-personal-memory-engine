from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake_store.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None))
    yield fake_store


def _make_record(text="hello", language="en", schema_version=3):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": "x", "cue": "y", "session_id": "z"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language=language,
        schema_version=schema_version,
        structure_hv=b"",
    )


def _seed_pre_migration_store(tmp_path, monkeypatch, n=20):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    records = []
    for i in range(n):
        rec = _make_record(text=f"row-{i}", schema_version=3)
        store.insert(rec)
        records.append(rec)
    return store, records


def test_migration_function_exists():
    from iai_mcp import migrate

    assert hasattr(migrate, "migrate_hd_vector_to_structure_hv_v3_to_v4")
    assert callable(migrate.migrate_hd_vector_to_structure_hv_v3_to_v4)


def test_migration_populates_structure_hv_and_bumps_schema_version(tmp_path, monkeypatch):
    store, records = _seed_pre_migration_store(tmp_path, monkeypatch, n=20)
    from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4
    from iai_mcp.types import STRUCTURE_HV_BYTES

    result = migrate_hd_vector_to_structure_hv_v3_to_v4(store)
    assert isinstance(result, dict)
    assert "processed" in result
    assert "updated" in result
    assert result["updated"] == 20
    assert result["processed"] == 20

    for rec in records:
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.schema_version == 4
        assert len(fetched.structure_hv) == STRUCTURE_HV_BYTES


def test_migration_is_idempotent(tmp_path, monkeypatch):
    store, _ = _seed_pre_migration_store(tmp_path, monkeypatch, n=10)
    from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4

    first = migrate_hd_vector_to_structure_hv_v3_to_v4(store)
    second = migrate_hd_vector_to_structure_hv_v3_to_v4(store)
    assert first["updated"] == 10
    assert second["updated"] == 0
    assert second["skipped"] >= 10


def test_migration_preserves_literal_surface_bytes(tmp_path, monkeypatch):
    store, records = _seed_pre_migration_store(tmp_path, monkeypatch, n=5)
    from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4

    pre_literals = {rec.id: rec.literal_surface for rec in records}
    migrate_hd_vector_to_structure_hv_v3_to_v4(store)
    for rid, literal in pre_literals.items():
        fetched = store.get(rid)
        assert fetched is not None
        assert fetched.literal_surface == literal


def test_migration_emits_audit_event(tmp_path, monkeypatch):
    store, _ = _seed_pre_migration_store(tmp_path, monkeypatch, n=3)
    from iai_mcp.events import query_events
    from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4

    migrate_hd_vector_to_structure_hv_v3_to_v4(store)
    events = query_events(store, kind="migration_v3_to_v4", limit=10)
    assert len(events) >= 1
    e = events[0]
    data = e["data"]
    for key in ("processed", "updated", "skipped", "duration_ms"):
        assert key in data, f"missing event payload key {key!r}"
    assert data["updated"] == 3


def test_migration_dry_run_does_not_mutate(tmp_path, monkeypatch):
    store, records = _seed_pre_migration_store(tmp_path, monkeypatch, n=4)
    from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4

    result = migrate_hd_vector_to_structure_hv_v3_to_v4(store, dry_run=True)
    assert result["updated"] == 4

    for rec in records:
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.schema_version == 3


def test_migration_uses_uuid_literal_guard(tmp_path, monkeypatch):
    store, _ = _seed_pre_migration_store(tmp_path, monkeypatch, n=2)
    from iai_mcp import store as store_mod

    call_count = {"n": 0}
    real_uuid_literal = store_mod._uuid_literal

    def spy(value):
        call_count["n"] += 1
        return real_uuid_literal(value)

    monkeypatch.setattr(store_mod, "_uuid_literal", spy)
    from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4

    migrate_hd_vector_to_structure_hv_v3_to_v4(store)
    assert call_count["n"] >= 2
