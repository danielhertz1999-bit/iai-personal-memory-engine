"""v2 -> v3 encryption migration.

Covers:
- Migration re-encrypts plaintext sensitive columns in place
- Dry-run leaves disk untouched
- Idempotent: running the migration a second time is a no-op
- Migration event written to events table
- schema_version stays at 2 (encryption migration is a data upgrade, not a schema bump in this plan;
  but we track the state via an events row so the dry-run reports zero on a fully-encrypted store)
- helper is `migrate_encryption_v2_to_v3`
- events.data column is also encrypted during migration
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    """In-memory keyring for deterministic tests."""
    import keyring as _keyring

    store_for_test: dict[tuple[str, str], str] = {}

    def fake_get(service: str, username: str):
        return store_for_test.get((service, username))

    def fake_set(service: str, username: str, password: str) -> None:
        store_for_test[(service, username)] = password

    def fake_delete(service: str, username: str) -> None:
        store_for_test.pop((service, username), None)

    monkeypatch.setattr(_keyring, "get_password", fake_get)
    monkeypatch.setattr(_keyring, "set_password", fake_set)
    monkeypatch.setattr(_keyring, "delete_password", fake_delete)
    yield store_for_test


def _make(text: str = "hello", language: str = "en"):
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
        profile_modulation_gain={"k": 0.1},
    )


def _write_plaintext_row(store, rec):
    """Bypass the store's encryption wrapper and write a fully-plaintext row."""
    from iai_mcp.store import RECORDS_TABLE

    row = {
        "id": str(rec.id),
        "tier": rec.tier,
        "literal_surface": rec.literal_surface,
        "aaak_index": rec.aaak_index,
        "embedding": [float(x) for x in rec.embedding],
        "structure_hv": b"",
        "community_id": "",
        "centrality": float(rec.centrality),
        "detail_level": int(rec.detail_level),
        "pinned": bool(rec.pinned),
        "stability": float(rec.stability),
        "difficulty": float(rec.difficulty),
        "last_reviewed": rec.last_reviewed,
        "never_decay": bool(rec.never_decay),
        "never_merge": bool(rec.never_merge),
        "provenance_json": json.dumps(rec.provenance),
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
        "tags_json": json.dumps(rec.tags),
        "language": rec.language,
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": json.dumps(rec.profile_modulation_gain or {}),
        "schema_version": 2,
    }
    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.add([row])


def test_migrate_encryption_helper_exists() -> None:
    """exposes migrate_encryption_v2_to_v3."""
    from iai_mcp import migrate
    assert hasattr(migrate, "migrate_encryption_v2_to_v3")


def test_migration_encrypts_plaintext_literal_surface(tmp_path):
    """A plaintext row becomes encrypted after migration."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    rec = _make(text="unencrypted secret")
    _write_plaintext_row(store, rec)

    # Sanity: before migration the row is plaintext.
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    pre = df[df["id"] == str(rec.id)].iloc[0]
    assert pre["literal_surface"] == "unencrypted secret"

    result = migrate_encryption_v2_to_v3(store)
    assert result["records_migrated"] >= 1

    df = store.db.open_table(RECORDS_TABLE).to_pandas()
    post = df[df["id"] == str(rec.id)].iloc[0]
    assert post["literal_surface"].startswith("iai:enc:v1:")


def test_migration_encrypts_provenance_and_profile_gain(tmp_path):
    """provenance_json AND profile_modulation_gain_json become encrypted."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    rec = _make(text="hello")
    _write_plaintext_row(store, rec)

    migrate_encryption_v2_to_v3(store)

    df = store.db.open_table(RECORDS_TABLE).to_pandas()
    post = df[df["id"] == str(rec.id)].iloc[0]
    assert post["provenance_json"].startswith("iai:enc:v1:")
    assert post["profile_modulation_gain_json"].startswith("iai:enc:v1:")


def test_migration_preserves_content_byte_for_byte(tmp_path):
    """decrypting the migrated row returns the original bytes."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    text = " verbatim: Привет, мир"
    rec = _make(text=text, language="ru")
    _write_plaintext_row(store, rec)

    migrate_encryption_v2_to_v3(store)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == text
    assert got.literal_surface.encode("utf-8") == text.encode("utf-8")
    assert got.provenance == rec.provenance


def test_migration_dry_run_does_not_mutate(tmp_path):
    """dry_run=True returns a count but leaves disk rows untouched."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    rec = _make(text="still plaintext")
    _write_plaintext_row(store, rec)

    out = migrate_encryption_v2_to_v3(store, dry_run=True)
    assert out["records_migrated"] >= 1  # Count is predictive

    df = store.db.open_table(RECORDS_TABLE).to_pandas()
    post = df[df["id"] == str(rec.id)].iloc[0]
    assert post["literal_surface"] == "still plaintext"


def test_migration_idempotent(tmp_path):
    """Second run returns records_migrated=0 on a fully-encrypted store."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make(text="x")
    _write_plaintext_row(store, rec)

    first = migrate_encryption_v2_to_v3(store)
    assert first["records_migrated"] >= 1
    second = migrate_encryption_v2_to_v3(store)
    assert second["records_migrated"] == 0


def test_migration_skips_already_encrypted_rows(tmp_path):
    """Records inserted via store.insert() are already encrypted; migration skips them."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make(text="already encrypted via insert")
    store.insert(rec)  # Normal encrypted path

    out = migrate_encryption_v2_to_v3(store)
    assert out["records_migrated"] == 0


def test_migration_writes_event(tmp_path):
    """A migration_v2_to_v3 event is recorded in the events table."""
    from iai_mcp.events import query_events
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make(text="record for event trail")
    _write_plaintext_row(store, rec)

    migrate_encryption_v2_to_v3(store)

    events = query_events(store, kind="migration_v2_to_v3", limit=1)
    assert len(events) == 1
    data = events[0]["data"]
    assert data.get("record_count", 0) >= 1


def test_migration_encrypts_events_data_column(tmp_path):
    """events.data_json for pre-existing events becomes encrypted post-migration."""
    from iai_mcp.events import write_event
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore, EVENTS_TABLE

    store = MemoryStore(path=tmp_path)
    # Write a plaintext event manually (bypass write_event's encryption wrap).
    # We simulate a pre-02-08 event by writing directly via the underlying table.
    tbl = store.db.open_table(EVENTS_TABLE)
    event_row = {
        "id": str(uuid4()),
        "kind": "test_plain_event",
        "severity": "info",
        "domain": "",
        "ts": datetime.now(timezone.utc),
        "data_json": json.dumps({"quote_from_user": "sensitive content"}),
        "session_id": "pre-0208",
        "source_ids_json": "[]",
    }
    tbl.add([event_row])

    migrate_encryption_v2_to_v3(store)

    df = store.db.open_table(EVENTS_TABLE).to_pandas()
    # Find our row
    row = df[df["kind"] == "test_plain_event"].iloc[0]
    assert row["data_json"].startswith("iai:enc:v1:")


def test_migration_reports_duration(tmp_path):
    """Result dict carries a duration_sec field."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _make()
    _write_plaintext_row(store, rec)

    out = migrate_encryption_v2_to_v3(store)
    assert "duration_sec" in out
    assert out["duration_sec"] >= 0


def test_migration_preserves_plaintext_columns(tmp_path):
    """language / tags / detail_level / embedding stay plaintext after migration."""
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore(path=tmp_path)
    rec = _make(text="plaintext-flags", language="ru")
    rec.tags = ["topic:auth", "topic:db"]
    _write_plaintext_row(store, rec)

    migrate_encryption_v2_to_v3(store)

    df = store.db.open_table(RECORDS_TABLE).to_pandas()
    post = df[df["id"] == str(rec.id)].iloc[0]
    assert post["language"] == "ru"
    assert json.loads(post["tags_json"]) == ["topic:auth", "topic:db"]
    assert post["detail_level"] == 2
    assert len(list(post["embedding"])) == EMBED_DIM
