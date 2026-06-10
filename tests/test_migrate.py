from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord, SCHEMA_VERSION_LEGACY


def _v1_record(
    text: str,
    *,
    language: str = "",
    tags: list[str] | None = None,
    dim: int = EMBED_DIM,
) -> MemoryRecord:
    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * dim,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": "2026-04-16T00:00:00Z", "cue": "seed", "session_id": "phase1"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language="en",
        schema_version=SCHEMA_VERSION_LEGACY,
    )
    if language:
        r.language = language
    else:
        r.language = ""
    return r


def test_migrate_v1_to_v2_sets_defaults(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("English legacy record for migration test with enough words")
    store.insert(r)
    result = migrate_v1_to_v2(store)
    assert result["records_migrated"] >= 1

    migrated = store.get(r.id)
    assert migrated is not None
    assert migrated.s5_trust_score == 0.5
    assert migrated.profile_modulation_gain == {}
    from iai_mcp.types import SCHEMA_VERSION_CURRENT
    assert migrated.schema_version == SCHEMA_VERSION_CURRENT
    assert migrated.schema_version >= 2


def test_migrate_v1_to_v2_defaults_language_to_en(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    en = _v1_record("This is a reasonable English sentence with enough words.")
    ru = _v1_record("Это осмысленное предложение с достаточным количеством слов.")
    store.insert(en)
    store.insert(ru)

    migrate_v1_to_v2(store)

    en_mig = store.get(en.id)
    ru_mig = store.get(ru.id)
    assert en_mig.language == "en"
    assert ru_mig.language == "en"


def test_migrate_v1_to_v2_preserves_existing_language_tag(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("legacy row carrying an explicit non-en tag", language="ru")
    store.insert(r)

    migrate_v1_to_v2(store)

    migrated = store.get(r.id)
    assert migrated.language == "ru"


def test_migrate_v1_to_v2_idempotent(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for i in range(5):
        store.insert(_v1_record(f"English record number {i} with enough content to detect."))

    first = migrate_v1_to_v2(store)
    assert first["records_migrated"] >= 5

    second = migrate_v1_to_v2(store)
    assert second["records_migrated"] == 0


def test_migrate_dry_run_no_writes(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("Dry run English text with enough words for language detection.")
    store.insert(r)
    before = store.get(r.id)
    assert before.schema_version == 1

    result = migrate_v1_to_v2(store, dry_run=True)
    assert result["records_migrated"] >= 1

    after = store.get(r.id)
    assert after.schema_version == 1


def test_migrate_writes_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    store.insert(_v1_record("English content one for migration event test."))

    migrate_v1_to_v2(store)

    events = query_events(store, kind="migration_v1_to_v2")
    assert len(events) == 1
    assert events[0]["data"]["record_count"] >= 1


def test_migrate_preserves_literal_surface_verbatim(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    verbatim = "SECRET_PHRASE_ABC_XYZ must survive the migration byte-for-byte exactly."
    r = _v1_record(verbatim)
    store.insert(r)

    migrate_v1_to_v2(store)

    migrated = store.get(r.id)
    assert migrated.literal_surface == verbatim


def test_migrate_preserves_provenance(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _v1_record("English content for provenance preservation test through migration.")
    store.insert(r)

    migrate_v1_to_v2(store)

    migrated = store.get(r.id)
    assert len(migrated.provenance) == 1
    assert migrated.provenance[0]["cue"] == "seed"
    assert migrated.provenance[0]["session_id"] == "phase1"


def test_migrate_skips_existing_v2_records(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    v2 = _v1_record("Already migrated record with language tag.", language="en")
    v2.schema_version = 2
    store.insert(v2)

    v1 = _v1_record("Legacy v1 record with enough content for detection.")
    store.insert(v1)

    result = migrate_v1_to_v2(store)
    assert result["records_migrated"] == 1

    v2_got = store.get(v2.id)
    assert v2_got.schema_version == 2


def test_migrate_result_carries_model_info(tmp_path):
    from iai_mcp.migrate import migrate_v1_to_v2
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    store.insert(_v1_record("English content for the migration model info check."))

    result = migrate_v1_to_v2(store)
    assert "previous_model" in result
    assert "new_model" in result
    assert "duration_sec" in result
