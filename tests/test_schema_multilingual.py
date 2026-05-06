"""Tests for 02-REVIEW.md (persist_schema hardcodes language='en' --
D-08a constitutional violation for multilingual users).

Bug: every schema hub record was created with language='en' regardless of
the language of the source cluster. A user storing Russian records saw
schema hubs derived from their Russian clusters tagged as English, so
language-filtered retrieval ('ru' filter) missed their own schemas.

Fix:
    - Add helper _majority_language(evidence_ids, store) -> str. Tie-break
      is deterministic (max with key=count on a stable input order).
    - persist_schema derives language from the helper; fallback 'en' only
      when evidence is empty or all evidence records are missing.

Constitutional contract (D-08a native-language storage):
    Records are stored in the language they were recorded in. This extends
    to derived records (schema hubs). mandates 7+ language support;
    hardcoded 'en' broke the contract silently.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.schema import SchemaCandidate, persist_schema
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------- helpers


def _rec(*, language: str, text: str = "seed") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language=language,
    )


def _seed_cluster(
    store: MemoryStore,
    lang_counts: dict[str, int],
) -> list[uuid4]:
    """Insert N records per language. Returns the list of evidence ids in
    INSERT ORDER (deterministic tie-break)."""
    evidence: list = []
    for lang, count in lang_counts.items():
        for i in range(count):
            r = _rec(language=lang, text=f"{lang}_seed_{i}")
            store.insert(r)
            evidence.append(r.id)
    return evidence


# ================================================= core cases


def test_persist_schema_derives_language_from_majority_evidence(tmp_path):
    """5 ru + 2 en + 1 ja evidence -> schema.language == 'ru'."""
    store = MemoryStore(path=tmp_path)
    evidence = _seed_cluster(store, {"ru": 5, "en": 2, "ja": 1})

    cand = SchemaCandidate(
        pattern="tags:tech+python",
        confidence=0.9,
        evidence_count=len(evidence),
        evidence_ids=list(evidence),
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    fresh = store.get(schema_id)
    assert fresh is not None
    assert fresh.language == "ru", (
        f"persist_schema must read majority language from evidence, got {fresh.language!r}"
    )


def test_persist_schema_fallback_en_on_empty_evidence(tmp_path):
    """No evidence -> fallback to 'en' (Phase-1 default, safe)."""
    store = MemoryStore(path=tmp_path)
    cand = SchemaCandidate(
        pattern="tags:orphan",
        confidence=0.9,
        evidence_count=0,
        evidence_ids=[],
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    fresh = store.get(schema_id)
    assert fresh is not None
    assert fresh.language == "en"


def test_persist_schema_tie_is_deterministic(tmp_path):
    """3 ru + 3 en (tied) -> deterministic winner governed by input order.
    max(..., key=list.count) with a list preserves first-seen-wins; 'ru'
    inserted first wins the tie."""
    store = MemoryStore(path=tmp_path)
    evidence = _seed_cluster(store, {"ru": 3, "en": 3})

    cand = SchemaCandidate(
        pattern="tags:tied",
        confidence=0.9,
        evidence_count=len(evidence),
        evidence_ids=list(evidence),
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    fresh = store.get(schema_id)
    assert fresh is not None
    # Tie-break: first distinct language in the evidence list wins.
    # Seeded as {ru:3, en:3} in that order -> 'ru' appears first.
    assert fresh.language == "ru"


def test_persist_schema_ignores_missing_evidence_records(tmp_path):
    """evidence_ids can point to records that were deleted/never existed.
    The helper must filter those out gracefully and use only the surviving
    records' language values."""
    store = MemoryStore(path=tmp_path)

    # Seed 2 real records in Japanese
    surviving = _seed_cluster(store, {"ja": 2})

    # Add 3 phantom ids that were never inserted
    phantom_ids = [uuid4() for _ in range(3)]

    cand = SchemaCandidate(
        pattern="tags:graceful",
        confidence=0.85,
        evidence_count=5,
        evidence_ids=list(surviving) + phantom_ids,
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    fresh = store.get(schema_id)
    assert fresh is not None
    # Only the 2 surviving Japanese records contribute -> 'ja'
    assert fresh.language == "ja", (
        f"persist_schema must ignore missing evidence records, got {fresh.language!r}"
    )


def test_persist_schema_no_hardcoded_english(tmp_path):
    """Structural guard: persist_schema source must not carry `language='en'`
    hardcoded; it must route language through _majority_language."""
    import inspect
    from iai_mcp import schema as schema_mod

    src = inspect.getsource(schema_mod.persist_schema)
    assert "language=\"en\"," not in src, (
        "persist_schema still hardcodes language='en'"
    )
    assert "_majority_language" in src, (
        "persist_schema must call _majority_language to derive schema language"
    )
    assert hasattr(schema_mod, "_majority_language"), (
        "_majority_language helper must exist at schema.py module scope"
    )
