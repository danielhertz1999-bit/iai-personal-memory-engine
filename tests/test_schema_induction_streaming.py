"""schema.py induce_schemas_tier0 + persist_schema migrate
to ``store.iter_record_columns(...)`` projection.

Two `all_records()` callers in `schema.py` are migrated so that the ≤1
all_records() invariant on `run_heavy_consolidation` becomes achievable.

Pre-migration architecture:

    run_heavy_consolidation
      ├── all_records() in sleep.py (records_by_id — kept)
      ├── _tier0_schema_surfacing (projection-only)
      └── induce_schemas_tier1
            └── induce_schemas_tier0
                  ├── all_records() in schema.py ← target A
                  └── (downstream) persist_schema
                        └── all_records() in schema.py ← target B

Total: 3 all_records() calls per heavy invocation (when auto-status candidates
fire).

Post-migration architecture:

    run_heavy_consolidation
      ├── all_records() in sleep.py (records_by_id — kept)
      ├── _tier0_schema_surfacing (projection-only)
      └── induce_schemas_tier1
            └── induce_schemas_tier0
                  ├── iter_record_columns(["id", "tags_json"]) ← A
                  └── persist_schema
                        └── iter_record_columns(["id", "tier", "tags_json"])
                              ← B (early-exit via break on first match)

Total: 1 all_records() call per heavy invocation. The invariant becomes
achievable; the invariant test in tests/test_sleep_consolidation_streaming.py
asserts ``count_all.call_count <= 1``.

Covered contracts:

  A — induce_schemas_tier0 migration:
    1. Calls iter_record_columns, NOT all_records (spy via monkeypatch)
    2. _decrypt_for_record fires zero times (proof of zero-AES-GCM)
    3. SchemaCandidate output is byte-identical to the prior implementation
       on a deterministic synthetic store (same patterns, same evidence_count,
       same confidence, same status)

  B — persist_schema migration:
    4. Calls iter_record_columns, NOT all_records (spy via monkeypatch)
    5. Early-exit via break on first matching pattern row works (the keeper
       scan must NOT iterate every record after a hit)
    6. Correct schema_id returned when keeper is mid-stream (the keeper's
       UUID is preserved across the iter_record_columns→str→UUID round-trip)

  Cross-cutting:
    7. existing_keeper_id remains a UUID (not a string from row["id"])
    8. The pattern_tag check is preserved byte-for-byte: tier == "semantic"
       AND f"pattern:{candidate.pattern}" in tags

 Note: every test uses a real ``MemoryRecord``
dataclass via ``_rec()`` — never a plain dict against attribute-access code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from iai_mcp.schema import (
    SchemaCandidate,
    induce_schemas_tier0,
    persist_schema,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Mirror tests/test_store_iter_records.py — process-isolated keyring so
    AES-256-GCM key generation does not poke the OS keychain inside CI."""
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


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch):
    """Avoid loading bge-m3 — persist_schema's insert path embeds the schema
    summary; without this fixture each test pays ~5s embedder load."""
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):  # noqa: ANN001
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):  # noqa: ANN001
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


def _rec(
    *,
    text: str = "t",
    tags: list[str] | None = None,
    tier: str = "episodic",
    detail_level: int = 2,
    language: str = "en",
) -> MemoryRecord:
    """Real-dataclass fixture (NEVER a plain dict — attribute-access code requires dataclass)."""
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail_level >= 3),
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Fresh MemoryStore in tmp_path/hippo (one per test, no cross-test bleed)."""
    return MemoryStore(path=tmp_path / "hippo")


# --------------------------------------------------------------------------- A: induce_schemas_tier0


def test_induce_schemas_tier0_uses_iter_record_columns_not_all_records(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Architecture flip: rewritten function uses
    ``iter_record_columns(["id", "tags_json"],...)`` and never calls
    ``all_records()``.

    Before: ``induce_schemas_tier0`` calls
    ``store.all_records()`` in schema.py — spy on ``all_records`` fires
    once and spy on ``iter_record_columns`` fires zero times → assertion
    fails RED.

    After: spy on ``iter_record_columns`` fires once and spy on
    ``all_records`` fires zero times → assertion passes GREEN.
    """
    # 5 records with the same tag pair (above CLUSTER_MIN_SIZE=3).
    for i in range(5):
        store.insert(_rec(text=f"r{i}", tags=["meeting", "notes"]))

    spy_all = MagicMock(wraps=store.all_records)
    spy_iter = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "all_records", spy_all)
    monkeypatch.setattr(store, "iter_record_columns", spy_iter)

    induce_schemas_tier0(store)

    assert spy_all.call_count == 0, (
        f"induce_schemas_tier0 must NOT call store.all_records(); "
        f"got {spy_all.call_count} call(s)"
    )
    assert spy_iter.call_count >= 1, (
        f"induce_schemas_tier0 must call store.iter_record_columns() at least "
        f"once; got {spy_iter.call_count} call(s)"
    )


def test_induce_schemas_tier0_zero_decrypt_calls(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero-decrypt contract: ``_decrypt_for_record`` fires zero times
    during the migrated path.

    Projection is ``["id", "tags_json"]`` — neither column is encrypted
    (``id`` is plain string UUID; ``tags_json`` is plain JSON string in
    store.py). Therefore the cipher cache is short-circuited entirely
    on this path, mirroring the ``_tier0_schema_surfacing`` win.

    Before: ``store.all_records()`` round-trips every row
    through ``_from_row``, which calls ``_decrypt_for_record`` on each of
    literal_surface + provenance_json + profile_modulation_gain_json
    (encrypted columns). For a 5-record store: up to 15 calls. Assertion
    ``call_count == 0`` fails RED.

    After: zero calls — assertion passes GREEN.
    """
    for i in range(5):
        store.insert(_rec(text=f"r{i}", tags=["meeting", "notes"]))

    decrypt_spy = MagicMock(wraps=store._decrypt_for_record)
    monkeypatch.setattr(store, "_decrypt_for_record", decrypt_spy)

    induce_schemas_tier0(store)

    assert decrypt_spy.call_count == 0, (
        f"induce_schemas_tier0 must NOT trigger ANY _decrypt_for_record "
        f"calls; got {decrypt_spy.call_count} call(s)"
    )


def test_induce_schemas_tier0_byte_identical_to_pre_d26_implementation(
    store: MemoryStore,
) -> None:
    """Contract: rewritten function produces identical SchemaCandidate
    output to the prior implementation on a deterministic synthetic
    store.

    Compute the expected output inline using the prior algorithm
    (``store.all_records()`` + ``_tag_cooccurrence``) and assert
    order-independent equality (sort by pattern) against the migrated
    function's output.

    Fixture (deterministic, 8 records):
      - 5 records tagged ["meeting", "notes"] → pair count = 5
      - 3 records tagged ["report", "deadline"] → pair count = 3

    Expected:
      - "tags:meeting+notes" — count=5, confidence=0.5, status="auto"
        (5 >= AUTO_INDUCT_COOCCURRENCE=5 BUT confidence < AUTO_INDUCT_CONFIDENCE
        =0.85, so it falls into pending_user_approval branch instead)
      - Wait — actually count=5 falls into the ``elif`` guard
        ``USER_APPROVAL_COOCCURRENCE <= count < AUTO_INDUCT_COOCCURRENCE``
        which is ``3 <= 5 < 5`` → False. So count=5 needs auto path
        ``count >= 5 AND confidence >= 0.85``; confidence 0.5 fails the
        confidence floor. Result: SKIPPED.
      - count=3 → ``elif 3 <= 3 < 5`` AND confidence=0.3 < 0.65 → SKIPPED.

    To get measurable output, raise count to clear the floors:
      - 9 records tagged ["meeting", "notes"]: count=9, conf=0.9 → "auto"
      - 4 records tagged ["report", "deadline"]: count=4, conf=0.4 →
        elif 3 <= 4 < 5 → True; conf 0.4 < 0.65 → SKIPPED
      - Add 4 records tagged ["alpha", "beta"]: count=4, conf=0.4 → SKIPPED
        same as above

    To exercise the user-approval path, we need conf >= 0.65. Confidence
    saturates at count/10, so count >= 7 with count < 5 is impossible.
    We accept that on this fixture only the auto path emits a candidate.
    """
    # 9 records with the same tag pair → count=9, confidence=0.9, status="auto"
    auto_recs: list[MemoryRecord] = []
    for i in range(9):
        r = _rec(text=f"auto-{i}", tags=["meeting", "notes"])
        auto_recs.append(r)
        store.insert(r)
    # 4 records with a different tag pair — below auto threshold (count<5),
    # below confidence threshold for user-approval (conf=0.4 < 0.65), so
    # contributes nothing to the candidate list.
    for i in range(4):
        store.insert(_rec(text=f"low-{i}", tags=["report", "deadline"]))

    # Compute expected via the prior algorithm inline. We re-implement
    # the contract directly so the test does not depend on the prior
    # implementation surviving the migration unchanged.
    from iai_mcp.schema import (
        AUTO_INDUCT_CONFIDENCE,
        AUTO_INDUCT_COOCCURRENCE,
        MAX_EVIDENCE_PER_SCHEMA,
        USER_APPROVAL_CONFIDENCE,
        USER_APPROVAL_COOCCURRENCE,
    )
    from iai_mcp.lilli.cycle.schema import _tag_cooccurrence

    expected_records = store.all_records()
    pair_counts = _tag_cooccurrence(expected_records)
    expected: list[dict] = []
    for pair, evidence in pair_counts.items():
        count = len(evidence)
        confidence = min(1.0, count / 10.0)
        if count >= AUTO_INDUCT_COOCCURRENCE and confidence >= AUTO_INDUCT_CONFIDENCE:
            status = "auto"
        elif (
            USER_APPROVAL_COOCCURRENCE <= count < AUTO_INDUCT_COOCCURRENCE
            and confidence >= USER_APPROVAL_CONFIDENCE
        ):
            status = "pending_user_approval"
        else:
            continue
        expected.append({
            "pattern": f"tags:{'+'.join(sorted(pair))}",
            "confidence": confidence,
            "evidence_count": count,
            "status": status,
            "evidence_ids_set": set(evidence[:MAX_EVIDENCE_PER_SCHEMA]),
        })

    actual = induce_schemas_tier0(store)

    expected_sorted = sorted(expected, key=lambda d: d["pattern"])
    actual_sorted = sorted(actual, key=lambda c: c.pattern)

    assert len(actual_sorted) == len(expected_sorted), (
        f"candidate count mismatch — expected={len(expected_sorted)} "
        f"actual={len(actual_sorted)}; expected={expected_sorted!r}; "
        f"actual={[(c.pattern, c.evidence_count, c.confidence, c.status) for c in actual_sorted]!r}"
    )
    for e, a in zip(expected_sorted, actual_sorted, strict=True):
        assert a.pattern == e["pattern"]
        assert a.evidence_count == e["evidence_count"]
        assert a.confidence == pytest.approx(e["confidence"])
        assert a.status == e["status"]
        # evidence_ids must round-trip back to the same UUIDs (set equality —
        # iter_record_columns batch order may differ from all_records pandas
        # iter order, but the underlying set must match).
        assert set(a.evidence_ids) == e["evidence_ids_set"]

    # Sanity: at least one auto candidate surfaced (the 9-records pair).
    assert any(c.status == "auto" for c in actual_sorted), (
        f"expected at least one status='auto' candidate on the 9-record "
        f"meeting+notes pair; got {[(c.pattern, c.evidence_count, c.status) for c in actual_sorted]!r}"
    )


def test_induce_schemas_tier0_evidence_ids_are_uuids(
    store: MemoryStore,
) -> None:
    """Boundary contract: ``iter_record_columns`` returns ``id`` as a
    string (per tests/test_store_iter_records.py) but
    ``SchemaCandidate.evidence_ids`` is typed ``list[UUID]``. The migration
    must convert at the boundary; without conversion, downstream code (e.g.
    ``store.boost_edges([(ev_id, schema_id) for ev_id in evidence_ids])``)
    would break.
    """
    inserted = []
    for i in range(9):
        r = _rec(text=f"r{i}", tags=["meeting", "notes"])
        store.insert(r)
        inserted.append(r.id)

    candidates = induce_schemas_tier0(store)
    auto = [c for c in candidates if c.status == "auto"]
    assert len(auto) >= 1, "expected at least one auto candidate"

    for c in auto:
        for ev_id in c.evidence_ids:
            assert isinstance(ev_id, UUID), (
                f"evidence_ids must be list[UUID]; got {type(ev_id).__name__} "
                f"for {ev_id!r}"
            )
        # Set equality with inserted ids — every evidence id must trace back
        # to a real record we inserted.
        assert set(c.evidence_ids).issubset(set(inserted))


# --------------------------------------------------------------------------- persist_schema


def test_persist_schema_uses_iter_record_columns_not_all_records_for_keeper_scan(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Architecture flip: the keeper-pattern scan in persist_schema
    uses ``iter_record_columns(["id", "tier", "tags_json"],...)``, NOT
    ``store.all_records()``.

    Fixture: empty store (no existing keeper); we are exercising the
    no-keeper-found branch, which still must execute the scan.

    Before: ``persist_schema`` calls ``store.all_records()``
    in schema.py — spy on ``all_records`` fires once. Assertion fails RED.

    After: spy on ``iter_record_columns`` fires (with at minimum
    ``["id", "tier", "tags_json"]`` projection); spy on ``all_records``
    fires zero times.

    Note: the fallback insert path in schema.py calls ``store.insert(...)``
    which internally uses ``boost_edges``/``merge_insert`` and may touch other
    tables — but it does NOT call ``store.all_records()`` (verified by reading
    store.py). So the spy on ``all_records`` cleanly captures only the
    keeper-scan path's calls.
    """
    # Seed 3 evidence records — minimum CLUSTER_MIN_SIZE.
    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    spy_all = MagicMock(wraps=store.all_records)
    spy_iter = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "all_records", spy_all)
    monkeypatch.setattr(store, "iter_record_columns", spy_iter)

    cand = SchemaCandidate(
        pattern="tags:meeting+notes",
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    persist_schema(store, cand)

    assert spy_all.call_count == 0, (
        f"persist_schema must NOT call store.all_records(); "
        f"got {spy_all.call_count} call(s)"
    )
    assert spy_iter.call_count >= 1, (
        f"persist_schema must call store.iter_record_columns() at least once "
        f"(keeper scan); got {spy_iter.call_count} call(s)"
    )


def test_persist_schema_early_exit_on_first_match(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keeper scan must break on the FIRST matching pattern row,
    matching the existing schema.py ``break`` semantics.

    Fixture: 50 schema-tier records, ALL carrying the keeper pattern tag.
    The migrated code must stop iterating after the first match — proven by
    counting how many rows the iterator yields before persist_schema returns.

    Strategy: monkeypatch-wrap ``iter_record_columns`` with a row counter.
    """
    # Insert 50 schema-tier records, all carrying the same pattern tag.
    pattern = "tags:meeting+notes"
    pattern_tag = f"pattern:{pattern}"
    keeper_ids: list[UUID] = []
    for i in range(50):
        r = _rec(
            text=f"schema-{i}",
            tags=["schema", "auto", pattern_tag],
            tier="semantic",
            detail_level=3,
        )
        store.insert(r)
        keeper_ids.append(r.id)

    # Wrap iter_record_columns with a row counter.
    real_iter = store.iter_record_columns
    yielded = {"count": 0}

    def counting_iter(columns, **kwargs):  # noqa: ANN001
        for row in real_iter(columns, **kwargs):
            yielded["count"] += 1
            yield row

    monkeypatch.setattr(store, "iter_record_columns", counting_iter)

    # Seed evidence records.
    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern=pattern,
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    # Returned id must be one of the existing keepers (the first matching row).
    assert schema_id in keeper_ids, (
        f"persist_schema must return an existing keeper id when a match exists; "
        f"got {schema_id} not in {keeper_ids[:3]}..."
    )

    # Early-exit invariant: substantially fewer than 50 rows iterated. Without
    # a `break` after first match, the wrap counter would see all 50 records.
    # Allow up to 2× CLUSTER_MIN_SIZE to absorb store batch boundaries —
    # iter_record_columns yields per row but the scanner reads in batches of
    # 1024, so the in-process generator stops cleanly on `break` from the
    # consuming code.
    assert yielded["count"] <= 50 // 2, (
        f"persist_schema must early-exit on first match; iterator yielded "
        f"{yielded['count']} rows on a 50-keeper-row store (expected break "
        f"after the first match — strictly < 50)"
    )


def test_persist_schema_returns_correct_id_when_keeper_is_mid_stream(
    store: MemoryStore,
) -> None:
    """When the keeper is the Nth row of the scan (not the first),
    the returned UUID must match the keeper's id, not a string from
    row["id"] or a different match-but-not-the-first-one row.

    Fixture: 5 schema records, only ONE of which carries the matching
    pattern tag. The migrated code must:
      1. Iterate through non-matching rows without misfiring.
      2. Find the matching row and capture its id (with str→UUID conversion).
      3. Break out of the loop.
      4. Return that captured UUID.
    """
    pattern = "tags:meeting+notes"
    pattern_tag = f"pattern:{pattern}"

    # Insert 5 schema-tier records, only ONE carries the matching tag.
    for i in range(2):
        store.insert(_rec(
            text=f"unrelated-{i}",
            tags=["schema", "auto", "pattern:other"],
            tier="semantic",
            detail_level=3,
        ))
    keeper = _rec(
        text="the-keeper",
        tags=["schema", "auto", pattern_tag],
        tier="semantic",
        detail_level=3,
    )
    store.insert(keeper)
    keeper_id = keeper.id
    for i in range(2):
        store.insert(_rec(
            text=f"trailing-{i}",
            tags=["schema", "auto", "pattern:something-else"],
            tier="semantic",
            detail_level=3,
        ))

    # Seed evidence records.
    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern=pattern,
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    returned_id = persist_schema(store, cand)

    assert returned_id == keeper_id, (
        f"persist_schema must return the matching keeper's UUID; "
        f"got {returned_id} expected {keeper_id}"
    )
    assert isinstance(returned_id, UUID), (
        f"persist_schema must return a UUID, not a string from row['id']; "
        f"got {type(returned_id).__name__}"
    )


def test_persist_schema_falls_through_to_insert_when_no_keeper(
    store: MemoryStore,
) -> None:
    """Byte-identical contract: when no existing schema carries the
    pattern tag, persist_schema falls through to the original insert path
    (line 371 ``store.insert(schema_rec)``) and returns a NEW UUID — not
    one of the existing-but-non-matching record ids.

    Fixture: 5 schema-tier records carrying DIFFERENT pattern tags. None
    matches our candidate; the function must insert a new schema record.
    """
    # Insert 5 schema-tier records, none matching the candidate pattern.
    other_ids: list[UUID] = []
    for i in range(5):
        r = _rec(
            text=f"other-{i}",
            tags=["schema", "auto", f"pattern:other-{i}"],
            tier="semantic",
            detail_level=3,
        )
        store.insert(r)
        other_ids.append(r.id)

    # Seed evidence.
    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:meeting+notes",
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    # Must be a fresh UUID, not one of the non-matching keepers.
    assert schema_id not in other_ids, (
        f"persist_schema must insert a new schema when no keeper matches; "
        f"got returned id {schema_id} which equals one of the existing "
        f"non-matching schema ids ({other_ids!r})"
    )
    # The new schema record exists in the store.
    new_rec = store.get(schema_id)
    assert new_rec is not None
    assert new_rec.tier == "semantic"
    assert new_rec.detail_level == 3
    assert "schema" in (new_rec.tags or [])
    assert f"pattern:{cand.pattern}" in (new_rec.tags or [])
