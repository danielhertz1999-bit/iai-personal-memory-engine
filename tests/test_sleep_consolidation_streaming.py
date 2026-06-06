"""_tier0_schema_surfacing rewritten on iter_record_columns(["tags_json"]).

Tests 1+2 verify that ``sleep._tier0_schema_surfacing`` calls
``store.iter_record_columns(["tags_json"], batch_size=1024)`` instead
of ``store.all_records()``. Tests 3-7 lock pre-existing filter semantics that
the rewrite must preserve byte-for-byte — record-count floor, raw:/domain:
filtering, count >= 3 floor, defensive JSON parse.

Covered contracts:

  Architecture flip:
    1. ``_tier0_schema_surfacing`` calls ``iter_record_columns(["tags_json"],...)``,
       not ``all_records()`` — verified via monkeypatched spies on both methods.

  Zero AES-GCM cost:
    2. Across the entire ``_tier0_schema_surfacing`` execution on a 16-record
       store, ``store._decrypt_for_record`` fires zero times — projection-only
       iteration skips encrypted columns entirely (literal_surface,
       provenance_json, profile_modulation_gain_json never touch disk).

  Filter semantics — byte-identical to the prior implementation (preserve all rules):
    3. ``raw:*`` and ``domain:*`` tags are filtered before counting (existing
       contract; new code must not regress).
    4. ``count >= 3`` per-tag floor preserved.
    5. ``len(records) < CLUSTER_MIN_SIZE`` global floor preserved (now expressed
       as ``record_count < CLUSTER_MIN_SIZE`` after single-pass iteration).
    6. Output dicts are byte-identical to the reference implementation on a
       deterministic 20-record fixture (compute expected via the same algorithm
       run inline against ``store.all_records()``).

  Defensive JSON parse:
    7. Malformed ``tags_json`` rows do NOT raise — defensive try/except absorbs
       JSONDecodeError and treats the row as having zero tags. Verified by
       monkeypatch-wrapping ``iter_record_columns`` to inject a malformed row
       AFTER the real rows.

Every test uses a real ``MemoryRecord`` dataclass via ``_make()`` — never a
plain dict against attribute-access code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from iai_mcp.sleep import CLUSTER_MIN_SIZE, _tier0_schema_surfacing
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


def _make(
    text: str = "hello world",
    tier: str = "episodic",
    tags: list[str] | None = None,
    detail: int = 2,
    language: str = "en",
) -> MemoryRecord:
    """Real-dataclass fixture (NEVER a plain dict)."""
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags if tags is not None else [],
        language=language,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Fresh MemoryStore in tmp_path/hippo (one per test, no cross-test bleed)."""
    return MemoryStore(path=tmp_path / "hippo")


def _populate_mixed_16(store: MemoryStore) -> None:
    """16-record fixture with mixed tier/tags payloads."""
    # 4 records with tag-a (single user-facing tag)
    for _ in range(4):
        store.insert(_make(text="alpha", tags=["tag-a"]))
    # 5 records with tag-b
    for _ in range(5):
        store.insert(_make(text="beta", tags=["tag-b"]))
    # 7 records with only filtered tags (raw:*, domain:*) — should contribute 0
    # candidates after the raw:/domain: filter.
    for _ in range(7):
        store.insert(_make(text="gamma", tags=["raw:noise", "domain:misc"]))


# --------------------------------------------------------------------------- architecture flip


def test_tier0_schema_surfacing_uses_iter_record_columns_not_all_records(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewritten function uses ``iter_record_columns(['tags_json'],...)``
    and never calls ``all_records()``.

    Spy on ``iter_record_columns`` must fire once and spy on
    ``all_records`` must fire zero times.
    """
    _populate_mixed_16(store)

    spy_all = MagicMock(wraps=store.all_records)
    spy_iter = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "all_records", spy_all)
    monkeypatch.setattr(store, "iter_record_columns", spy_iter)

    _tier0_schema_surfacing(store)

    assert spy_all.call_count == 0, (
        f"_tier0_schema_surfacing must NOT call store.all_records(); "
        f"got {spy_all.call_count} call(s)"
    )
    assert spy_iter.call_count == 1, (
        f"_tier0_schema_surfacing must call store.iter_record_columns() exactly "
        f"once; got {spy_iter.call_count} call(s)"
    )

    # Defense-in-depth: verify the columns parameter is exactly ["tags_json"]
    # — caller is paying for projection, so reading any other column would
    # spend AES-GCM cost we are explicitly avoiding.
    args, kwargs = spy_iter.call_args
    if args:
        cols = args[0]
    else:
        cols = kwargs.get("columns")
    assert cols == ["tags_json"], (
        f"projection must be exactly ['tags_json'] (zero AES-GCM cost); "
        f"got columns={cols!r}"
    )


# --------------------------------------------------------------------------- zero-decrypt contract


def test_tier0_schema_surfacing_zero_decrypt_calls(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_decrypt_for_record`` fires zero times during the projection path.

    The contract is that projection-only iteration with
    ``columns=["tags_json"]`` skips every encrypted column at the disk-read
    layer; the cipher cache is short-circuited entirely on this path.

    An ``all_records()``-based implementation would round-trip every row
    through ``_from_row``, which calls ``_decrypt_for_record`` on each of
    literal_surface + provenance_json + profile_modulation_gain_json (encrypted
    columns) — up to 48 calls on a 16-record store. The projection path makes
    zero calls.
    """
    _populate_mixed_16(store)

    decrypt_spy = MagicMock(wraps=store._decrypt_for_record)
    monkeypatch.setattr(store, "_decrypt_for_record", decrypt_spy)

    _tier0_schema_surfacing(store)

    assert decrypt_spy.call_count == 0, (
        f"_tier0_schema_surfacing must NOT trigger ANY _decrypt_for_record "
        f"calls (zero AES-GCM cost on the projection path); "
        f"got {decrypt_spy.call_count} call(s)"
    )


# --------------------------------------------------------------------------- raw: / domain: filter


def test_tier0_schema_surfacing_filters_raw_and_domain_tags(
    store: MemoryStore,
) -> None:
    """Existing contract: ``raw:*`` and ``domain:*`` tags are skipped (they are
    classification metadata, not schema-candidate signals).

    5 records with both ``raw:literal`` AND ``tag-real``: only ``tag-real``
    should appear in the candidates output (count=5, confidence=0.5).
    Same for ``domain:foo`` + ``tag-real-2``.
    """
    # Empty fresh store from the fixture; populate with 10 records:
    # 5 with raw: + tag-real, 5 with domain: + tag-real-2.
    # CLUSTER_MIN_SIZE = 3 so 10 records easily clears the floor.
    for _ in range(5):
        store.insert(_make(text="r1", tags=["raw:literal", "tag-real"]))
    for _ in range(5):
        store.insert(_make(text="r2", tags=["domain:foo", "tag-real-2"]))

    candidates = _tier0_schema_surfacing(store)
    patterns = sorted(c["pattern"] for c in candidates)

    # Only the unfiltered tags should surface; both raw: and domain: must NOT.
    assert "tag:tag-real" in patterns
    assert "tag:tag-real-2" in patterns
    assert "tag:raw:literal" not in patterns
    assert "tag:domain:foo" not in patterns

    # Count and confidence preserved (5 occurrences each, confidence = 0.5).
    by_pattern = {c["pattern"]: c for c in candidates}
    assert by_pattern["tag:tag-real"]["evidence_count"] == 5
    assert by_pattern["tag:tag-real"]["confidence"] == pytest.approx(0.5)
    assert by_pattern["tag:tag-real-2"]["evidence_count"] == 5
    assert by_pattern["tag:tag-real-2"]["confidence"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- count >= 3 floor


def test_tier0_schema_surfacing_floor_count_3(store: MemoryStore) -> None:
    """Existing contract: per-tag count must be >= 3 to surface as a candidate.

    Fixture: 6 records, 3 with ``tag-a`` and 3 with ``tag-b``. Both clear the
    >= 3 floor and the global ``CLUSTER_MIN_SIZE`` floor (6 >= 3).

    Note: this isolates the per-tag count >= 3 floor from the global
    ``len(records) < CLUSTER_MIN_SIZE`` floor (test 5 covers the latter).
    """
    for _ in range(3):
        store.insert(_make(text="a", tags=["tag-a"]))
    for _ in range(3):
        store.insert(_make(text="b", tags=["tag-b"]))

    candidates = _tier0_schema_surfacing(store)
    assert len(candidates) == 2

    expected = sorted(
        [
            {"pattern": "tag:tag-a", "confidence": 0.3, "evidence_count": 3},
            {"pattern": "tag:tag-b", "confidence": 0.3, "evidence_count": 3},
        ],
        key=lambda d: d["pattern"],
    )
    actual = sorted(candidates, key=lambda d: d["pattern"])
    # Confidence is a float; use approx equality.
    for e, a in zip(expected, actual, strict=True):
        assert a["pattern"] == e["pattern"]
        assert a["evidence_count"] == e["evidence_count"]
        assert a["confidence"] == pytest.approx(e["confidence"])


# --------------------------------------------------------------------------- CLUSTER_MIN_SIZE global floor


def test_tier0_schema_surfacing_below_cluster_min_size_returns_empty(
    store: MemoryStore,
) -> None:
    """Existing contract: when total records < CLUSTER_MIN_SIZE, return [].

    Expressed as ``record_count < CLUSTER_MIN_SIZE`` after single-pass
    iteration. Must return ``[]`` on stores with fewer than
    ``CLUSTER_MIN_SIZE`` records.
    """
    # Insert exactly CLUSTER_MIN_SIZE - 1 records. With CLUSTER_MIN_SIZE = 3
    # this is 2 records — below the floor.
    for _ in range(CLUSTER_MIN_SIZE - 1):
        store.insert(_make(text="below-floor", tags=["any-tag"]))

    candidates = _tier0_schema_surfacing(store)
    assert candidates == [], (
        f"expected [] when record count ({CLUSTER_MIN_SIZE - 1}) is below "
        f"CLUSTER_MIN_SIZE ({CLUSTER_MIN_SIZE}); got {candidates!r}"
    )


# --------------------------------------------------------------------------- byte-identical output


def test_tier0_schema_surfacing_byte_identical_to_pre_w3(
    store: MemoryStore,
) -> None:
    """Contract: the projection-based function produces byte-identical output
    to the reference algorithm on a deterministic 20-record fixture.

    Compute the expected output inline using the reference algorithm against
    ``store.all_records()``; assert order-independent equality (sort by pattern)
    against the projection implementation's output.

    Fixture (deterministic, 20 records):
      - 5 records with tags=["a"]
      - 5 records with tags=["b"]
      - 4 records with tags=["c"]
      - 3 records with tags=["a", "raw:noise"] -> 'a' count + 3
      - 3 records with tags=["b", "domain:x"] -> 'b' count + 3

    Expected counts: a=8, b=8, c=4. All clear the count >= 3 floor.
    """
    for _ in range(5):
        store.insert(_make(text="a", tags=["a"]))
    for _ in range(5):
        store.insert(_make(text="b", tags=["b"]))
    for _ in range(4):
        store.insert(_make(text="c", tags=["c"]))
    for _ in range(3):
        store.insert(_make(text="ar", tags=["a", "raw:noise"]))
    for _ in range(3):
        store.insert(_make(text="bd", tags=["b", "domain:x"]))

    # Compute expected via the reference algorithm inline.
    records = store.all_records()
    tag_counts: dict[str, int] = {}
    for r in records:
        for t in r.tags or []:
            if t.startswith("raw:") or t.startswith("domain:"):
                continue
            tag_counts[t] = tag_counts.get(t, 0) + 1
    expected = [
        {
            "pattern": f"tag:{tag}",
            "confidence": min(1.0, count / 10.0),
            "evidence_count": count,
        }
        for tag, count in tag_counts.items()
        if count >= 3
    ]

    actual = _tier0_schema_surfacing(store)

    # Sort both sides by pattern for order-independent equality (dict-iter
    # order is insertion-order in py3.7+ but iter_record_columns batch order
    # is not guaranteed identical to all_records pandas-iterrows order).
    expected_sorted = sorted(expected, key=lambda d: d["pattern"])
    actual_sorted = sorted(actual, key=lambda d: d["pattern"])

    assert len(actual_sorted) == len(expected_sorted)
    for e, a in zip(expected_sorted, actual_sorted, strict=True):
        assert a["pattern"] == e["pattern"]
        assert a["evidence_count"] == e["evidence_count"]
        assert a["confidence"] == pytest.approx(e["confidence"])

    # Sanity: 3 patterns surface (a, b, c) — neither raw:noise nor domain:x.
    assert {c["pattern"] for c in actual} == {"tag:a", "tag:b", "tag:c"}
    by_pattern = {c["pattern"]: c["evidence_count"] for c in actual}
    assert by_pattern["tag:a"] == 8
    assert by_pattern["tag:b"] == 8
    assert by_pattern["tag:c"] == 4


# --------------------------------------------------------------------------- defensive JSON parse


def test_tier0_schema_surfacing_handles_malformed_tags_json_gracefully(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive try/except contract: malformed ``tags_json`` rows MUST
    NOT raise — they contribute zero tag counts, valid rows still surface.

    Strategy: monkeypatch-wrap ``store.iter_record_columns`` to inject a
    malformed row AFTER the real rows. The real iter yields 5 valid rows
    + 1 malformed row; the defensive
    ``try: json.loads... except json.JSONDecodeError`` in the function body
    absorbs the malformed row → no exception, candidates still surface for
    ``tag-good``.

    The malformed row is injected via the iterator wrap, never written to the
    store directly — ``_from_row`` parses ``tags_json`` without try/except
    and would crash ``all_records()`` on read, breaking test isolation.
    """
    # 5 valid records — well above CLUSTER_MIN_SIZE = 3.
    for _ in range(5):
        store.insert(_make(text="g", tags=["tag-good"]))

    # Capture the real iter and wrap it to append one malformed row at the end.
    real_iter = store.iter_record_columns

    def iter_with_malformed_tail(columns, **kwargs):  # noqa: ANN001 — match arg shape
        yield from real_iter(columns, **kwargs)
        # Malformed JSON — the defensive try/except must absorb this without
        # raising. (Real production data with a corrupted row column might look
        # like this if a write was interrupted mid-flush.)
        yield {"tags_json": "not valid json {{{"}

    monkeypatch.setattr(store, "iter_record_columns", iter_with_malformed_tail)

    # Must not raise: the projection path consumes the malformed row but
    # absorbs the JSONDecodeError.
    candidates = _tier0_schema_surfacing(store)

    # tag-good still surfaces (5 records, count=5, confidence=0.5).
    by_pattern = {c["pattern"]: c for c in candidates}
    assert "tag:tag-good" in by_pattern, (
        f"valid records' tag must still surface despite malformed-row tail; "
        f"got candidates={candidates!r}"
    )
    assert by_pattern["tag:tag-good"]["evidence_count"] == 5
    assert by_pattern["tag:tag-good"]["confidence"] == pytest.approx(0.5)


# ============================================================================
# run_heavy_consolidation single-materialisation invariant
# ============================================================================
#
# The ≤1 all_records() invariant on run_heavy_consolidation holds once the two
# `all_records()` callers in schema.py (induce_schemas_tier0 + persist_schema)
# use iter_record_columns projection instead.
#
# Before that migration: 2 calls when only induce_schemas_tier0 fires; 3 calls
# when persist_schema fires for an auto-status candidate.
# After: 1 call total (the records_by_id materialisation in
# run_heavy_consolidation, kept by the minimum-change branch).
#
# These tests ALSO lock the public contract of run_heavy_consolidation's
# return dict — protects against drive-by changes.


@pytest.fixture
def _patch_schema_embedder(monkeypatch: pytest.MonkeyPatch):
    """persist_schema's insert path embeds the schema summary; without this
    fixture each test pays ~5s embedder load. Mirrors test_schema_dedup.py."""
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


def _populate_for_heavy(store: MemoryStore) -> list[MemoryRecord]:
    """10 records on a single tag pair — clears (a) CLUSTER_MIN_SIZE record-count
    floor, (b) per-tag count >= 3 floor, (c) AUTO_INDUCT_COOCCURRENCE = 5 +
    AUTO_INDUCT_CONFIDENCE = 0.85 thresholds (count=10, confidence=1.0). This
    forces the FULL schema-induction path including persist_schema's keeper
    scan, exercising the single-materialisation invariant against all three
    former all_records() call sites."""
    from iai_mcp.types import EMBED_DIM as _EMBED_DIM
    from datetime import datetime as _dt, timezone as _tz
    from uuid import uuid4 as _uuid

    inserted: list[MemoryRecord] = []
    for i in range(10):
        r = MemoryRecord(
            id=_uuid(),
            tier="episodic",
            literal_surface=f"meeting-rec-{i}",
            aaak_index="",
            embedding=[1.0] + [0.0] * (_EMBED_DIM - 1),
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=_dt.now(_tz.utc),
            updated_at=_dt.now(_tz.utc),
            tags=["meeting", "notes"],
            language="en",
        )
        store.insert(r)
        inserted.append(r)
    return inserted


def test_run_heavy_consolidation_calls_all_records_at_most_once(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    _patch_schema_embedder,
) -> None:
    """Single-materialisation invariant: run_heavy_consolidation
    calls store.all_records() AT MOST ONCE per invocation.

    Before the schema.py migration: 2 or 3 calls — one from the
    records_by_id materialisation in run_heavy_consolidation, one from
    induce_schemas_tier0, and one from persist_schema's keeper scan when an
    auto-status candidate is persisted.

    After: 1 call (only the records_by_id materialisation; the schema.py paths
    use iter_record_columns instead).

    The test seeds 10 records on a single tag pair. count=10, confidence=1.0
    → status="auto" → persist_schema fires → all three former call sites
    are exercised in one heavy invocation, so the assertion ``call_count <= 1``
    exercises the full path.
    """
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    _populate_for_heavy(store)

    spy = MagicMock(wraps=store.all_records)
    monkeypatch.setattr(store, "all_records", spy)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    run_heavy_consolidation(
        store, session_id="s-w4-inv", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )

    assert spy.call_count <= 1, (
        f"run_heavy_consolidation must call store.all_records() "
        f"AT MOST ONCE per invocation; got {spy.call_count} call(s). "
        f"Former contributors: records_by_id materialisation, "
        f"induce_schemas_tier0, and persist_schema's keeper scan."
    )


def test_run_heavy_consolidation_iter_record_columns_called_at_least_once(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    _patch_schema_embedder,
) -> None:
    """Companion to the single-materialisation invariant: proves the
    schema-surfacing and schema.py paths actually executed via
    iter_record_columns. Without this companion, a buggy implementation that
    elided BOTH all_records() AND iter_record_columns would silently pass the
    ≤1 invariant."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    _populate_for_heavy(store)

    spy = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "iter_record_columns", spy)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    run_heavy_consolidation(
        store, session_id="s-w4-iter", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )

    assert spy.call_count >= 1, (
        f"run_heavy_consolidation must call store.iter_record_columns() at "
        f"least once per invocation (_tier0_schema_surfacing path + "
        f"schema.py paths); got {spy.call_count} call(s)."
    )


def test_run_heavy_consolidation_returns_expected_keys(
    store: MemoryStore,
    _patch_schema_embedder,
) -> None:
    """Lock the public contract of run_heavy_consolidation's return dict.
    Protects against drive-by changes to the function body."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

    _populate_for_heavy(store)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    result = run_heavy_consolidation(
        store, session_id="s-w4-keys", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )

    expected_keys = {
        "mode",
        "tier",
        "summaries_created",
        "decay_result",
        "schema_candidates",
        "schemas_induced",
    }
    assert set(result.keys()) == expected_keys, (
        f"run_heavy_consolidation public contract: expected keys "
        f"{sorted(expected_keys)}; got {sorted(result.keys())}"
    )
    assert result["mode"] == "heavy"
    assert result["tier"] in ("tier0", "tier1")
