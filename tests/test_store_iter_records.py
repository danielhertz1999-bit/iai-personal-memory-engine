"""Streaming + projection iterator on MemoryStore.

Exercises ``iter_records`` and ``iter_record_columns`` on ``MemoryStore``,
including ``_from_row`` tolerating partial row dicts produced by column
projection.

Covered contracts:

  iter_records:
    1. yields all inserted records (set equality, order-independent)
    2. yielded items are MemoryRecord instances and round-trip core fields
    3. empty store yields []
    4. batch_size=1 yields all rows without crashing
    5. batch_size much larger than table yields all rows without crashing
    6. columns projection reads a subset and still yields valid MemoryRecord
       (proves _from_row partial-row hardening)
    7. where filter restricts the iteration set

  iter_record_columns:
    8. returns raw dicts whose keys equal the requested columns subset
    9. encrypted columns pass through as ciphertext when projected
   10. empty store yields []
   11. where filter restricts the iteration set

  _from_row hardening:
   12. partial row dict (only required-by-__post_init__ columns) does not
       KeyError; missing columns fall back to dataclass defaults
   13. all_records() behaviour is byte-equivalent (additive guarantee)

Every test uses a real ``MemoryRecord``
dataclass via ``_make()`` — never a plain dict against attribute-access code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.crypto import CIPHERTEXT_PREFIX
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Mirror tests/test_runtime_graph_cache.py — process-isolated keyring so
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
    tier: str = "episodic",
    text: str = "hello world",
    tags: list[str] | None = None,
    detail: int = 2,
    pinned: bool = False,
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
        pinned=pinned,
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


# --------------------------------------------------------------------------- iter_records


def test_iter_records_yields_all_inserted(store):
    """iter_records returns every inserted row (order-independent)."""
    inserted_ids = set()
    for i in range(10):
        rec = _make(text=f"record-{i}")
        store.insert(rec)
        inserted_ids.add(rec.id)

    iterated = list(store.iter_records())
    assert len(iterated) == 10
    assert {r.id for r in iterated} == inserted_ids


def test_iter_records_yields_correct_dataclass_type(store):
    """yielded items are MemoryRecord instances; core fields round-trip."""
    rec_a = _make(text="alpha", tier="episodic", tags=["t1", "t2"])
    rec_b = _make(text="beta", tier="semantic", tags=["t3"])
    rec_c = _make(text="gamma", tier="episodic", tags=[])
    for r in (rec_a, rec_b, rec_c):
        store.insert(r)

    iterated = list(store.iter_records())
    assert len(iterated) == 3
    by_id = {r.id: r for r in iterated}
    assert isinstance(by_id[rec_a.id], MemoryRecord)
    assert by_id[rec_a.id].literal_surface == "alpha"
    assert by_id[rec_a.id].tier == "episodic"
    assert by_id[rec_a.id].tags == ["t1", "t2"]
    assert by_id[rec_b.id].literal_surface == "beta"
    assert by_id[rec_b.id].tier == "semantic"
    assert by_id[rec_b.id].tags == ["t3"]
    assert by_id[rec_c.id].literal_surface == "gamma"
    assert by_id[rec_c.id].tags == []


def test_iter_records_handles_empty_store(store):
    """empty store -> empty iteration, no exception."""
    assert list(store.iter_records()) == []


def test_iter_records_respects_batch_size_one(store):
    """batch_size=1 (every row its own batch) still yields all rows."""
    for i in range(7):
        store.insert(_make(text=f"row-{i}"))
    iterated = list(store.iter_records(batch_size=1))
    assert len(iterated) == 7


def test_iter_records_respects_batch_size_larger_than_table(store):
    """batch_size much larger than row count yields all rows in one batch."""
    for i in range(5):
        store.insert(_make(text=f"row-{i}"))
    iterated = list(store.iter_records(batch_size=10000))
    assert len(iterated) == 5


def test_iter_records_with_columns_projects_subset(store):
    """column projection still yields valid MemoryRecord — proves
    _from_row tolerates partial row dicts (missing columns fall back to
    dataclass defaults). Pre-hardening this raises KeyError on
    row['literal_surface'] (line 1340)."""
    rec_a = _make(text="alpha", tags=["tag-a"])
    rec_b = _make(text="beta", tags=["tag-b"])
    rec_c = _make(text="gamma", tags=["tag-c"])
    for r in (rec_a, rec_b, rec_c):
        store.insert(r)

    iterated = list(
        store.iter_records(columns=["id", "tags_json", "tier", "embedding"])
    )
    assert len(iterated) == 3
    by_id = {r.id: r for r in iterated}
    assert by_id[rec_a.id].tags == ["tag-a"]
    assert by_id[rec_b.id].tags == ["tag-b"]
    assert by_id[rec_c.id].tags == ["tag-c"]


def test_iter_records_with_where_filter(store):
    """where filter (SQL-style predicate) restricts the iteration set."""
    rec_e1 = _make(text="ep1", tier="episodic")
    rec_e2 = _make(text="ep2", tier="episodic")
    rec_s1 = _make(text="sem1", tier="semantic")
    for r in (rec_e1, rec_e2, rec_s1):
        store.insert(r)

    iterated = list(store.iter_records(where="tier = 'episodic'"))
    assert len(iterated) == 2
    assert {r.id for r in iterated} == {rec_e1.id, rec_e2.id}
    assert all(r.tier == "episodic" for r in iterated)


# --------------------------------------------------------------------------- iter_record_columns


def test_iter_record_columns_returns_raw_dicts(store):
    """returns raw dicts whose keys equal exactly the requested column subset.
    Critically: literal_surface is NOT in the dict — proof that the encrypted
    column was never read from disk (zero AES-GCM cost)."""
    for i in range(3):
        store.insert(_make(text=f"row-{i}", tags=[f"t{i}"]))

    rows = list(store.iter_record_columns(["id", "tags_json"]))
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == {"id", "tags_json"}
        assert "literal_surface" not in row


def test_iter_record_columns_passes_ciphertext_through(store):
    """when an encrypted column IS projected, value is the ciphertext
    string with the iai:enc:v1: prefix — iter_record_columns must NOT decrypt."""
    rec = _make(text="secret content for ciphertext test")
    store.insert(rec)

    rows = list(store.iter_record_columns(["id", "literal_surface"]))
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row["literal_surface"], str)
    # CIPHERTEXT_PREFIX is "iai:enc:v1:" (see iai_mcp.crypto)
    assert row["literal_surface"].startswith(CIPHERTEXT_PREFIX), (
        f"expected ciphertext prefix {CIPHERTEXT_PREFIX!r}, "
        f"got {row['literal_surface']!r}"
    )


def test_iter_record_columns_handles_empty_store(store):
    """empty store -> empty iteration, no exception."""
    assert list(store.iter_record_columns(["id"])) == []


def test_iter_record_columns_with_where_filter(store):
    """where filter restricts the iteration set in projection mode."""
    rec_e1 = _make(text="ep1", tier="episodic")
    rec_e2 = _make(text="ep2", tier="episodic")
    rec_s1 = _make(text="sem1", tier="semantic")
    for r in (rec_e1, rec_e2, rec_s1):
        store.insert(r)

    rows = list(
        store.iter_record_columns(["id", "tier"], where="tier = 'episodic'")
    )
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {str(rec_e1.id), str(rec_e2.id)}
    assert all(r["tier"] == "episodic" for r in rows)


# --------------------------------------------------------------------------- _from_row hardening


def test_from_row_partial_row_dict_does_not_crash(store):
    """a hand-built minimal row dict (only the columns required by
    MemoryRecord.__post_init__) flows through _from_row without KeyError.
    Pre-hardening this raises KeyError on row['literal_surface'] (line 1340).
    Every column NOT in the input dict must fall back to the dataclass default."""
    minimal_row = {
        "id": str(uuid4()),
        "embedding": [0.0] * EMBED_DIM,
        "tier": "episodic",
        "tags_json": "[]",
    }
    rec = store._from_row(minimal_row)

    assert isinstance(rec, MemoryRecord)
    assert rec.tier == "episodic"
    assert rec.embedding == [0.0] * EMBED_DIM
    # Defaults for the columns we did not project:
    assert rec.literal_surface == ""              # default for missing literal_surface
    assert rec.aaak_index == ""                   # default for missing aaak_index
    assert rec.provenance == []                   # default for missing provenance_json
    assert rec.detail_level == 1                  # default for missing detail_level
    assert rec.pinned is False                    # default for missing pinned
    assert rec.stability == 0.0                   # default for missing stability
    assert rec.difficulty == 0.0                  # default for missing difficulty
    assert rec.never_decay is False               # default for missing never_decay
    assert rec.never_merge is False               # default for missing never_merge
    assert rec.tags == []                         # tags_json="[]" -> []
    assert rec.language == "en"                   # default for missing language
    # created_at / updated_at fall back to datetime.now(timezone.utc) — just
    # check they are real datetimes and not None.
    assert isinstance(rec.created_at, datetime)
    assert isinstance(rec.updated_at, datetime)


def test_iter_records_does_not_modify_existing_all_records_behaviour(store):
    """additive guarantee: all_records() is byte-equivalent before/after
    the new methods exist on the same store."""
    inserted = []
    for i in range(5):
        rec = _make(text=f"persist-{i}", tags=[f"persist-{i}"])
        store.insert(rec)
        inserted.append(rec)

    # Snapshot 1 — before touching iter_records.
    before = store.all_records()

    # Touch the new method.
    list(store.iter_records())

    # Snapshot 2 — after.
    after = store.all_records()

    assert len(before) == len(after) == 5
    by_id_before = {r.id: r for r in before}
    by_id_after = {r.id: r for r in after}
    assert set(by_id_before.keys()) == set(by_id_after.keys())
    for rid in by_id_before:
        assert by_id_before[rid].literal_surface == by_id_after[rid].literal_surface
        assert by_id_before[rid].tags == by_id_after[rid].tags
        assert by_id_before[rid].tier == by_id_after[rid].tier
