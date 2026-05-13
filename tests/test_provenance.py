"""Tests for provenance append on recall and edge-based contradict."""
from __future__ import annotations

from uuid import UUID

from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM
from tests.test_store import _make


def test_recall_appends_provenance(tmp_path):
    """every recall creates a new provenance entry on every returned record."""
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    before = store.get(r.id).provenance
    dispatch(
        store,
        "memory_recall",
        {"cue": "test cue", "session_id": "s1", "cue_embedding": r.embedding},
    )
    after = store.get(r.id).provenance
    assert len(after) == len(before) + 1
    assert after[-1]["cue"] == "test cue"
    assert after[-1]["session_id"] == "s1"
    assert "ts" in after[-1]


def test_recall_appends_provenance_twice(tmp_path):
    """Two recalls -> two new provenance entries (reconsolidation never idempotent)."""
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    dispatch(store, "memory_recall", {"cue": "first", "session_id": "s1", "cue_embedding": r.embedding})
    dispatch(store, "memory_recall", {"cue": "second", "session_id": "s2", "cue_embedding": r.embedding})
    after = store.get(r.id).provenance
    assert len(after) == 2
    assert after[0]["cue"] == "first"
    assert after[1]["cue"] == "second"


def test_contradict_creates_linked_record_without_rewrite(tmp_path):
    """ edge-based: original preserved, new record linked."""
    store = MemoryStore(path=tmp_path)
    r = _make(text="Original fact")
    store.insert(r)
    original_text = store.get(r.id).literal_surface

    result = dispatch(
        store,
        "memory_contradict",
        {"id": str(r.id), "new_fact": "Contradicting fact", "cue_embedding": r.embedding},
    )
    assert result["edge_type"] == "contradicts"
    assert result["original_id"] == str(r.id)

    # original remains unchanged (full rewrite is ).
    assert store.get(r.id).literal_surface == original_text

    # New record contains the contradicting fact.
    new_rec = store.get(UUID(result["new_record_id"]))
    assert new_rec is not None
    assert new_rec.literal_surface == "Contradicting fact"
    assert "contradict" in new_rec.tags


def test_contradict_unknown_record_raises(tmp_path):
    """Tampering resistance (T-01-01): unknown UUID -> ValueError (RPC error -32000)."""
    import pytest
    store = MemoryStore(path=tmp_path)
    phantom_id = "11111111-2222-3333-4444-555555555555"
    with pytest.raises(ValueError):
        dispatch(
            store,
            "memory_contradict",
            {"id": phantom_id, "new_fact": "x", "cue_embedding": [0.0] * EMBED_DIM},
        )


# -------------------------------------------------------- H-02 guard


def test_contradict_rejects_cyrillic_new_fact_without_raw_tag(tmp_path):
    """H-02: memory_contradict enforces English-raw on the new record.

    Without the constitutional guard a Cyrillic new_fact would store silently
    and corrupt the invariant that established.
    """
    import pytest

    store = MemoryStore(path=tmp_path)
    r = _make(text="Original fact")
    store.insert(r)

    with pytest.raises(ValueError, match="constitutional"):
        dispatch(
            store,
            "memory_contradict",
            {
                "id": str(r.id),
                "new_fact": "Не правда, было не так",
                "cue_embedding": r.embedding,
            },
        )


def test_contradict_new_record_has_aaak_index_stamped(tmp_path):
    """H-02: aaak_index is generated on the new record -- not left empty."""
    store = MemoryStore(path=tmp_path)
    r = _make(text="Original English fact")
    store.insert(r)

    result = dispatch(
        store,
        "memory_contradict",
        {
            "id": str(r.id),
            "new_fact": "English correction",
            "cue_embedding": r.embedding,
        },
    )
    new_rec = store.get(UUID(result["new_record_id"]))
    assert new_rec is not None
    # generate_aaak_index yields W:<wing>/R:<room>/E:<entities>/T:<tags>.
    # It must be non-empty and contain at least the wing segment.
    assert new_rec.aaak_index != ""
    assert new_rec.aaak_index.startswith("W:")


# -------------------------------------- append_provenance_batch


def test_append_provenance_batch_basic(tmp_path):
    """3 records, 3 distinct entries, one batch call: each record gets its own entry."""
    store = MemoryStore(path=tmp_path)
    r1 = _make(text="a")
    r2 = _make(text="b")
    r3 = _make(text="c")
    for r in (r1, r2, r3):
        store.insert(r)
    e1 = {"ts": "2026-04-17T00:00:00Z", "cue": "one", "session_id": "s1"}
    e2 = {"ts": "2026-04-17T00:00:01Z", "cue": "two", "session_id": "s1"}
    e3 = {"ts": "2026-04-17T00:00:02Z", "cue": "three", "session_id": "s1"}
    store.append_provenance_batch([(r1.id, e1), (r2.id, e2), (r3.id, e3)])
    got1 = store.get(r1.id).provenance
    got2 = store.get(r2.id).provenance
    got3 = store.get(r3.id).provenance
    assert got1[-1]["cue"] == "one"
    assert got2[-1]["cue"] == "two"
    assert got3[-1]["cue"] == "three"


def test_append_provenance_batch_multiple_entries_same_id(tmp_path):
    """Two entries for the same record: provenance list grows by exactly 2 entries, in order."""
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    before = len(store.get(r.id).provenance)
    e1 = {"ts": "t1", "cue": "first", "session_id": "s"}
    e2 = {"ts": "t2", "cue": "second", "session_id": "s"}
    store.append_provenance_batch([(r.id, e1), (r.id, e2)])
    after = store.get(r.id).provenance
    assert len(after) == before + 2
    assert after[-2]["cue"] == "first"
    assert after[-1]["cue"] == "second"


def test_append_provenance_batch_empty_list(tmp_path):
    """Empty input -> no-op, no exception."""
    store = MemoryStore(path=tmp_path)
    store.append_provenance_batch([])  # must not raise
    # Sanity: store still functional.
    r = _make()
    store.insert(r)
    assert store.get(r.id) is not None


def test_append_provenance_batch_unknown_id(tmp_path):
    """Unknown id is silently skipped; known id is appended correctly."""
    from uuid import uuid4
    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)
    phantom = uuid4()
    e_known = {"ts": "t1", "cue": "known", "session_id": "s"}
    e_unknown = {"ts": "t0", "cue": "unknown", "session_id": "s"}
    # Should NOT raise on the phantom id (matches append_provenance semantics).
    store.append_provenance_batch([(phantom, e_unknown), (r.id, e_known)])
    after = store.get(r.id).provenance
    assert after[-1]["cue"] == "known"


def test_append_provenance_batch_equivalence_with_single(tmp_path):
    """Byte parity: N single calls on store A == 1 batch call on store B, modulo updated_at."""
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    store_a = MemoryStore(path=path_a)
    store_b = MemoryStore(path=path_b)
    # Seed same 3 records into both stores with SAME uuids so we can compare directly.
    import copy
    from uuid import uuid4
    base_records = [_make(text=f"r{i}") for i in range(3)]
    for r in base_records:
        store_a.insert(r)
        store_b.insert(copy.deepcopy(r))  # avoid shared refs
    entries = [
        {"ts": "t1", "cue": "e1", "session_id": "sA"},
        {"ts": "t2", "cue": "e2", "session_id": "sB"},
        {"ts": "t3", "cue": "e3", "session_id": "sC"},
    ]
    # Path A: N single calls.
    for r, e in zip(base_records, entries):
        store_a.append_provenance(r.id, e)
    # Path B: one batch call.
    store_b.append_provenance_batch(list(zip([r.id for r in base_records], entries)))
    # Compare provenance lists (ignore updated_at since it is clock-based).
    for r in base_records:
        prov_a = store_a.get(r.id).provenance
        prov_b = store_b.get(r.id).provenance
        assert prov_a == prov_b, f"provenance diverged for {r.id}: {prov_a} vs {prov_b}"


def test_append_provenance_batch_scan_count(tmp_path, monkeypatch):
    """N+1 collapse: 5 single calls -> 5 to_pandas scans; 1 batch call -> 1 scan."""
    from iai_mcp.store import MemoryStore as _MS, RECORDS_TABLE

    store = MemoryStore(path=tmp_path)
    recs = [_make(text=f"r{i}") for i in range(5)]
    for r in recs:
        store.insert(r)

    # Monkey-counter on the *records table*'s to_pandas by shimming open_table.
    counter = [0]
    original_open_table = store.db.open_table

    def _counting_open_table(name, *args, **kwargs):
        tbl = original_open_table(name, *args, **kwargs)
        if name == RECORDS_TABLE:
            original_to_pandas = tbl.to_pandas

            def _counting_to_pandas(*a, **k):
                counter[0] += 1
                return original_to_pandas(*a, **k)

            tbl.to_pandas = _counting_to_pandas  # type: ignore[assignment]
        return tbl

    store.db.open_table = _counting_open_table  # type: ignore[assignment]

    # --- N singles ---
    counter[0] = 0
    for r in recs:
        store.append_provenance(r.id, {"ts": "t", "cue": "c", "session_id": "s"})
    single_scans = counter[0]

    # --- 1 batch ---
    counter[0] = 0
    store.append_provenance_batch(
        [(r.id, {"ts": "t", "cue": "c2", "session_id": "s"}) for r in recs]
    )
    batch_scans = counter[0]

    assert single_scans == 5, f"append_provenance (single) did {single_scans} records-table scans; expected 5"
    assert batch_scans == 1, f"append_provenance_batch did {batch_scans} scans; expected 1 (N+1 collapse)"
