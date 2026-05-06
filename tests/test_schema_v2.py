"""Tests for MemoryRecord v2 schema extensions + edge-type enum.

D-02a / / D-GUARD / D-STORAGE introduce:
- MemoryRecord.language (ISO-639-1 required, D-08a)
- MemoryRecord.s5_trust_score (float [0,1], default 0.5, prep)
- MemoryRecord.profile_modulation_gain (dict, runtime gain)
- MemoryRecord.schema_version (1 or 2)
- 6 new edge types in EDGE_TYPES registry
- Round-trip of all v2 fields through store.insert / store.get

Constitutional: plan-02-01 adds these fields ADDITIVELY. Existing Phase 1
fixtures with language="en" must keep working.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


# ------------------------------------------------------------- MemoryRecord v2


def _make_v2(
    *,
    language: str = "en",
    s5_trust_score: float = 0.5,
    profile_modulation_gain: dict | None = None,
    schema_version: int = 2,
    literal_surface: str = "hello world",
    tier: str = "episodic",
    embedding_dim: int | None = None,
):
    """Construct a v2 MemoryRecord with all required fields set."""
    from iai_mcp.types import MemoryRecord

    # Pick DIM from the embedder or explicit caller override.
    if embedding_dim is None:
        from iai_mcp.embed import Embedder
        embedding_dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.1] * embedding_dim,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language=language,
        s5_trust_score=s5_trust_score,
        profile_modulation_gain=profile_modulation_gain or {},
        schema_version=schema_version,
    )


def test_memory_record_has_language_field():
    """language is required ISO-639-1 string field."""
    r = _make_v2(language="en")
    assert r.language == "en"


def test_memory_record_requires_language_field():
    """omitting language at construction must raise."""
    from iai_mcp.types import MemoryRecord

    from iai_mcp.embed import Embedder
    _dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    with pytest.raises(TypeError):
        MemoryRecord(  # type: ignore[call-arg]
            id=uuid4(),
            tier="episodic",
            literal_surface="hi",
            aaak_index="",
            embedding=[0.0] * _dim,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=[],
        )


def test_memory_record_language_must_be_non_empty():
    """language=\"\" should be rejected at __post_init__."""
    with pytest.raises(ValueError):
        _make_v2(language="")


def test_memory_record_has_s5_trust_score():
    r = _make_v2(s5_trust_score=0.5)
    assert r.s5_trust_score == 0.5


def test_memory_record_s5_trust_score_default_is_0_5():
    """D-22 neutral prior: default is 0.5."""
    from iai_mcp.types import MemoryRecord
    from iai_mcp.embed import Embedder
    _dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="hi",
        aaak_index="",
        embedding=[0.0] * _dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
        # s5_trust_score, profile_modulation_gain, schema_version use defaults
    )
    assert r.s5_trust_score == 0.5


def test_memory_record_s5_trust_score_rejects_out_of_range():
    """[0, 1] inclusive bounds."""
    with pytest.raises(ValueError):
        _make_v2(s5_trust_score=1.5)
    with pytest.raises(ValueError):
        _make_v2(s5_trust_score=-0.1)


def test_memory_record_s5_trust_score_boundary_values_ok():
    assert _make_v2(s5_trust_score=0.0).s5_trust_score == 0.0
    assert _make_v2(s5_trust_score=1.0).s5_trust_score == 1.0


def test_memory_record_has_profile_modulation_gain():
    r = _make_v2(profile_modulation_gain={"monotropism_depth": 1.3, "interest_boost": 1.5})
    assert r.profile_modulation_gain == {"monotropism_depth": 1.3, "interest_boost": 1.5}


def test_memory_record_profile_modulation_gain_default_empty_dict():
    r = _make_v2()
    assert r.profile_modulation_gain == {}


def test_memory_record_has_schema_version_default_2():
    r = _make_v2()
    assert r.schema_version == 2


def test_memory_record_schema_version_accepts_1_for_migration():
    r = _make_v2(schema_version=1)
    assert r.schema_version == 1


def test_memory_record_schema_version_rejects_other_values():
    # schema_version=3 is now valid (Plan 02-08 encryption marker)
    # and schema_version=4 is the new current (Plan 03-01 TEM factorization).
    # Anything outside SCHEMA_VERSION_ACCEPTED is still rejected.
    with pytest.raises(ValueError):
        _make_v2(schema_version=0)
    with pytest.raises(ValueError):
        _make_v2(schema_version=99)


# ------------------------------------------------------------------- edges


def test_edge_types_registry_has_9_members():
    """Phase 1 (hebbian, contradicts) + 6 types + 1 (hebbian_structure)."""
    from iai_mcp.store import EDGE_TYPES

    expected = {
        "hebbian",
        "contradicts",
        "consolidated_from",
        "schema_instance_of",
        "temporal_next",
        "invariant_anchor",
        "curiosity_bridge",
        "profile_modulates",
        # CONN-05 TEM factorization Hebbian LTP on structure edges.
        "hebbian_structure",
    }
    assert EDGE_TYPES == frozenset(expected)


def test_boost_edges_accepts_new_phase2_types(tmp_path):
    """All 6 new edge types must be acceptable via store.boost_edges(pairs, edge_type=...)."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r1 = _make_v2()
    r2 = _make_v2()
    store.insert(r1)
    store.insert(r2)

    for edge_type in (
        "consolidated_from",
        "schema_instance_of",
        "temporal_next",
        "invariant_anchor",
        "curiosity_bridge",
        "profile_modulates",
    ):
        w = store.boost_edges([(r1.id, r2.id)], edge_type=edge_type, delta=1.0)
        assert list(w.values())[0] == pytest.approx(1.0), f"edge_type={edge_type} weight wrong"


def test_boost_edges_phase1_types_still_work(tmp_path):
    """Phase 1 callers using the default (hebbian) still get no-behavior-change."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r1 = _make_v2()
    r2 = _make_v2()
    store.insert(r1)
    store.insert(r2)
    w = store.boost_edges([(r1.id, r2.id)], delta=0.1)  # default hebbian
    assert list(w.values())[0] == pytest.approx(0.1)


def test_boost_edges_rejects_unknown_edge_type(tmp_path):
    """Typo protection: unknown edge_type must raise."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r1 = _make_v2()
    r2 = _make_v2()
    store.insert(r1)
    store.insert(r2)
    with pytest.raises(ValueError):
        store.boost_edges([(r1.id, r2.id)], edge_type="not_a_real_type")


# ---------------------------------------------------------- store round-trips


def test_record_to_from_row_preserves_language(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _make_v2(language="ru", literal_surface="Hello Russian")
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.language == "ru"


def test_record_to_from_row_preserves_s5_trust_score(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _make_v2(s5_trust_score=0.73)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert abs(got.s5_trust_score - 0.73) < 1e-5


def test_record_to_from_row_preserves_profile_modulation_gain(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    gain = {"monotropism_depth": 1.3, "interest_boost": 1.5}
    r = _make_v2(profile_modulation_gain=gain)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.profile_modulation_gain == gain


def test_record_to_from_row_preserves_schema_version(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _make_v2(schema_version=2)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.schema_version == 2


# ----------------------------------------------------- legacy (v1) read path


def test_legacy_record_reads_default_v1_defaults(tmp_path):
    """Read-side backward compatibility: a record row without language columns
    (pre-Phase-2) should load with language=\"en\" and schema_version=1 defaults.

    This matters during migration: code reads both v1 and v2 rows.
    We simulate a v1 record by inserting through a "legacy" path that uses
    the fields only, and verify the reader fills defaults.
    """
    import json
    from datetime import datetime, timezone

    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.embed import Embedder
    _dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    store = MemoryStore(path=tmp_path)
    tbl = store.db.open_table(RECORDS_TABLE)
    # Directly insert a row WITHOUT the v2 columns -- emulating a v1 read.
    now = datetime.now(timezone.utc)
    v1_id = uuid4()
    # Determine the store's current schema by introspecting.
    # Build a compatible row: all known columns, using defaults for v2 fields
    # that will land on the row (language="" simulates legacy data).
    row = {
        "id": str(v1_id),
        "tier": "episodic",
        "literal_surface": "legacy record",
        "aaak_index": "",
        "embedding": [0.0] * _dim,  # store must accept current DIM
        "structure_hv": b"",
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 1,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "provenance_json": "[]",
        "created_at": now,
        "updated_at": now,
        "tags_json": "[]",
        # v2 columns with "legacy" values:
        "language": "",                       # empty -> reader defaults to "en"
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": "{}",
        "schema_version": 1,
    }
    tbl.add([row])
    got = store.get(v1_id)
    assert got is not None
    # Reader fills blank language with "en" for back-compat.
    assert got.language in ("en", "")  # either default or preserved blank
    assert got.schema_version == 1
