"""Tests for R1 — schema-pattern dedup in persist_schema.

Locked decisions covered (06-CONTEXT.md):
- persist_schema dedups by tag `pattern:{candidate.pattern}` against
  existing tier="semantic" records; reinforces schema_instance_of edges
  onto the keeper instead of inserting a duplicate row.
- new event kind `schema_reinforced` with payload
  `{schema_id, pattern, evidence_added, total_evidence}`; severity "info";
  source_ids `[keeper_id, *new_evidence_ids[:5]]`.
- single test file, pytest convention (`tmp_path` LanceDB root).

R1 acceptance (06-SPEC.md): N persist_schema calls for the same pattern
collapse to ONE schema record, with the keeper's incoming
`schema_instance_of` edge count equal to the cumulative distinct evidence
count across all calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.events import query_events
from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------- helpers


def _rec(
    *,
    text: str = "t",
    tags: list[str] | None = None,
    language: str = "en",
    tier: str = "episodic",
    detail_level: int = 2,
) -> MemoryRecord:
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
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    """Avoid loading bge-m3 during dedup tests — perf hygiene."""
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


# ---------------------------------------------------------------- Task 1: events taxonomy + write-event smoke


def test_events_module_docstring_lists_schema_reinforced():
    """events.py module docstring documents the new `schema_reinforced` kind."""
    import iai_mcp.events as events_mod

    doc = events_mod.__doc__ or ""
    assert "schema_reinforced" in doc, (
        "events.py module docstring missing `schema_reinforced` taxonomy entry "
        "(Plan 06-01 D-10). Add a additions block after the "
        "section listing the new event kind, payload schema, and source_ids note."
    )


def test_write_event_accepts_schema_reinforced_kind(tmp_path):
    """schema_reinforced event round-trips through write_event + query_events."""
    from iai_mcp.events import write_event

    store = MemoryStore(path=tmp_path)
    keeper_id = uuid4()
    ev_id = uuid4()
    write_event(
        store,
        kind="schema_reinforced",
        data={
            "schema_id": str(keeper_id),
            "pattern": "tags:capture+role:user",
            "evidence_added": 1,
            "total_evidence": 5,
        },
        severity="info",
        source_ids=[keeper_id, ev_id],
    )
    rows = query_events(store, kind="schema_reinforced")
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "schema_reinforced"
    assert row["severity"] == "info"
    payload = row["data"]
    assert payload["pattern"] == "tags:capture+role:user"
    assert payload["evidence_added"] == 1
    assert payload["total_evidence"] == 5
    assert payload["schema_id"] == str(keeper_id)


# ---------------------------------------------------------------- Task 2: persist_schema dedup branch (R1)


def _seed_evidence(store: MemoryStore, n: int) -> list[MemoryRecord]:
    """Insert n fresh episodic evidence records (one per call iteration).

    Each record carries the canonical capture/role tags so a downstream
    induced schema for `tags:capture+role:user` traces back to genuine
    evidence. Returns the list in insertion order.
    """
    recs = [_rec(text=f"ev{i}", tags=["capture", "role:user"]) for i in range(n)]
    for r in recs:
        store.insert(r)
    return recs


def test_persist_schema_dedups_same_pattern(tmp_path):
    """R1: 10 persist_schema calls for the same pattern produce ONE schema record."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    pattern = "tags:capture+role:user"
    pattern_tag = f"pattern:{pattern}"

    for _ in range(10):
        ev = _seed_evidence(store, 1)
        cand = SchemaCandidate(
            pattern=pattern,
            confidence=0.9,
            evidence_count=1,
            evidence_ids=[ev[0].id],
            status="auto",
        )
        persist_schema(store, cand)

    schemas = [
        r for r in store.all_records()
        if r.tier == "semantic" and pattern_tag in (r.tags or [])
    ]
    assert len(schemas) == 1, (
        f"expected exactly one schema for pattern {pattern!r}, got {len(schemas)}"
    )


def test_persist_schema_reinforces_edges_on_dedup(tmp_path):
    """R1: schema_instance_of edge count to keeper == cumulative evidence count."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    pattern = "tags:capture+role:user"
    pattern_tag = f"pattern:{pattern}"

    keeper_id = None
    cumulative_evidence = 0
    for _ in range(10):
        ev = _seed_evidence(store, 1)
        cand = SchemaCandidate(
            pattern=pattern,
            confidence=0.9,
            evidence_count=1,
            evidence_ids=[ev[0].id],
            status="auto",
        )
        sid = persist_schema(store, cand)
        keeper_id = keeper_id or sid
        cumulative_evidence += 1

    # store.boost_edges canonicalises (src, dst) to a sorted tuple, so the
    # keeper appears in EITHER column depending on the string ordering of
    # the paired evidence UUID. OR-count both columns to recover the true
    # edge-incidence count (each edge row has the keeper in exactly one
    # column — no double-count).
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    keeper_str = str(keeper_id)
    sio = edges_df[
        (edges_df["edge_type"] == "schema_instance_of")
        & ((edges_df["dst"] == keeper_str) | (edges_df["src"] == keeper_str))
    ]
    assert len(sio) == cumulative_evidence, (
        f"expected {cumulative_evidence} schema_instance_of edges incident on keeper, "
        f"got {len(sio)}"
    )

    # Sanity: exactly one keeper survives.
    keepers = [
        r for r in store.all_records()
        if r.tier == "semantic" and pattern_tag in (r.tags or [])
    ]
    assert len(keepers) == 1


def test_persist_schema_emits_schema_reinforced_event(tmp_path):
    """R1 + 9 reinforced events + 1 induction event after 10 calls."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    pattern = "tags:capture+role:user"

    for _ in range(10):
        ev = _seed_evidence(store, 1)
        cand = SchemaCandidate(
            pattern=pattern,
            confidence=0.9,
            evidence_count=1,
            evidence_ids=[ev[0].id],
            status="auto",
        )
        persist_schema(store, cand)

    induction_events = query_events(store, kind="schema_induction_run")
    reinforced_events = query_events(store, kind="schema_reinforced", limit=100)

    matching_inductions = [
        e for e in induction_events if e["data"].get("pattern") == pattern
    ]
    matching_reinforcements = [
        e for e in reinforced_events if e["data"].get("pattern") == pattern
    ]
    assert len(matching_inductions) == 1, (
        f"expected 1 schema_induction_run event, got {len(matching_inductions)}"
    )
    assert len(matching_reinforcements) == 9, (
        f"expected 9 schema_reinforced events, got {len(matching_reinforcements)}"
    )

    # query_events sorts newest first; the FIRST in the list is the most
    # recent reinforcement and must carry the highest total_evidence.
    payloads = [e["data"] for e in matching_reinforcements]
    for p in payloads:
        assert "schema_id" in p
        assert p["pattern"] == pattern
        assert isinstance(p["evidence_added"], int)
        assert isinstance(p["total_evidence"], int)
    totals = [p["total_evidence"] for p in payloads]
    # Newest first → totals should be monotonically non-increasing in list order.
    assert totals == sorted(totals, reverse=True), (
        f"total_evidence should grow over time; saw {totals}"
    )


def test_persist_schema_returns_keeper_id(tmp_path):
    """R1: persist_schema returns the SAME UUID across N calls for same pattern."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    pattern = "tags:capture+role:user"

    returned_ids = []
    for _ in range(10):
        ev = _seed_evidence(store, 1)
        cand = SchemaCandidate(
            pattern=pattern,
            confidence=0.9,
            evidence_count=1,
            evidence_ids=[ev[0].id],
            status="auto",
        )
        returned_ids.append(persist_schema(store, cand))

    first = returned_ids[0]
    assert all(rid == first for rid in returned_ids), (
        f"persist_schema should return the keeper id on every call; got {returned_ids}"
    )


def test_persist_schema_does_not_collapse_distinct_patterns(tmp_path):
    """R1 negative: distinct patterns produce distinct schema records."""
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)

    ev_a = _seed_evidence(store, 1)
    sid_a = persist_schema(
        store,
        SchemaCandidate(
            pattern="A",
            confidence=0.9,
            evidence_count=1,
            evidence_ids=[ev_a[0].id],
            status="auto",
        ),
    )
    ev_b = _seed_evidence(store, 1)
    sid_b = persist_schema(
        store,
        SchemaCandidate(
            pattern="B",
            confidence=0.9,
            evidence_count=1,
            evidence_ids=[ev_b[0].id],
            status="auto",
        ),
    )
    assert sid_a != sid_b

    schemas = [
        r for r in store.all_records()
        if r.tier == "semantic" and any(
            t in ("pattern:A", "pattern:B") for t in (r.tags or [])
        )
    ]
    assert len(schemas) == 2
    patterns = sorted(
        t.split(":", 1)[1]
        for r in schemas
        for t in r.tags
        if t.startswith("pattern:")
    )
    assert patterns == ["A", "B"]
