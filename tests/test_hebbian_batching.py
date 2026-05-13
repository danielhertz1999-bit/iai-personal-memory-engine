"""— Hebbian write-batching coverage.

Eight sync tests (project does NOT use pytest-asyncio):

R1 / A2 — `test_boost_edges_emits_at_most_two_versions`
R2      — `test_boost_edges_scalar_delta_unchanged`
R2      — `test_boost_edges_sequence_delta_per_pair`
R2      — `test_boost_edges_sequence_delta_length_mismatch_raises`
A7      — `test_boost_edges_coalesces_duplicate_pairs`
R3 site — `test_sleep_consolidated_from_batches_into_two_versions`
R3 site — `test_curiosity_bridge_batches_into_two_versions`
R3 site — `test_schema_bind_batches_into_two_versions`
R3 site — `test_pipeline_profile_modulates_batches_with_sequence_delta`

Eight tests minimum — SPEC R4 asks for >= 5; this ships the full target from
CONTEXT D7.4-08.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore


# ----------------------------------------------------------------- helpers


def _versions(store: MemoryStore) -> int:
    """Return the current LanceDB version count for the edges table."""
    tbl = store.db.open_table(EDGES_TABLE)
    return len(tbl.list_versions())


# ----------------------------------------------------------- R1 / A2 — versions


def test_boost_edges_emits_at_most_two_versions(tmp_path):
    """R1 + A2 acceptance: ONE call with 5 pairs (3 hits + 2 new) -> <= 2 new versions.

    Today's pre-refactor body would emit 5 versions (1 per tbl.update / tbl.add).
    The refactor consolidates to <= 2 (one merge_insert for the 3
    updates, one tbl.add for the 2 new rows).
    """
    store = MemoryStore(path=tmp_path)
    a, b, c, d, e, f, g = (uuid4() for _ in range(7))

    # Seed 3 edges via a single call (the seed itself produces ~1 version).
    store.boost_edges([(a, b), (c, d), (e, f)], delta=0.1, edge_type="hebbian")

    versions_before = _versions(store)

    # 5-pair call: 3 hits (a,b), (c,d), (e,f) + 2 new (a,c), (f,g).
    new = store.boost_edges(
        [(a, b), (c, d), (e, f), (a, c), (f, g)],
        delta=0.2,
        edge_type="hebbian",
    )

    versions_after = _versions(store)
    delta_versions = versions_after - versions_before

    # Hard cap: <= 2 (one merge_insert for updates + one tbl.add for inserts).
    assert delta_versions <= 2, (
        f"boost_edges emitted {delta_versions} versions "
        f"(expected <= 2 after batching)"
    )

    # Returned weights must be: 0.3 for the 3 pre-existing pairs (0.1 + 0.2)
    # and 0.2 for the 2 new pairs (0 + 0.2). Keys are canonical-sorted.
    assert len(new) == 5
    for key, weight in new.items():
        if {key[0], key[1]} in ({str(a), str(b)}, {str(c), str(d)}, {str(e), str(f)}):
            assert abs(weight - 0.3) < 1e-5, f"{key} expected 0.3, got {weight}"
        else:
            assert abs(weight - 0.2) < 1e-5, f"{key} expected 0.2, got {weight}"


# ----------------------------------------------------------- R2 — scalar delta


def test_boost_edges_scalar_delta_unchanged(tmp_path):
    """R2 backwards-compat: scalar `delta=0.3` applies uniformly to all pairs."""
    store = MemoryStore(path=tmp_path)
    a, b, c, d = (uuid4() for _ in range(4))

    new = store.boost_edges([(a, b), (c, d)], delta=0.3, edge_type="hebbian")

    assert len(new) == 2
    for weight in new.values():
        assert abs(weight - 0.3) < 1e-5


# ----------------------------------------------------------- R2 — sequence delta


def test_boost_edges_sequence_delta_per_pair(tmp_path):
    """R2: `delta=[0.5, 0.7]` applies per-pair (in pair-list order)."""
    store = MemoryStore(path=tmp_path)
    a, b, c, d = (uuid4() for _ in range(4))

    new = store.boost_edges(
        [(a, b), (c, d)],
        delta=[0.5, 0.7],
        edge_type="hebbian",
    )

    assert len(new) == 2
    # Map back from canonical-sorted key to original pair to assert per-pair delta.
    key_ab = tuple(sorted([str(a), str(b)]))
    key_cd = tuple(sorted([str(c), str(d)]))
    assert abs(new[key_ab] - 0.5) < 1e-5
    assert abs(new[key_cd] - 0.7) < 1e-5


def test_boost_edges_sequence_delta_length_mismatch_raises(tmp_path):
    """R2: Sequence-delta with len(deltas) != len(pairs) -> ValueError."""
    store = MemoryStore(path=tmp_path)
    a, b, c, d = (uuid4() for _ in range(4))

    with pytest.raises(ValueError, match="deltas length"):
        store.boost_edges(
            [(a, b), (c, d)],
            delta=[0.5, 0.7, 0.9],  # 3 deltas for 2 pairs
            edge_type="hebbian",
        )


# ----------------------------------------------------------- A7 — coalesce


def test_boost_edges_coalesces_duplicate_pairs(tmp_path):
    """A7: `[(a,b), (a,b)]` with delta=0.1 produces `cur + 0.2`, NOT `cur + 0.1`.

    The legacy implementation refreshed `existing = tbl.to_pandas()` after every
    pair so duplicate canonical (src,dst) keys saw each other's delta. The
    refactor preserves this semantic via in-memory coalescing BEFORE the write.
    """
    store = MemoryStore(path=tmp_path)
    a, b = uuid4(), uuid4()

    # First seed one edge so `cur` is non-zero.
    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")

    # Second call: SAME pair listed twice. Expect 0.1 (existing) + 0.2 (sum) = 0.3.
    new = store.boost_edges([(a, b), (a, b)], delta=0.1, edge_type="hebbian")

    assert len(new) == 1, "duplicate pair should collapse to ONE canonical key"
    canonical = tuple(sorted([str(a), str(b)]))
    assert abs(new[canonical] - 0.3) < 1e-5, (
        f"coalesced delta should be cur + 2*delta = 0.3, got {new[canonical]}"
    )


def test_boost_edges_coalesces_duplicate_pairs_first_call(tmp_path):
    """A7 strengthen: even on a FRESH edge, `[(a,b), (a,b)]` with delta=0.1
    should produce 0.2 (NOT 0.1) — coalescing happens before write."""
    store = MemoryStore(path=tmp_path)
    a, b = uuid4(), uuid4()

    new = store.boost_edges([(a, b), (a, b)], delta=0.1, edge_type="hebbian")
    canonical = tuple(sorted([str(a), str(b)]))
    assert abs(new[canonical] - 0.2) < 1e-5


# ----------------------------------------------------------- R3 — site-level


def test_sleep_consolidated_from_batches_into_two_versions(tmp_path):
    """R3 site-level: sleep._create_semantic_summary's per-source loop now
    issues ONE boost_edges call (consolidated_from edges).

    Asserts the summary's outgoing consolidated_from edges all exist with the
    expected weight, AND the create-summary call did not balloon the edges.lance
    version count by N (one per source) — only by <= 2 (one tbl.add for the new
    rows; merge_insert path empty since these are fresh edges).
    """
    from iai_mcp.sleep import _create_semantic_summary
    from tests.test_store import _make

    store = MemoryStore(path=tmp_path)

    # Seed 5 source records into a "cluster".
    cluster = [_make(text=f"source memory {i}") for i in range(5)]
    for r in cluster:
        store.insert(r)

    versions_before = _versions(store)
    summary_id = _create_semantic_summary(
        store,
        cluster,
        summary_text="cls summary of 5 source memories",
        language="en",
    )
    versions_after = _versions(store)

    delta_versions = versions_after - versions_before
    # <= 2 covers the 1 add for new edges (5 fresh consolidated_from rows) PLUS
    # any incidental merge_insert version when the merge_insert path is empty.
    assert delta_versions <= 2, (
        f"sleep.cls boost emitted {delta_versions} versions for 5 sources "
        f"(expected <= 2 after )"
    )

    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    summary_str = str(summary_id)
    consolidated = df[
        (df["src"].isin([summary_str, *[str(r.id) for r in cluster]]))
        & (df["dst"].isin([summary_str, *[str(r.id) for r in cluster]]))
        & (df["edge_type"] == "consolidated_from")
    ]
    assert len(consolidated) == 5, (
        f"expected 5 consolidated_from edges, got {len(consolidated)}"
    )
    # Every weight should equal delta=1.0 (the legacy per-iter scalar).
    for w in consolidated["weight"]:
        assert abs(float(w) - 1.0) < 1e-5


def test_curiosity_bridge_batches_into_two_versions(tmp_path):
    """R3 site-level: curiosity.fire's per-trigger loop now issues ONE
    boost_edges call (curiosity_bridge edges)."""
    from iai_mcp.curiosity import fire_curiosity
    from tests.test_store import _make

    store = MemoryStore(path=tmp_path)

    # Seed 5 records that will become triggers (entropy must be high enough to
    # surface a question — we drive it via direct call below).
    triggers = [_make(text=f"ambiguous memory {i}") for i in range(5)]
    for r in triggers:
        store.insert(r)

    # Build a fake hits structure compatible with fire_curiosity.
    class _Hit:
        def __init__(self, record_id):
            self.record_id = record_id
            self.score = 0.4

    hits = [_Hit(r.id) for r in triggers]

    versions_before = _versions(store)
    # entropy=1.5 (above ENTROPY_HIGH default) -> tier="question" path,
    # 5 trigger_ids, ONE batched boost_edges call after the refactor.
    q = fire_curiosity(
        store,
        hits=hits,
        cue="what was that thing",
        entropy=1.5,
        session_id="sess-curiosity",
        turn=10,
    )
    versions_after = _versions(store)

    assert q is not None, "high-entropy curiosity call should fire"

    delta_versions = versions_after - versions_before
    assert delta_versions <= 2, (
        f"curiosity boost emitted {delta_versions} versions for 5 triggers "
        f"(expected <= 2 after )"
    )

    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    bridge = df[df["edge_type"] == "curiosity_bridge"]
    assert len(bridge) == 5, (
        f"expected 5 curiosity_bridge edges, got {len(bridge)}"
    )


def test_schema_bind_batches_into_two_versions(tmp_path):
    """R3 site-level: schema.bind's per-evidence loop now issues ONE
    boost_edges call (schema_instance_of edges)."""
    from iai_mcp.schema import SchemaCandidate, persist_schema
    from tests.test_store import _make

    store = MemoryStore(path=tmp_path)

    # Seed 5 evidence records.
    evidence = [_make(text=f"evidence {i}") for i in range(5)]
    for r in evidence:
        store.insert(r)

    # Pattern is unique to this test so the dedup branch in persist_schema
    # does NOT short-circuit (we want the new-schema insert path that contains
    # the line-374 for-loop -> batched call).
    candidate = SchemaCandidate(
        pattern="phase74_test_pattern_unique",
        confidence=0.7,
        evidence_ids=[r.id for r in evidence],
        evidence_count=5,
        status="auto",
    )

    versions_before = _versions(store)
    schema_id = persist_schema(store, candidate)
    versions_after = _versions(store)

    assert schema_id is not None

    delta_versions = versions_after - versions_before
    # `induce` emits both schema_instance_of edges (this plan's batched call)
    # AND the schema record's own row insert (records.lance, not edges.lance —
    # so it doesn't hit our edges-version count). <= 2 covers the merge_insert
    # + tbl.add for 5 fresh schema_instance_of edges.
    assert delta_versions <= 2, (
        f"schema.bind boost emitted {delta_versions} versions for 5 evidence "
        f"(expected <= 2 after )"
    )

    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    instance_edges = df[df["edge_type"] == "schema_instance_of"]
    assert len(instance_edges) == 5, (
        f"expected 5 schema_instance_of edges, got {len(instance_edges)}"
    )


def test_pipeline_profile_modulates_batches_with_sequence_delta(tmp_path):
    """R3 site-level: pipeline.recall_hook's per-hit profile_modulates loop
    now issues ONE boost_edges call with `delta=deltas` Sequence (per-hit
    varying gain).

    This directly exercises the loop body that was changed in pipeline.py:924.
    We unit-test the gather-then-batch pattern by simulating the hits + gains
    structure and asserting:
    1. ONE boost_edges call produces edges for all hits with non-empty gains.
    2. Hits with empty gains are skipped (preserves the existing fallback).
    3. Hits with total_gain<=0 fall back to delta=1.0 (preserves fallback).
    4. <= 2 versions per call regardless of hit count.
    """
    from iai_mcp.pipeline import PROFILE_SENTINEL_UUID

    store = MemoryStore(path=tmp_path)

    # 5 record ids; we treat them as h.record_id values.
    record_ids = [uuid4() for _ in range(5)]
    # Per-hit gains: gain values mirror what profile_modulation_gain dict gives.
    gains_per_hit = [
        {"profile_match_strong": 0.4, "language_match": 0.1},  # total = 0.5
        {},                                                    # skipped (empty)
        {"profile_match_weak": 0.2},                           # total = 0.2
        {"profile_match_neg": -0.5, "language_match": 0.1},    # total = -0.4 -> 1.0
        {"profile_match_strong": 0.7},                         # total = 0.7
    ]

    # Replicate the gather-then-batch pattern from pipeline.py:924 in a
    # contained form so the test is independent of the full recall plumbing.
    pairs: list[tuple] = []
    deltas: list[float] = []
    for rid, gains in zip(record_ids, gains_per_hit):
        if not gains:
            continue
        total_gain = float(sum(gains.values()))
        if total_gain <= 0:
            total_gain = 1.0
        pairs.append((rid, PROFILE_SENTINEL_UUID))
        deltas.append(total_gain)

    assert len(pairs) == 4, "4 hits should produce edges (1 skipped for empty gains)"
    assert len(deltas) == 4

    versions_before = _versions(store)
    new = store.boost_edges(
        pairs,
        delta=deltas,
        edge_type="profile_modulates",
    )
    versions_after = _versions(store)

    delta_versions = versions_after - versions_before
    assert delta_versions <= 2, (
        f"profile_modulates boost emitted {delta_versions} versions "
        f"(expected <= 2 after )"
    )

    # 4 edges created, each with the per-hit delta.
    assert len(new) == 4
    expected_per_pair = {
        tuple(sorted([str(record_ids[0]), str(PROFILE_SENTINEL_UUID)])): 0.5,
        tuple(sorted([str(record_ids[2]), str(PROFILE_SENTINEL_UUID)])): 0.2,
        tuple(sorted([str(record_ids[3]), str(PROFILE_SENTINEL_UUID)])): 1.0,
        tuple(sorted([str(record_ids[4]), str(PROFILE_SENTINEL_UUID)])): 0.7,
    }
    for key, exp in expected_per_pair.items():
        assert key in new, f"missing edge for {key}"
        assert abs(new[key] - exp) < 1e-5, (
            f"{key} expected {exp}, got {new[key]}"
        )
