"""Regression fence — the community gate is a MODE-DEPENDENT diagnostic,
not a hard filter.

The load-bearing claim has two parts:
  1. Records OUTSIDE the top-3 gated communities can still surface in
     `scored_hits[:K]` when their cosine rank is high. The gate never
     filters; it only biases.
  2. The bias is mode-dependent:
       - verbatim mode -> 0.0 (no categorical bias)
       - concept mode -> 0.1 (soft +10% categorical hint to records
                                inside top-3 gated communities)

Previously the gate was a HARD FILTER: `pipeline_recall` reduced
`candidates` to records inside the top-3 communities. On a degenerate
one-record-per-community graph (the cold-start bug class) only 3
candidates survived; gold (12-24 records) could not. The redesign closes
this by reading the candidate
pool from cosine top-K_CANDIDATES instead, and applying a mode-dependent
soft bias only at the Stage-5 ranking step.

This fence catches both:
  (a) someone re-introducing a hard filter (test 1 below);
  (b) someone changing the bias constants or removing the mode
      dispatch (test 2 below).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import (
    COMMUNITY_BIAS_CONCEPT,
    COMMUNITY_BIAS_VERBATIM,
    _gate_bias_for_mode,
    _recall_core,
    recall_for_benchmark,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------- test fixtures


class _FakeEmbedder:
    """Stand-in embedder; cue's embedding is configurable."""

    DIM = EMBED_DIM

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec if vec is not None else [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _make(vec: list[float], text: str = "rec") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _build_one_record_per_community(
    tmp_path,
    n: int = 50,
) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord], CommunityAssignment]:
    """Replicates the cold-start bug class: 50 records, 1 community each.

    Each record's embedding is the i-th unit basis vector in EMBED_DIM
    space, so all records are mutually orthogonal AND aligned to a
    distinct primary axis. The assignment is constructed directly
    (bypassing Leiden), placing each record in its OWN community whose
    centroid equals the record's embedding. This means the community
    nearest the cue (by centroid cosine) is the community containing
    the record nearest the cue (by record cosine).

    Mirrors the deleted tests/test_pipeline_community_gate_augment.py
    helper `_build_degenerate_graph_and_assignment` (patch
    era). Kept as a private helper here since the patch tests are gone.
    """
    store = MemoryStore(path=tmp_path / "hippo")
    recs: list[MemoryRecord] = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        rec = _make(vec, text=f"rec{i}")
        store.insert(rec)
        recs.append(rec)

    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(
            rec.id, community_id=None, embedding=list(rec.embedding),
        )
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })

    # One record per community: centroid = record's embedding.
    cids = [uuid4() for _ in recs]
    centroids = {cids[i]: list(recs[i].embedding) for i in range(len(recs))}
    node_to_community = {recs[i].id: cids[i] for i in range(len(recs))}
    mid_regions = {cids[i]: [recs[i].id] for i in range(len(recs))}
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test-degenerate",
        top_communities=cids[:3],
        mid_regions=mid_regions,
    )
    return store, graph, recs, assignment


# ------------------------------------------------------------------- tests


def test_records_outside_gated_communities_surface_via_cosine(tmp_path):
    """anti-hard-filter fence: gold OUTSIDE top-3 communities still surfaces.

    Build a 50-record fixture where each record has a distinct primary
    axis. The cue points at axis 5 (rec[5] is the gold). The community
    gate (top-3 by centroid cosine) returns the community of rec[5]
    plus two arbitrary others (the orthogonal axes all tie at cosine 0
    so the secondary order is by stable-sort UUID — out of our control,
    but reliably NOT covering all 50 communities).

    The cue points at axis 5, NOT at axis 0; rec[5] is therefore in
    its own community (because each record is in its own community in
    this fixture). The cosine top-K pool surfaces rec[5] regardless of
    whether the gate's secondary picks happen to include it.

    Mode is "concept" so the +0.1*cos bias for top-3-gated records is
    active; the gold record (cosine 1.0 to the cue) wins on its raw
    cosine alone, even when the gate's bias goes to other communities.

    If a future change re-introduces a hard filter (where `candidates`
    are reduced to gate members only), this test
    fails: rec[5] has cosine 1.0 but only the 3 gated communities
    survive, and on the orthogonal-axes geometry the gate may rank
    rec[5]'s community OUTSIDE the top-3, dropping the gold record
    from the candidate pool.
    """
    store, graph, recs, assignment = _build_one_record_per_community(tmp_path, n=50)

    # Cue points at axis 5; rec[5] has cosine 1.0; all others are 0.
    cue_vec = [0.0] * EMBED_DIM
    cue_vec[5] = 1.0
    embedder = _FakeEmbedder(vec=cue_vec)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 5", session_id="s-gate-diag-1",
        k_hits=10, mode="concept",
    )

    found_ids = {h.record_id for h in resp.hits}
    assert recs[5].id in found_ids, (
        "gold record (cosine 1.0 to cue, on axis 5) "
        "is NOT in top-10 hits. The gate must NEVER filter — only "
        "bias. If this fails, someone re-introduced the hard-filter "
        "behavior (candidates restricted to top-3 "
        "gated-community members)."
    )
    # Stronger version: the gold record is the TOP hit (cosine 1.0 vs
    # all others tied at 0; even with concept-mode +0.1*cos bias for
    # records in the gated set, the gold's cosine 1.0 beats anything
    # the bias can synthesize on a 0-cosine record).
    assert resp.hits[0].record_id == recs[5].id, (
        f"gold should be top-1 by cosine alone (1.0 vs ~0); "
        f"got {resp.hits[0].record_id} as top hit. Possible cause: "
        "Stage 5 weights were re-tuned, or community-bias scalar is "
        "being applied multiplicatively/subtractively instead of "
        "additively to records inside the gated set."
    )


def test_mode_bias_verbatim_zero_concept_nonzero(tmp_path):
    """canonical fence: verbatim mode bias=0.0; concept mode bias=0.1.

    Records inside top-3 gated communities get a score bonus ONLY in
    concept mode. A record outside top-3 communities never gets the
    bonus regardless of mode. The same fixture is recalled in both
    modes; we assert:
      - Both calls return the SAME record list (gate never filters,
        only biases ranking).
      - In verbatim mode, the gated record's score reflects ZERO
        community contribution (cosine + AAAK + degree + age only).
      - In concept mode, the gated record's score is approximately
        `verbatim_score + 0.1 * cos` higher than its verbatim
        counterpart.
      - The non-gated control record's score is unchanged across modes
        (the bias only applies to records inside top-3 gated communities).

    This catches: (a) someone changing COMMUNITY_BIAS_VERBATIM away
    from 0.0 or COMMUNITY_BIAS_CONCEPT away from 0.1; (b) someone
    removing the `mode` dispatch from `_gate_bias_for_mode` or
    `_recall_core`'s Stage 5; (c) someone reintroducing a hard filter
    that drops non-gated records.

    Symbol-level pre-flight: `_gate_bias_for_mode("verbatim") == 0.0`
    and `_gate_bias_for_mode("concept") == 0.1` (constants intact).

    Fixture geometry — keep it simple to make scores byte-identical
    across the two modes for the non-bias terms:
      - All records have the SAME aaak (empty), SAME tier (episodic),
        SAME literal_surface length (so age, deg_norm contribute
        identically across records).
      - No edges in the graph -> max_deg = 0 -> log_max_deg = 0 ->
        deg_norm == 0 for every record -> W_DEGREE * deg_norm == 0.
      - No profile_state -> no per-record gain product.
      - No structural_weight -> no structural-similarity term.
      => base_s = W_COSINE * cos - W_AGE * age (everything else
         constant or zero across records).
    """
    # Symbol-level pre-flight assertions — contract surface intact.
    assert COMMUNITY_BIAS_VERBATIM == 0.0
    assert COMMUNITY_BIAS_CONCEPT == 0.1
    assert _gate_bias_for_mode("verbatim") == 0.0
    assert _gate_bias_for_mode("concept") == 0.1
    assert _gate_bias_for_mode("unknown") == 0.0  # defensive default

    # Build a 50-record fixture: 1 record per community on distinct
    # primary axes (orthogonal). The cue points at axis 0; rec[0] sits
    # in community c0 whose centroid is the axis-0 unit vector — so the
    # gate places c0 first by centroid cosine.
    store, graph, recs, assignment = _build_one_record_per_community(tmp_path, n=50)

    # Cue points at axis 0 (matching rec[0]'s primary axis).
    cue_vec = [0.0] * EMBED_DIM
    cue_vec[0] = 1.0
    embedder = _FakeEmbedder(vec=cue_vec)

    # Identify the GATED record (rec[0], in community c0 at top-1 by
    # centroid cosine) and the CONTROL record. The control is whichever
    # record sits in a community OUTSIDE the top-3 gated set; we pick
    # rec[20] (community c20, axis 20 — definitely orthogonal to cue,
    # cosine 0.0). We also pre-compute the gold expectation that rec[0]
    # gets the +0.1 community bonus in concept mode.
    rec_GATED = recs[0]
    rec_CONTROL = recs[20]

    # --- recall in verbatim mode ---
    result_v = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 0", session_id="s-mode-bias-v",
        mode="verbatim",
    )

    # --- recall in concept mode ---
    result_c = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 0", session_id="s-mode-bias-c",
        mode="concept",
    )

    # --- 1. Same record list — the gate must NEVER filter. ---
    verbatim_ids = {h.record_id for h in result_v.scored_hits}
    concept_ids = {h.record_id for h in result_c.scored_hits}
    assert verbatim_ids == concept_ids, (
        "gate must NEVER filter; mode change should not "
        f"alter the record list. verbatim_only={verbatim_ids - concept_ids}, "
        f"concept_only={concept_ids - verbatim_ids}"
    )

    # --- 2. Lookup GATED and CONTROL records' scores in both modes. ---
    v_gated = next(h for h in result_v.scored_hits if h.record_id == rec_GATED.id)
    c_gated = next(h for h in result_c.scored_hits if h.record_id == rec_GATED.id)
    v_ctrl = next(h for h in result_v.scored_hits if h.record_id == rec_CONTROL.id)
    c_ctrl = next(h for h in result_c.scored_hits if h.record_id == rec_CONTROL.id)

    # --- 3. Concept mode: GATED record gains COMMUNITY_BIAS_CONCEPT * cos. ---
    # The score delta is the *only* term that changes across modes for
    # the gated record (everything else is identical: same record, same
    # cue, same fixture, same time, same profile_state). The cosine of
    # rec_GATED to the cue is 1.0 (axis 0 vs axis-0 cue), so the
    # expected bonus is 0.1 * 1.0 == 0.1.
    cos_GATED = 1.0
    expected_bonus = COMMUNITY_BIAS_CONCEPT * cos_GATED
    delta_gated = c_gated.score - v_gated.score
    assert delta_gated == pytest.approx(expected_bonus, abs=1e-4), (
        f"concept mode: GATED record (rec[0], cosine={cos_GATED}) "
        f"should gain ~{expected_bonus:.4f} from "
        f"COMMUNITY_BIAS_CONCEPT * cos when transitioning verbatim -> "
        f"concept. Got delta = c_gated.score - v_gated.score = "
        f"{delta_gated:.4f}.\n"
        f"v_gated.score = {v_gated.score:.4f}; "
        f"c_gated.score = {c_gated.score:.4f}."
    )

    # --- 4. CONTROL record (outside top-3 gated): score UNCHANGED. ---
    # rec_CONTROL is in c20 — definitely NOT in top_communities[:3] for
    # this fixture (c0/c1/c2 dominate by centroid cosine since cue is
    # at axis 0; the orthogonal-axes geometry sorts the rest by stable-
    # sort UUID). The control record's score must be byte-identical
    # across modes.
    delta_ctrl = c_ctrl.score - v_ctrl.score
    assert delta_ctrl == pytest.approx(0.0, abs=1e-6), (
        f"concept mode: CONTROL record (rec[20], cosine 0 to cue, "
        f"in non-gated community c20) must NOT receive the community "
        f"bias. Got delta = c_ctrl.score - v_ctrl.score = {delta_ctrl:.6f}; "
        f"expected 0.0.\n"
        f"v_ctrl.score = {v_ctrl.score:.6f}; "
        f"c_ctrl.score = {c_ctrl.score:.6f}."
    )

    # --- 5. Verbatim mode: bias contribution is identically zero. ---
    # In verbatim mode COMMUNITY_BIAS_VERBATIM == 0.0, so the gated
    # record's score does NOT receive any community contribution.
    # Because cosine for rec_CONTROL is 0 (axis 20 vs axis-0 cue) and
    # rec_GATED has cosine 1.0, the verbatim-mode score difference is
    # purely W_COSINE * (1.0 - 0.0) = W_COSINE — the cosine term alone.
    # No additive bias term sneaks in; all other contributions
    # (aaak, deg_norm, age) are identical by fixture construction.
    from iai_mcp.pipeline import W_COSINE
    expected_verbatim_delta = W_COSINE * (1.0 - 0.0)
    actual_verbatim_delta = v_gated.score - v_ctrl.score
    assert actual_verbatim_delta == pytest.approx(
        expected_verbatim_delta, abs=1e-4
    ), (
        f"verbatim mode: gated vs control score delta should be "
        f"W_COSINE * cos_diff = {W_COSINE} * 1.0 = {expected_verbatim_delta:.4f} "
        f"with NO community-bias contribution. Got delta = "
        f"{actual_verbatim_delta:.4f}. If this differs, either "
        f"COMMUNITY_BIAS_VERBATIM is non-zero, or the mode dispatch "
        f"in _gate_bias_for_mode is broken."
    )
