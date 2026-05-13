"""redesign (08-CONTEXT.md ): production answer-packing entry-point contract.

Tests the new public function `recall_for_response(...)` introduced by
. Contract:

- Signature: store, graph, assignment, rich_club, embedder, cue,
  session_id, budget_tokens=1500, profile_state=None, turn=0, mode='concept'.
- NO `k_hits` parameter — calling with `k_hits=10` MUST raise TypeError.
- Returns RecallResponse (not _RecallCoreResult).
- Packs hits under `budget_tokens` per the pre-Phase-8 production
  contract: each hit contributes `len(literal_surface) // 4` tokens to
  the running budget; loop breaks when `budget_used + tokens > budget_tokens`
  AND `len(hits) >= 1` (always at least one hit when one exists).
- mode plumbing: the `mode` parameter threads through to
  `_recall_core` unchanged.

Cross-file: see `tests/test_recall_for_benchmark.py` for the top-K
contract, and `tests/test_recall_core_unit.py` for the underlying
`_recall_core` shape and stage-internal behaviour.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord, RecallResponse


# ------------------------------------------------------------ test fixtures


class _FakeEmbedder:
    """Stand-in embedder. The cue's embedding is configurable per-test."""

    DIM = EMBED_DIM

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec if vec is not None else [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _make(
    vec: list[float], text: str = "rec", aaak: str = "", tier: str = "episodic",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index=aaak,
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


def _build_store_and_graph(
    tmp_path, n: int, surface_len: int = 4,
) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord]]:
    """Build N records with primary-axis distinct embeddings + matching graph.

    Each record's literal_surface has `surface_len` characters so the
    per-hit token estimate is `surface_len // 4`. Tune `surface_len` to
    control budget-pack behaviour deterministically.
    """
    store = MemoryStore(path=tmp_path / "lancedb")
    recs: list[MemoryRecord] = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        text = "x" * surface_len
        rec = _make(vec, text=text)
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(
            rec.id, community_id=None, embedding=list(rec.embedding),
        )
        graph._nx.nodes[str(rec.id)].update({
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    return store, graph, recs


def _flat_assignment(recs: list[MemoryRecord]) -> CommunityAssignment:
    """Single flat community covering all records (healthy graph baseline)."""
    cid = uuid4()
    centroid = [1.0] + [0.0] * (EMBED_DIM - 1)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )


# -------------------------------------------------- contract / signature tests


def test_recall_for_response_no_k_hits_param(tmp_path) -> None:
    """Test 1: calling with `k_hits=10` raises TypeError.

    The contract split is the whole point: production answer-packing
    cannot accept a top-K cap parameter, otherwise an optional argument
    would let the two contracts silently swap semantics.
    """
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    with pytest.raises(TypeError):
        recall_for_response(
            store=store, graph=graph, assignment=assignment,
            rich_club=[], embedder=_FakeEmbedder(),
            cue="test", session_id="s1",
            k_hits=10,    # this kwarg does not exist
        )


def test_recall_for_response_returns_recall_response_type(tmp_path) -> None:
    """Test 2: returns a RecallResponse with all 7 fields populated."""
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s2",
    )

    assert isinstance(resp, RecallResponse)
    assert isinstance(resp.hits, list)
    assert isinstance(resp.anti_hits, list)
    assert isinstance(resp.activation_trace, list)
    assert isinstance(resp.budget_used, int)
    assert isinstance(resp.hints, list)
    assert isinstance(resp.cue_mode, str)
    assert isinstance(resp.patterns_observed, list)


def test_recall_for_response_packs_under_budget(tmp_path) -> None:
    """Test 3: hits packed under `budget_tokens` per the pre-Phase-8 contract.

    Each record's literal_surface = 200 chars -> tokens = 200 // 4 = 50.
    With budget_tokens=120, the loop breaks after the first hit
    (50 tokens). Adding a second would push us to 100; adding a third
    would push us to 150 > 120 AND len(hits) >= 1, so we break.
    """
    from iai_mcp.pipeline import recall_for_response

    # surface_len=200 -> 50 tokens per hit.
    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=200)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s3", budget_tokens=120,
    )

    # Tight budget: 1 fits (50 tokens, budget_used=50), 2nd would push
    # to 100 (still <= 120, fits), 3rd would push to 150 > 120 AND
    # len(hits) >= 1, break. So we get exactly 2 hits.
    assert len(resp.hits) == 2
    assert resp.budget_used == 100


def test_recall_for_response_returns_all_with_unlimited_budget(tmp_path) -> None:
    """Test 4: with budget_tokens=10000 (effectively unlimited), all hits are returned.

    The exhaustion is the ranker's natural stop, not the budget cap.
    """
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=4)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s4", budget_tokens=10000,
    )

    # All 5 records fit (5 * 1 token = 5 tokens, budget = 10000).
    assert len(resp.hits) == 5


def test_recall_for_response_minimum_one_hit(tmp_path) -> None:
    """Test 5: with extremely tight budget, the minimum-1-hit guard returns 1 hit.

    Even when the first hit's tokens exceed `budget_tokens`, the contract
    guarantees `len(hits) >= 1` when at least one ranked hit exists.
    """
    from iai_mcp.pipeline import recall_for_response

    # surface_len=400 -> 100 tokens per hit; budget=50 (tighter than even 1 hit).
    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=400)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s5", budget_tokens=50,
    )

    # One hit always survives (the production "always at least one" guard).
    assert len(resp.hits) == 1


def test_recall_for_response_threads_mode_to_core(tmp_path) -> None:
    """Test 5b: wiring — `mode` flows from entry point to `_recall_core` unchanged.

    Calling with `mode="verbatim"` must produce a response whose
    `cue_mode == "verbatim"` (proves the parameter threaded through).
    """
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    resp_v = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s5b", budget_tokens=10000,
        mode="verbatim",
    )
    assert resp_v.cue_mode == "verbatim", (
        f"verbatim mode did not propagate; cue_mode={resp_v.cue_mode}"
    )

    resp_c = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s5c", budget_tokens=10000,
        mode="concept",
    )
    assert resp_c.cue_mode == "concept"


def test_recall_for_response_signature_has_no_k_hits_param() -> None:
    """The function signature exposes `budget_tokens` and `mode` but NOT `k_hits`."""
    import inspect
    from iai_mcp.pipeline import recall_for_response

    sig = inspect.signature(recall_for_response)
    assert "budget_tokens" in sig.parameters
    assert "mode" in sig.parameters
    assert "k_hits" not in sig.parameters, (
        "recall_for_response signature must NOT carry a k_hits parameter "
        "(D-07 contract split — the entry-point split exists so the two "
        "response shapes can never silently swap via an optional kwarg)."
    )


def test_recall_for_response_default_budget_tokens_1500() -> None:
    """The default budget_tokens is 1500 (matches pre-Phase-8 production default)."""
    import inspect
    from iai_mcp.pipeline import recall_for_response

    sig = inspect.signature(recall_for_response)
    assert sig.parameters["budget_tokens"].default == 1500


# ------------------------------------------------------ shared / parity tests


def test_recall_for_response_shares_core_with_benchmark(tmp_path) -> None:
    """Both entry points share `_recall_core` — only the final pack/cap differs.

    This test proves ("only the final pack/cap differs"): when
    called with the same fixture and the same `mode`, the cue-matched
    record (cosine=1.0) must be the top hit on BOTH entry points, and
    both must surface the same set of record_ids (only ordering of
    tied-cosine records may differ across calls due to age-penalty
    floating-point drift between the two `datetime.now()` calls).
    """
    from iai_mcp.pipeline import recall_for_benchmark, recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=8, surface_len=4)
    assignment = _flat_assignment(recs)

    resp_y = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-shared-r",
        budget_tokens=10000,    # unlimited so all ranked hits surface
    )
    resp_b = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-shared-b",
        k_hits=100,             # > graph size so all ranked hits surface
    )

    # Top hit must be the cue-matched record (cosine=1.0 vs orthogonal 0.0
    # for the rest) on both entry points — this is the load-bearing
    # ranking claim of D-07.
    assert resp_y.hits[0].record_id == resp_b.hits[0].record_id, (
        "top scored hit (cosine=1.0 cue-match) must be identical across "
        "entry points; only the final pack/cap is supposed to differ"
    )
    # Both entry points must surface the same SET of record_ids when
    # neither cap is binding. The within-set ordering may vary among
    # tied-cosine records due to age-penalty floating-point drift.
    r_set = {h.record_id for h in resp_y.hits}
    b_set = {h.record_id for h in resp_b.hits}
    assert r_set == b_set, (
        f"recall_for_response and recall_for_benchmark must surface the "
        f"same record-id set when neither cap binds; got\n"
        f"  response only: {r_set - b_set}\n"
        f"  benchmark only: {b_set - r_set}"
    )
