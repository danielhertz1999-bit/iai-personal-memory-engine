"""Phase 8 redesign (08-CONTEXT.md D-07): benchmark top-K entry-point contract.

Tests the new public function `recall_for_benchmark(...)` introduced by
Plan 08-02. Contract:

- Signature: store, graph, assignment, rich_club, embedder, cue,
  session_id, k_hits=10, profile_state=None, turn=0, mode='concept'.
- NO `budget_tokens` parameter — calling with `budget_tokens=1500`
  MUST raise TypeError.
- Returns RecallResponse with `len(hits) <= k_hits` (cap honoured).
- Hits are sorted by score descending (R5 deterministic tie-break by
  UUID-asc preserved from `_recall_core`).
- mode plumbing: bench callers pass `mode="concept"`; the parameter
  threads through to `_recall_core` unchanged.

Cross-file: see `tests/test_recall_for_response.py` for the production
budget-pack contract, and `tests/test_recall_core_unit.py` for the
underlying `_recall_core` shape.
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


def test_recall_for_benchmark_no_budget_tokens_param(tmp_path) -> None:
    """Test 6: calling with `budget_tokens=1500` raises TypeError.

    The contract split is the whole point: top-K benchmark cannot accept
    a token-budget parameter, otherwise an optional argument would let
    the two contracts silently swap semantics.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    with pytest.raises(TypeError):
        recall_for_benchmark(
            store=store, graph=graph, assignment=assignment,
            rich_club=[], embedder=_FakeEmbedder(),
            cue="test", session_id="s6",
            budget_tokens=1500,    # this kwarg does not exist
        )


def test_recall_for_benchmark_returns_at_most_k_hits(tmp_path) -> None:
    """Test 7: `len(hits) <= k_hits` — the cap is honoured.

    Build 12 records; ask for k_hits=5; assert len(hits) == 5.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=12)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s7", k_hits=5,
    )

    assert isinstance(resp, RecallResponse)
    assert len(resp.hits) == 5


def test_recall_for_benchmark_hits_sorted_by_score_desc(tmp_path) -> None:
    """Test 8: hits are sorted by `score` descending (R5 deterministic order)."""
    from iai_mcp.pipeline import recall_for_benchmark

    # 8 records on distinct axes; cue at axis 0 -> rank ordered by axis index.
    store, graph, recs = _build_store_and_graph(tmp_path, n=8)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s8", k_hits=10,
    )

    scores = [h.score for h in resp.hits]
    assert scores == sorted(scores, reverse=True), (
        f"recall_for_benchmark hits not sorted desc by score: {scores}"
    )


def test_recall_for_benchmark_returns_fewer_when_pool_is_small(tmp_path) -> None:
    """Test 9: with k_hits=20 and only 8 ranked records, returns 8 hits.

    The cap is the natural exhaustion of `_recall_core.scored_hits`, not k_hits.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=8)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s9", k_hits=20,
    )

    # Pool is 8; k_hits=20 caps at 8.
    assert len(resp.hits) == 8


def test_recall_for_benchmark_budget_used_is_informational(tmp_path) -> None:
    """Test 10: `budget_used` reflects the per-hit token estimate sum (not a cap)."""
    from iai_mcp.pipeline import recall_for_benchmark

    # surface_len=200 -> 50 tokens per hit. With k_hits=3 and 5 records,
    # budget_used = 3 * 50 = 150 (informational; no cap).
    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=200)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s10", k_hits=3,
    )

    assert len(resp.hits) == 3
    assert resp.budget_used == 150


def test_recall_for_benchmark_threads_mode_to_core(tmp_path) -> None:
    """D-02 mode plumbing: `mode='concept'` (bench default) flows through."""
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-mode", k_hits=10, mode="concept",
    )
    assert resp.cue_mode == "concept"


def test_recall_for_benchmark_signature_has_no_budget_tokens_param() -> None:
    """The function signature exposes `k_hits` and `mode` but NOT `budget_tokens`."""
    import inspect
    from iai_mcp.pipeline import recall_for_benchmark

    sig = inspect.signature(recall_for_benchmark)
    assert "k_hits" in sig.parameters
    assert "mode" in sig.parameters
    assert "budget_tokens" not in sig.parameters, (
        "recall_for_benchmark signature must NOT carry a budget_tokens "
        "parameter (D-07 contract split — the entry-point split exists so "
        "the two response shapes can never silently swap via an optional kwarg)."
    )


def test_recall_for_benchmark_default_k_hits_10() -> None:
    """The default k_hits is 10 (matches LongMemEval-S protocol convention)."""
    import inspect
    from iai_mcp.pipeline import recall_for_benchmark

    sig = inspect.signature(recall_for_benchmark)
    assert sig.parameters["k_hits"].default == 10
