"""Tests for iai_mcp.pipeline (D-13 5-stage retrieval pipeline).

Uses a FakeEmbedder fixture so tests don't pull BAAI/bge-small-en-v1.5 from
HuggingFace during every run. The Embedder contract verified separately in
test_embed.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import (
    W_AAAK,
    W_AGE,
    W_COSINE,
    W_DEGREE,
    _aaak_overlap,
    _community_gate,
    _cosine,
    _pick_seeds,
    recall_for_response,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


class _FakeEmbedder:
    """Stand-in for the configured embedder so tests don't require the model.

    Returns a deterministic primary-axis vector for any input. recall_for_response
    uses embedder.embed() only for the cue, so this is sufficient.

    DIM follows the default registry (bge-m3 = 1024d). Plan-02 tests
    that hand-build vectors must use `[1.0] + [0.0] * (DIM - 1)` style via this
    constant so the store.insert() dim-check passes.
    """

    DIM = EMBED_DIM  # 1024 under default (bge-m3)

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make(vec: list[float], text: str = "rec", aaak: str = "", detail: int = 2) -> MemoryRecord:
    """Construct a MemoryRecord for pipeline tests.

    Uses `tier="episodic"` to stay within TIER_ENUM; created_at at current UTC.
    language="en" required.
    """
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index=aaak,
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
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


# ---------------------------------------------------------- stage-unit tests


def test_community_gate_picks_nearest() -> None:
    """: top-1 gate on 3 centroids picks the one nearest the cue."""
    c0 = uuid4()
    c1 = uuid4()
    c2 = uuid4()
    centroids = {
        c0: [1.0] + [0.0] * (EMBED_DIM - 1),
        c1: [0.0] * 384,
        c2: [-1.0] + [0.0] * (EMBED_DIM - 1),
    }
    a = CommunityAssignment(community_centroids=centroids)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)
    gated = _community_gate(cue, a, top_n=1)
    assert len(gated) == 1
    assert gated[0] == c0


def test_community_gate_returns_top_n_in_order() -> None:
    c0 = uuid4()
    c1 = uuid4()
    c2 = uuid4()
    centroids = {
        c0: [1.0] + [0.0] * (EMBED_DIM - 1),
        c1: [0.5, 0.5] + [0.0] * (EMBED_DIM - 2),
        c2: [-1.0] + [0.0] * (EMBED_DIM - 1),
    }
    a = CommunityAssignment(community_centroids=centroids)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)
    gated = _community_gate(cue, a, top_n=3)
    assert gated == [c0, c1, c2]


def test_aaak_overlap_basic_jaccard() -> None:
    assert _aaak_overlap("", "anything") == 0.0
    assert _aaak_overlap("x", "") == 0.0
    assert _aaak_overlap("a b", "a b") == 1.0  # identical
    # Jaccard({a,b}, {b,c}) = 1 / 3
    assert abs(_aaak_overlap("a b", "b c") - 1 / 3) < 1e-9


def test_aaak_overlap_slash_split_symmetric() -> None:
    """AAAK tokens use '/' as separator; both sides must split on it."""
    # Identical slash-delimited paths -> 1.0 (bug fix: cue side also splits).
    assert _aaak_overlap("auth/login", "auth/login") == 1.0
    # Partial share: {auth, login} vs {auth, logout} -> Jaccard = 1/3.
    assert abs(_aaak_overlap("auth/login", "auth/logout") - 1 / 3) < 1e-9
    # Case-insensitive.
    assert _aaak_overlap("AUTH/Login", "auth/login") == 1.0


def test_cosine_basic_properties() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector guard


def test_score_weight_constants_match_d13() -> None:
    """D-13 score = cos + 0.3*aaak + 0.1*log(1+deg) − 0.05*age."""
    assert W_COSINE == 1.0
    assert W_AAAK == 0.3
    assert W_DEGREE == 0.1
    assert W_AGE == 0.05


# ------------------------------------------------------------- end-to-end


def test_pipeline_returns_hits_with_adjacent_suggestions(tmp_path) -> None:
    """End-to-end: pipeline returns ranked hits with non-empty activation_trace
    and adjacent_suggestions is a list on every hit (AUTIST-07 contract)."""
    store = MemoryStore(path=tmp_path)
    records = [
        _make([1.0] + [0.0] * (EMBED_DIM - 1), text="primary match", aaak="test match"),
        _make([0.9, 0.1] + [0.0] * (EMBED_DIM - 2), text="close match"),
        _make([0.0, 1.0] + [0.0] * (EMBED_DIM - 2), text="orthogonal"),
        _make([-1.0] + [0.0] * (EMBED_DIM - 1), text="opposite"),
        _make([0.5, 0.5] + [0.0] * (EMBED_DIM - 2), text="mid"),
    ]
    for r in records:
        store.insert(r)
    graph = MemoryGraph()
    for r in records:
        graph.add_node(r.id, community_id=None, embedding=r.embedding)
    for i in range(len(records) - 1):
        graph.add_edge(records[i].id, records[i + 1].id)

    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: community_id for r in records},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r.id for r in records]},
    )

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="test match",
        session_id="s1",
    )
    assert len(resp.hits) >= 1
    # Primary record has aaak overlap ("test match" in both cue and aaak_index),
    # cosine=1.0, and degree=1: score = 1.0 + 0.3*1.0 + 0.1*log(2) ≈ 1.369.
    # Close record has cos≈0.994, no aaak, degree=2: 0.994 + 0.1*log(3) ≈ 1.104.
    # Primary must win thanks to the AAAK overlap bonus.
    assert resp.hits[0].literal_surface == "primary match"
    # Opposite record must NOT appear as a top hit (negative cosine).
    assert all(h.literal_surface != "opposite" for h in resp.hits[:2])
    # adjacent_suggestions must be a list on every hit.
    for h in resp.hits:
        assert isinstance(h.adjacent_suggestions, list)
    # activation_trace = seeds ∪ spread; must not be empty here.
    assert len(resp.activation_trace) >= 1


def test_pipeline_provenance_appended_to_every_hit(tmp_path) -> None:
    """ regression: every hit returned gets a provenance entry."""
    store = MemoryStore(path=tmp_path)
    r1 = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="primary")
    store.insert(r1)
    graph = MemoryGraph()
    graph.add_node(r1.id, community_id=None, embedding=r1.embedding)
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r1.id: community_id},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r1.id]},
    )
    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="anything",
        session_id="session-42",
    )
    refreshed = store.get(r1.id)
    assert refreshed is not None
    assert len(refreshed.provenance) == 1
    assert refreshed.provenance[0]["session_id"] == "session-42"
    assert refreshed.provenance[0]["cue"] == "anything"


def test_pipeline_budget_caps_hit_count(tmp_path) -> None:
    """Budget enforcement: when tokens exceeded, pipeline stops adding hits."""
    store = MemoryStore(path=tmp_path)
    # 5 records each with ~200 chars (~50 tokens). Budget=60 -> only first fits.
    long_text = "x" * 200
    records = []
    for i in range(5):
        r = _make(
            [1.0, float(i) * 0.001] + [0.0] * (EMBED_DIM - 2),
            text=f"{long_text}-{i}",
        )
        records.append(r)
        store.insert(r)
    graph = MemoryGraph()
    for r in records:
        graph.add_node(r.id, community_id=None, embedding=r.embedding)
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: community_id for r in records},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r.id for r in records]},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="c",
        session_id="s",
        budget_tokens=60,
    )
    # With 50-token records and 60-token budget, at most 1 hit fits then loop breaks.
    # (We always admit 1 even if it exceeds budget, per the len(hits)>=1 guard.)
    assert len(resp.hits) == 1


def test_pipeline_anti_hits_from_contradicts_edge(tmp_path) -> None:
    """D-13 anti-hit contract: contradicts-edge neighbours of a top hit surface."""
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r1 = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="original")
    store.insert(r1)
    dispatch(
        store,
        "memory_contradict",
        {
            "id": str(r1.id),
            "new_fact": "refuted version",
            "cue_embedding": r1.embedding,
        },
    )

    graph = MemoryGraph()
    graph.add_node(r1.id, community_id=None, embedding=[1.0] + [0.0] * (EMBED_DIM - 1))
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r1.id: community_id},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r1.id]},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="anything",
        session_id="s1",
    )
    assert len(resp.anti_hits) >= 1
    assert "refuted" in resp.anti_hits[0].literal_surface


def test_pipeline_activation_trace_includes_seeds(tmp_path) -> None:
    """activation_trace = seeds ∪ spread; must contain each seed."""
    store = MemoryStore(path=tmp_path)
    a = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="A")
    b = _make([0.9, 0.1] + [0.0] * (EMBED_DIM - 2), text="B")
    c = _make([0.0, 1.0] + [0.0] * (EMBED_DIM - 2), text="C")
    for r in (a, b, c):
        store.insert(r)
    graph = MemoryGraph()
    for r in (a, b, c):
        graph.add_node(r.id, community_id=None, embedding=r.embedding)
    graph.add_edge(a.id, b.id)
    graph.add_edge(b.id, c.id)
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: community_id for r in (a, b, c)},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [a.id, b.id, c.id]},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="c",
        session_id="s",
    )
    # The top-cosine seed is A; its 2-hop neighbourhood is {B, C}. Trace must contain A.
    assert a.id in resp.activation_trace


def test_pick_seeds_ranks_by_blended_score(tmp_path) -> None:
    """Stage 3 blend: 0.6*cos + 0.4*centrality picks the high-blend record first.

    redesign: `_pick_seeds` now operates over a
    precomputed shared cosine array; positions, not UUIDs, flow through.
    Reproduces the pre-redesign assertion: r2 (cos=0.707, cen=1.0,
    blend=0.82) beats r1 (cos=1.0, cen=0.0, blend=0.6) at n=1.
    """
    import numpy as np

    # Pool layout: position 0 = r1, position 1 = r2.
    # cue = axis 0 -> shared_cos = [1.0, 0.707].
    shared_cos = np.array([1.0, 0.7071068], dtype=np.float32)
    centrality_arr = np.array([0.0, 1.0], dtype=np.float32)
    candidate_indices = np.array([0, 1], dtype=np.int64)
    seed_indices = _pick_seeds(
        candidate_indices, shared_cos, centrality_arr, n=1,
    )
    # r2 (position 1): blend = 0.6 * 0.707 + 0.4 * 1.0 = 0.824 > r1's 0.6.
    assert list(seed_indices) == [1]


def test_pipeline_core_dispatch_integration(tmp_path, monkeypatch) -> None:
    """core.dispatch("memory_recall", ...) routes to pipeline for non-empty store."""
    import iai_mcp.pipeline as pipeline_mod
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="integration")
    store.insert(r)

    # Stub out Embedder inside core to avoid HF download.
    class _StubEmbedder:
        DIM = 384

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

    # core.py imports Embedder lazily inside dispatch -> patch at module level.
    import iai_mcp.embed as embed_mod
    monkeypatch.setattr(embed_mod, "Embedder", _StubEmbedder)

    resp = dispatch(
        store,
        "memory_recall",
        {"cue": "integration", "session_id": "s-int"},
    )
    assert "hits" in resp
    assert isinstance(resp["hits"], list)
    # activation_trace field always present, list of string UUIDs.
    assert isinstance(resp["activation_trace"], list)
    assert "budget_used" in resp


def test_pipeline_empty_gate_falls_back_to_all_nodes(tmp_path) -> None:
    """If community gate returns no candidates, pipeline falls back to all nodes."""
    store = MemoryStore(path=tmp_path)
    r = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="lonely")
    store.insert(r)
    graph = MemoryGraph()
    graph.add_node(r.id, community_id=None, embedding=r.embedding)
    # Assignment whose mid_regions is empty (degenerate) -> pipeline must fall back.
    assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="flat",
        top_communities=[],
        mid_regions={},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="c",
        session_id="s",
    )
    # The lone record is still reachable via the fallback.
    assert len(resp.hits) == 1
