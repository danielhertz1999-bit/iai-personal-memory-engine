"""Regression-fence — exactly one cue-vs-pool cosine pass per recall.

The redesign's load-bearing claim is that the rank-stage cosine term
reads from a shared array built ONCE at the top of `_recall_core`.
This file fences the claim at the entry-point level: for both public
entry points (`recall_for_response`, `recall_for_benchmark`) the
matmul that computes `pool_embs @ cue_vec` fires exactly ONCE per
call. The L0 fast-path bypasses the pool entirely (zero pool matmuls).

The rank-stage previously used a separate `E @ cue_vec` matmul plus a
helper that added a third independent cosine pass. The redesign collapses
all three into one shared pass — the matmul-counter assertions in this
file fence that contract for the public entry points (the
`_recall_core`-level fence lives in `test_recall_core_unit.py`).

Implementation note: the matmul-counter is the canonical approach with
no sentinel-content fallback. The wrapper counts only "cue-vs-large-pool"
matmul calls — 2D matrix shaped (N >= 50, D) against 1D cue vector
shaped (D). The community-gate centroid matmul (which has K =
#communities < 50 in our fixtures) is excluded from the count by the
>= 50 row floor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------- test fixtures


class _FakeEmbedder:
    """Stand-in embedder; cue's embedding is configurable per-test."""

    DIM = EMBED_DIM

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec if vec is not None else [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _make(vec: list[float], text: str = "rec", tier: str = "episodic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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


def _build_store_and_graph(tmp_path, n: int) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord]]:
    """Build N records with distinct primary-axis embeddings + matching graph."""
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
        # Mirror build_runtime_graph: write the payload into the sidecar so
        # _collect_graph_pool's fast path hits via graph.get_embedding.
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": f"rec{recs.index(rec)}",
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


# ----------------------------------------------------- matmul counter helper


def _matmul_with_counter(counter: dict[str, int]):
    """Wrap np.matmul with a shape-discriminating counter.

    Counts only the "cue-vs-large-pool" matmul: 2D matrix shaped
    (N >= 50, D) against a 1D cue vector shaped (D). The community-gate
    centroid matmul (which has K = #communities < 50 in our fixtures)
    is excluded from the count by the >= 50 row floor.

    This is the canonical approach; there is no
    fallback to a sentinel-based content test.
    """
    orig = np.matmul

    def wrapped(a, b, **kw):
        try:
            if (
                hasattr(a, "shape")
                and hasattr(b, "shape")
                and len(a.shape) == 2
                and len(b.shape) == 1
                and a.shape[1] == b.shape[0]
                and a.shape[0] >= 50
            ):
                counter["count"] = counter.get("count", 0) + 1
        except Exception:
            pass
        return orig(a, b, **kw)

    return wrapped


# ----------------------------------------------------------------- tests


def test_recall_for_benchmark_runs_one_pool_cosine(tmp_path, monkeypatch):
    """recall_for_benchmark fires the cue-vs-pool matmul EXACTLY once.

    50+-node fixture so the >= 50 row floor in the matmul counter
    discriminates the load-bearing pool matmul from the small
    community-centroid matmul. With the entry point plumbed
    onto _recall_core, the only cue-vs-large-pool matmul should fire
    inside _recall_core's shared cosine pass; Stage 5 reads from
    `shared_cos[reachable_indices]` — never another pool matmul.
    """
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="primary", session_id="s-bench-cosine-1",
        k_hits=10, mode="concept",
    )

    assert counter["count"] == 1, (
        f"cue-vs-large-pool matmul fired "
        f"{counter['count']} times via recall_for_benchmark; expected "
        "exactly 1 (the shared cosine pass at the top of _recall_core)."
    )


def test_recall_for_response_runs_one_pool_cosine(tmp_path, monkeypatch):
    """recall_for_response fires the cue-vs-pool matmul EXACTLY once.

    Production entry-point analogue of the bench test above. budget_tokens
    is generous (4000) so the budget-pack loop does not influence whether
    a second matmul could fire (it cannot, but we keep the cap loose so
    the test is not gated on budget arithmetic).
    """
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="primary", session_id="s-resp-cosine-2",
        budget_tokens=4000, mode="concept",
    )

    assert counter["count"] == 1, (
        f"cue-vs-large-pool matmul fired "
        f"{counter['count']} times via recall_for_response; expected "
        "exactly 1 (the shared cosine pass at the top of _recall_core)."
    )


def test_l0_fastpath_runs_zero_pool_cosines(tmp_path, monkeypatch):
    """L0 fast-path: should_skip_retrieval triggers BEFORE any pool walk.

    When the active-inference gate decides to skip retrieval, _recall_core
    returns the L0 sentinel hit without ever calling _collect_graph_pool
    or the shared-cosine matmul. The matmul counter must therefore stay
    at 0 across the entry-point call.

    This fences the "L0 path is genuinely a fast-path" contract: if a
    future change accidentally moved the pool walk before the L0 gate,
    this test would surface a non-zero count even when retrieval was
    skipped.
    """
    import iai_mcp.gate as gate_mod
    from iai_mcp.pipeline import recall_for_benchmark

    # Force should_skip_retrieval to fire, simulating an L0 hit.
    monkeypatch.setattr(
        gate_mod,
        "should_skip_retrieval",
        lambda cue: (True, "test L0 reason"),
    )

    # Insert the deterministic L0 sentinel record + a small fixture pool.
    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    l0_uuid = UUID("00000000-0000-0000-0000-000000000001")
    now = datetime.now(timezone.utc)
    l0_rec = MemoryRecord(
        id=l0_uuid,
        tier="episodic",
        literal_surface="L0 identity literal",
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
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
    store.insert(l0_rec)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="hi", session_id="s-l0-fast-3",
        k_hits=10, mode="concept",
    )

    # The L0 fast-path returns exactly 1 hit (the L0 sentinel).
    assert len(resp.hits) == 1, (
        f"L0 fast-path should return exactly 1 hit; got {len(resp.hits)}"
    )
    assert resp.hits[0].record_id == l0_uuid, (
        "L0 fast-path returned a non-L0 record; gate fired but pool walk "
        "happened anyway."
    )
    assert counter["count"] == 0, (
        f"L0 fast-path violation: cue-vs-large-pool matmul fired "
        f"{counter['count']} times even though the L0 gate fired; "
        "expected 0 (the L0 path bypasses the pool walk entirely)."
    )
