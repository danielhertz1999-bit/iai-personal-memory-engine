"""redesign — load-bearing infrastructure tests.

Verifies the new shared-cosine helpers introduced by against
the locked decisions in `internal architecture spec
08-CONTEXT.md`:

- single shared cosine pass — `_collect_graph_pool` is the (ids, embs)
  pool collector that feeds the one-shot matmul at the top of `_recall_core`.
- mode-dependent community-gate soft bias — `COMMUNITY_BIAS_VERBATIM`
  (0.0, HIPPEA pure / EPF literal / hippocampal episodic) and
  `COMMUNITY_BIAS_CONCEPT` (0.1, CLS neocortical semantic / categorical
  hint), dispatched by `_gate_bias_for_mode(mode)` from the cue-classifier
  in `core.dispatch` (R5).
- candidate-pool size — `K_CANDIDATES = 200`, justified by the
  empirical 99th-percentile gold rank from the LongMemEval-S v1 trace
  plus 30% margin.
- `_RecallCoreResult` — dataclass shape returned by `_recall_core`.

These probes are the first wave of the redesign tests; the
heavier behavioural fixture (matmul-counter, gate-as-diagnostic, verbatim
filter placement, etc.) lives in `tests/test_recall_core_unit.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _make(vec: list[float], text: str = "rec") -> MemoryRecord:
    """Construct a MemoryRecord for shared-cosine pool tests."""
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


# --------------------------------------------------- _collect_graph_pool tests


def test_collect_graph_pool_returns_aligned_ids_and_embeddings(tmp_path) -> None:
    """D-01 fast path: graph._nx node attr "embedding" is the cheap source.

    Build 5 nodes whose primary-axis embeddings are pre-installed onto
    the NetworkX node dict (the `build_runtime_graph` shape). Assert the
    returned (ids, embs) pair is row-aligned: pool_embs[i] == records[i].embedding
    when pool_ids[i] == records[i].id.
    """
    from iai_mcp.pipeline import _collect_graph_pool

    store = MemoryStore(path=tmp_path / "lancedb")
    records: list[MemoryRecord] = []
    for i in range(5):
        vec = [0.0] * EMBED_DIM
        vec[i] = 1.0
        rec = _make(vec, text=f"rec{i}")
        store.insert(rec)
        records.append(rec)
    graph = MemoryGraph()
    for rec in records:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        # Mirror what build_runtime_graph does: pour the payload onto the
        # NetworkX node attr dict so _collect_graph_pool's fast path hits.
        graph._nx.nodes[str(rec.id)].update({"embedding": list(rec.embedding)})

    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)

    assert len(pool_ids) == 5
    assert pool_embs.shape == (5, EMBED_DIM)
    assert pool_embs.dtype == np.float32
    # Row alignment: pool_embs[i] reflects pool_ids[i]'s record.
    id_to_rec = {r.id: r for r in records}
    for i, rid in enumerate(pool_ids):
        rec = id_to_rec[rid]
        np.testing.assert_allclose(
            pool_embs[i], np.asarray(rec.embedding, dtype=np.float32)
        )


def test_collect_graph_pool_empty_graph(tmp_path) -> None:
    """Empty graph returns ([], np.zeros((0, embed_dim), dtype=float32)).

    The shape and dtype contract is load-bearing: downstream callers
    (_recall_core) need a 2D float32 array even when the pool is empty,
    so `pool_embs @ cue_vec` short-circuits cleanly to an empty result.
    """
    from iai_mcp.pipeline import _collect_graph_pool

    store = MemoryStore(path=tmp_path / "lancedb")
    graph = MemoryGraph()
    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)
    assert pool_ids == []
    assert pool_embs.shape == (0, store.embed_dim)
    assert pool_embs.dtype == np.float32


def test_collect_graph_pool_falls_back_to_store_get(tmp_path) -> None:
    """When _nx.nodes has no embedding, _collect_graph_pool falls back to store.get."""
    from iai_mcp.pipeline import _collect_graph_pool

    store = MemoryStore(path=tmp_path / "lancedb")
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    rec = _make(vec, text="store-only")
    store.insert(rec)
    graph = MemoryGraph()
    graph.add_node(rec.id, community_id=None, embedding=list(vec))
    # Ensure the _nx node attr does NOT carry the embedding (force fallback).
    if "embedding" in graph._nx.nodes[str(rec.id)]:
        del graph._nx.nodes[str(rec.id)]["embedding"]

    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)

    assert pool_ids == [rec.id]
    assert pool_embs.shape == (1, EMBED_DIM)
    np.testing.assert_allclose(
        pool_embs[0], np.asarray(vec, dtype=np.float32)
    )


# --------------------------------------------------------- module-level constants


def test_K_CANDIDATES_is_200() -> None:
    """K_CANDIDATES = 200, single module constant (no tier branch)."""
    from iai_mcp.pipeline import K_CANDIDATES

    assert K_CANDIDATES == 200
    assert isinstance(K_CANDIDATES, int)


def test_COMMUNITY_BIAS_constants_are_mode_dependent() -> None:
    """verbatim=0.0 (HIPPEA pure) and concept=0.1 (CLS neocortical).

    Constants live at module level for downstream (`_recall_core` Stage 5)
    + test introspection. They are floats, never strings or ints.
    """
    from iai_mcp.pipeline import COMMUNITY_BIAS_CONCEPT, COMMUNITY_BIAS_VERBATIM

    assert COMMUNITY_BIAS_VERBATIM == 0.0
    assert COMMUNITY_BIAS_CONCEPT == 0.1
    assert isinstance(COMMUNITY_BIAS_VERBATIM, float)
    assert isinstance(COMMUNITY_BIAS_CONCEPT, float)


def test_gate_bias_for_mode_returns_correct_value() -> None:
    """D-02 helper: dispatch off mode parameter; defensive default is 0.0.

    Anything other than the literal string "concept" returns
    COMMUNITY_BIAS_VERBATIM (0.0) so a malformed / missing / case-mismatched
    mode never accidentally biases recall toward categorical filtering.
    """
    from iai_mcp.pipeline import _gate_bias_for_mode

    assert _gate_bias_for_mode("verbatim") == 0.0
    assert _gate_bias_for_mode("concept") == 0.1
    # Defensive defaults — "never accidentally bias" rule.
    assert _gate_bias_for_mode("unknown") == 0.0
    assert _gate_bias_for_mode("") == 0.0
    # Case-sensitive: "CONCEPT" is NOT "concept".
    assert _gate_bias_for_mode("CONCEPT") == 0.0


def test_RecallCoreResult_dataclass_has_required_fields() -> None:
    """`_RecallCoreResult` is the shape returned by `_recall_core`.

    Default-constructed instance has all 7 fields present with
    correct empty/default values so downstream entry points
    (recall_for_response / recall_for_benchmark in 08-02) can apply
    pack/cap on a fully-populated structure.
    """
    from iai_mcp.pipeline import _RecallCoreResult

    r = _RecallCoreResult()
    assert r.scored_hits == []
    assert r.activation_trace == []
    assert r.anti_hits == []
    assert r.hints == []
    assert r.patterns_observed == []
    assert r.cue_mode == "concept"
    assert r.budget_used == 0
