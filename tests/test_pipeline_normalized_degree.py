"""R2 acceptance suite — bounded graph-bonus + max_degree cache.

Two-tier coverage:

  Task 1 (cache + build_runtime_graph contract):
    - test_build_runtime_graph_sets_max_degree_attribute
    - test_cache_round_trip_preserves_max_degree
    - test_empty_store_max_degree_is_zero

  Task 2 (rank-stage R2 acceptance):
    - test_normalized_degree_lets_verbatim_outrank_hub
    - test_old_formula_would_have_ranked_hub_above_verbatim   (regression direction lock)
    - test_pipeline_reason_contains_deg_norm_not_raw_log
    - test_zero_max_degree_does_not_raise_division_error

The hub/verbatim fixtures use HAND-CRAFTED 384d unit vectors so the cosine
window between hub and verbatim is precisely controllable. _PerfEmbedder's
sha256-based vectors collapse to ≈0 for distinct text and 1.0 for identical
text — they cannot produce the 0.3 < gap < 0.42 window the R2 math demands
(W_DEGREE=0.1 × log(1+64) ≈ 0.42 = max old-formula degree contribution).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------- Fixture machinery


class _ControlledEmbedder:
    """Embedder whose output for a given text is deterministic AND
    overridable. ``self.fixed`` maps cue text → 384d unit vector; any
    other text falls through to a sha256-derived vector (the same
    pattern as _PerfEmbedder for parity with seed-time use).

    Used by R2 tests to pin the cue's vector so the dot product against
    each candidate is the controlled cosine.
    """

    DIM = EMBED_DIM

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self.fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self.fixed:
            return list(self.fixed[text])
        # Deterministic fallback for anything we didn't pre-program.
        import hashlib
        import random
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _unit_vector_with_cosine(cue_vec: list[float], target_cos: float) -> list[float]:
    """Build a unit vector v such that dot(cue_vec, v) == target_cos.

    Construction: v = α * cue + β * orth, where orth is a fixed unit
    vector orthogonal to cue, α = target_cos, β = sqrt(1 - target_cos²).
    Both cue and orth are unit vectors, so v is a unit vector with the
    requested cosine. Deterministic across runs.
    """
    cue = np.asarray(cue_vec, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue))
    if cue_norm == 0.0:
        raise ValueError("cue_vec must be non-zero")
    cue = cue / cue_norm

    # Pick a probe along axis 1 if not parallel to cue, else axis 0.
    probe = np.zeros(EMBED_DIM, dtype=np.float32)
    probe[1] = 1.0
    if abs(float(np.dot(cue, probe))) > 0.999:
        probe = np.zeros(EMBED_DIM, dtype=np.float32)
        probe[0] = 1.0
    orth = probe - float(np.dot(cue, probe)) * cue
    orth = orth / float(np.linalg.norm(orth))

    alpha = float(target_cos)
    beta = float(math.sqrt(max(0.0, 1.0 - alpha * alpha)))
    v = alpha * cue + beta * orth
    # Re-normalise to absorb float32 round-off; the result is essentially
    # already a unit vector (alpha² + beta² == 1 by construction).
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.astype(np.float32).tolist()


def _make_episodic(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
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


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


# ------------------------------------------------------------- Task 1 tests


def test_build_runtime_graph_sets_max_degree_attribute(tmp_path):
    """After build_runtime_graph the graph carries an integer _max_degree."""
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    # Seed 5 isolated records so the degree distribution is the trivial
    # all-zeros (one isolated node per record).
    for i in range(5):
        vec = embedder.embed(f"isolated-{i}")
        store.insert(_make_episodic(vec, f"text {i}"))

    graph, _, _ = build_runtime_graph(store)
    assert hasattr(graph, "_max_degree"), "graph must carry _max_degree attribute"
    assert isinstance(graph._max_degree, int), "_max_degree must be int"
    assert graph._max_degree >= 0


def test_cache_round_trip_preserves_max_degree(tmp_path):
    """A second build_runtime_graph (cache HIT) reads max_degree from
    runtime_graph_cache.json — no recompute required."""
    from iai_mcp import runtime_graph_cache
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    ids = []
    for i in range(6):
        vec = embedder.embed(f"node-{i}")
        rec = _make_episodic(vec, f"surface {i}")
        store.insert(rec)
        ids.append(rec.id)
    # Manufacture a small star: ids[0] linked to ids[1..5] (deg=5 hub).
    store.boost_edges(
        [(ids[0], ids[j]) for j in range(1, 6)],
        edge_type="hebbian",
        delta=1.0,
    )

    graph1, _, _ = build_runtime_graph(store)
    md1 = graph1._max_degree
    assert md1 >= 5, f"expected hub degree >= 5, got {md1}"

    # Inspect cache directly: max_degree key must be present.
    cache = runtime_graph_cache.try_load(store)
    assert cache is not None, "cache must round-trip"
    # try_load now returns a 4-tuple (assignment, rich_club, node_payload, max_degree).
    assert len(cache) == 4, f"try_load must return 4-tuple, got {len(cache)}"
    _assignment, _rich_club, _node_payload, cached_md = cache
    assert int(cached_md) == md1

    # Second build: cache HIT must rehydrate the same value.
    graph2, _, _ = build_runtime_graph(store)
    assert graph2._max_degree == md1


def test_empty_store_max_degree_is_zero(tmp_path):
    """Empty / single-isolated-node store: max_degree == 0 (no division
    by zero downstream — Task 2 rank stage falls back to deg_norm=0.0)."""
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    rec = _make_episodic(embedder.embed("only"), "only one")
    store.insert(rec)

    graph, _, _ = build_runtime_graph(store)
    # One isolated node -> deg=0 -> max_degree=0
    assert graph._max_degree == 0


# ------------------------------------------------------------- Task 2 tests
# Hub vs verbatim fixture geometry:
#   cue text:        "verbatim cue marker A"
#   verbatim record: cos = 0.60 to cue
#   hub record:      cos = 0.30 to cue, deg = 64 (max in graph)
#   filler records:  64 distractors carrying isolated edges to make hub deg=64
#
# OLD formula (W_DEGREE * log(1+deg)):
#   hub_score      ≈ 0.30 + 0.1 * log(65) ≈ 0.30 + 0.4170 = 0.7170
#   verbatim_score ≈ 0.60 + 0.1 * log(1)  ≈ 0.60 + 0.0000 = 0.6000
#   → hub wins by ≈ 0.117  (old regression direction)
#
# NEW formula (W_DEGREE * log(1+deg)/log(1+max_deg)):
#   hub_score      ≈ 0.30 + 0.1 * 1.0     = 0.4000
#   verbatim_score ≈ 0.60 + 0.1 * 0.0     = 0.6000
#   → verbatim wins by 0.20  (R2 acceptance)


def _seed_hub_vs_verbatim(tmp_path, hub_degree: int = 64):
    """Seed a store with one hub (deg=hub_degree, cos=0.30 to cue) and
    one verbatim (deg=0, cos=0.60 to cue), plus N=hub_degree distractor
    records connected only to the hub via Hebbian edges.

    Returns (store, embedder, graph, assignment, rich_club, hub_id, verbatim_id, cue_text).
    """
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()

    cue_text = "verbatim cue marker A"
    # Pin the cue vector to a known direction. Using a sha256-derived
    # vector so the embedder's hash path would have produced the same.
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    hub_vec = _unit_vector_with_cosine(cue_vec, 0.30)
    verbatim_vec = _unit_vector_with_cosine(cue_vec, 0.60)

    hub_rec = _make_episodic(hub_vec, "hub schema record")
    verbatim_rec = _make_episodic(
        verbatim_vec, "the exact verbatim quote you are looking for"
    )
    store.insert(hub_rec)
    store.insert(verbatim_rec)

    # Create distractor records and link each to the hub. Each link adds
    # 1 to the hub's degree (Hebbian undirected). We use distinct edges
    # so the hub ends with degree = hub_degree.
    distractor_ids = []
    edge_pairs = []
    for i in range(hub_degree):
        # Use an orthogonal-ish vector — far from cue so distractors never
        # outrank either hub or verbatim by cosine alone.
        d_vec = embedder.embed(f"distractor-{i}-far-from-cue")
        d_rec = _make_episodic(d_vec, f"unrelated junk {i}")
        store.insert(d_rec)
        distractor_ids.append(d_rec.id)
        edge_pairs.append((hub_rec.id, d_rec.id))

    store.boost_edges(edge_pairs, edge_type="hebbian", delta=1.0)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        hub_rec.id, verbatim_rec.id, cue_text,
    )


def test_normalized_degree_lets_verbatim_outrank_hub(tmp_path):
    """R2 acceptance: under the NEW formula the verbatim record outranks
    the hub on a cue where verbatim has cos=0.60 and hub has cos=0.30
    plus deg=64. Verbatim must land at or before position the hub does."""
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     hub_id, verbatim_id, cue_text) = _seed_hub_vs_verbatim(tmp_path)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="r2_acceptance",
        budget_tokens=1500,
    )
    hit_ids = [h.record_id for h in resp.hits]
    assert verbatim_id in hit_ids, f"verbatim must appear in hits; got {hit_ids}"
    if hub_id in hit_ids:
        verb_pos = hit_ids.index(verbatim_id)
        hub_pos = hit_ids.index(hub_id)
        assert verb_pos < hub_pos, (
            f"verbatim must rank above hub under new formula. "
            f"verbatim@{verb_pos} hub@{hub_pos} hits={hit_ids}"
        )
    # Stronger acceptance: verbatim is at position 0.
    assert hit_ids[0] == verbatim_id, (
        f"verbatim must be position-0 under new formula; got {hit_ids[0]} "
        f"(verbatim_id={verbatim_id}, hits={hit_ids})"
    )


def test_old_formula_would_have_ranked_hub_above_verbatim(tmp_path):
    """Regression direction lock: hand-compute the OLD score using the
    same fixture and confirm hub > verbatim. Proves the fix actually
    changed ordering, not a flaky test that happened to pass."""
    from math import log

    (store, embedder, graph, _assignment, _rich_club,
     hub_id, verbatim_id, cue_text) = _seed_hub_vs_verbatim(tmp_path)

    # Resolve hub + verbatim cosines and degrees from the live graph.
    cue_vec = np.asarray(embedder.embed(cue_text), dtype=np.float32)
    cue_vec = cue_vec / float(np.linalg.norm(cue_vec))

    def _live_cos(rid):
        node = graph._nx.nodes[str(rid)]
        v = np.asarray(node["embedding"], dtype=np.float32)
        return float(np.dot(cue_vec, v))

    hub_cos = _live_cos(hub_id)
    verbatim_cos = _live_cos(verbatim_id)

    deg_dict = dict(graph._nx.degree())
    hub_deg = float(deg_dict.get(str(hub_id), 0))
    verbatim_deg = float(deg_dict.get(str(verbatim_id), 0))

    # OLD formula constants (from pipeline.py:115-118, NOT changed by ).
    W_COSINE = 1.0
    W_DEGREE = 0.1
    # AAAK is 0 (no aaak_index seeded). Age penalty is ~0 for fresh records.
    hub_old = W_COSINE * hub_cos + W_DEGREE * log(1.0 + hub_deg)
    verbatim_old = W_COSINE * verbatim_cos + W_DEGREE * log(1.0 + verbatim_deg)
    assert hub_old > verbatim_old, (
        "OLD formula must rank hub above verbatim — otherwise the R2 "
        "fix would not change ordering and the new test would be vacuous. "
        f"hub_old={hub_old:.4f} verbatim_old={verbatim_old:.4f} "
        f"hub_cos={hub_cos:.4f} verbatim_cos={verbatim_cos:.4f} "
        f"hub_deg={hub_deg} verbatim_deg={verbatim_deg}"
    )


def test_pipeline_reason_contains_deg_norm_not_raw_log(tmp_path):
    """The reason string must show `deg_norm` (the bounded value), NOT
    `log(deg+1)`, on both structural branches."""
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     _hub_id, _verbatim_id, cue_text) = _seed_hub_vs_verbatim(tmp_path)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="r2_reason_check",
        budget_tokens=1500,
    )
    assert resp.hits, "fixture must produce at least one hit"
    for h in resp.hits:
        assert "deg_norm" in h.reason, (
            f"reason must contain 'deg_norm'; got: {h.reason!r}"
        )
        assert "log(deg+1)" not in h.reason, (
            f"reason must NOT contain raw 'log(deg+1)'; got: {h.reason!r}"
        )


def test_zero_max_degree_does_not_raise_division_error(tmp_path):
    """When the live graph has max_degree==0 (all isolated nodes / cold
    start) the rank stage must not raise ZeroDivisionError. deg_norm
    falls back to 0.0 and cosine carries the recall on its own."""
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    cue_text = "cold start cue with no graph topology"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    # Seed 3 isolated records — no edges anywhere — max_degree must be 0.
    for i in range(3):
        v = _unit_vector_with_cosine(cue_vec, 0.5 - 0.1 * i)
        store.insert(_make_episodic(v, f"isolated-cold-{i}"))

    graph, assignment, rich_club = build_runtime_graph(store)
    assert graph._max_degree == 0, (
        f"isolated graph must have max_degree=0, got {graph._max_degree}"
    )

    # The call must not raise.
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="cold_start_zero_max_deg",
        budget_tokens=1500,
    )
    # And it must return *something* (cosine alone ranks the candidates).
    assert len(resp.hits) >= 1
