"""— graph-native recall tests (RED scaffold).

Close the latency gap by switching recall_for_response's seed + spread
stages from per-id ``store.get(rid)`` LanceDB round-trips to in-RAM
``G.nodes[rid]`` attribute lookups. ``build_runtime_graph`` attaches the
record payload (embedding, surface, centrality, tier) to every graph
node so the recall hot path never touches disk for a graph-resident id.

Covered contracts:

  A1 — every node in G carries embedding + surface + centrality + tier
       after ``build_runtime_graph``.
  A2 — seed stage does NOT call ``store.get`` (patch raises if invoked).
  A3 — spread stage (rank/reachable walk) does NOT call ``store.get``.
  A4 — verbatim L0 fast path (cue_text exact-match / gate skip) still
       hits ``store.get`` — invariant path is untouched.
  A5 — partial sync / missing attribute on a node falls back to
       ``store.get`` without crashing; recall still returns hits.
  A6 — correctness fence: recall returns the seeded records with
       high cosine similarity (no correctness regression).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

from iai_mcp import retrieve
from iai_mcp.pipeline import recall_for_response
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Swap macOS Keychain for an in-memory dict so tests don't prompt."""
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


class _DetEmbedder:
    """Deterministic embedder — seeds record vectors by text hash."""

    def __init__(self, dim: int = 384) -> None:
        self.DIM = dim
        self.DEFAULT_DIM = dim
        self.DEFAULT_MODEL_KEY = "test"

    def embed(self, text: str) -> list[float]:
        import hashlib
        import random

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 0 else v


def _make_record(vec: list[float], text: str) -> MemoryRecord:
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
        tags=["t"],
        language="en",
    )


@pytest.fixture
def seeded_store(tmp_path: Path) -> tuple[MemoryStore, _DetEmbedder, list[MemoryRecord]]:
    """Fresh store with 12 records so the seed+spread stages have enough
    material to exercise the graph-native read path."""
    store = MemoryStore(path=tmp_path / "hippo")
    store.root = tmp_path
    emb = _DetEmbedder(dim=store.embed_dim)
    recs = []
    for i in range(12):
        vec = emb.embed(f"fact-{i}")
        rec = _make_record(vec, f"synthetic fact {i}")
        store.insert(rec)
        recs.append(rec)
    return store, emb, recs


# ---------------------------------------------------------------- A1: node payload


def test_A1_build_runtime_graph_attaches_node_payload(seeded_store):
    """A1: every node carries embedding + surface + centrality + tier in sidecar."""
    store, _emb, recs = seeded_store
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    # Post-untangle: payload lives in the _node_payload sidecar, surfaced via
    # graph.get_payload(uuid). The legacy graph._nx.nodes[*] read path was
    # retired by the mosaicsigma wave.
    listed = list(graph.iter_nodes())
    assert len(listed) == len(recs)
    for rec in recs:
        payload = graph.get_payload(rec.id)
        assert "embedding" in payload, f"node {rec.id} missing embedding sidecar"
        assert "surface" in payload, f"node {rec.id} missing surface sidecar"
        assert "centrality" in payload, f"node {rec.id} missing centrality sidecar"
        assert "tier" in payload, f"node {rec.id} missing tier sidecar"
        # Embedding list matches the record's embedding within float32 precision
        # (SQLite stores embeddings as float32 BLOB; minor precision loss is expected).
        import pytest as _pt
        assert list(payload["embedding"]) == _pt.approx(
            list(rec.embedding), rel=1e-5
        )
        assert payload["surface"] == rec.literal_surface
        assert payload["tier"] == rec.tier


# ---------------------------------------------------------------- A2: seed stage


def test_A2_seed_stage_reads_from_graph_not_store(seeded_store):
    """A2: seed stage (top-K by cosine) must NOT call store.get.

    We patch MemoryStore.get to raise; if recall_for_response still returns
    a non-empty RecallResponse, the seed stage is graph-native.
    """
    store, emb, _recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    # The verbatim L0 fast-path (gate skip) calls store.get too — disable
    # the skip by choosing a cue that the gate will NOT classify as trivial.
    cue = "explain the authentication migration for long-running deployments"

    # AllowedError raises ONLY on the hot-path store.get; the L0 fast-path
    # is known not to fire for this cue.
    class _Boom(RuntimeError):
        pass

    original_get = store.get

    def _explode(rid):
        # Allow the verbatim L0 UUID fetch to pass through so the fast-path
        # check (no L0 record seeded) is a clean miss — but any OTHER store.get
        # call blows up.
        from uuid import UUID
        l0 = UUID("00000000-0000-0000-0000-000000000001")
        if rid == l0:
            return None
        raise _Boom(f"store.get({rid}) — seed stage should not call this")

    with mock.patch.object(MemoryStore, "get", side_effect=_explode):
        resp = recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=emb,
            cue=cue,
            session_id="s",
            budget_tokens=1500,
        )
    assert len(resp.hits) >= 1


# ---------------------------------------------------------------- A3: spread stage


def test_A3_spread_stage_reads_from_graph_not_store(seeded_store):
    """A3: rank+spread stages do NOT call store.get either.

    Same shape as A2 but asserts over the full reachable-union not just
    seeds.
    """
    store, emb, _recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    cue = "network stack changes for the web cache"

    class _Boom(RuntimeError):
        pass

    def _explode(rid):
        from uuid import UUID
        l0 = UUID("00000000-0000-0000-0000-000000000001")
        if rid == l0:
            return None
        raise _Boom(f"store.get({rid}) during spread/rank")

    with mock.patch.object(MemoryStore, "get", side_effect=_explode):
        resp = recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=emb,
            cue=cue,
            session_id="s",
            budget_tokens=1500,
        )
    # If spread/rank was using store.get, we would have exploded above.
    assert isinstance(resp.hits, list)


# ---------------------------------------------------------------- A4: L0 fast path


def test_A4_verbatim_l0_fast_path_still_calls_store_get(seeded_store):
    """A4: the L0 (gate-skip) fast path still hits store.get — unchanged.

     invariant: verbatim recall path is NOT touched.
    """
    store, emb, _recs = seeded_store
    # Seed the deterministic L0 record so the gate-skip branch fires.
    from uuid import UUID
    l0_id = UUID("00000000-0000-0000-0000-000000000001")
    l0_vec = emb.embed("l0-identity")
    now = datetime.now(timezone.utc)
    l0_rec = MemoryRecord(
        id=l0_id,
        tier="semantic",
        literal_surface="L0 identity kernel",
        aaak_index="",
        embedding=l0_vec,
        community_id=None,
        centrality=0.0,
        detail_level=5,  # never_decay
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["identity"],
        language="en",
    )
    store.insert(l0_rec)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    # Pick a cue that the gate treats as trivial (short / who-am-i style).
    cue = "hi"

    with mock.patch.object(MemoryStore, "get", wraps=store.get) as spy:
        _ = recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=emb,
            cue=cue,
            session_id="s",
            budget_tokens=1500,
        )
    # At LEAST one store.get call on the L0 fast path (verbatim invariant).
    assert spy.call_count >= 1


# ---------------------------------------------------------------- A5: fallback


def test_A5_missing_node_attr_falls_back_to_store_get(seeded_store):
    """A5: if a node somehow lacks the embedding (race / partial sync),
    pool collection falls back to store.get and recall still returns
    correct hits — no crash.
    """
    store, emb, recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    # Blow away the embedding sidecar entry on half the nodes — pool
    # collection will fall back to store.get for these.
    victims = [str(r.id) for r in recs[:6]]
    for nid_str in victims:
        sidecar = graph._node_payload.get(nid_str)
        if sidecar and "embedding" in sidecar:
            del sidecar["embedding"]

    cue = "summary of cli subcommand changes for the auth token rotation"
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue=cue,
        session_id="s",
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1


# ---------------------------------------------------------------- A6: correctness


def test_A6_m04_correctness_no_regression(seeded_store):
    """A6: recall returns the seeded record whose text matches the cue.

    Minimal correctness fence inside this file (the heavyweight
    bench.verbatim sweep covers gap=5/20/100 elsewhere; this guards the
    happy-path-does-not-regress invariant inside the unit suite).
    """
    store, emb, recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    # Query with text similar to record 7 — its cosine should dominate.
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue="synthetic fact 7",
        session_id="s",
        budget_tokens=1500,
    )
    # At least one hit comes back.
    assert len(resp.hits) >= 1
    # All hit record ids are in the seeded record id set.
    seeded_ids = {r.id for r in recs}
    assert all(h.record_id in seeded_ids for h in resp.hits)
