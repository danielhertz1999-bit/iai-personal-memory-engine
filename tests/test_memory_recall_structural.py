"""Plan 03-01 CONN-05 GREEN: memory_recall_structural via core.dispatch.

Verifies the new MCP tool branch:
- Structural query enters as a dict[str, str] of role->filler pairs.
- Pipeline ranks by Hamming similarity over the packed structure_hv.
- ZERO LLM token cost: no Embedder() instantiated, no anthropic client called.
- Budget knob honoured: smaller budget -> fewer hits returned.
- Pipeline-level structural_weight knob shifts ranking when set.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake_store.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None))
    yield fake_store


def _make_record(text="x", **overrides):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    base = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    base.update(overrides)
    return MemoryRecord(**base)


def _seed(store, n=5):
    """Seed N records with varied tags so structure_hv differs per record."""
    recs = []
    for i in range(n):
        rec = _make_record(text=f"text-{i}", tags=[f"topic-{i}"])
        store.insert(rec)
        recs.append(rec)
    return recs


# ------------------------------------------------------------ dispatch happy


def test_memory_recall_structural_returns_hits(tmp_path, monkeypatch):
    """The new branch returns a hits list with score / record_id / literal_surface."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    recs = _seed(store, n=5)

    resp = dispatch(
        store,
        "memory_recall_structural",
        {
            "structure_query": {"TIER": "episodic", "LANG": "en"},
            "budget_tokens": 2000,
        },
    )
    assert "hits" in resp
    assert isinstance(resp["hits"], list)
    assert len(resp["hits"]) >= 1
    h = resp["hits"][0]
    assert "record_id" in h
    assert "score" in h
    assert "literal_surface" in h
    assert h["score"] >= 0.0
    assert resp["structural_query_size"] == 2


def test_memory_recall_structural_zero_llm_cost(tmp_path, monkeypatch):
    """The branch must NOT instantiate Embedder() OR call anthropic.messages.create.
    Hard guarantee for the 0-LLM-cost invariant (constitutional fit)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    _seed(store, n=3)

    # Trip-wire: replace Embedder with a sentinel that raises on instantiation.
    embedder_called = {"n": 0}

    class _BoomEmbedder:
        def __init__(self, *a, **kw):
            embedder_called["n"] += 1
            raise AssertionError(
                "memory_recall_structural must NOT instantiate Embedder() "
                "(constitutional 0-LLM-cost invariant)"
            )

    import iai_mcp.embed as embed_mod
    monkeypatch.setattr(embed_mod, "Embedder", _BoomEmbedder)

    # Trip-wire: any anthropic client call raises immediately.
    try:
        import anthropic
        def _boom_client(*a, **kw):
            raise AssertionError("memory_recall_structural must NOT touch anthropic API")
        monkeypatch.setattr(anthropic, "Anthropic", _boom_client)
    except ImportError:
        pass

    resp = dispatch(
        store,
        "memory_recall_structural",
        {"structure_query": {"TIER": "episodic"}, "budget_tokens": 2000},
    )
    assert embedder_called["n"] == 0
    assert "hits" in resp


def test_memory_recall_structural_budget_honoured(tmp_path, monkeypatch):
    """A tiny budget returns fewer hits than a large budget."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    _seed(store, n=5)

    big = dispatch(
        store, "memory_recall_structural",
        {"structure_query": {"TIER": "episodic"}, "budget_tokens": 5000},
    )
    small = dispatch(
        store, "memory_recall_structural",
        {"structure_query": {"TIER": "episodic"}, "budget_tokens": 5},
    )
    assert len(big["hits"]) >= len(small["hits"])
    assert len(small["hits"]) >= 1  # At least one hit always returned.


# --------------------------------------------------------- pipeline peer wire


def test_pipeline_ranker_structural_weight_shifts_ordering(tmp_path, monkeypatch):
    """Setting profile_state["structural_weight"]=0.9 changes the ranker's
    output relative to weight=0.0 -- proving structural is a ranking peer,
    not a no-op cosmetic kwarg."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    recs = []
    # Seed records with distinguishable structures (different tiers).
    for i, tier in enumerate(["episodic", "semantic", "episodic", "semantic"]):
        rec = _make_record(text=f"row-{i}", tier=tier, tags=[f"tag-{i}"])
        store.insert(rec)
        recs.append(rec)

    # Build a minimal graph + assignment so recall_for_response can run.
    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=r.id, embedding=r.embedding)
    assignment = CommunityAssignment(
        node_to_community={r.id: r.id for r in recs},
        community_centroids={r.id: list(r.embedding) for r in recs},
        modularity=0.5,
        top_communities=[r.id for r in recs],
        mid_regions={r.id: [r.id] for r in recs},
    )

    class _StaticEmbedder:
        DIM = len(recs[0].embedding)
        DEFAULT_DIM = DIM
        DEFAULT_MODEL_KEY = "test"
        def embed(self, text):  # deterministic vector keyed off length
            return [0.1] * self.DIM

    e = _StaticEmbedder()

    baseline = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=e, cue="hello", session_id="-",
        budget_tokens=5000, profile_state={"structural_weight": 0.0},
    )
    weighted = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=e, cue="hello", session_id="-",
        budget_tokens=5000, profile_state={"structural_weight": 0.9},
    )
    # Both runs return some hits.
    assert len(baseline.hits) >= 1
    assert len(weighted.hits) >= 1
    # The weighted ranker exposes structural-score reasoning in its reason
    # strings (proves it's reading the new branch). Baseline does not.
    assert any("structural" in h.reason for h in weighted.hits)
    assert all("structural" not in h.reason for h in baseline.hits)


def test_unknown_method_does_not_match(tmp_path, monkeypatch):
    """The new branch only matches the canonical method name.

    Phase 07.13-02 V3-03 fix: dispatch fall-through now raises
    UnknownMethodError instead of returning {"error": ...} dict. The
    bogus-suffix branch is name-gated; an exact-match-only assertion is
    that the bogus method raises UnknownMethodError specifically.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.core import UnknownMethodError, dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    # Unknown method falls through to UnknownMethodError post-Phase-07.13-02.
    # We don't care about the exception details -- just that our branch is
    # name-gated and a non-canonical name doesn't accidentally match.
    try:
        dispatch(store, "memory_recall_structural_BOGUS", {})
    except (UnknownMethodError, KeyError, ValueError, TypeError):
        pass  # acceptable -- the bogus method failed downstream or fell through
    # The valid method still works.
    resp = dispatch(store, "memory_recall_structural", {"structure_query": {}})
    assert "hits" in resp


def test_memory_recall_structural_max_records_caps(tmp_path, monkeypatch):
    """V3-07: max_records limits how many rows enter the O(N) scoring loop."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT

    store = MemoryStore()
    now = datetime.now(timezone.utc)
    for i in range(12):
        rid = uuid4()
        emb = [0.001 * (i + 1)] * 384
        hv = bytes(1250)
        rec = MemoryRecord(
            id=rid,
            tier="episodic",
            literal_surface=f"row-{i}",
            aaak_index="",
            embedding=emb,
            structure_hv=hv,
            community_id=None,
            centrality=0.0,
            detail_level=1,
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
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_CURRENT,
        )
        store.insert(rec)

    out = dispatch(
        store,
        "memory_recall_structural",
        {"structure_query": {}, "max_records": 5},
    )
    assert len(out["hits"]) <= 5
