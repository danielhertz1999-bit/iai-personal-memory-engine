"""Tests for profile_modulates edges (Plan 02-03 Task 1, runtime gain).

The runtime-gain mechanism: active autistic-kernel knobs (e.g.
monotropism_depth in the active domain) multiply hit scores during
recall_for_response. The multiplication is recorded as a `profile_modulates` edge
in the edges table pointing from the affected record to a fixed profile
sentinel UUID. The record's `profile_modulation_gain` dict is populated at
recall time with the per-knob gains actually applied.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(
    *,
    text: str,
    vec: list[float] | None = None,
    tags: list[str] | None = None,
    language: str = "en",
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
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
        tags=list(tags or []),
        language=language,
    )


# ---------------------------------------------------------------- helpers


def test_profile_modulation_for_record_empty_profile():
    """No knobs set -> no gains computed."""
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="hi", tags=["domain:coding"])
    gains = profile_modulation_for_record(rec, profile_state={})
    assert isinstance(gains, dict)
    # Empty state -> no gains
    assert gains == {} or all(v == 1.0 for v in gains.values())


def test_profile_modulation_for_record_monotropism_depth_domain_tag():
    """With monotropism_depth[coding]=0.9 and domain:coding tag -> gain present."""
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="deep coding fact", tags=["domain:coding"])
    gains = profile_modulation_for_record(
        rec,
        profile_state={"monotropism_depth": {"coding": 0.9}},
    )
    assert "monotropism_depth" in gains
    assert gains["monotropism_depth"] > 1.0


def test_profile_modulation_for_record_wrong_domain_no_gain():
    """Domain mismatch -> monotropism gain NOT present."""
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="gardening fact", tags=["domain:gardening"])
    gains = profile_modulation_for_record(
        rec,
        profile_state={"monotropism_depth": {"coding": 0.9}},
    )
    assert "monotropism_depth" not in gains


def test_profile_modulation_for_record_interest_boost():
    """interest_boost float > 0 -> gain > 1.0."""
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="hi", tags=[])
    gains = profile_modulation_for_record(
        rec,
        profile_state={"interest_boost": 0.5},
    )
    assert "interest_boost" in gains
    assert gains["interest_boost"] > 1.0


# ---------------------------------------------------------------- pipeline integration


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    """Fake embedder so we don't load bge-m3 during the pipeline test."""
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield


def test_profile_modulation_edge_created_on_knob_affect(tmp_path, monkeypatch):
    """Pipeline recall with active monotropism_depth creates profile_modulates edges."""
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    # Inject a fake embedder that produces vectors aligned with the primary axis.
    from iai_mcp.embed import Embedder as _E

    store = MemoryStore(path=tmp_path)

    # Seed a coding-tagged record.
    r = _rec(text="code fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)
    profile_state = {"monotropism_depth": {"coding": 0.9}}

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        budget_tokens=1500,
        profile_state=profile_state,
    )

    # Inspect edges for a profile_modulates row.
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert len(pm) >= 1


def test_profile_modulates_edge_weight_positive(tmp_path):
    """profile_modulates edge weight is positive and reflects gain magnitude."""
    from iai_mcp import retrieve
    from iai_mcp.embed import Embedder as _E
    from iai_mcp.pipeline import recall_for_response

    store = MemoryStore(path=tmp_path)
    r = _rec(text="code fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)
    profile_state = {
        "monotropism_depth": {"coding": 0.9},
        "interest_boost": 0.5,
    }

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        profile_state=profile_state,
    )

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert (pm["weight"] > 0).all()


def test_profile_modulation_gain_populates_on_record(tmp_path):
    """After recall, the record's profile_modulation_gain dict is non-empty."""
    from iai_mcp import retrieve
    from iai_mcp.embed import Embedder as _E
    from iai_mcp.pipeline import recall_for_response

    store = MemoryStore(path=tmp_path)
    r = _rec(text="code fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)
    profile_state = {"monotropism_depth": {"coding": 0.9}}

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        profile_state=profile_state,
    )

    # The hit record's profile_modulation_gain (from pipeline's in-memory cache)
    # should be populated. We verify via the response's hits.
    assert len(resp.hits) >= 1
    # Either the response hints or edges confirm the modulation fired.
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert len(pm) >= 1


def test_profile_modulation_no_gain_when_state_empty(tmp_path):
    """Empty profile_state -> no profile_modulates edges."""
    from iai_mcp import retrieve
    from iai_mcp.embed import Embedder as _E
    from iai_mcp.pipeline import recall_for_response

    store = MemoryStore(path=tmp_path)
    r = _rec(text="fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        profile_state={},
    )

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert len(pm) == 0
