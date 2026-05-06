"""Tests for TOK-07 subagent delegation (Plan 02-04 Task 3, D-27).

serialize_session_for_subagent emits a JSON-safe dict containing:
- l0, l1, l2, rich_club segments (D-10 session-start payload)
- hashes dict (D-28 delta-encoding integration)
- proxy_tools list (5 Phase-1 memory tools; no 02-04 user-introspection tools)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
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


def _seeded_store(tmp_path) -> MemoryStore:
    store = MemoryStore(path=tmp_path)
    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)
    now = datetime.now(timezone.utc)
    for i in range(3):
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"fact {i}",
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
        store.insert(rec)
    return store


def test_serialize_session_keys(tmp_path):
    from iai_mcp.delegate import serialize_session_for_subagent
    from iai_mcp.retrieve import build_runtime_graph

    store = _seeded_store(tmp_path)
    _graph, assignment, rc = build_runtime_graph(store)
    out = serialize_session_for_subagent(store, assignment, rc)
    assert set(out.keys()) == {"l0", "l1", "l2", "rich_club", "hashes", "proxy_tools"}


def test_serialize_hashes_for_each_component(tmp_path):
    from iai_mcp.delegate import serialize_session_for_subagent
    from iai_mcp.retrieve import build_runtime_graph

    store = _seeded_store(tmp_path)
    _graph, assignment, rc = build_runtime_graph(store)
    out = serialize_session_for_subagent(store, assignment, rc)
    hashes = out["hashes"]
    for k in ("l0", "l1", "l2", "rich_club"):
        assert k in hashes
        assert isinstance(hashes[k], str)
        assert len(hashes[k]) == 16


def test_serialize_is_json_safe(tmp_path):
    from iai_mcp.delegate import serialize_session_for_subagent
    from iai_mcp.retrieve import build_runtime_graph

    store = _seeded_store(tmp_path)
    _graph, assignment, rc = build_runtime_graph(store)
    out = serialize_session_for_subagent(store, assignment, rc)
    # Round-trips through json without raising.
    blob = json.dumps(out)
    restored = json.loads(blob)
    assert restored["proxy_tools"] == out["proxy_tools"]


def test_subagent_proxy_tools_returns_five(tmp_path):
    from iai_mcp.delegate import subagent_proxy_tools

    tools = subagent_proxy_tools()
    assert len(tools) == 5
    names = {t["name"] for t in tools}
    assert names == {
        "memory_recall",
        "memory_reinforce",
        "memory_contradict",
        "memory_consolidate",
        "profile_get_set",
    }


def test_subagent_proxy_tools_excludes_02_04_new_tools():
    """Subagent doesn't get curiosity_pending / schema_list / events_query
    (those are user-introspection, not subagent tooling)."""
    from iai_mcp.delegate import subagent_proxy_tools

    names = {t["name"] for t in subagent_proxy_tools()}
    assert "curiosity_pending" not in names
    assert "schema_list" not in names
    assert "events_query" not in names


def test_serialize_l2_is_list_of_strings(tmp_path):
    from iai_mcp.delegate import serialize_session_for_subagent
    from iai_mcp.retrieve import build_runtime_graph

    store = _seeded_store(tmp_path)
    _graph, assignment, rc = build_runtime_graph(store)
    out = serialize_session_for_subagent(store, assignment, rc)
    assert isinstance(out["l2"], list)
    for item in out["l2"]:
        assert isinstance(item, str)
