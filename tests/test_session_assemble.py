"""Phase 5 RED-state test scaffold. Tasks 2-5 turn these GREEN.

Covers TOK-11 / / D5-02: wake_depth-branched session-start payload
shape + token budget enforcement at each branch.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity
from iai_mcp.session import SessionStartPayload, assemble_session_start
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------- token helpers
def _tok(text: str) -> int:
    """cl100k tokeniser with char/4 fallback. Self-contained per test convention."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4) if text else 0


def _empty_assignment() -> CommunityAssignment:
    return CommunityAssignment()


def _one_community_assignment() -> CommunityAssignment:
    cid = uuid4()
    return CommunityAssignment(
        node_to_community={uuid4(): cid},
        community_centroids={cid: [0.1] * EMBED_DIM},
        modularity=0.5,
        backend="leiden-networkx",
        top_communities=[cid],
        mid_regions={cid: [uuid4()]},
    )


def _seed_a_few_pinned(store: MemoryStore, n: int = 3) -> None:
    """Seed a handful of pinned records so standard/deep have content to render."""
    now = datetime.now(timezone.utc)
    for i in range(n):
        rec = MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {i}: important context for standard mode.",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.5,
            detail_level=5,
            pinned=True,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)


# ---------------------------------------------------------------- minimal mode
def test_minimal_payload_le_30_tokens(tmp_path):
    """TOK-11: minimal wake_depth yields ≤30 raw tok across new pointer fields."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "minimal"

    payload = assemble_session_start(
        store, _empty_assignment(), [], session_id="abc12345",
        profile_state=state,
    )
    total = (
        _tok(payload.identity_pointer)
        + _tok(payload.brain_handle)
        + _tok(payload.topic_cluster_hint)
    )
    assert total <= 30, (
        f"minimal payload {total} tok > 30; fields: "
        f"id={payload.identity_pointer!r} handle={payload.brain_handle!r} "
        f"topic={payload.topic_cluster_hint!r}"
    )


def test_minimal_payload_legacy_fields_empty(tmp_path):
    """D5-10 back-compat: minimal wake_depth leaves legacy fields empty."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    _seed_a_few_pinned(store, 3)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "minimal"

    payload = assemble_session_start(
        store, _empty_assignment(), [], session_id="abc12345",
        profile_state=state,
    )
    assert payload.l0 == ""
    assert payload.l1 == ""
    assert payload.l2 == []
    assert payload.rich_club == ""


def test_minimal_payload_has_new_fields(tmp_path):
    """D5-02: minimal payload populates identity_pointer/brain_handle/topic_cluster_hint."""
    import re
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "minimal"

    payload = assemble_session_start(
        store, _one_community_assignment(), [], session_id="abc12345",
        profile_state=state,
    )
    # identity_pointer: <id:XXXXXXXX> (8 hex) when L0 seeded
    assert re.match(r"<id:[0-9a-f]{8}>", payload.identity_pointer), payload.identity_pointer
    # brain_handle: <sess:... pend:N>
    assert re.match(r"<sess:.+ pend:\d+>", payload.brain_handle), payload.brain_handle
    # topic_cluster_hint: <topic:...>
    assert re.match(r"<topic:.+>", payload.topic_cluster_hint), payload.topic_cluster_hint


def test_minimal_payload_wake_depth_echoed(tmp_path):
    """Minimal payload echoes wake_depth='minimal' for introspection."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "minimal"

    payload = assemble_session_start(
        store, _empty_assignment(), [], session_id="s1",
        profile_state=state,
    )
    assert payload.wake_depth == "minimal"


# ---------------------------------------------------------------- standard mode
def test_standard_payload_preserves_phase1_behavior(tmp_path):
    """D5-10: wake_depth=standard reproduces Phase-1 1388-tok payload shape."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    _seed_a_few_pinned(store, 3)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "standard"

    payload = assemble_session_start(
        store, _empty_assignment(), [], session_id="s1",
        profile_state=state,
    )
    assert "IAI-MCP" in payload.l0, f"standard L0 should contain IAI-MCP: {payload.l0!r}"
    assert payload.wake_depth == "standard"


# ------------------------------------------------------------------ deep mode
def test_deep_payload_allows_2000_budget(tmp_path):
    """D5-02: deep mode lifts rich_club budget to 2000."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    _seed_a_few_pinned(store, 3)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "deep"

    payload = assemble_session_start(
        store, _empty_assignment(), [], session_id="s1",
        profile_state=state,
    )
    assert payload.total_cached_tokens <= 2000
    assert payload.wake_depth == "deep"


# --------------------------------------------------------- fallback behaviour
def test_unknown_wake_depth_falls_back_to_minimal(tmp_path):
    """D5-10 silent fallback: unknown wake_depth → minimal shape."""
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    from iai_mcp import profile
    state = profile.default_state()
    state["wake_depth"] = "invalid_value"

    payload = assemble_session_start(
        store, _empty_assignment(), [], session_id="s1",
        profile_state=state,
    )
    # minimal shape: legacy fields empty, new pointers populated
    assert payload.l0 == ""
    assert payload.l1 == ""
    assert payload.l2 == []
    assert payload.rich_club == ""
    # wake_depth echo either 'minimal' (silent rewrite) acceptable
    assert payload.wake_depth == "minimal"
