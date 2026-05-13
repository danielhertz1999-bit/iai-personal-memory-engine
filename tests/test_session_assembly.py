"""Tests for the session-start assembler (, , ).

the DEFAULT wake_depth flipped to `minimal` (lazy <=30
tok payload). Tests that assert Phase-1 eager-dump behaviour now pass
``profile_state={"wake_depth": "standard"}`` explicitly to continue
exercising the back-compat legacy path.

Covers:
- Graceful empty-store path (total_cached_tokens == 0, l0 == "").
- L0 identity rendering -- "IAI-MCP" appears in payload.l0 when seeded.
- Total cached budget respected (<= 2000 tok) on realistic pinned content.
- L2 community cap at 7 ( Yeo-like).
- Rich-club segment truncation at 1500-tok budget.
- core.py `session_start_payload` dispatch wiring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity, dispatch
from iai_mcp.session import (
    L0_RECORD_UUID,
    L2_COMMUNITY_CAP,
    RICH_CLUB_BUDGET_TOKENS,
    TOTAL_CACHED_BUDGET,
    SessionStartPayload,
    _approx_tokens,
    assemble_session_start,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# D5-02: Phase-1 eager behaviour lives behind wake_depth="standard"
# now that the default flipped to "minimal". Legacy tests opt in explicitly.
_STANDARD = {"wake_depth": "standard"}


# ------------------------------------------------------------- helpers


def _l0_record(store: MemoryStore) -> None:
    """Seed the fixed-UUID L0 identity record (matches core._seed_l0_identity)."""
    _seed_l0_identity(store)


def _pinned_record(
    store: MemoryStore,
    text: str,
    community_id: UUID | None = None,
    tags: list[str] | None = None,
) -> MemoryRecord:
    r = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.5,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language="en",
    )
    store.insert(r)
    return r


# -------------------------------------------------- graceful empty-store path


def test_empty_store_graceful(tmp_path):
    """Empty store -> all segments empty, token totals zero on cached.

    assert on the standard (Phase-1) path — minimal mode would
    emit pointer handles even on empty stores, which is by design.
    """
    store = MemoryStore(path=tmp_path)
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert payload.l0 == ""
    assert payload.l1 == ""
    assert payload.l2 == []
    assert payload.rich_club == ""
    assert payload.total_cached_tokens == 0
    # Dynamic tail is a fixed reserve even on empty stores.
    assert payload.total_dynamic_tokens > 0


# ---------------------------------------------------------- identity


def test_l0_renders_identity(tmp_path):
    """a seeded L0 record puts 'IAI-MCP' into the L0 segment (standard mode)."""
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert "IAI-MCP" in payload.l0


def test_l0_uses_fixed_uuid(tmp_path):
    """The assembler MUST read from the canonical L0 UUID. Standard mode."""
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    # Confirm the seed landed at the fixed UUID, not some random new UUID.
    assert store.get(L0_RECORD_UUID) is not None
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert payload.l0 != ""


def test_l0_segment_excludes_literal_only_at_cap(tmp_path):
    """L0 segment contains aaak_index header plus the literal (truncated if long)."""
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    # The L0 record's aaak_index is stamped at seed time -> shows up in payload.
    assert "W:" in payload.l0  # wing marker from generate_aaak_index


# ------------------------------------------------------------- budget


def test_total_cached_budget_respected(tmp_path):
    """L0 + L1 + L2 + rich_club <= TOTAL_CACHED_BUDGET (2000 tok)."""
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    # 10 pinned L1 records with reasonable short content.
    for i in range(10):
        _pinned_record(store, f"Pinned fact #{i}: short verbatim content here.")
    payload = assemble_session_start(store, CommunityAssignment(), [])
    assert payload.total_cached_tokens <= TOTAL_CACHED_BUDGET


def test_l1_caps_at_max_records(tmp_path):
    """L1 segment stays bounded even with many pinned records (10-entry cap)."""
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    # Seed 20 pinned records -- L1 should truncate to 10.
    for i in range(20):
        _pinned_record(store, f"Pinned fact #{i}")
    payload = assemble_session_start(store, CommunityAssignment(), [])
    l1_lines = payload.l1.split("\n") if payload.l1 else []
    assert len(l1_lines) <= 10


# -------------------------------------------------- Yeo-like cap


def test_l2_capped_at_seven(tmp_path):
    """: L2 summaries never exceed 7 regardless of input community count."""
    store = MemoryStore(path=tmp_path)
    # Create 10 fake communities each with one member record.
    assignment = CommunityAssignment()
    for i in range(10):
        cid = uuid4()
        rec = _pinned_record(
            store, f"member of community {i}", community_id=cid
        )
        assignment.top_communities.append(cid)
        assignment.mid_regions[cid] = [rec.id]
        assignment.community_centroids[cid] = [0.0] * EMBED_DIM
    payload = assemble_session_start(store, assignment, [])
    assert len(payload.l2) <= L2_COMMUNITY_CAP
    assert L2_COMMUNITY_CAP == 7


# ----------------------------------------------- rich-club budget truncation


def test_rich_club_truncation_under_budget(tmp_path):
    """Passing 50 records with long surfaces still keeps rich_club <= 1500 tok."""
    store = MemoryStore(path=tmp_path)
    # Build 50 records with ~300 chars each (~75 tok each).
    rich_uuids: list[UUID] = []
    for i in range(50):
        r = _pinned_record(store, f"rich-club entry {i}: " + ("x" * 280))
        rich_uuids.append(r.id)
    payload = assemble_session_start(store, CommunityAssignment(), rich_uuids)
    assert _approx_tokens(payload.rich_club) <= RICH_CLUB_BUDGET_TOKENS


# ---------------------------------------------- core.py dispatch integration


def test_session_start_payload_dispatch_empty(tmp_path):
    """core.dispatch('session_start_payload') returns the canonical shape even on empty store."""
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "session_start_payload", {})
    # Shape keys are all present regardless of whether the store is populated.
    for key in (
        "l0",
        "l1",
        "l2",
        "rich_club",
        "total_cached_tokens",
        "total_dynamic_tokens",
        "breakpoint_marker",
    ):
        assert key in result
    # On a fresh store the L0 segment is empty (no seed yet).
    assert result["l0"] == ""
    assert result["total_cached_tokens"] == 0


def test_session_start_payload_dispatch_with_l0(tmp_path):
    """Once L0 is seeded, dispatch returns identity content.

    D5-10: per-process wake_depth stays at the 'minimal' default,
    so we temporarily flip it to 'standard' for this back-compat assertion
    and restore afterwards. Thread-safety is not a concern for unit tests.
    """
    import iai_mcp.core as core
    original = core._profile_state.get("wake_depth", "minimal")
    core._profile_state["wake_depth"] = "standard"
    try:
        store = MemoryStore(path=tmp_path)
        _seed_l0_identity(store)
        result = dispatch(store, "session_start_payload", {})
        assert "IAI-MCP" in result["l0"]
        assert result["breakpoint_marker"] == "--<cache-breakpoint>--"
    finally:
        core._profile_state["wake_depth"] = original


def test_payload_type_is_session_start_payload(tmp_path):
    """Direct assemble_session_start returns a SessionStartPayload instance."""
    store = MemoryStore(path=tmp_path)
    payload = assemble_session_start(store, CommunityAssignment(), [])
    assert isinstance(payload, SessionStartPayload)
    assert payload.breakpoint_marker == "--<cache-breakpoint>--"
