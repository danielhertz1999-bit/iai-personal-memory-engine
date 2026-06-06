"""Cross-session e2e simulation — querying across sessions works end-to-end.

Scope:
- store path: session B's captured turn is retrievable from session A.
- live path: session A's PENDING live turn is returned for session A
  WITHOUT any drain, as seen by a cross-session caller.

SAFETY: uses tmp_path only, never ~/.iai-mcp/ or the live daemon socket.
"""
from __future__ import annotations

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from tests.conftest_recall import make_tmp_store


def test_cross_session_recency_query(tmp_path):
    """REQ-5: session B's capture is retrievable from session A's query context.

    Session B inserts a distinctive phrase with its own session_id.
    Session A calls episodes_recent (global + session-filtered) and must
    find B's phrase in both queries.

    Requires the episodes_recent dispatch handler.
    """
    # Shared store simulates the global ~/.iai-mcp/ store available to both sessions.
    store = make_tmp_store(tmp_path)

    # --- Session B activity: captures a distinctive phrase ---
    b_session_id = "b-session-111"
    distinctive_phrase = "distinctive phrase bxyz phase59 cross session marker"

    result_b = capture_turn(
        store,
        cue="session b distinctive phrase",
        text=distinctive_phrase,
        tier="episodic",
        session_id=b_session_id,
        role="user",
    )
    assert result_b["status"] == "inserted", f"session B insert failed: {result_b}"

    # Also insert some session A turns so the global query has more context.
    a_session_id = "a-session-222"
    for i in range(2):
        capture_turn(
            store,
            cue=f"session a turn {i}",
            text=f"session a turn {i} normal content for e2e test phase59",
            tier="episodic",
            session_id=a_session_id,
            role="user",
        )

    # --- Session A query: global episodes_recent ---
    global_result = dispatch(store, "episodes_recent", {"n": 5})
    # Requires the episodes_recent dispatch handler.
    assert "turns" in global_result, (
        f"episodes_recent global query missing 'turns': {global_result!r}"
    )
    global_surfaces = [t.get("literal_surface", "") for t in global_result["turns"]]
    assert any(distinctive_phrase in s for s in global_surfaces), (
        f"session B's phrase not found in global query result; "
        f"surfaces: {global_surfaces!r}"
    )

    # --- Session A query: session-filtered episodes_recent ---
    filtered_result = dispatch(
        store,
        "episodes_recent",
        {"n": 5, "session_id": b_session_id},
    )
    assert "turns" in filtered_result, (
        f"episodes_recent session-filtered query missing 'turns': {filtered_result!r}"
    )
    filtered_turns = filtered_result["turns"]
    assert filtered_turns, "session-filtered query returned no turns"
    top = filtered_turns[0]
    assert top.get("literal_surface") == distinctive_phrase, (
        f"top session-filtered turn must be B's phrase; "
        f"got {top.get('literal_surface')!r}"
    )
    assert top.get("session_id") == b_session_id, (
        f"session_id on filtered top turn must be {b_session_id!r}; "
        f"got {top.get('session_id')!r}"
    )


# ---------------------------------------------------------------------------
#: pending live turn visible in cross-session query (REQ-1)
# ---------------------------------------------------------------------------


def test_pending_live_visible_across_sessions(tmp_path, monkeypatch):
    """REQ-1 (live path): session A's pending live turn is returned for session A.

    Session A has a pending live file (no drain ran); a caller invokes
    episodes_recent(session_id=A) and A's live turn is returned. This exercises
    the cross-session live-read path (caller != session owner — session A
    querying for session A's turns from "session B's context").
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))

    store = make_tmp_store(tmp_path)

    a_session_id = "cross-live-a-session"
    live_text = "cross session pending live turn text long enough for capture test"

    from iai_mcp.capture import write_deferred_event
    write_deferred_event(a_session_id, "user", live_text)

    # No drain called — the live file has not been drained into the store.
    result = dispatch(
        store,
        "episodes_recent",
        {"n": 10, "session_id": a_session_id},
    )
    assert "turns" in result, f"episodes_recent missing 'turns': {result!r}"
    turns = result["turns"]
    assert len(turns) >= 1, f"expected >= 1 pending turn; got {len(turns)}"
    surfaces = [t.get("literal_surface", "") for t in turns]
    assert any(live_text in s for s in surfaces), (
        f"pending live turn must be returned without drain; "
        f"got surfaces: {surfaces!r}"
    )
    # record_id must be pending:..., not literal "None"
    assert all(t["record_id"] != "None" for t in turns), (
        f"pending turn record_id must not be literal 'None'"
    )
