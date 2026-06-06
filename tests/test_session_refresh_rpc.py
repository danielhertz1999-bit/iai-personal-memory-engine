"""Verify the session_refresh_if_stale daemon RPC and max_record_created_at helper.

Tests:
    test_max_created_at_empty_store -- empty store returns None
    test_max_created_at_after_insert -- returns the newest created_at ISO string
    test_not_stale_returns_empty -- watermark == MAX -> rendered=""
    test_stale_returns_nonempty -- watermark < MAX -> returns rendered + new_max_ts
    test_emit_free -- no session_started event emitted by refresh
    test_sc4_drain_before_read -- pending deferred entry absent before call,
                                          present in store after call (drain precedes read)
    test_sc5_global_store_cross_cwd -- record inserted under cwd A is recallable
                                          under cwd B from the same HOME-based store
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated HOME with keyring + crypto passphrase
# ---------------------------------------------------------------------------


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """Isolate HOME to tmp_path so tests never touch ~/.iai-mcp."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-session-refresh-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "hippo"))

    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    """Open a MemoryStore that respects the iai_home env overrides."""
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _insert_record(store, text: str):
    """Insert a record directly to bypass deferred capture."""
    from iai_mcp.capture import capture_turn
    return capture_turn(store, text=text, cue="test cue", tier="episodic", role="user")


def _write_drainable_deferred(home: Path, session_id: str, text: str) -> Path:
    """Write a non-active (drainable) deferred JSONL file.

    Uses a timestamped filename (no '.live.jsonl' suffix) so the MVP drain
    picks it up. Mirrors the format drain_deferred_captures expects.
    """
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": text,
        "cue": f"session {session_id} deferred cue",
        "tier": "episodic",
        "role": "user",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(
        json.dumps(header, ensure_ascii=False) + "\n"
        + json.dumps(event, ensure_ascii=False) + "\n"
    )
    return out


# ---------------------------------------------------------------------------
# max_record_created_at helper
# ---------------------------------------------------------------------------


def test_max_created_at_empty_store(iai_home):
    """Empty store: max_record_created_at returns None."""
    from iai_mcp.session import max_record_created_at
    store = _open_store()
    assert max_record_created_at(store) is None


def test_max_created_at_after_insert(iai_home):
    """After inserting records, max_record_created_at returns a non-None ISO string."""
    from iai_mcp.session import max_record_created_at
    store = _open_store()
    _insert_record(store, "alice said: first record inserted for max_created_at test")
    result = max_record_created_at(store)
    assert result is not None
    assert isinstance(result, str)
    # Second record; result should be >= previous (non-decreasing)
    _insert_record(store, "alice said: second record, must be at least 12 chars")
    result2 = max_record_created_at(store)
    assert result2 is not None
    assert result2 >= result


# ---------------------------------------------------------------------------
# dispatch session_refresh_if_stale: no-op when not stale
# ---------------------------------------------------------------------------


def test_not_stale_returns_empty(iai_home):
    """Watermark equals MAX(created_at) -> rendered="" (nothing newer)."""
    from iai_mcp.session import max_record_created_at
    from iai_mcp.core import dispatch
    store = _open_store()
    _insert_record(store, "alice said: record present before watermark is set")

    current_max = max_record_created_at(store)
    assert current_max is not None

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": current_max,
        "session_id": "test-session-no-op",
    })
    assert result["rendered"] == ""


# ---------------------------------------------------------------------------
# dispatch session_refresh_if_stale: recompose when stale
# ---------------------------------------------------------------------------


def test_stale_returns_nonempty(iai_home):
    """Watermark < MAX(created_at) -> non-empty rendered, new_max_ts > watermark,
    rendered length <= SESSION_START_CACHE_MAX_CHARS."""
    from iai_mcp.session import SESSION_START_CACHE_MAX_CHARS, max_record_created_at
    from iai_mcp.core import dispatch
    store = _open_store()

    # Insert enough text to get a non-empty standard brief.
    for i in range(5):
        _insert_record(
            store,
            f"alice said: memory entry {i} with sufficient length for brief composition test"
        )

    old_watermark = "2000-01-01T00:00:00+00:00"  # guaranteed older than any real record

    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": old_watermark,
        "session_id": "test-session-stale",
    })
    # Rendered brief should be non-empty.
    assert result["rendered"] != "", "Expected non-empty rendered when stale"
    assert result["new_max_ts"] > old_watermark
    assert len(result["rendered"]) <= SESSION_START_CACHE_MAX_CHARS


# ---------------------------------------------------------------------------
# Emit-free assertion
# ---------------------------------------------------------------------------


def test_emit_free(iai_home):
    """session_refresh_if_stale MUST NOT emit any session_started event."""
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.core import dispatch
    store = _open_store()

    for i in range(3):
        _insert_record(
            store,
            f"alice emit-free test record {i} with enough length to pass min_capture_len"
        )

    # Flush any buffered events before counting so the baseline is accurate.
    flush_event_buffer(store)
    before = len(query_events(store, kind="session_started"))

    dispatch(store, "session_refresh_if_stale", {
        "watermark": "2000-01-01T00:00:00+00:00",
        "session_id": "test-session-emit-free",
    })

    # Flush again so any newly emitted events are counted.
    flush_event_buffer(store)
    after = len(query_events(store, kind="session_started"))
    assert after == before, (
        f"session_refresh_if_stale emitted {after - before} session_started event(s); "
        "expected 0 (emit-free path required)"
    )


# ---------------------------------------------------------------------------
# SC4: drain-before-read
# ---------------------------------------------------------------------------


def test_sc4_drain_before_read(iai_home):
    """A pending deferred buffer entry is absent before the call and present
    in the store after session_refresh_if_stale (drain precedes the read).

    Satisfies SC4: the handler drains before querying MAX(created_at), so a
    just-deferred turn that has NOT yet been drained is captured and surfaced
    by the recompose in the same RPC call.
    """
    from iai_mcp.session import max_record_created_at
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore

    # Fresh store — no records yet.
    store = _open_store()
    old_watermark = max_record_created_at(store) or "2000-01-01T00:00:00+00:00"

    # Write a drainable deferred file (non-.live suffix) with a unique text.
    unique_text = "alice SC4 deferred unique content for drain-before-read assertion test"
    _write_drainable_deferred(iai_home, "sc4-test-session", unique_text)

    # BEFORE the call: the deferred entry must NOT be in the store yet.
    assert not any(
        "SC4 deferred unique content" in (r.literal_surface or "")
        for r in store.all_records()
    ), "Entry must be absent from store before drain runs"

    # Call the RPC — drain should run FIRST, then re-query MAX.
    result = dispatch(store, "session_refresh_if_stale", {
        "watermark": old_watermark,
        "session_id": "sc4-test-session",
    })

    # AFTER the call: the drained entry is now present in the store.
    assert any(
        "SC4 deferred unique content" in (r.literal_surface or "")
        for r in store.all_records()
    ), "Entry must be present in store after drain-before-read RPC"

    post_max = max_record_created_at(store)
    assert post_max is not None, "Store should have records after drain"
    assert post_max > old_watermark, (
        f"MAX(created_at) should have advanced: pre={old_watermark}, post={post_max}"
    )
    # new_max_ts in result must be the post-drain MAX.
    assert result["new_max_ts"] == post_max


# ---------------------------------------------------------------------------
# SC5: global store — cross-cwd record recallable
# ---------------------------------------------------------------------------


def test_sc5_global_store_cross_cwd(iai_home):
    """Record inserted under one cwd is recallable from a different cwd.

    The store root is HOME-based (~/.iai-mcp), NOT cwd-based. Inserting a
    record while process is in directory A and reading it while in directory B
    must succeed with the same record found.
    """
    import os
    from iai_mcp.store import MemoryStore

    # Create two separate tmp dirs for cwd simulation.
    dir_a = iai_home / "project_a"
    dir_b = iai_home / "project_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    try:
        # --- cwd = dir_a: insert a record ---
        os.chdir(str(dir_a))
        store_a = _open_store()
        result = _insert_record(
            store_a,
            "alice project-b work: unique cross-cwd test record for SC5 global store assertion"
        )
        assert result.get("status") in ("inserted", "reinforced"), (
            f"Insert failed: {result}"
        )

        # --- cwd = dir_b: open a new store handle, verify record is found ---
        os.chdir(str(dir_b))
        store_b = _open_store()
        all_records = store_b.all_records()
        texts = [r.literal_surface for r in all_records]
        found = any(
            "SC5 global store" in t or "cross-cwd test record" in t
            for t in texts
        )
        assert found, (
            "SC5 FAIL: record inserted under dir_a not found under dir_b. "
            f"Store root must be HOME-based, not cwd-based. "
            f"Records found: {[t[:60] for t in texts]}"
        )
    finally:
        os.chdir(original_cwd)
