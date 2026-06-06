"""Regression tests for Synaptic Tagging-and-Capture (STC).

Locks the integration contracts of the Frey-Morris 1997 STC analogue:

    - PeriEventBuffer ring buffer respects maxlen (deque eviction).
    - PeriEventBuffer.flush_within_window filters by time window and
      does NOT mutate the deque.
    - End-to-end strong-event upgrade -- N semantic records added
      to the buffer + write_event(strong_kind) -> all in-window records
      upgraded to episodic, one stc_upgrade_pass event per record;
      out-of-window records are NOT upgraded.
    - MemoryStore.upgrade_tier refuses non-upgrade moves
      (ValueError naming the direction) and emits exactly one
      stc_upgrade_pass event with the documented body shape.
    - dry_run preserves no-mutation -- zero row tier mutations, N events
      emitted with dry_run_mode=True, buffer entries survive.
    - every one of the 4 STC env vars fails loud with ValueError naming
      the offending var.
    - Recursion prevention: stc_upgrade_pass is NOT in the default
      strong_event_types so emitting a fake stc_upgrade_pass MUST NOT
      cascade into another upgrade pass (defense-in-depth, prevents
      unbounded recursion if a user override added it).

Fixtures are inline. Synthetic stores use tmp_path with user_id='alice'.
No live embedder call, no real LifecycleStateMachine construction.
"""
# Standard-library imports first so optional iai_mcp.* imports fail loud
# with a clear ImportError if the package layout changes.
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from iai_mcp.daemon import _load_stc_config
from iai_mcp.events import query_events, write_event
from iai_mcp.peri_event_buffer import (
    PeriEventBuffer,
    PeriEventEntry,
    get_buffer,
    set_buffer,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction stays isolated from any on-disk store, AND wipe
# every IAI_MCP_* env var so each test starts from documented
# defaults. Tests that need a specific value re-set after this fixture.
@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_PERI_EVENT_BUFFER_SIZE",
        "IAI_MCP_PERI_EVENT_WINDOW_SEC",
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        "IAI_MCP_STC_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# fake-daemon-wrapper: register a PeriEventBuffer singleton for the
# test, tear down at end. No real LifecycleStateMachine construction.
# Idempotent across tests because set_buffer(None) runs in finally.
@pytest.fixture
def stc_buffer() -> Iterator[PeriEventBuffer]:
    buf = PeriEventBuffer(maxlen=20)
    set_buffer(buf)
    try:
        yield buf
    finally:
        set_buffer(None)


def _make_store(tmp_path: Path) -> MemoryStore:
    """Build a per-test MemoryStore rooted at tmp_path."""
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


# Build a minimal MemoryRecord. tier defaults to "semantic" so the STC
# upgrade path (semantic -> episodic) is exercised end-to-end. Fixed
# small-magnitude embedding skips the live embedder cost.
def _make_record(
    *,
    embed_dim: int,
    tier: str = "semantic",
    literal: str = "alice prefers tea over coffee",
    created_at: datetime | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier=tier,
        literal_surface=literal,
        aaak_index="",
        embedding=[0.01] * embed_dim,
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
        language="en",
        tags=["t"],
    )


# ---------------------------------------------------------------------------
# Test 1: buffer maxlen respected (deque eviction)
# ---------------------------------------------------------------------------


def test_buffer_maxlen_respected() -> None:
    """deque(maxlen=N) evicts the oldest entry transparently. Adding
    N+K entries leaves exactly N in the buffer; flush_within_window over
    a wide window returns those N entries."""
    buf = PeriEventBuffer(maxlen=3)
    now = datetime.now(timezone.utc)
    for i in range(5):
        buf.add(uuid.uuid4(), now, "semantic")
    assert len(buf) == 3
    entries = buf.flush_within_window(now, 3600)
    assert len(entries) == 3
    # All survivors are PeriEventEntry instances tagged "semantic".
    assert all(isinstance(e, PeriEventEntry) for e in entries)
    assert all(e.original_tier == "semantic" for e in entries)


# ---------------------------------------------------------------------------
# Test 2: flush_within_window filters by time + is non-destructive
# ---------------------------------------------------------------------------


def test_flush_within_window_filters_by_time() -> None:
    """flush_within_window returns only entries whose age <= window_sec
    AND does NOT mutate the underlying deque (subsequent flush over a
    wider window must still see every entry that was ever added)."""
    buf = PeriEventBuffer(maxlen=10)
    now = datetime.now(timezone.utc)
    # Three entries at offsets 10s / 100s / 5000s from now.
    buf.add(uuid.uuid4(), now - timedelta(seconds=10), "semantic")
    buf.add(uuid.uuid4(), now - timedelta(seconds=100), "semantic")
    buf.add(uuid.uuid4(), now - timedelta(seconds=5000), "semantic")

    # 1800s window: only the 10s + 100s entries qualify.
    in_window = buf.flush_within_window(now, 1800)
    assert len(in_window) == 2

    # Non-destructive: a second flush over a wider window still sees all 3.
    all_entries = buf.flush_within_window(now, 10_000)
    assert len(all_entries) == 3
    # Buffer itself is unchanged (no implicit mutation).
    assert len(buf) == 3


# ---------------------------------------------------------------------------
# Test 3: strong-event end-to-end upgrade pass
# ---------------------------------------------------------------------------


def test_strong_event_triggers_upgrade_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stc_buffer: PeriEventBuffer,
) -> None:
    """End-to-end: 5 semantic records in the buffer + 1 record
    outside the peri-event window. A strong-event write_event fan-out
    upgrades exactly the 5 in-window records to episodic, emits 5
    stc_upgrade_pass events with the documented body shape, and leaves
    the out-of-window record untouched.
    """
    # Force live mutation; pytest defaults dry_run=True at daemon.py
    # under PYTEST_CURRENT_TEST.
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    # 5 in-window semantic records (captured just now -- well inside 1800s).
    in_window_records: list[MemoryRecord] = []
    for i in range(5):
        rec = _make_record(
            embed_dim=embed_dim,
            tier="semantic",
            literal=f"alice said something {i}",
        )
        store.insert(rec)
        stc_buffer.add(rec.id, rec.created_at, "semantic")
        in_window_records.append(rec)

    # 1 out-of-window record (captured_at = now - 4000s; outside default
    # 1800s window). Buffer-add uses the deliberately-stale timestamp so
    # flush_within_window filters it out.
    old_ts = datetime.now(timezone.utc) - timedelta(seconds=4000)
    rec_old = _make_record(
        embed_dim=embed_dim,
        tier="semantic",
        literal="bob said something old",
        created_at=old_ts,
    )
    store.insert(rec_old)
    stc_buffer.add(rec_old.id, old_ts, "semantic")

    # Strong-event trigger: write_event with kind="memory_capture" (in
    # default strong_event_types). The post-emit STC hook fan-outs
    # to buf.trigger_stc -> upgrade_tier on each in-window entry.
    write_event(store, "memory_capture", {"text": "alice noted a thing"})

    # Every in-window record now has tier=="episodic".
    for rec in in_window_records:
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.tier == "episodic", (
            f"in-window record {rec.id} must be upgraded to episodic, "
            f"got {fetched.tier!r}"
        )

    # Out-of-window record stays semantic.
    fetched_old = store.get(rec_old.id)
    assert fetched_old is not None
    assert fetched_old.tier == "semantic", (
        f"out-of-window record must NOT be upgraded, got {fetched_old.tier!r}"
    )

    # Exactly 5 stc_upgrade_pass events emitted (one per in-window record).
    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    assert len(events) == 5, (
        f"expected 5 stc_upgrade_pass events, got {len(events)}"
    )
    # Per-event body shape: 5 documented keys with the correct values.
    expected_keys = {
        "record_id", "from_tier", "to_tier",
        "trigger_event_type", "dry_run_mode",
    }
    in_window_ids = {str(r.id) for r in in_window_records}
    seen_ids: set[str] = set()
    for ev in events:
        body = ev["data"]
        assert set(body.keys()) == expected_keys, (
            f"event body keys must be {expected_keys}, got {set(body.keys())}"
        )
        assert body["from_tier"] == "semantic"
        assert body["to_tier"] == "episodic"
        assert body["trigger_event_type"] == "memory_capture"
        assert body["dry_run_mode"] is False
        assert body["record_id"] in in_window_ids
        seen_ids.add(body["record_id"])
    assert seen_ids == in_window_ids, (
        f"every in-window record must have one event; "
        f"missing={in_window_ids - seen_ids}, extra={seen_ids - in_window_ids}"
    )


# ---------------------------------------------------------------------------
# Test 4: downgrade raises + event body shape
# ---------------------------------------------------------------------------


def test_upgrade_tier_downgrade_raises_and_event_body_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """episodic -> semantic is a downgrade; upgrade_tier raises
    ValueError naming the refusal (contract: never downgrade).
    Then exercise the upgrade path: semantic -> episodic returns True,
    mutates the row, and emits exactly one stc_upgrade_pass event with
    the documented body shape including dry_run_mode=False."""
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")
    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    # Downgrade attempt: episodic -> semantic raises ValueError.
    rec_ep = _make_record(
        embed_dim=embed_dim, tier="episodic", literal="alice already episodic",
    )
    store.insert(rec_ep)
    with pytest.raises(ValueError, match="refusing non-upgrade"):
        store.upgrade_tier(
            rec_ep.id, "semantic",
            trigger_event_type="test_downgrade",
            dry_run=False,
        )
    # No mutation on the refused call.
    refetched = store.get(rec_ep.id)
    assert refetched is not None
    assert refetched.tier == "episodic"

    # Same-tier attempt: episodic -> episodic also refused (<= guard).
    with pytest.raises(ValueError, match="refusing non-upgrade"):
        store.upgrade_tier(
            rec_ep.id, "episodic",
            trigger_event_type="test_sametier",
            dry_run=False,
        )

    # Invalid tier name: bare ValueError naming the invalid token.
    with pytest.raises(ValueError, match="invalid new_tier"):
        store.upgrade_tier(
            rec_ep.id, "not_a_tier",
            trigger_event_type="test_invalid",
            dry_run=False,
        )

    # Legitimate upgrade: semantic -> episodic, returns True, mutates row.
    rec_sem = _make_record(
        embed_dim=embed_dim, tier="semantic", literal="bob said hello",
    )
    store.insert(rec_sem)
    ok = store.upgrade_tier(
        rec_sem.id, "episodic",
        trigger_event_type="test_event",
        dry_run=False,
    )
    assert ok is True
    refetched = store.get(rec_sem.id)
    assert refetched is not None
    assert refetched.tier == "episodic"

    # Exactly one stc_upgrade_pass event for this record_id.
    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    matching = [
        ev for ev in events
        if ev["data"].get("record_id") == str(rec_sem.id)
    ]
    assert len(matching) == 1
    body = matching[0]["data"]
    assert set(body.keys()) == {
        "record_id", "from_tier", "to_tier",
        "trigger_event_type", "dry_run_mode",
    }
    assert body["from_tier"] == "semantic"
    assert body["to_tier"] == "episodic"
    assert body["trigger_event_type"] == "test_event"
    assert body["dry_run_mode"] is False


# ---------------------------------------------------------------------------
# Test 5: dry_run preserves zero row mutations + buffer survives
# ---------------------------------------------------------------------------


def test_dry_run_preserves_no_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stc_buffer: PeriEventBuffer,
) -> None:
    """With IAI_MCP_STC_DRY_RUN=true, 3 semantic records added to the
    buffer + a memory_capture event:
      - event still fires for each candidate with dry_run_mode=True
      - zero row tier mutations (store.get() returns tier='semantic')
      - buffer entries are NOT cleared (dry-run preserves the deque)
    """
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "true")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    records: list[MemoryRecord] = []
    for i in range(3):
        rec = _make_record(
            embed_dim=embed_dim, tier="semantic",
            literal=f"alice noted {i}",
        )
        store.insert(rec)
        stc_buffer.add(rec.id, rec.created_at, "semantic")
        records.append(rec)

    pre_buffer_len = len(stc_buffer)
    assert pre_buffer_len == 3

    write_event(store, "memory_capture", {"text": "alice triggered a strong event"})

    # Zero row mutation: every record still tier='semantic'.
    for rec in records:
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.tier == "semantic", (
            f"dry-run must NOT mutate row tier; got {fetched.tier!r} for {rec.id}"
        )

    # Exactly 3 stc_upgrade_pass events with dry_run_mode=True.
    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    assert len(events) == 3, (
        f"expected 3 stc_upgrade_pass events in dry-run, got {len(events)}"
    )
    for ev in events:
        body = ev["data"]
        assert body["dry_run_mode"] is True
        assert body["from_tier"] == "semantic"
        assert body["to_tier"] == "episodic"
        assert body["trigger_event_type"] == "memory_capture"

    # Dry-run leaves the buffer entries intact (clear_processed skipped).
    assert len(stc_buffer) == 3, (
        f"dry-run must preserve buffer entries, got len={len(stc_buffer)}"
    )


# ---------------------------------------------------------------------------
# Test 6: every STC env var fails loud with ValueError naming the var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        # peri_event_buffer_size: int in [1, 1000]
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "0"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "not-an-int"),
        # peri_event_window_sec: int in [1, 86400]
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "-1"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "0"),
        # strong_event_types: non-empty CSV with no empty tokens
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", ""),
        # dry_run: must be in documented vocab
        ("IAI_MCP_STC_DRY_RUN", "banana"),
    ],
)
def test_env_var_fail_loud(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Every malformed STC knob fails loud at _load_stc_config(). The
    error message MUST name the offending env var so operators can act."""
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_stc_config()


# ---------------------------------------------------------------------------
# Test 7: threat-flag -- stc_upgrade_pass NOT in default strong
# events -> no recursion when a fake stc_upgrade_pass is emitted directly.
# ---------------------------------------------------------------------------


def test_stc_upgrade_pass_NOT_in_default_strong_events_no_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stc_buffer: PeriEventBuffer,
) -> None:
    """Threat flag (recursion-on-misconfig defense-in-depth):
    the default strong_event_types deliberately excludes stc_upgrade_pass
    so the post-emit STC hook in write_event does NOT fan out when a
    stc_upgrade_pass event is observed. Emit a fake stc_upgrade_pass with
    a semantic record in the buffer; assert the record stays semantic
    and the events table contains exactly one stc_upgrade_pass (the one
    we just wrote -- no cascade)."""
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")
    # Sanity-check the default exclusion (defense-in-depth invariant).
    cfg = _load_stc_config()
    assert "stc_upgrade_pass" not in cfg.strong_event_types

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    rec = _make_record(
        embed_dim=embed_dim, tier="semantic",
        literal="alice peri-event candidate",
    )
    store.insert(rec)
    stc_buffer.add(rec.id, rec.created_at, "semantic")

    # Directly emit a fake stc_upgrade_pass. The post-emit hook in
    # write_event MUST treat this kind as a non-strong event (default
    # config excludes it) and NOT cascade into trigger_stc -> upgrade_tier
    # -> another stc_upgrade_pass ->... unbounded recursion.
    write_event(
        store,
        "stc_upgrade_pass",
        {
            "record_id": str(uuid.uuid4()),
            "from_tier": "semantic",
            "to_tier": "episodic",
            "trigger_event_type": "fake_test_emit",
            "dry_run_mode": False,
        },
    )

    # Record stays semantic -- the upgrade pass was NOT triggered.
    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.tier == "semantic", (
        f"defense-in-depth: stc_upgrade_pass must NOT fan out by default; "
        f"got {fetched.tier!r}"
    )

    # Exactly ONE stc_upgrade_pass event: the one we explicitly emitted.
    # If recursion had fired we would see >= 2 events (our fake + the
    # cascaded upgrade for the buffered record).
    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    assert len(events) == 1, (
        f"expected exactly one stc_upgrade_pass (the fake emit, no cascade); "
        f"got {len(events)}"
    )
    assert events[0]["data"]["trigger_event_type"] == "fake_test_emit"


if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
