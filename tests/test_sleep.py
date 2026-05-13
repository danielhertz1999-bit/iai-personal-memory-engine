"""Tests for iai_mcp.sleep — CLS replay scheduler + light/heavy consolidation (, , , ).

D-16 scheduler: ACTIVITY / TIME / MANUAL modes; 48h force-run; TZ-aware quiet window.
D-19 FSRS decay sweep: `_decay_edges` on hebbian edges only; invariant edges spared.
D-29 unified: light at session_exit, heavy in quiet window.
D-GUARD: `should_call_llm` ladder consulted before any Tier-1 path.

Test constructors use vectors sized to `store.embed_dim` so they work under
the bge-m3 1024d default.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------- helpers

def _record(
    *,
    text: str = "hi",
    vec: list[float] | None = None,
    tags: list[str] | None = None,
    tier: str = "episodic",
    detail_level: int = 2,
    language: str = "en",
    never_decay: bool = False,
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=never_decay,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )


# ============================================================== SleepMode + SleepConfig


def test_sleep_mode_enum_has_three_values():
    from iai_mcp.sleep import SleepMode

    assert SleepMode.ACTIVITY.value == "activity"
    assert SleepMode.TIME.value == "time"
    assert SleepMode.MANUAL.value == "manual"


def test_sleep_config_defaults():
    from iai_mcp.sleep import SleepConfig, SleepMode

    cfg = SleepConfig()
    assert cfg.mode == SleepMode.ACTIVITY
    assert cfg.quiet_window == (22, 6)
    assert cfg.require_idle_minutes == 30
    assert cfg.max_defer_hours == 48
    assert cfg.llm_enabled is False
    assert cfg.light_on_exit is True


# ================================================================ should_run_heavy


def test_should_run_heavy_activity_mode_inside_window():
    """ACTIVITY mode + 40min idle + 23:30 user-local -> (True, "")."""
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    # 23:30 UTC
    now = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is True
    assert reason == ""


def test_should_run_heavy_activity_mode_outside_window():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    # 15:00 is outside (22, 6) quiet window
    now = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False
    assert "quiet window" in reason.lower() or "outside" in reason.lower()


def test_should_run_heavy_activity_mode_too_recent():
    """Idle < 30min -> blocked."""
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    last = now - timedelta(minutes=5)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False
    assert "idle" in reason.lower()


def test_should_run_heavy_time_mode_only_at_3am():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.TIME)
    tz = ZoneInfo("UTC")
    # Hour != 3 -> False
    now_2am = datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc)
    ok_2, _ = should_run_heavy(now_2am, now_2am - timedelta(hours=1), cfg, tz)
    assert ok_2 is False

    now_3am = datetime(2026, 1, 1, 3, 30, tzinfo=timezone.utc)
    ok_3, _ = should_run_heavy(now_3am, now_3am - timedelta(hours=1), cfg, tz)
    assert ok_3 is True


def test_should_run_heavy_manual_mode_never_auto():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.MANUAL)
    tz = ZoneInfo("UTC")
    # Even with 80h idle and in quiet window, MANUAL returns False.
    now = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False
    assert "manual" in reason.lower()


def test_should_run_heavy_48h_force():
    """idle > 48h -> force-run regardless of window."""
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    # 15:00 local (outside window) but 50h idle -> force run
    now = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=50)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is True
    assert "defer" in reason.lower() or "48" in reason


def test_should_run_heavy_respects_user_tz_tokyo():
    """quiet_window(22,6) with Asia/Tokyo; UTC 13:00 = JST 22:00 -> inside window."""
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("Asia/Tokyo")
    # UTC 13:00 = JST 22:00 (inside window)
    now = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is True


def test_should_run_heavy_respects_user_tz_utc():
    """Same UTC 13:00 with UTC tz -> 13:00 is OUT of (22,6)."""
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False


# ============================================================== light consolidation


def test_run_light_consolidation_returns_expected_shape(tmp_path):
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = run_light_consolidation(store, session_id="s-light")
    assert isinstance(result, dict)
    assert "fsrs_ticked" in result
    assert "cooccurrence_updates" in result
    assert result["mode"] == "light"


def test_run_light_consolidation_no_llm_call(tmp_path, monkeypatch):
    """Light phase must NOT touch should_call_llm -- pure local."""
    from iai_mcp import sleep as sleep_mod
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    call_count = {"n": 0}
    original_should = sleep_mod.should_call_llm

    def _counting(*args, **kwargs):
        call_count["n"] += 1
        return original_should(*args, **kwargs)

    monkeypatch.setattr(sleep_mod, "should_call_llm", _counting)

    store = MemoryStore(path=tmp_path)
    # Seed a record
    store.insert(_record())

    run_light_consolidation(store, session_id="s-light")
    assert call_count["n"] == 0


def test_run_light_consolidation_emits_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    run_light_consolidation(store, session_id="s-x")
    events = query_events(store, kind="cls_consolidation_run")
    assert len(events) >= 1
    ev = events[0]
    assert ev["data"]["mode"] == "light"
    assert ev["session_id"] == "s-x"


# ============================================================== heavy consolidation


def test_run_heavy_consolidation_uses_d_guard(tmp_path, monkeypatch):
    """When should_call_llm returns False (no api key), heavy completes via Tier 0."""
    from iai_mcp.events import query_events
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    # Seed with 3 records so a trivial cluster is possible
    recs = [_record(text=f"rec {i}") for i in range(3)]
    for r in recs:
        store.insert(r)
    # Boost a Hebbian triangle among them
    store.boost_edges([(recs[0].id, recs[1].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[1].id, recs[2].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[0].id, recs[2].id)], edge_type="hebbian", delta=0.5)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    result = run_heavy_consolidation(
        store, session_id="s-heavy", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )
    assert result["mode"] == "heavy"
    assert result["tier"] == "tier0"

    events = query_events(store, kind="cls_consolidation_run")
    heavy_events = [e for e in events if e["data"].get("mode") == "heavy"]
    assert len(heavy_events) >= 1
    assert heavy_events[0]["data"]["tier"] == "tier0"


def test_run_heavy_consolidation_creates_consolidated_from_edges(tmp_path):
    """3+ cohesive records produce one summary record + consolidated_from edges."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    # Seed 3 cohesive records
    recs = [_record(text=f"fact {i}") for i in range(3)]
    for r in recs:
        store.insert(r)
    # All three linked by hebbian triangle -> clusters as one component
    store.boost_edges([(recs[0].id, recs[1].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[1].id, recs[2].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[0].id, recs[2].id)], edge_type="hebbian", delta=0.5)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    result = run_heavy_consolidation(
        store, session_id="s-cons", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )
    assert result["summaries_created"] >= 1

    # consolidated_from edges exist
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    cf = edges_df[edges_df["edge_type"] == "consolidated_from"]
    assert len(cf) >= 3  # summary -> each of 3 sources


def test_run_heavy_consolidation_mem01_preserves_sources(tmp_path):
    """ verbatim: source literal_surfaces untouched after consolidation."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literals = ["fact alpha", "fact beta", "fact gamma"]
    recs = [_record(text=t) for t in literals]
    for r in recs:
        store.insert(r)
    store.boost_edges(
        [(recs[0].id, recs[1].id), (recs[1].id, recs[2].id), (recs[0].id, recs[2].id)],
        edge_type="hebbian", delta=0.5,
    )

    run_heavy_consolidation(
        store, session_id="s", config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store), rate=RateLimitLedger(store),
        has_api_key=False,
    )

    # Re-read each source and assert literal_surface unchanged.
    for rec, expected in zip(recs, literals):
        reloaded = store.get(rec.id)
        assert reloaded is not None
        assert reloaded.literal_surface == expected


def test_run_heavy_consolidation_empty_store(tmp_path):
    """Empty store -> no summaries, no failures."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = run_heavy_consolidation(
        store, session_id="s", config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store), rate=RateLimitLedger(store),
        has_api_key=False,
    )
    assert result["summaries_created"] == 0


def test_run_heavy_consolidation_no_cluster_below_threshold(tmp_path):
    """A pair of connected records (<3) does NOT produce a cluster."""
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    r1, r2 = _record(text="a"), _record(text="b")
    store.insert(r1)
    store.insert(r2)
    store.boost_edges([(r1.id, r2.id)], edge_type="hebbian", delta=0.5)

    run_heavy_consolidation(
        store, session_id="s", config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store), rate=RateLimitLedger(store),
        has_api_key=False,
    )

    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    cf = edges_df[edges_df["edge_type"] == "consolidated_from"]
    assert len(cf) == 0
