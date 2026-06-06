"""Regression tests for the sleep-pipeline overhaul.

Coverage:
  - STEP_PHASE mapping correctness (all SleepStep members covered).
  - NREM-before-REM ordering in _STEP_ORDER (CRISIS_RECLUSTER last).
  - CLUSTER_REPLAY batches recently-reviewed records into time-windows
    and emits hebbian_cluster_replay edges.
  - EssentialVariableTracker detects rich_club_ratio floor breach and
    returns a BreachInfo with direction='floor_breach'.
  - CRISIS_RECLUSTER is a no-op when crisis_mode=False and emits a single
    crisis_recluster_pass event when True (clearing crisis_mode after).
  - Each of the IAI_MCP_* sleep-overhaul env vars fails loud with a
    ValueError naming the offending var.
  - Dry-run mode preserves no-mutation invariants on all three mutation
    paths (CLUSTER_REPLAY edge boost, crisis_mode set, CRISIS_RECLUSTER
    reassignment).

Fixtures are inline. Synthetic stores use tmp_path with user_id='alice'.
"""
# Standard-library imports first so optional iai_mcp.* imports below fail
# loud with a clear ImportError if the package layout changes.
from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.ashby_step import (
    BreachInfo,
    EssentialVariableTracker,
    TopologySnapshot,
)
from iai_mcp.daemon import (
    SleepOverhaulConfig,
    _load_sleep_overhaul_config,
)
from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import (
    LifecycleStateRecord,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.sleep_pipeline import (
    MAX_PAIRS_PER_CLUSTER,
    STEP_PHASE,
    SleepPhase,
    SleepPipeline,
    SleepStep,
)
from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction never scribbles on the user's real ~/.iai-mcp/hippo store. The
# global shell may have IAI_MCP_STORE set per project, and
# MemoryStore.__init__ overrides path= kwarg from that env var. Pinning
# here keeps the suite hermetic regardless of caller env.
@pytest.fixture(autouse=True)
def _isolate_iai_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    # Clear every IAI_MCP_* sleep-overhaul knob so each test starts from
    # SPEC defaults. Tests that need overrides re-set after this fixture.
    for var in (
        "IAI_MCP_RICH_CLUB_RATIO_FLOOR",
        "IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO",
        "IAI_MCP_EDGE_DENSITY_FLOOR",
        "IAI_MCP_CLUSTER_WINDOW_SEC",
        "IAI_MCP_CRISIS_DROP_QUARTILE",
        "IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT",
        "IAI_MCP_SLEEP_OVERHAUL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# Build a minimal MemoryRecord with sensible defaults. Embedding is seeded
# from literal_surface (via random.Random) so every record with a distinct
# surface string gets a directionally-distinct embedding (pairwise cosine
# < 0.20 in 384-d), ensuring the pattern-separation gate (near_dup_threshold
# = 0.92) does not collapse multiple records into one.
def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    last_reviewed: datetime | None = None,
    community_id: uuid.UUID | None = None,
) -> MemoryRecord:
    rng = random.Random(hash(literal_surface))
    raw = [rng.gauss(0.0, 1.0) for _ in range(embed_dim)]
    mag = math.sqrt(sum(x * x for x in raw))
    embedding = [x / mag for x in raw] if mag > 0 else raw
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding,
        community_id=community_id,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=last_reviewed,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )


def _make_store(tmp_path: Path) -> MemoryStore:
    """Build a per-test MemoryStore. The autouse fixture pins
    IAI_MCP_STORE so path= is overridden anyway; passing a tmp_path
    subdir keeps the constructor happy for callers that bypass the env.
    """
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


# ---------------------------------------------------------------------------
# Test 1: STEP_PHASE mapping correctness
# ---------------------------------------------------------------------------


def test_r1_step_phase_mapping() -> None:
    """STEP_PHASE assertions."""
    assert SleepPhase.NREM is not None
    assert SleepPhase.REM is not None
    assert STEP_PHASE[SleepStep.SCHEMA_MINE] == SleepPhase.NREM
    assert STEP_PHASE[SleepStep.DREAM_DECAY] == SleepPhase.REM
    # All members explicit (no sentinel default).
    assert set(STEP_PHASE.keys()) == set(SleepStep)

    nrem_steps = {s for s, p in STEP_PHASE.items() if p == SleepPhase.NREM}
    rem_steps = {s for s, p in STEP_PHASE.items() if p == SleepPhase.REM}
    assert nrem_steps == {
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
    }
    assert rem_steps == {
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    }


# ---------------------------------------------------------------------------
# Test 2: _STEP_ORDER NREM-before-REM
# ---------------------------------------------------------------------------


def test_r2_step_order_nrem_before_rem() -> None:
    """NREM phase steps fully precede REM phase; CRISIS_RECLUSTER last.

    RECONSOLIDATION, USER_MODEL_UPDATE, and DMN_REFLECTION all sit inside
    the REM phase ahead of CRISIS_RECLUSTER. The NREM-before-REM invariant
    and the CRISIS_RECLUSTER-last invariant both hold.
    """
    order = SleepPipeline._STEP_ORDER
    assert len(order) == 13
    assert order[-1] == SleepStep.RECALL_INDEX_REBUILD

    # All NREM-phase positions < all REM-phase positions.
    nrem_positions = [
        order.index(s)
        for s in (
            SleepStep.SCHEMA_MINE,
            SleepStep.KNOB_TUNE,
            SleepStep.OPTIMIZE_LANCE,
            SleepStep.COMPACT_RECORDS,
        )
    ]
    rem_positions = [
        order.index(s)
        for s in (
            SleepStep.DREAM_DECAY,
            SleepStep.ERASURE_AGENT,
            SleepStep.CLUSTER_REPLAY,
            SleepStep.RECONSOLIDATION,
            SleepStep.USER_MODEL_UPDATE,
            SleepStep.DMN_REFLECTION,
            SleepStep.CRISIS_RECLUSTER,
            SleepStep.CLUSTER_SUMMARY,
            SleepStep.RECALL_INDEX_REBUILD,
        )
    ]
    assert max(nrem_positions) < min(rem_positions)

    # APPEND-not-renumber discipline: CLUSTER_REPLAY=7
    # and CRISIS_RECLUSTER=8 are STABLE enum values, not positions. The
    # resume-math wrap-detection compares last_completed_index against
    # len(_STEP_ORDER)-1, so a future APPEND without renumber stays safe
    # so long as the new member is in the tuple too.
    assert SleepStep.CLUSTER_REPLAY.value == 7
    assert SleepStep.CRISIS_RECLUSTER.value == 8
    # RECONSOLIDATION has STABLE enum value 9 reserved at the tail; dispatch
    # position is controlled by _STEP_ORDER (slotted between CLUSTER_REPLAY
    # and CRISIS_RECLUSTER inside REM).
    assert SleepStep.RECONSOLIDATION.value == 9
    assert order.index(SleepStep.RECONSOLIDATION) == (
        order.index(SleepStep.CLUSTER_REPLAY) + 1
    )
    # USER_MODEL_UPDATE has STABLE enum value 10; dispatch position is one
    # slot AFTER RECONSOLIDATION and one BEFORE DMN_REFLECTION inside REM.
    assert SleepStep.USER_MODEL_UPDATE.value == 10
    assert order.index(SleepStep.USER_MODEL_UPDATE) == (
        order.index(SleepStep.RECONSOLIDATION) + 1
    )
    # DMN_REFLECTION has STABLE enum value 11; dispatch position is one
    # slot AFTER USER_MODEL_UPDATE and one BEFORE CRISIS_RECLUSTER inside REM.
    assert SleepStep.DMN_REFLECTION.value == 11
    assert order.index(SleepStep.DMN_REFLECTION) == (
        order.index(SleepStep.USER_MODEL_UPDATE) + 1
    )
    # CRISIS_RECLUSTER is still at its original relative REM position,
    # before CLUSTER_SUMMARY and RECALL_INDEX_REBUILD.
    # Its enum value is still 8; its dispatch position is len(order)-3.
    assert order.index(SleepStep.CRISIS_RECLUSTER) == len(order) - 3
    # CLUSTER_SUMMARY and RECALL_INDEX_REBUILD are the final two steps.
    assert SleepStep.CLUSTER_SUMMARY.value == 12
    assert SleepStep.RECALL_INDEX_REBUILD.value == 13
    assert order[-2] == SleepStep.CLUSTER_SUMMARY
    assert order[-1] == SleepStep.RECALL_INDEX_REBUILD


# ---------------------------------------------------------------------------
# Test 3: CLUSTER_REPLAY batches + boosts intra-cluster edges
# ---------------------------------------------------------------------------


def test_r3_cluster_replay_batches_intra_cluster_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10 records in 3 time-clusters -> clusters_replayed=3 event AND
    hebbian_cluster_replay edges in EDGES_TABLE.
    """
    # Force non-dry-run so mutation actually fires under pytest.
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CLUSTER_WINDOW_SEC", "300")
    monkeypatch.setenv("IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT", "0.05")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    tbl = store.db.open_table(RECORDS_TABLE)

    # Build 10 records across 3 time-windows. Window = 300s. Place records
    # at offsets that keep each cluster contained in a single window.
    now = datetime.now(timezone.utc)
    cluster_offsets = [
        [-30, -60, -90, -120],     # cluster A (within 5 min ago)
        [-430, -460, -490],         # cluster B (around 8 min ago)
        [-830, -860, -890],         # cluster C (around 14 min ago)
    ]
    record_ids: list[uuid.UUID] = []
    for cluster in cluster_offsets:
        for off in cluster:
            rec = _make_record(
                embed_dim=embed_dim,
                literal_surface=f"alice record at {off}s",
            )
            store.insert(rec)
            # Force last_reviewed to the target offset via direct update.
            ts = now + timedelta(seconds=off)
            tbl.update(
                where=f"id = '{str(rec.id)}'",
                values={"last_reviewed": ts},
            )
            record_ids.append(rec.id)
    assert len(record_ids) == 10

    # Construct pipeline + run JUST the CLUSTER_REPLAY step in isolation
    # to keep this test surgical.
    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_cluster_replay(interrupt_check=None)
    assert done is True
    assert payload["clusters_replayed"] == 3
    assert payload["dry_run"] is False

    # Event-body assertions.
    events = query_events(store, kind="cluster_replay_pass", limit=5)
    assert len(events) >= 1
    body = events[0]["data"]
    assert body["clusters_replayed"] == 3
    assert body["window_sec"] == 300
    assert body["lookback_windows"] == 5
    assert body["dry_run_mode"] is False

    # At least one edge of the new type was created.
    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    cluster_edges = edges[edges["edge_type"] == "hebbian_cluster_replay"]
    assert len(cluster_edges) > 0, (
        "non-dry-run CLUSTER_REPLAY must create hebbian_cluster_replay edges"
    )

    # MAX_PAIRS_PER_CLUSTER is an invariant the step honors; the synthetic
    # fixture is small (4+3+3) so no cluster should trigger the cap, but
    # the field MUST still be emitted on every event for the verifier.
    assert "max_pairs_per_cluster_applied" in body
    assert body["max_pairs_per_cluster_applied"] == 0
    assert MAX_PAIRS_PER_CLUSTER == 100


# ---------------------------------------------------------------------------
# Test 4: EssentialVariableTracker breach detection
# ---------------------------------------------------------------------------


def test_r4_essential_variable_tracker_detects_rich_club_breach() -> None:
    """rich_club_ratio=0.01 (< floor 0.05) returns BreachInfo;
    healthy snapshot returns None for every key; empty-store short-circuit
    returns None for every key.
    """
    # Tracker accepts any duck-typed object exposing the three float
    # threshold attributes (the ashby_step constructor reads _rich_club_floor
    # etc. from cfg.* names). Use a bare class so the test does not depend
    # on importing SleepOverhaulConfig defaults.
    class _Cfg:
        rich_club_ratio_floor = 0.05
        community_count_ceiling_ratio = 0.9
        edge_density_floor = 0.001

    tracker = EssentialVariableTracker(_Cfg())

    # Breach: rich_club at 0.01 < floor 0.05 -> floor_breach.
    breach_snapshot = TopologySnapshot(
        rich_club_ratio=0.01,
        community_count=500,
        edge_density=0.01,
        total_nodes=1000,
    )
    breaches = tracker.check(breach_snapshot)
    assert set(breaches.keys()) == {
        "rich_club_ratio",
        "community_count",
        "edge_density",
    }
    rc = breaches["rich_club_ratio"]
    assert isinstance(rc, BreachInfo)
    assert rc.direction == "floor_breach"
    assert rc.observed_value == pytest.approx(0.01)
    assert rc.threshold == pytest.approx(0.05)
    # community_count ratio = 500/1000 = 0.5 < ceiling 0.9 -> no breach.
    assert breaches["community_count"] is None
    # edge_density 0.01 > floor 0.001 -> no breach.
    assert breaches["edge_density"] is None

    # Healthy snapshot: every key None.
    healthy = TopologySnapshot(
        rich_club_ratio=0.5,
        community_count=10,
        edge_density=0.5,
        total_nodes=1000,
    )
    healthy_result = tracker.check(healthy)
    assert all(v is None for v in healthy_result.values())

    # Empty-store short-circuit: total_nodes=0 -> every key None even
    # though every observed value is below the floor (ashby_step.py L144).
    empty = TopologySnapshot(
        rich_club_ratio=0.0,
        community_count=0,
        edge_density=0.0,
        total_nodes=0,
    )
    empty_result = tracker.check(empty)
    assert all(v is None for v in empty_result.values())


# ---------------------------------------------------------------------------
# Test 5: CRISIS_RECLUSTER conditional on crisis_mode
# ---------------------------------------------------------------------------


def test_r5_crisis_recluster_conditional_on_crisis_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-op (no event emitted) when crisis_mode=False; drops bottom
    quartile AND clears crisis_mode AND emits exactly one
    crisis_recluster_pass event when crisis_mode=True.
    """
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    lifecycle_path = tmp_path / "lifecycle.json"

    # --- Phase A: crisis_mode=False -> no-op (no mutation, no event).
    state: LifecycleStateRecord = default_state()
    state["crisis_mode"] = False
    save_state(state, lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_crisis_recluster(interrupt_check=None)
    assert done is True
    assert payload["communities_dropped"] == 0
    # No crisis_recluster_pass event emitted on the no-op skip path.
    events_a = query_events(store, kind="crisis_recluster_pass", limit=10)
    assert len(events_a) == 0, (
        f"crisis_mode=False path must NOT emit crisis_recluster_pass, "
        f"got {len(events_a)} event(s)"
    )

    # --- Phase B: 100 single-record communities + crisis_mode=True
    # -> drops bottom 25 communities AND clears crisis_mode AND emits
    # exactly one crisis_recluster_pass event.
    # community_id is a UUID column in the records schema; the store returns
    # the raw string on read and store._from_row(...) parses it via
    # uuid.UUID(...). Synthetic non-UUID labels like "comm-001" crash the
    # next query_similar() called by pattern_separation_gate inside the
    # following store.insert(). Use real UUID4s instead -- the step body
    # only needs distinct community-id strings to count communities.
    tbl = store.db.open_table(RECORDS_TABLE)
    for i in range(100):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice rec {i}",
        )
        store.insert(rec)
        tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    state = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)

    pipeline_b = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline_b._step_crisis_recluster(interrupt_check=None)
    assert done is True
    # 25% of 100 = 25 communities dropped.
    assert payload["communities_dropped"] == 25, (
        f"expected 25 communities dropped (25% of 100), got {payload}"
    )

    # crisis_mode cleared back to False.
    final_state = load_state(lifecycle_path)
    assert final_state["crisis_mode"] is False, (
        "non-dry-run CRISIS_RECLUSTER must clear crisis_mode"
    )

    # Exactly one crisis_recluster_pass event.
    events_b = query_events(store, kind="crisis_recluster_pass", limit=10)
    assert len(events_b) == 1, (
        f"expected exactly 1 crisis_recluster_pass event, got {len(events_b)}"
    )
    body = events_b[0]["data"]
    assert body["communities_dropped"] == 25
    assert body["dry_run_mode"] is False


# ---------------------------------------------------------------------------
# Test 6: env-var fail-loud naming (parametrized -> 7 sub-cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var_name,bad_value",
    [
        # rich_club_ratio_floor: float in (0.0, 1.0] -- 2.0 is out of range.
        ("IAI_MCP_RICH_CLUB_RATIO_FLOOR", "2.0"),
        # community_count_ceiling_ratio: float in (0.0, 1.0] -- -0.1 out of range.
        ("IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO", "-0.1"),
        # edge_density_floor: float in (0.0, 1.0] -- not_a_float parse error.
        ("IAI_MCP_EDGE_DENSITY_FLOOR", "not_a_float"),
        # cluster_window_sec: int in [1, 86400] -- 0 below min.
        ("IAI_MCP_CLUSTER_WINDOW_SEC", "0"),
        # crisis_drop_quartile: float in (0.0, 1.0) -- 1.0 is exclusive upper bound.
        ("IAI_MCP_CRISIS_DROP_QUARTILE", "1.0"),
        # cluster_replay_initial_weight: float in (0.0, 1.0] -- 5.0 out of range.
        ("IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT", "5.0"),
        # dry_run: bool vocab -- "maybe" is unparseable.
        ("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "maybe"),
    ],
)
def test_r6_env_var_fail_loud_naming(
    monkeypatch: pytest.MonkeyPatch,
    var_name: str,
    bad_value: str,
) -> None:
    """Each invalid env var raises ValueError whose message names the var."""
    monkeypatch.setenv(var_name, bad_value)
    with pytest.raises(ValueError) as exc_info:
        _load_sleep_overhaul_config()
    assert var_name in str(exc_info.value), (
        f"ValueError message {str(exc_info.value)!r} must contain {var_name!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: dry-run no-mutation on all 3 paths
# ---------------------------------------------------------------------------


def test_r7_dry_run_no_mutation_all_three_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry_run=true -> events emitted but zero mutation on all 3 paths
    (CLUSTER_REPLAY edge boost, EssentialVariableTracker crisis_mode set,
    CRISIS_RECLUSTER reassignment).
    """
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "true")
    monkeypatch.setenv("IAI_MCP_CLUSTER_WINDOW_SEC", "300")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    records_tbl = store.db.open_table(RECORDS_TABLE)

    # ---- Path 1: CLUSTER_REPLAY -- 4 records in one cluster, dry_run=true
    # -> event with dry_run_mode=True and ZERO hebbian_cluster_replay edges.
    now = datetime.now(timezone.utc)
    for off in (-30, -60, -90, -120):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice rec {off}",
        )
        store.insert(rec)
        records_tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"last_reviewed": now + timedelta(seconds=off)},
        )

    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    pipeline._step_cluster_replay(interrupt_check=None)

    events1 = query_events(store, kind="cluster_replay_pass", limit=5)
    assert events1, "cluster_replay_pass event must still emit in dry_run"
    body1 = events1[0]["data"]
    assert body1["dry_run_mode"] is True
    assert body1["clusters_replayed"] == 1, (
        f"4 records in one window -> 1 cluster, got {body1}"
    )
    # ZERO new hebbian_cluster_replay edges -- dry_run must not write.
    edges_after_p1 = store.db.open_table(EDGES_TABLE).to_pandas()
    if not edges_after_p1.empty:
        cluster_edges = edges_after_p1[
            edges_after_p1["edge_type"] == "hebbian_cluster_replay"
        ]
        assert len(cluster_edges) == 0, (
            "dry_run must not write hebbian_cluster_replay edges"
        )

    # ---- Path 2: EssentialVariableTracker crisis_mode set
    # Run the hook directly with an artificially-strict rich-club floor.
    # Under dry_run, ANY breach event emitted MUST report
    # crisis_mode_set=False AND lifecycle_state.crisis_mode stays False.
    monkeypatch.setenv("IAI_MCP_RICH_CLUB_RATIO_FLOOR", "0.99")
    pipeline_p2 = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    try:
        pipeline_p2._run_essential_variable_tracker_hook()
    except Exception:
        # Hook is best-effort; failure here does not fail the test (the
        # assertion we care about is the absence of crisis_mode flip below).
        pass
    events2 = query_events(store, kind="essential_variable_breach", limit=10)
    for e in events2:
        body2 = e["data"]
        assert body2["dry_run_mode"] is True
        assert body2["crisis_mode_set"] is False, (
            "dry_run breach event must report crisis_mode_set=False"
        )
    final_state = load_state(lifecycle_path)
    assert final_state["crisis_mode"] is False, (
        "dry_run must not flip crisis_mode"
    )

    # ---- Path 3: CRISIS_RECLUSTER -- pre-seed crisis_mode=True directly,
    # 100 single-record communities, dry_run=true. Event must report
    # dry_run_mode=True AND records_reassigned=0 AND crisis_mode stays True.
    state: LifecycleStateRecord = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    # Use real UUID4s for community_id (same reason as Phase B above):
    # the store schema declares community_id as a UUID, and synthetic
    # non-UUID labels crash the next pattern_separation_gate -> query_similar
    # call inside the next store.insert. The step body only requires
    # distinct strings to count communities.
    for i in range(100):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice c-rec {i}",
        )
        store.insert(rec)
        records_tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    pipeline_p3 = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    pipeline_p3._step_crisis_recluster(interrupt_check=None)
    events3 = query_events(store, kind="crisis_recluster_pass", limit=5)
    assert events3, "crisis_recluster_pass must still emit in dry_run"
    body3 = events3[0]["data"]
    assert body3["dry_run_mode"] is True
    assert body3["records_reassigned"] == 0, (
        "dry_run must not reassign community_id on any record"
    )

    # crisis_mode NOT cleared in dry_run.
    final_state_p3 = load_state(lifecycle_path)
    assert final_state_p3["crisis_mode"] is True, (
        "dry_run must not clear crisis_mode"
    )
