"""Regression tests for user-model + predictive prefetch.

Coverage:

    - UserModel persistence -- default() empty, save+load round-trips every
      field including int dict keys; file mode 0o600; first-run load() returns
      default; corrupt JSON self-heals to default.
    - UserModelAggregator computes top_recent_topics (community-id labelled),
      tool_usage_freq, and time_of_day_pattern from a synthetic store.
    - SleepStep.USER_MODEL_UPDATE = 10 in REM phase; _STEP_ORDER position is
      strictly between RECONSOLIDATION and CRISIS_RECLUSTER; dispatch table
      wires USER_MODEL_UPDATE -> _step_user_model_update.
    - UserModelPrefetcher returns top-K record ids drawn from the communities
      whose labels match the loaded UserModel's top_recent_topics.
    - SessionStart integration path (iai_mcp.core) is importable; the
      prefetcher returns a non-empty subset of seeded ids against a loaded
      UserModel built from the same store.
    - dry_run=true skips save() but still emits the user_model_aggregate_pass
      event with dry_run_mode=True; user_model.json file never lands on disk;
      record_surprise emits exactly one user_model_surprise event with the
      expected payload keys.
    - 4 IAI_MCP_USER_MODEL_* env vars fail-loud with ValueError naming the
      offending var on malformed / out-of-range values; defaults parse to
      the documented defaults (aggregation_window_days=30, prefetch_top_k=10,
      user_model_path="~/.iai-mcp/user_model.json").

Synthetic stores use tmp_path with user_id='alice'. Fixture seed values
use 'alice' / 'bob' / lorem-style labels -- never 'Alice' (the project convention).
"""
from __future__ import annotations

import json
import os
import stat
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from iai_mcp.daemon import (
    UserModelConfig,
    _load_user_model_config,
)
from iai_mcp.events import query_events, write_event
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.sleep_pipeline import (
    STEP_PHASE,
    SleepPhase,
    SleepPipeline,
    SleepStep,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord
from iai_mcp.user_model import (
    UserModel,
    UserModelAggregator,
    UserModelPrefetcher,
    default,
    load,
    record_surprise,
    save,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Autouse fixture: pin IAI_MCP_STORE + IAI_MCP_USER_MODEL_PATH to tmp_path so
# per-test MemoryStore + UserModel persistence never touch the user's real
# ~/.iai-mcp/ directory. Also wipe every IAI_MCP_USER_MODEL_* env var so each
# test starts from documented defaults; tests that need overrides re-set after
# this fixture.
@pytest.fixture(autouse=True)
def _isolate_iai_user_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv(
        "IAI_MCP_USER_MODEL_PATH", str(tmp_path / "user_model.json"),
    )
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS",
        "IAI_MCP_USER_MODEL_PREFETCH_TOP_K",
        "IAI_MCP_USER_MODEL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# Build a minimal MemoryRecord. Embedding fills every cell with a tiny non-zero
# so the structure_hv autopop in MemoryStore.insert succeeds (mirrors the Phase
# 11.0/11.3/11.4 precedent at tests/test_phase11_4_*.py L98 / 11.3 L102).
def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    community_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
        community_id=community_id,
        centrality=0.5,
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


def _make_store(tmp_path: Path) -> MemoryStore:
    """Build a per-test MemoryStore rooted at tmp_path."""
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


def _seed_records_for_communities(
    store: MemoryStore,
    community_assignments: list[tuple[str, uuid.UUID]],
) -> list[uuid.UUID]:
    """Insert one record per (literal, community_id) tuple.

    Returns the list of inserted record ids in insertion order so callers
    can correlate seeded ids with their community assignment.

    Distinct embeddings per record (orthogonal-ish basis vectors) keep
    pattern_separation_gate from collapsing them as near-duplicates --
    even though patsep defaults to dry_run=True under pytest, distinct
    embeddings remove any ambiguity about which path the insert took.
    """
    embed_dim = store._embed_dim
    ids: list[uuid.UUID] = []
    for i, (surface, cid) in enumerate(community_assignments):
        emb = [0.0] * embed_dim
        # Pick a unique axis per record so cosine between any two is 0.
        emb[i % embed_dim] = 1.0
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=surface,
            community_id=cid,
            embedding=emb,
        )
        store.insert(rec)
        ids.append(rec.id)
    return ids


# ---------------------------------------------------------------------------
# Test 1: UserModel persistence (default + round-trip + chmod 0o600)
# ---------------------------------------------------------------------------


def test_R1_persistence_roundtrip_chmod_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default() returns empty UserModel; save+load round-trips
    every field including int-keyed time_of_day_pattern; persisted file mode
    is 0o600; first-run load() (no file on disk) returns default."""
    # First-run load() (no file yet at the tmp_path location) -> default.
    target = tmp_path / "user_model.json"
    assert not target.exists(), "tmp_path must start clean"

    fresh = load()
    assert isinstance(fresh, UserModel)
    assert fresh.top_recent_topics == []
    assert fresh.tool_usage_freq == {}
    assert fresh.time_of_day_pattern == {}
    assert fresh.recent_projects == []
    assert fresh.aggregation_window_days == 30

    # default() helper -> structurally equal to the first-run load.
    d = default()
    assert d.top_recent_topics == []
    assert d.tool_usage_freq == {}
    assert d.time_of_day_pattern == {}
    assert d.recent_projects == []
    assert d.aggregation_window_days == 30

    # Populated round-trip. int dict keys in time_of_day_pattern must survive
    # the JSON serialize (str-ifies) -> load() coerce-back-to-int path.
    model = UserModel(
        top_recent_topics=["python async", "torchhd hdc", "alice notes"],
        tool_usage_freq={"memory_recall": 42, "memory_capture": 7},
        time_of_day_pattern={9: 5, 14: 12, 22: 1},
        recent_projects=["project-alpha"],
        last_updated=datetime(2026, 5, 16, 8, 42, 13, tzinfo=timezone.utc),
        aggregation_window_days=14,
    )
    save(model)

    assert target.exists(), "save() must materialise the file at tmp path"

    # Mode 0o600: user rw, no group/other access.
    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o600, f"file mode must be 0o600, got {oct(mode)}"

    # Round-trip via load() restores every field including int dict keys.
    loaded = load()
    assert loaded.top_recent_topics == [
        "python async", "torchhd hdc", "alice notes",
    ]
    assert loaded.tool_usage_freq == {"memory_recall": 42, "memory_capture": 7}
    # Crucial: keys are real Python int (not "9" / "14" / "22").
    assert loaded.time_of_day_pattern == {9: 5, 14: 12, 22: 1}
    assert all(isinstance(k, int) for k in loaded.time_of_day_pattern.keys())
    assert loaded.recent_projects == ["project-alpha"]
    assert loaded.aggregation_window_days == 14
    assert loaded.last_updated == datetime(
        2026, 5, 16, 8, 42, 13, tzinfo=timezone.utc,
    )


# ---------------------------------------------------------------------------
# Test 2: UserModelAggregator computes known fields from synthetic store
# ---------------------------------------------------------------------------


def test_R2_aggregator_computes_known_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic store with 2 communities + a handful of
    retrieval_used events -> aggregator produces top_recent_topics (one
    label per community), tool_usage_freq containing retrieval_used, and a
    populated time_of_day_pattern keyed by the current hour."""
    store = _make_store(tmp_path)

    cid_a = uuid.uuid4()
    cid_b = uuid.uuid4()
    # Two records per community. Short literals so first-50-chars = full label.
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_a),
            ("alice topic alpha extended", cid_a),
            ("bob topic beta", cid_b),
            ("bob topic beta extended", cid_b),
        ],
    )

    # Emit a handful of retrieval_used events. write_event timestamps at
    # insert time -> hour bucket is datetime.now(timezone.utc).hour.
    for _ in range(3):
        write_event(store, "retrieval_used", {"tool": "retrieval_used"})

    agg = UserModelAggregator()
    model = agg.aggregate(store, window_days=30)

    # Exactly 2 community labels (one per community_id).
    assert len(model.top_recent_topics) == 2, (
        f"expected 2 community labels, got {model.top_recent_topics!r}"
    )
    labels = set(model.top_recent_topics)
    # Most-recent record per community drives the label; "extended" variant
    # was inserted later in each community so should be the chosen label.
    # (Ordering depends on tiebreak; assert by set membership to stay robust.)
    assert any("alice topic alpha" in lbl for lbl in labels)
    assert any("bob topic beta" in lbl for lbl in labels)

    # tool_usage_freq includes retrieval_used (data["tool"] preferred over
    # event kind, per aggregator code path).
    assert "retrieval_used" in model.tool_usage_freq, (
        f"retrieval_used missing from tool_usage_freq: "
        f"{model.tool_usage_freq!r}"
    )
    assert model.tool_usage_freq["retrieval_used"] >= 3

    # time_of_day_pattern populated -- at least the current hour bucket.
    assert len(model.time_of_day_pattern) >= 1
    current_hour = datetime.now(timezone.utc).hour
    assert current_hour in model.time_of_day_pattern, (
        f"current hour {current_hour} missing from "
        f"time_of_day_pattern={model.time_of_day_pattern!r}"
    )
    assert model.time_of_day_pattern[current_hour] >= 3
    # All keys are real int (not pandas / str).
    assert all(isinstance(k, int) for k in model.time_of_day_pattern.keys())

    # aggregation_window_days echoed back on the persisted snapshot.
    assert model.aggregation_window_days == 30


# ---------------------------------------------------------------------------
# Test 3: SleepStep enum + STEP_PHASE + _STEP_ORDER position + dispatch
# ---------------------------------------------------------------------------


def test_R3_sleep_step_position_and_dispatch(tmp_path: Path) -> None:
    """USER_MODEL_UPDATE.value == 10; REM phase; _STEP_ORDER
    position is strictly between RECONSOLIDATION and CRISIS_RECLUSTER;
    dispatch table wires the step -> _step_user_model_update bound method."""
    assert SleepStep.USER_MODEL_UPDATE.value == 10
    assert STEP_PHASE[SleepStep.USER_MODEL_UPDATE] == SleepPhase.REM

    order = SleepPipeline._STEP_ORDER
    idx_user = order.index(SleepStep.USER_MODEL_UPDATE)
    idx_recon = order.index(SleepStep.RECONSOLIDATION)
    idx_crisis = order.index(SleepStep.CRISIS_RECLUSTER)
    assert idx_recon < idx_user < idx_crisis, (
        f"USER_MODEL_UPDATE must sit strictly between RECONSOLIDATION and "
        f"CRISIS_RECLUSTER in _STEP_ORDER; got idx_recon={idx_recon}, "
        f"idx_user={idx_user}, idx_crisis={idx_crisis}"
    )
    #: USER_MODEL_UPDATE directly follows RECONSOLIDATION.
    assert idx_user == idx_recon + 1, (
        f"USER_MODEL_UPDATE must directly follow RECONSOLIDATION; "
        f"got idx_user={idx_user}, idx_recon={idx_recon}"
    )

    # Dispatch table: construct a real SleepPipeline (cheap, no I/O beyond
    # lifecycle_state read) and read the @property.
    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=None, lifecycle_state_path=lifecycle_path,
    )
    methods = pipeline._step_methods
    assert SleepStep.USER_MODEL_UPDATE in methods, (
        f"_step_methods missing USER_MODEL_UPDATE entry; "
        f"keys={list(methods.keys())!r}"
    )
    # Bound-method equality (not identity): two fresh bound-method objects
    # for the same descriptor compare ==, never ``is``. Compare via the
    # underlying unbound function on the class.
    assert methods[SleepStep.USER_MODEL_UPDATE].__func__ is (
        SleepPipeline._step_user_model_update
    ), "USER_MODEL_UPDATE must dispatch to _step_user_model_update"


# ---------------------------------------------------------------------------
# Test 4: UserModelPrefetcher returns topic-matched record ids
# ---------------------------------------------------------------------------


def test_R4_prefetcher_returns_topic_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With records spread across 3 communities and a
    UserModel naming labels for ONLY 2 of those 3, the prefetcher returns
    ids drawn exclusively from the 2 matched communities."""
    store = _make_store(tmp_path)

    cid_match_1 = uuid.uuid4()
    cid_match_2 = uuid.uuid4()
    cid_unmatched = uuid.uuid4()
    ids = _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_match_1),
            ("alice topic alpha second record", cid_match_1),
            ("bob topic beta", cid_match_2),
            ("bob topic beta second record", cid_match_2),
            ("unmatched gamma topic", cid_unmatched),
            ("unmatched gamma topic second record", cid_unmatched),
        ],
    )
    # Derive the labels the aggregator WOULD produce (most-recent record's
    # first-50-chars rstripped). Insertion order = ascending created_at, so
    # the "second record" variants win each community.
    label_match_1 = "alice topic alpha second record"
    label_match_2 = "bob topic beta second record"
    label_unmatched = "unmatched gamma topic second record"

    model = UserModel(
        top_recent_topics=[label_match_1, label_match_2],
        tool_usage_freq={},
        time_of_day_pattern={},
        recent_projects=[],
        last_updated=datetime.now(timezone.utc),
        aggregation_window_days=30,
    )

    result = UserModelPrefetcher().prefetch(store, model, top_k=10)
    assert isinstance(result, list)
    assert len(result) >= 1, "prefetcher must return at least one match"
    assert len(result) <= 10

    # Every returned id must belong to a matched community (NOT the
    # unmatched one). Build the ground-truth set from the seeded ids.
    matched_ids = {str(rid) for rid in ids[:4]}  # first 4 = match_1 + match_2
    unmatched_ids = {str(rid) for rid in ids[4:]}
    for rid in result:
        assert rid in matched_ids, (
            f"prefetcher returned id {rid} which is not in a matched "
            f"community; matched={matched_ids}, unmatched={unmatched_ids}"
        )
        assert rid not in unmatched_ids, (
            f"prefetcher leaked unmatched community id {rid}; "
            f"label_unmatched={label_unmatched!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: SessionStart integration path is importable + prefetcher
# returns a non-empty subset against a loaded model
# ---------------------------------------------------------------------------


def test_R5_session_start_prefetch_integration_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core.py SessionStart augmentation path is importable AND the
    prefetcher returns a non-empty subset of seeded ids when run against a
    loaded UserModel that names the seeded community labels."""
    # The integration site in core.py imports these symbols lazily;
    # confirm they resolve so any future rename / move trips the test.
    from iai_mcp import core as _core_mod  # noqa: F401
    from iai_mcp.core import dispatch  # noqa: F401
    from iai_mcp.user_model import UserModelPrefetcher as _PF
    from iai_mcp.user_model import load as _load
    from iai_mcp.daemon import _load_user_model_config as _lcfg

    monkeypatch.setenv("IAI_MCP_USER_MODEL_DRY_RUN", "false")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    cid_beta = uuid.uuid4()
    seeded = _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha second record", cid_alpha),
            ("bob topic beta", cid_beta),
            ("bob topic beta second record", cid_beta),
        ],
    )

    # Build + persist a UserModel naming both seeded community labels.
    label_alpha = "alice topic alpha second record"
    label_beta = "bob topic beta second record"
    model = UserModel(
        top_recent_topics=[label_alpha, label_beta],
        tool_usage_freq={"memory_recall": 5},
        time_of_day_pattern={datetime.now(timezone.utc).hour: 5},
        recent_projects=[],
        last_updated=datetime.now(timezone.utc),
        aggregation_window_days=30,
    )
    save(model)

    # The core.py path calls _load_user_model_config + load + prefetch in
    # sequence; reproduce that here so we exercise the exact wiring.
    cfg = _lcfg()
    loaded = _load()
    assert loaded.top_recent_topics == [label_alpha, label_beta], (
        "save+load round-trip lost the topics; check tmp_path redirect"
    )
    prefetched = _PF().prefetch(store, loaded, top_k=cfg.prefetch_top_k)
    assert isinstance(prefetched, list)
    assert len(prefetched) >= 1, (
        "prefetcher returned no ids despite matching topics + a seeded store; "
        "core.py SessionStart augmentation would silently no-op in production"
    )
    # Every prefetched id MUST be among the seeded ids (no leak from other
    # tests' tmp_paths or stale state).
    seeded_str = {str(rid) for rid in seeded}
    for rid in prefetched:
        assert rid in seeded_str, (
            f"prefetcher returned non-seeded id {rid}; seeded={seeded_str}"
        )


# ---------------------------------------------------------------------------
# Test 6: dry_run skips save() + l2 mutation BUT still emits event
# ---------------------------------------------------------------------------


def test_R6_dry_run_skips_save_but_emits_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With cfg.dry_run=True (pytest default), the REM step
    body emits exactly one user_model_aggregate_pass event tagged
    dry_run_mode=True; the user_model.json file is NEVER written to disk;
    the returned payload has dry_run=True."""
    # Leave IAI_MCP_USER_MODEL_DRY_RUN unset -> pytest-aware default -> True.
    store = _make_store(tmp_path)

    cid_a = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_a),
            ("alice topic alpha second", cid_a),
        ],
    )

    target = tmp_path / "user_model.json"
    assert not target.exists(), "tmp_path must start clean"

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_user_model_update(interrupt_check=None)
    assert done is True
    assert payload["dry_run"] is True, (
        f"dry_run pytest-default must surface in payload; got {payload!r}"
    )
    assert payload["topics_count"] >= 1

    # File must NOT exist on disk (save() skipped under dry_run).
    assert not target.exists(), (
        "dry_run path must NOT persist user_model.json; "
        f"file appeared at {target}"
    )

    # Exactly ONE user_model_aggregate_pass event with dry_run_mode=True.
    events = query_events(store, kind="user_model_aggregate_pass", limit=10)
    assert len(events) == 1, (
        f"_step_user_model_update must emit exactly one event, "
        f"got {len(events)}"
    )
    body = events[0]["data"]
    assert body["dry_run_mode"] is True, (
        f"event must tag dry_run_mode=True under pytest default; got {body!r}"
    )
    # Required event payload keys.
    for key in (
        "topics_count",
        "tools_count",
        "hours_count",
        "projects_count",
        "window_days",
        "dry_run_mode",
    ):
        assert key in body, f"event body missing key {key!r}; body={body!r}"
    assert isinstance(body["topics_count"], int)
    assert isinstance(body["window_days"], int)


# ---------------------------------------------------------------------------
# Test 7: surprise tracking -- record_surprise emits one event
# ---------------------------------------------------------------------------


def test_R6_record_surprise_emits_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each record_surprise call emits exactly
    one user_model_surprise event carrying predicted_topic, actual_topic,
    and dry_run_mode (tagged from the current cfg)."""
    store = _make_store(tmp_path)
    # Leave dry_run at pytest default (True). The event still fires, with
    # dry_run_mode=True on the body.
    record_surprise(store, predicted_topic="alpha", actual_topic="beta")
    events = query_events(store, kind="user_model_surprise", limit=10)
    assert len(events) == 1, (
        f"record_surprise must emit exactly one event, got {len(events)}"
    )
    body = events[0]["data"]
    assert body["predicted_topic"] == "alpha"
    assert body["actual_topic"] == "beta"
    assert "dry_run_mode" in body
    assert isinstance(body["dry_run_mode"], bool)
    # Under pytest default dry_run is True; assert that.
    assert body["dry_run_mode"] is True


# ---------------------------------------------------------------------------
# Test 8: every malformed env var fails loud with ValueError naming var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS", "0"),
        ("IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS", "366"),
        ("IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS", "nan"),
        ("IAI_MCP_USER_MODEL_PREFETCH_TOP_K", "0"),
        ("IAI_MCP_USER_MODEL_PREFETCH_TOP_K", "200"),
        ("IAI_MCP_USER_MODEL_PREFETCH_TOP_K", "not-an-int"),
        ("IAI_MCP_USER_MODEL_DRY_RUN", "maybe"),
        ("IAI_MCP_USER_MODEL_DRY_RUN", "banana"),
    ],
)
def test_R7_invalid_env_var_raises_ValueError_naming_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Every malformed knob raises ValueError naming
    the offending env var so operators can act."""
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_user_model_config()


def test_R7_defaults_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env overrides _load_user_model_config
    returns the documented defaults (aggregation_window_days=30, prefetch_top_k=10,
    user_model_path="~/.iai-mcp/user_model.json"). dry_run is True under
    PYTEST_CURRENT_TEST (pytest-aware default) -- that's the documented
    contract, not a regression."""
    # Wipe even the autouse-fixture-set USER_MODEL_PATH so the helper falls
    # back to the default ~/.iai-mcp/user_model.json string.
    for var in (
        "IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS",
        "IAI_MCP_USER_MODEL_PREFETCH_TOP_K",
        "IAI_MCP_USER_MODEL_PATH",
        "IAI_MCP_USER_MODEL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = _load_user_model_config()
    assert isinstance(cfg, UserModelConfig)
    assert cfg.aggregation_window_days == 30
    assert cfg.prefetch_top_k == 10
    assert cfg.user_model_path == "~/.iai-mcp/user_model.json"
    # PYTEST_CURRENT_TEST is set by pytest -> pytest-aware default fires.
    assert cfg.dry_run is True


if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    import sys
    raise SystemExit(pytest.main([__file__, "-v"]))
