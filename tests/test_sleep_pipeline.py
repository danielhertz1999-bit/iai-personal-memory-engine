"""SleepPipeline tests.

Covers:
- 6-step ordering (SCHEMA_MINE ->... -> ERASURE_AGENT ->... -> COMPACT_RECORDS).
- progress cleared on full success.
- resume-from-step-N: with last_completed_index=1 (KNOB_TUNE done),
  only DREAM_DECAY / ERASURE_AGENT / OPTIMIZE_LANCE / COMPACT_RECORDS run.
- failure persists progress (last_completed_index=idx(step)-1, attempt+1,
  last_error).
- 3-strike threshold triggers 24h auto-quarantine.
- quarantined run() short-circuits with quarantine_triggered=True.
- quarantine auto-recovery once until_ts is in the past.
- reset_quarantine() clears immediately.
- force_run() ignores quarantine.
- bounded deferral persists chunk_idx in deferral marker; run returns interrupted=True.
- atomic-step crash leaves progress consistent (no partial state corruption).
-: legacy `last_completed_step` field is migrated to
  `last_completed_index` on read via enum-lookup (NOT subtraction),
  so a pre- crash-window file resumes at the correct step
  without skipping ERASURE_AGENT on a future cycle.

All tests run with a stub `store` (None) and step methods replaced via
monkeypatch — no real store I/O, no real embedder load. Combined
wall-clock target < 1 sec.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import (
    LifecycleState,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.sleep_pipeline import (
    SleepPipeline,
    SleepPipelineResult,
    SleepStep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    """Isolated lifecycle_state.json path inside tmp_path."""
    return tmp_path / "lifecycle_state.json"


@pytest.fixture
def event_log_dir(tmp_path: Path) -> Path:
    """Isolated lifecycle event log directory."""
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def event_log(event_log_dir: Path) -> LifecycleEventLog:
    """LifecycleEventLog rooted at the test event_log_dir."""
    return LifecycleEventLog(log_dir=event_log_dir)


@pytest.fixture
def pipeline(state_path: Path, event_log: LifecycleEventLog) -> SleepPipeline:
    """Standard SleepPipeline instance with stub store and isolated paths."""
    return SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )


def _patch_steps_to_noop(
    pipeline: SleepPipeline, monkeypatch: pytest.MonkeyPatch,
    *,
    record: list[SleepStep] | None = None,
    payloads: dict[SleepStep, dict] | None = None,
) -> list[SleepStep]:
    """Replace all 13 _step_* methods with no-ops that track call order.

    Returns the (mutable) list of recorded SleepStep values; if `record`
    was passed, it is returned as-is so the caller can inspect it.

    : ERASURE_AGENT is patched alongside the original five so
    the dispatcher's full 6-step sequence is exercised without invoking
    the real eligibility / tombstone logic.

    : CLUSTER_REPLAY and CRISIS_RECLUSTER are patched too so
    the dispatcher's full NREM-first-REM sequence is exercised
    without invoking the real cluster-replay / re-cluster logic.

    : CLUSTER_SUMMARY and RECALL_INDEX_REBUILD patched so the
    full 13-step sequence is exercised without real store I/O.
    """
    calls = record if record is not None else []
    payloads = payloads or {}

    def _make_step(step: SleepStep):
        payload = payloads.get(step, {})

        def _noop(_interrupt_check):
            calls.append(step)
            return True, dict(payload)

        return _noop

    monkeypatch.setattr(
        pipeline, "_step_schema_mine", _make_step(SleepStep.SCHEMA_MINE),
    )
    monkeypatch.setattr(
        pipeline, "_step_knob_tune", _make_step(SleepStep.KNOB_TUNE),
    )
    monkeypatch.setattr(
        pipeline, "_step_dream_decay", _make_step(SleepStep.DREAM_DECAY),
    )
    monkeypatch.setattr(
        pipeline, "_step_erasure_agent",
        _make_step(SleepStep.ERASURE_AGENT),
    )
    monkeypatch.setattr(
        pipeline, "_step_optimize_lance", _make_step(SleepStep.OPTIMIZE_LANCE),
    )
    monkeypatch.setattr(
        pipeline, "_step_compact_records",
        _make_step(SleepStep.COMPACT_RECORDS),
    )
    monkeypatch.setattr(
        pipeline, "_step_cluster_replay",
        _make_step(SleepStep.CLUSTER_REPLAY),
    )
    monkeypatch.setattr(
        pipeline, "_step_reconsolidation",
        _make_step(SleepStep.RECONSOLIDATION),
    )
    monkeypatch.setattr(
        pipeline, "_step_user_model_update",
        _make_step(SleepStep.USER_MODEL_UPDATE),
    )
    monkeypatch.setattr(
        pipeline, "_step_dmn_reflection",
        _make_step(SleepStep.DMN_REFLECTION),
    )
    monkeypatch.setattr(
        pipeline, "_step_crisis_recluster",
        _make_step(SleepStep.CRISIS_RECLUSTER),
    )
    monkeypatch.setattr(
        pipeline, "_step_cluster_summary",
        _make_step(SleepStep.CLUSTER_SUMMARY),
    )
    monkeypatch.setattr(
        pipeline, "_step_recall_index_rebuild",
        _make_step(SleepStep.RECALL_INDEX_REBUILD),
    )
    return calls


# ---------------------------------------------------------------------------
# Ordering + happy path
# ---------------------------------------------------------------------------


def test_pipeline_runs_9_steps_in_order(
    pipeline: SleepPipeline, monkeypatch: pytest.MonkeyPatch,
):
    """All 13 steps execute exactly once, in declared order.

    NREM phase (stabilization) runs first; REM phase (pruning + abstraction)
    runs second. CRISIS_RECLUSTER is conditional on crisis_mode (no-op fixture
    replaces the real body here). CLUSTER_SUMMARY and RECALL_INDEX_REBUILD
    are the final two REM steps (appended in 62-02).

    Test name pinned from -- the body now asserts the
    62-02 13-step order. Renaming would churn pytest filters used by ops
    dashboards; the docstring carries the corrected count.
    """
    calls = _patch_steps_to_noop(pipeline, monkeypatch)

    result: SleepPipelineResult = pipeline.run()

    assert calls == [
        # NREM phase
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        # REM phase
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]
    assert result["completed_steps"] == calls
    assert result["failed_step"] is None
    assert result["error"] is None
    assert result["quarantine_triggered"] is False
    assert result.get("interrupted", False) is False


def test_pipeline_clears_progress_on_success(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Full successful run -> sleep_cycle_progress=None on disk."""
    _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()
    record = load_state(state_path)
    assert record["sleep_cycle_progress"] is None


def test_pipeline_emits_started_and_completed_events(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    event_log: LifecycleEventLog,
):
    """Each step emits one started + one completed event."""
    _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()
    events = event_log.read_all()
    started = [e for e in events if e["event"] == "sleep_step_started"]
    completed = [e for e in events if e["event"] == "sleep_step_completed"]
    # 62-02 appended CLUSTER_SUMMARY and RECALL_INDEX_REBUILD to the REM
    # phase. Step count is now 13.
    assert len(started) == 13
    assert len(completed) == 13
    # Started events appear in step order (: NREM phase
    # first then REM phase; RECALL_INDEX_REBUILD last per 62-02).
    assert [e["step"] for e in started] == [
        s.name for s in (
            # NREM phase
            SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
            SleepStep.OPTIMIZE_LANCE, SleepStep.COMPACT_RECORDS,
            # REM phase
            SleepStep.DREAM_DECAY, SleepStep.ERASURE_AGENT,
            SleepStep.CLUSTER_REPLAY,
            SleepStep.RECONSOLIDATION,
            SleepStep.USER_MODEL_UPDATE,
            SleepStep.DMN_REFLECTION,
            SleepStep.CRISIS_RECLUSTER,
            SleepStep.CLUSTER_SUMMARY,
            SleepStep.RECALL_INDEX_REBUILD,
        )
    ]


# ---------------------------------------------------------------------------
# Resume from step N
# ---------------------------------------------------------------------------


def test_pipeline_resume_from_step_N(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """last_completed_index=idx(KNOB_TUNE) -> resume at OPTIMIZE_LANCE.

    : index-based resume math. rewrites
    the step order to NREM-first-then-REM, so the step AFTER KNOB_TUNE is
    OPTIMIZE_LANCE (NREM pos 2), and the remaining suffix is 6 steps long:
    OPTIMIZE_LANCE, COMPACT_RECORDS, DREAM_DECAY, ERASURE_AGENT,
    CLUSTER_REPLAY, CRISIS_RECLUSTER.
    """
    # Seed lifecycle_state.json with prior progress.
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": SleepPipeline._STEP_ORDER.index(
            SleepStep.KNOB_TUNE,
        ),
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]


def test_pipeline_resume_after_cycle_complete_treated_as_fresh(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """last_completed_index==len(_STEP_ORDER)-1 -> restart from position 0.

    : the wrap-detection check compares against the tuple
    length, not the enum's max numeric value.
    """
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": len(SleepPipeline._STEP_ORDER) - 1,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    # Defensive: a stale completed-cycle marker must not become a no-op.
    assert calls == [
        # NREM phase
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        # REM phase
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


def _patch_step_to_raise(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    failing_step: SleepStep,
    *,
    error_msg: str = "synthetic failure",
) -> None:
    """Replace `failing_step` body with one that raises RuntimeError;
    leave the other 4 as no-ops.
    """
    _patch_steps_to_noop(pipeline, monkeypatch)
    method_name = {
        SleepStep.SCHEMA_MINE: "_step_schema_mine",
        SleepStep.KNOB_TUNE: "_step_knob_tune",
        SleepStep.DREAM_DECAY: "_step_dream_decay",
        SleepStep.ERASURE_AGENT: "_step_erasure_agent",
        SleepStep.OPTIMIZE_LANCE: "_step_optimize_lance",
        SleepStep.COMPACT_RECORDS: "_step_compact_records",
        SleepStep.CLUSTER_REPLAY: "_step_cluster_replay",
        SleepStep.RECONSOLIDATION: "_step_reconsolidation",
        SleepStep.USER_MODEL_UPDATE: "_step_user_model_update",
        SleepStep.DMN_REFLECTION: "_step_dmn_reflection",
        SleepStep.CRISIS_RECLUSTER: "_step_crisis_recluster",
    }[failing_step]

    def _raiser(_interrupt_check):
        raise RuntimeError(error_msg)

    monkeypatch.setattr(pipeline, method_name, _raiser)


def test_pipeline_failure_persists_progress(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Failure mid-DREAM_DECAY -> last_completed_index=idx(COMPACT_RECORDS),
    attempt=1, last_error set.

     : under the NREM-first-REM order, DREAM_DECAY (REM pos
    4) is preceded by COMPACT_RECORDS (NREM pos 3), so a DREAM_DECAY
    failure persists last_completed_index = idx(COMPACT_RECORDS) and the
    completed_steps so-far list is the full 4-step NREM phase.
    """
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.DREAM_DECAY)

    result = pipeline.run()

    assert result["failed_step"] == SleepStep.DREAM_DECAY
    assert result["error"] is not None
    assert "synthetic failure" in result["error"]
    assert result["completed_steps"] == [
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
    ]

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.COMPACT_RECORDS,
    )
    assert progress["attempt"] == 1
    assert "synthetic failure" in (progress["last_error"] or "")


def test_pipeline_resume_then_fail_again_increments_attempt(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Two consecutive failures of the same step -> attempt=2.

     : same predecessor logic as test_pipeline_failure_persists_progress
    — DREAM_DECAY is preceded by COMPACT_RECORDS in the new ordering.
    """
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.DREAM_DECAY)

    pipeline.run()  # attempt=1
    pipeline.run()  # attempt=2

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.COMPACT_RECORDS,
    )
    assert progress["attempt"] == 2
    # No quarantine yet -- 3-strike threshold is exclusive of attempt 2.
    assert record["quarantine"] is None


# ---------------------------------------------------------------------------
# 3-strike quarantine
# ---------------------------------------------------------------------------


def test_pipeline_3_strike_quarantine(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Three consecutive failures of the same step -> quarantine entered."""
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.OPTIMIZE_LANCE)

    pipeline.run()  # attempt=1
    pipeline.run()  # attempt=2
    result = pipeline.run()  # attempt=3 -> quarantine

    assert result["quarantine_triggered"] is True
    assert result["failed_step"] == SleepStep.OPTIMIZE_LANCE

    record = load_state(state_path)
    assert record["quarantine"] is not None
    quarantine = record["quarantine"]
    assert "OPTIMIZE_LANCE" in quarantine["reason"]
    assert "3x" in quarantine["reason"]
    # until_ts approximately 24h after since_ts.
    until = datetime.fromisoformat(quarantine["until_ts"])
    since = datetime.fromisoformat(quarantine["since_ts"])
    delta = until - since
    assert timedelta(hours=23, minutes=59) <= delta <= timedelta(
        hours=24, minutes=1,
    )


def test_pipeline_quarantined_run_short_circuits(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """While quarantined, run() returns immediately and runs zero steps."""
    # Seed an active quarantine.
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "manual seed",
        "since_ts": now.isoformat(),
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    result = pipeline.run()

    assert result["quarantine_triggered"] is True
    assert result["completed_steps"] == []
    assert calls == []  # No steps executed.


def test_pipeline_quarantine_auto_recovery_after_ttl(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
    event_log: LifecycleEventLog,
):
    """Quarantine.until_ts in the past -> auto-cleared, cycle proceeds."""
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now - timedelta(hours=1)).isoformat(),
        "reason": "expired seed",
        "since_ts": (now - timedelta(hours=25)).isoformat(),
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    result = pipeline.run()

    assert result["quarantine_triggered"] is False
    assert calls == [
        # NREM phase
        SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE, SleepStep.COMPACT_RECORDS,
        # REM phase
        SleepStep.DREAM_DECAY, SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]
    # Quarantine record cleared post-recovery.
    record_after = load_state(state_path)
    assert record_after["quarantine"] is None
    # Auto-recovery event emitted.
    events = event_log.read_all()
    lifted = [e for e in events if e["event"] == "quarantine_lifted"]
    assert len(lifted) >= 1
    assert lifted[0]["reason"] == "auto_recovery_after_ttl"


def test_pipeline_reset_quarantine_clears(
    pipeline: SleepPipeline,
    state_path: Path,
):
    """reset_quarantine() wipes quarantine + resets attempt counter."""
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "stuck",
        "since_ts": now.isoformat(),
    }
    record["sleep_cycle_progress"] = {
        "last_completed_index": SleepPipeline._STEP_ORDER.index(
            SleepStep.DREAM_DECAY,
        ),
        "attempt": 3,
        "last_error": "boom",
        "started_at": now.isoformat(),
    }
    save_state(record, state_path)

    assert pipeline.is_quarantined() is True
    pipeline.reset_quarantine()
    assert pipeline.is_quarantined() is False

    record_after = load_state(state_path)
    assert record_after["quarantine"] is None
    # Attempt reset, but last_completed_index preserved (resume still works).
    progress = record_after["sleep_cycle_progress"]
    assert progress is not None
    assert progress["attempt"] == 0
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.DREAM_DECAY,
    )


def test_pipeline_force_run_ignores_quarantine(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """force_run() executes all steps even when quarantine is active."""
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    result = pipeline.force_run()

    assert result["quarantine_triggered"] is False
    #: 6 steps (ERASURE_AGENT appended between DREAM_DECAY
    # and OPTIMIZE_LANCE).
    #: 8 steps total (CLUSTER_REPLAY + CRISIS_RECLUSTER
    # APPENDed to the REM phase).
    #: 9 steps total (RECONSOLIDATION APPENDed inside
    # REM between CLUSTER_REPLAY and CRISIS_RECLUSTER).
    #: 10 steps total (USER_MODEL_UPDATE APPENDed inside
    # REM between RECONSOLIDATION and CRISIS_RECLUSTER).
    #: 11 steps total (DMN_REFLECTION APPENDed inside
    # REM between USER_MODEL_UPDATE and CRISIS_RECLUSTER).
    # 62-02: 13 steps total (CLUSTER_SUMMARY + RECALL_INDEX_REBUILD
    # APPENDed after CRISIS_RECLUSTER).
    assert len(calls) == 13
    # force_run does NOT clear the quarantine record on its own.
    record_after = load_state(state_path)
    assert record_after["quarantine"] is not None


# ---------------------------------------------------------------------------
# Bounded deferral
# ---------------------------------------------------------------------------


def test_pipeline_bounded_deferral_persists_chunk_idx(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Interrupt fires during step 3 -> progress shows step 3 deferral marker."""
    # Step 1 + 2 succeed; step 3 (real method) sees interrupt_check
    # return True on its first chunk and bails.
    _patch_steps_to_noop(pipeline, monkeypatch)

    # Restore real _step_dream_decay so _check_interrupt path runs.
    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)

    # interrupt_check: returns True after step 3 starts (first call).
    call_counter = {"n": 0}

    def _trigger():
        call_counter["n"] += 1
        return True  # always defer

    result = pipeline.run(interrupt_check=_trigger)

    # Expected: NREM phase + ERASURE_AGENT (REM pos 5 — wait, ERASURE_AGENT
    # is at REM pos 5 but DREAM_DECAY is REM pos 4, so the deferral fires
    # at DREAM_DECAY and only the 4-step NREM phase completed).
    # ordering: NREM (SCHEMA_MINE, KNOB_TUNE, OPTIMIZE_LANCE,
    # COMPACT_RECORDS) all complete before the deferral at DREAM_DECAY (REM pos 4).
    assert result.get("interrupted") is True
    assert result["completed_steps"] == [
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
    ]
    assert result["failed_step"] is None

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    # last_completed_index is idx(COMPACT_RECORDS) because DREAM_DECAY
    # did not finish; under the NREM-first-REM ordering
    # the predecessor of DREAM_DECAY is COMPACT_RECORDS (last NREM step).
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.COMPACT_RECORDS,
    )
    # last_error contains the deferral marker (NOT a real error).
    err = progress["last_error"] or ""
    assert err.startswith("deferred:")
    assert "DREAM_DECAY" in err
    assert "chunk_idx=0" in err


def test_pipeline_resumes_after_deferral(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """After a deferral on step 3, the next run re-attempts step 3."""
    # First run: defer at DREAM_DECAY (REM pos 4 under order).
    _patch_steps_to_noop(pipeline, monkeypatch)
    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)
    pipeline.run(interrupt_check=lambda: True)

    # Second run: replace DREAM_DECAY with no-op (so it can pass) and
    # confirm we ran the entire REM-phase suffix (:
    # DREAM_DECAY -> ERASURE_AGENT -> CLUSTER_REPLAY -> CRISIS_RECLUSTER).
    #: RECONSOLIDATION inserted between CLUSTER_REPLAY
    # and CRISIS_RECLUSTER.
    calls: list[SleepStep] = []
    _patch_steps_to_noop(pipeline, monkeypatch, record=calls)
    pipeline.run()
    assert calls == [
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]


def test_pipeline_deferral_does_not_increment_attempt(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Bounded deferral is a cooperative yield, NOT a strike."""
    _patch_steps_to_noop(pipeline, monkeypatch)
    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)

    pipeline.run(interrupt_check=lambda: True)
    pipeline.run(interrupt_check=lambda: True)
    pipeline.run(interrupt_check=lambda: True)

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    # attempt stayed at 0 across 3 deferrals (no strike triggered).
    assert progress["attempt"] == 0
    assert record["quarantine"] is None


# ---------------------------------------------------------------------------
# Atomic-step crash safety
# ---------------------------------------------------------------------------


def test_pipeline_atomic_no_corruption_on_step_crash(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """A step crash leaves lifecycle_state.json well-formed (load_state OK)."""
    _patch_step_to_raise(
        pipeline, monkeypatch, SleepStep.OPTIMIZE_LANCE,
        error_msg="lance shard corrupt",
    )
    pipeline.run()

    # File parses cleanly via load_state — atomic-replace path held.
    record = load_state(state_path)
    assert record["sleep_cycle_progress"] is not None
    progress = record["sleep_cycle_progress"]
    # OPTIMIZE_LANCE failed; the predecessor in _STEP_ORDER is
    # ERASURE_AGENT -- but moves
    # OPTIMIZE_LANCE up into the NREM phase (pos 2), so its predecessor
    # is now KNOB_TUNE (pos 1). last_completed_index points at it.
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.KNOB_TUNE,
    )
    assert progress["attempt"] == 1
    # Other invariants (default WAKE state, shadow_run flag) preserved.
    # shadow_run default flipped True -> False.
    assert record["current_state"] == LifecycleState.WAKE.value
    assert record["shadow_run"] is False


def test_pipeline_run_does_not_mutate_other_state_fields(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Sleep-cycle writes must NOT touch current_state / shadow_run / etc."""
    record = default_state()
    record["current_state"] = LifecycleState.SLEEP.value
    record["wrapper_event_seq"] = 42
    save_state(record, state_path)

    _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    after = load_state(state_path)
    assert after["current_state"] == LifecycleState.SLEEP.value
    assert after["wrapper_event_seq"] == 42
    assert after["sleep_cycle_progress"] is None  # cleared on success


# ---------------------------------------------------------------------------
# is_quarantined edge cases
# ---------------------------------------------------------------------------


def test_is_quarantined_false_when_no_record(
    pipeline: SleepPipeline,
):
    """No state file at all -> is_quarantined() returns False."""
    assert pipeline.is_quarantined() is False


def test_is_quarantined_false_for_malformed_until_ts(
    pipeline: SleepPipeline,
    state_path: Path,
):
    """Malformed quarantine.until_ts -> is_quarantined() returns False
    (do not lock the user out on corrupted entry).
    """
    record = default_state()
    record["quarantine"] = {
        "until_ts": "not a timestamp",
        "reason": "test",
        "since_ts": "also not a timestamp",
    }
    save_state(record, state_path)
    assert pipeline.is_quarantined() is False


# ---------------------------------------------------------------------------
# — index-based resume math + legacy field migration
# ---------------------------------------------------------------------------


def test_pipeline_resume_after_cycle_complete_wraps_to_schema_mine(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """After a successful cycle persists last_completed_index ==
    len(_STEP_ORDER)-1, the next run starts at SCHEMA_MINE.

      acceptance: with step.value math the wrap would
    fire prematurely (ERASURE_AGENT.value=6 > COMPACT_RECORDS.value=5),
    causing the next cycle to start at OPTIMIZE_LANCE (wrong). With
    position-based math, the wrap fires exactly once after
    COMPACT_RECORDS, and SCHEMA_MINE runs next.
    """
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": len(SleepPipeline._STEP_ORDER) - 1,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls[0] == SleepStep.SCHEMA_MINE


def test_pipeline_failed_erasure_agent_resumes_at_erasure_agent(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """ERASURE_AGENT raises -> persisted last_completed_index ==
    idx(DREAM_DECAY); next run resumes at ERASURE_AGENT.

      acceptance: with step.value math the on-failure
    persist would write step.value-1 = 5 (COMPACT_RECORDS.value), and
    the next run would skip OPTIMIZE_LANCE entirely. Position-based math
    writes idx(ERASURE_AGENT)-1 = idx(DREAM_DECAY) so the resume math
    re-runs ERASURE_AGENT on the next cycle.
    """
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.ERASURE_AGENT)
    pipeline.run()  # raises mid-ERASURE_AGENT

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.DREAM_DECAY,
    )

    # Second run: resume should re-attempt ERASURE_AGENT (still raises).
    pipeline.run()
    record_after = load_state(state_path)
    progress_after = record_after["sleep_cycle_progress"]
    assert progress_after is not None
    assert progress_after["attempt"] == 2  # strike-counter incremented


def test_pipeline_three_strike_quarantine_on_erasure_agent(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """ERASURE_AGENT failing 3x triggers 24h auto-quarantine.

      acceptance: position-based strike check correctly
    accumulates failures of ERASURE_AGENT (the new step).
    """
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.ERASURE_AGENT)
    pipeline.run()  # attempt=1
    pipeline.run()  # attempt=2
    result = pipeline.run()  # attempt=3 -> quarantine

    assert result["quarantine_triggered"] is True
    assert result["failed_step"] == SleepStep.ERASURE_AGENT
    record = load_state(state_path)
    assert record["quarantine"] is not None
    assert "ERASURE_AGENT" in record["quarantine"]["reason"]


def test_pipeline_legacy_last_completed_step_field_migrated(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """A legacy `lifecycle_state.json` with `last_completed_step=4`
    (legacy OPTIMIZE_LANCE.value) resumes at COMPACT_RECORDS next cycle.

    The migration MUST use
    enum-lookup (SleepStep(4) -> SleepStep.OPTIMIZE_LANCE) and then
    look up the NEW position (_STEP_ORDER.index(OPTIMIZE_LANCE)) — NOT
    subtract 1 from the legacy value. With subtraction the next cycle
    would skip COMPACT_RECORDS, then ALSO silently regress ERASURE_AGENT
    on a future cycle. With enum-lookup, the resume points at exactly
    COMPACT_RECORDS without skipping ERASURE_AGENT on subsequent passes.
    """
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": 4,  # legacy OPTIMIZE_LANCE.value
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    # Crash-window resume: OPTIMIZE_LANCE (legacy value 4 ->
    # SleepStep.OPTIMIZE_LANCE) had just completed in the legacy run.
    #: OPTIMIZE_LANCE is now NREM pos 2; the next step
    # is COMPACT_RECORDS (NREM pos 3), followed by the full REM phase
    # in the same cycle. The migration MUST NOT skip any step.
    assert calls == [
        SleepStep.COMPACT_RECORDS,
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]

    # Second cycle: should be a fresh start (the persisted progress was
    # cleared by the successful end of cycle 1).
    calls2: list[SleepStep] = []
    _patch_steps_to_noop(pipeline, monkeypatch, record=calls2)
    pipeline.run()
    assert calls2 == [
        # NREM phase
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        # REM phase
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]


def test_pipeline_legacy_last_completed_step_zero_starts_fresh(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Legacy `last_completed_step=0` migrates to last_completed_index=-1
    (no step completed yet) and the cycle runs all 13 steps.

    Defensive: legacy value 0 is NOT a valid SleepStep enum value, so
    the migration falls back to -1 (start fresh) rather than crashing.
    """
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": 0,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
        # NREM phase
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        # REM phase
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]
