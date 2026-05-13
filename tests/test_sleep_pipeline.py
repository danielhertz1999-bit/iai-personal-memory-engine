"""Task 1.4 -- SleepPipeline tests.

Covers:
- 5-step ordering (SCHEMA_MINE -> ... -> COMPACT_RECORDS).
- progress cleared on full success.
- resume-from-step-N: with last_completed_step=2, only steps 3/4/5 run.
- failure persists progress (last_completed_step=N-1, attempt+1, last_error).
- 3-strike threshold triggers 24h auto-quarantine.
- quarantined run() short-circuits with quarantine_triggered=True.
- quarantine auto-recovery once until_ts is in the past.
- reset_quarantine() clears immediately.
- force_run() ignores quarantine.
- bounded deferral persists chunk_idx in deferral marker; run returns interrupted=True.
- atomic-step crash leaves progress consistent (no partial state corruption).

All tests run with a stub `store` (None) and step methods replaced via
monkeypatch — no real LanceDB I/O, no real embedder load. Combined
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
    """Replace all 5 _step_* methods with no-ops that track call order.

    Returns the (mutable) list of recorded SleepStep values; if `record`
    was passed, it is returned as-is so the caller can inspect it.
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
        pipeline, "_step_optimize_lance", _make_step(SleepStep.OPTIMIZE_LANCE),
    )
    monkeypatch.setattr(
        pipeline, "_step_compact_records",
        _make_step(SleepStep.COMPACT_RECORDS),
    )
    return calls


# ---------------------------------------------------------------------------
# Ordering + happy path
# ---------------------------------------------------------------------------


def test_pipeline_runs_5_steps_in_order(
    pipeline: SleepPipeline, monkeypatch: pytest.MonkeyPatch,
):
    """All 5 steps execute exactly once, in declared order."""
    calls = _patch_steps_to_noop(pipeline, monkeypatch)

    result: SleepPipelineResult = pipeline.run()

    assert calls == [
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.DREAM_DECAY,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
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
    assert len(started) == 5
    assert len(completed) == 5
    # Started events appear in step order
    assert [e["step"] for e in started] == [
        s.name for s in (
            SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
            SleepStep.DREAM_DECAY, SleepStep.OPTIMIZE_LANCE,
            SleepStep.COMPACT_RECORDS,
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
    """last_completed_step=2 -> only steps 3/4/5 execute on next run."""
    # Seed lifecycle_state.json with prior progress.
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": 2,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
        SleepStep.DREAM_DECAY,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
    ]


def test_pipeline_resume_after_step_5_treated_as_fresh(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """last_completed_step==5 should restart the cycle from step 1."""
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": 5,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    # Defensive: a stale '5' must not become a no-op.
    assert calls == [
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.DREAM_DECAY,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
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
        SleepStep.OPTIMIZE_LANCE: "_step_optimize_lance",
        SleepStep.COMPACT_RECORDS: "_step_compact_records",
    }[failing_step]

    def _raiser(_interrupt_check):
        raise RuntimeError(error_msg)

    monkeypatch.setattr(pipeline, method_name, _raiser)


def test_pipeline_failure_persists_progress(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Failure mid-step 3 -> last_completed_step=2, attempt=1, last_error set."""
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.DREAM_DECAY)

    result = pipeline.run()

    assert result["failed_step"] == SleepStep.DREAM_DECAY
    assert result["error"] is not None
    assert "synthetic failure" in result["error"]
    assert result["completed_steps"] == [
        SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
    ]

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_step"] == 2
    assert progress["attempt"] == 1
    assert "synthetic failure" in (progress["last_error"] or "")


def test_pipeline_resume_then_fail_again_increments_attempt(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    """Two consecutive failures of the same step -> attempt=2."""
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.DREAM_DECAY)

    pipeline.run()  # attempt=1
    pipeline.run()  # attempt=2

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_step"] == 2
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
        SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
        SleepStep.DREAM_DECAY, SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
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
        "last_completed_step": 3,
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
    # Attempt reset, but last_completed_step preserved (resume still works).
    progress = record_after["sleep_cycle_progress"]
    assert progress is not None
    assert progress["attempt"] == 0
    assert progress["last_completed_step"] == 3


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
    assert len(calls) == 5
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

    # Expected: step 1 + 2 completed; step 3 deferred at chunk_idx=0.
    assert result.get("interrupted") is True
    assert result["completed_steps"] == [
        SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
    ]
    assert result["failed_step"] is None

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    # last_completed_step is 2 because step 3 did not finish.
    assert progress["last_completed_step"] == 2
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
    # First run: defer at step 3.
    _patch_steps_to_noop(pipeline, monkeypatch)
    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)
    pipeline.run(interrupt_check=lambda: True)

    # Second run: replace step 3 with no-op (so it can pass) and confirm
    # we ran steps 3, 4, 5 only.
    calls: list[SleepStep] = []
    _patch_steps_to_noop(pipeline, monkeypatch, record=calls)
    pipeline.run()
    assert calls == [
        SleepStep.DREAM_DECAY,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
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
    assert progress["last_completed_step"] == 3
    assert progress["attempt"] == 1
    # Other invariants (default WAKE state, shadow_run flag) preserved.
    # Task 1.6: shadow_run default flipped True -> False.
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
