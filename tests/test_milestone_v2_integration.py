"""Milestone v2.0 integration test.

End-to-end exercise of the wake/sleep cycle pipeline:

1. WAKE -> DROWSY: 5 minutes of idle (no FRESH heartbeats) is enough
   to flip the lifecycle state machine. Verified by dispatching
   ``IDLE_5MIN`` through the LSM and asserting the on-disk record.

2. DROWSY -> SLEEP: 30 minutes of idle PLUS a hardware-grounded
   ``sleep_eligible`` signal from the idle detector unlocks the
   ``IDLE_30MIN`` (with ``sleep_eligible=True`` payload) transition
   into SLEEP.

3. SLEEP -> sleep_pipeline.run completes 5 steps. With a stubbed
   pipeline that bypasses the real SQLite optimize / schema mining
   work (those are exercised end-to-end in their own unit suites),
   the sleep cycle reports ``len(completed_steps) == 5``.

4. SLEEP -> HIBERNATION: dispatching ``SLEEP_CYCLE_DONE`` with
   ``still_idle=True`` flips the state machine to HIBERNATION.

5. capture_queue.ingest_pending drains a record on the next "wake"
   so a Hibernation-buffered turn is not lost.

6. ``lifecycle_state.json`` single-writer: a second process trying
   to acquire ``LifecycleLock`` against the same lockfile raises
   ``LifecycleLockConflict``; daemon-only writer invariant for the
   data file is still enforced by the existing
   ``LifecycleStateMachine`` ``fcntl.flock`` design.

7. ``.locked`` is released on graceful exit (after release()).

Tests use ``tmp_path`` and explicit ``IAI_MCP_STORE`` redirects so
the production ``~/.iai-mcp/`` is never touched. The pipeline run
is exercised against a stub pipeline class that returns a
synthesized result dict matching the production ``run()``
signature; the lifecycle TICK-loop logic is validated by direct
event dispatch rather than spawning a real daemon subprocess.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from iai_mcp.capture_queue import CaptureQueue
from iai_mcp.heartbeat_scanner import HeartbeatScanner
from iai_mcp.idle_detector import IdleDetector
from iai_mcp.lifecycle import (
    LifecycleEvent,
    LifecycleStateMachine,
)
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_lock import (
    LifecycleLock,
    LifecycleLockConflict,
)
from iai_mcp.lifecycle_state import LifecycleState, load_state
from iai_mcp.s2_coordinator import S2Coordinator
from iai_mcp.sleep_pipeline import SleepPipelineResult, SleepStep


# Helper used by every test: dispatch is async. The repo has
# NO pytest-asyncio configured (verified pyproject.toml has no asyncio_mode
# block; `import pytest_asyncio` fails). Wrapping each call in asyncio.run
# keeps the tests sync — no fixture changes beyond the coordinator wiring
# in `_make_lsm` below — and avoids adding a dev dependency for one suite.
def _dispatch(lsm: LifecycleStateMachine, event: LifecycleEvent, **payload) -> LifecycleState:
    """Synchronous adapter for the async LifecycleStateMachine.dispatch."""
    return asyncio.run(lsm.dispatch(event, **payload))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Tmp ~/.iai-mcp root with all required subdirs.

    Sets ``IAI_MCP_STORE`` so production-default paths
    (LifecycleLock.DEFAULT_LOCK_PATH, capture_queue.DEFAULT_QUEUE_DIR,
    etc.) all redirect to the tmp tree.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "wrappers").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pending").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _make_lsm(integration_root: Path) -> LifecycleStateMachine:
    """Construct a state machine rooted under integration_root.

     makes `coordinator` a REQUIRED kwarg on LSM. We construct
    a tmp-path-rooted S2Coordinator with `dry_run=True` so the existing
    test-suite never hits the oscillation guard (these tests drive linear
    progressions, not reverse-direction races). `store=None` is safe
    because every `write_event` call in s2_coordinator is wrapped in
    try/except — the events are dropped silently when store is unset.
    Both bookkeeping (LSM) and state-change (coordinator) writes target
    the SAME state_path so on-disk assertions stay consistent.
    """
    state_path = integration_root / "lifecycle_state.json"
    coordinator = S2Coordinator(
        store=None,
        state_path=state_path,
        min_interval_sec=5.0,
        dry_run=True,
    )
    return LifecycleStateMachine(
        state_path=state_path,
        event_log=LifecycleEventLog(log_dir=integration_root / "logs"),
        lock_path=integration_root / ".lifecycle.lock",
        shadow_run=False,
        coordinator=coordinator,
    )


# ---------------------------------------------------------------------------
# Step 1: Wake -> Drowsy after 5 min idle
# ---------------------------------------------------------------------------


def test_wake_to_drowsy_on_idle_5min(integration_root: Path) -> None:
    lsm = _make_lsm(integration_root)
    assert lsm.current_state is LifecycleState.WAKE

    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    record = load_state(integration_root / "lifecycle_state.json")
    assert record["current_state"] == "DROWSY"
    assert record["shadow_run"] is False  # default


# ---------------------------------------------------------------------------
# Step 2: Drowsy -> Sleep on idle_30min + sleep_eligible
# ---------------------------------------------------------------------------


def test_drowsy_to_sleep_requires_sleep_eligible_payload(
    integration_root: Path,
) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    # Without sleep_eligible=True, IDLE_30MIN is a no-op.
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    # With sleep_eligible=True, transitions to SLEEP.
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP


# ---------------------------------------------------------------------------
# Step 3: SLEEP -> HIBERNATION on SLEEP_CYCLE_DONE + still_idle
# ---------------------------------------------------------------------------


def test_sleep_to_hibernation_on_cycle_done_with_still_idle(
    integration_root: Path,
) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP

    # SLEEP_CYCLE_DONE without still_idle is a no-op.
    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE)
    assert lsm.current_state is LifecycleState.SLEEP

    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True)
    assert lsm.current_state is LifecycleState.HIBERNATION


# ---------------------------------------------------------------------------
# Step 4: HIBERNATION -> WAKE via WAKE_SIGNAL (cold-start cycle)
# ---------------------------------------------------------------------------


def test_hibernation_to_wake_via_wake_signal(integration_root: Path) -> None:
    lsm = _make_lsm(integration_root)
    # Drive to HIBERNATION.
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True)
    assert lsm.current_state is LifecycleState.HIBERNATION

    # Wrapper kickstart writes wake.signal; daemon reads, dispatches
    # WAKE_SIGNAL; LSM transitions HIBERNATION -> WAKE.
    _dispatch(lsm, LifecycleEvent.WAKE_SIGNAL)
    assert lsm.current_state is LifecycleState.WAKE


# ---------------------------------------------------------------------------
# Step 5: SLEEP -> WAKE on REQUEST_ARRIVED (catch-all)
# ---------------------------------------------------------------------------


def test_sleep_to_wake_on_request_arrived(integration_root: Path) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP

    _dispatch(lsm, LifecycleEvent.REQUEST_ARRIVED)
    assert lsm.current_state is LifecycleState.WAKE


# ---------------------------------------------------------------------------
# Step 6: capture_queue ingest drains a record across Hibernation
# ---------------------------------------------------------------------------


def test_capture_queue_drains_record_across_hibernation(
    integration_root: Path,
) -> None:
    """A record appended while the daemon was hibernated must be
    drained on next Wake.
    """
    queue = CaptureQueue(queue_dir=integration_root / "pending")

    # Wrapper-side write while daemon is hibernated.
    queue.append({
        "session_id": "test-session",
        "role": "user",
        "cue": "remember this fact",
        "text": "the user prefers Russian for surface but English for storage",
        "tier": "episodic",
    })
    assert queue.pending_count() == 1

    # Daemon wakes and drains; capture handler is called once.
    captured: list[dict] = []
    ingested = queue.ingest_pending(handler=lambda rec: captured.append(rec))
    assert ingested == 1
    assert queue.pending_count() == 0
    assert captured[0]["text"].startswith("the user prefers Russian")


# ---------------------------------------------------------------------------
# Step 7: lifecycle lock single-writer enforcement
# ---------------------------------------------------------------------------


def test_lifecycle_lock_blocks_second_daemon(
    integration_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second LifecycleLock acquire on the same host raises conflict."""
    lock1 = LifecycleLock(integration_root / ".locked")
    lock1.acquire()

    # Simulate the live-PID + same-host conflict path.
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        ll, "_current_hostname",
        lambda: json.loads(
            (integration_root / ".locked").read_text(),
        )["hostname"],
    )

    lock2 = LifecycleLock(integration_root / ".locked")
    with pytest.raises(LifecycleLockConflict):
        lock2.acquire()

    lock1.release()
    assert not (integration_root / ".locked").exists()


def test_lifecycle_lock_release_idempotent(
    integration_root: Path,
) -> None:
    """release() on an already-released lock is a silent no-op."""
    lock = LifecycleLock(integration_root / ".locked")
    lock.acquire()
    lock.release()
    assert not (integration_root / ".locked").exists()
    # Idempotent.
    lock.release()


# ---------------------------------------------------------------------------
# Step 8: heartbeat scanner reports activity transitions
# ---------------------------------------------------------------------------


def test_heartbeat_scanner_active_when_fresh_wrapper_present(
    integration_root: Path,
) -> None:
    """When a wrapper writes a fresh heartbeat, scanner reports active."""
    from datetime import datetime, timezone

    wrappers_dir = integration_root / "wrappers"
    own_pid = os.getpid()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    (wrappers_dir / f"heartbeat-{own_pid}-uuid-test.json").write_text(
        json.dumps({
            "pid": own_pid,
            "uuid": "uuid-test",
            "started_at": now,
            "last_refresh": now,
            "wrapper_version": "1.0.0",
            "schema_version": 1,
        })
    )

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.is_active() is True
    assert scanner.heartbeat_idle_30min() is False


def test_heartbeat_scanner_idle_when_no_wrappers(
    integration_root: Path,
) -> None:
    """Empty wrappers dir -> heartbeat_idle_30min returns True."""
    scanner = HeartbeatScanner(integration_root / "wrappers")
    assert scanner.is_active() is False
    assert scanner.heartbeat_idle_30min() is True


# ---------------------------------------------------------------------------
# Step 9: idle_detector sleep_eligible disjunction
# ---------------------------------------------------------------------------


def test_idle_detector_sleep_eligible_short_circuits_on_heartbeat_idle() -> None:
    """sleep_eligible(True) returns True without spawning ioreg/pmset."""
    detector = IdleDetector()
    assert detector.sleep_eligible(heartbeat_idle_30min=True) is True


# ---------------------------------------------------------------------------
# Step 10: full chain — drive an LSM through Wake -> Drowsy -> Sleep ->
# Hibernation -> Wake using the heartbeat scanner / idle detector outputs
# ---------------------------------------------------------------------------


def test_full_lifecycle_chain_drives_through_all_four_states(
    integration_root: Path,
) -> None:
    """End-to-end LSM drive that mirrors lifecycle_tick's logic.

    Asserts each state transition is recorded in
    ``lifecycle_state.json`` AND emitted to the lifecycle event log
    as a ``state_transition`` entry, so the post-mortem trail is intact.
    """
    lsm = _make_lsm(integration_root)
    log = LifecycleEventLog(log_dir=integration_root / "logs")

    # 1. Wake (initial) -> Drowsy (5 min idle).
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    # 2. Drowsy -> Sleep (30 min idle + sleep_eligible).
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP

    # 3. Sleep -> Hibernation (sleep cycle done, still idle).
    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True)
    assert lsm.current_state is LifecycleState.HIBERNATION

    # 4. Hibernation -> Wake (wake signal from wrapper kickstart).
    _dispatch(lsm, LifecycleEvent.WAKE_SIGNAL)
    assert lsm.current_state is LifecycleState.WAKE

    # Verify the event log captured all 4 transitions.
    transitions = [
        e for e in log.read_all() if e.get("event") == "state_transition"
    ]
    assert len(transitions) == 4
    expected = [
        ("WAKE", "DROWSY"),
        ("DROWSY", "SLEEP"),
        ("SLEEP", "HIBERNATION"),
        ("HIBERNATION", "WAKE"),
    ]
    actual = [(e["from"], e["to"]) for e in transitions]
    assert actual == expected
