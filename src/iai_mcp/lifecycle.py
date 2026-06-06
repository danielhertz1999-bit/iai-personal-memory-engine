"""-- Lifecycle State Machine + Shadow-Run Mode.

Realises LOCKED contracts L1 (hibernation depth: kill process) and
L2 (state authority: daemon-only writer for `lifecycle_state.json`).

The four lifecycle states (WAKE, DROWSY, SLEEP, HIBERNATION) form a
deterministic FSM. Transitions are pure functions of the current state
and the dispatched event (with optional payload guards); side effects
(persistence + event-log append + shadow-run warning) happen ONLY in
`dispatch`.

  Task 1.6: flipped `shadow_run` default from
True to False. HIBERNATION transitions now actually exit the daemon
process via the global shutdown event in `daemon.main()`'s lifecycle
tick. The legacy `_rss_watchdog_loop` was removed in Task 1.4; this
state machine is the sole owner of shutdown authority.

Shadow-run mode is preserved as an opt-in for testing: passing
`shadow_run=True` to `LifecycleStateMachine.__init__` keeps the old
"persist + log + emit shadow_run_warning, do NOT exit" behaviour so
tests can drive transitions without terminating the daemon process.

Single-writer enforcement (L2): a separate lock file
`~/.iai-mcp/.lifecycle.lock` carries the `fcntl.flock(LOCK_EX|LOCK_NB)`.
The data file `lifecycle_state.json` is atomically replaced via
`os.replace` (pattern), which swaps the inode — any lock
held on the data file's fd would not protect the new file. The lock
file is never renamed, so the lock survives `save_state` cycles.
"""
from __future__ import annotations

import asyncio
import errno
import fcntl
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import (
    LIFECYCLE_STATE_PATH,
    LifecycleState,
    LifecycleStateRecord,
    default_state,
    load_state,
    save_state,
)

# Default lock path lives next to lifecycle_state.json. Hidden so it
# does not show up in `ls`. Pattern matches `daemon-state.json` /
# `.daemon-state.json` precedent.
DEFAULT_LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lifecycle.lock"


class LifecycleStateLocked(RuntimeError):
    """Raised when another process holds the lifecycle_state.json lock.

    Per L2 the daemon is the sole authority. A wrapper that finds the
    lock held by the daemon should signal events via Unix socket
    (when daemon alive) or write `~/.iai-mcp/wake.signal` (when
    daemon hibernated) — never bypass the lock with a direct write.
    """


class LifecycleEvent(str, Enum):
    """Events that drive state-machine transitions."""

    HEARTBEAT_REFRESH = "heartbeat_refresh"
    IDLE_5MIN = "idle_5min"
    IDLE_30MIN = "idle_30min"
    SLEEP_ELIGIBLE = "sleep_eligible"
    REQUEST_ARRIVED = "request_arrived"
    SLEEP_CYCLE_DONE = "sleep_cycle_done"
    HIBERNATION_GRACE_EXPIRED = "hibernation_grace_expired"
    WAKE_SIGNAL = "wake_signal"
    TICK = "tick"
    # FORCE_SLEEP: explicit consolidation trigger from force-rem / user-sleep.
    # Routes any state -> DROWSY first (so the DROWSY-edge teardown / drain runs),
    # then DROWSY -> SLEEP (bypassing idle/eligibility), so iai-mcp daemon force-rem
    # still triggers one full pipeline run via the canonical path.
    FORCE_SLEEP = "force_sleep"


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp; central so tests can monkey-patch."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pure transition function — exposed at module scope for property tests
# ---------------------------------------------------------------------------

def compute_transition(
    state: LifecycleState,
    event: LifecycleEvent,
    payload: dict[str, Any] | None = None,
) -> LifecycleState | None:
    """Return the target state, or None if `event` is a no-op for `state`.

    Pure function — no I/O, no side effects, deterministic. The
    transition table is encoded inline here rather than a dict because
    the guard-bearing rows (`(DROWSY, IDLE_30MIN)` AND `sleep_eligible`)
    are easier to read as straight-line code than a `(state, event,
    guard) -> state` lookup with conditional fallback.

    Transition table:

      | From | Event | To |
      | WAKE | IDLE_5MIN | DROWSY |
      | DROWSY | HEARTBEAT_REFRESH | WAKE |
      | DROWSY | IDLE_30MIN AND sleep_eligible | SLEEP |
      | SLEEP | REQUEST_ARRIVED | WAKE |
      | SLEEP | SLEEP_CYCLE_DONE AND still_idle | HIBERNATION |
      | HIBERNATION | WAKE_SIGNAL | WAKE |
      | * | REQUEST_ARRIVED | WAKE (catch-all)

    Catch-all: REQUEST_ARRIVED from any state goes to WAKE; that
    matches the SLEEP-specific rule above and adds DROWSY/HIBERNATION
    coverage. (HIBERNATION → WAKE on REQUEST_ARRIVED is a future-phase
    cold-start path — a wrapper that has REQUEST_ARRIVED to dispatch
    has already woken the daemon via wake.signal first; this branch
    exists for in-process test scaffolding and defence-in-depth.)
    """
    payload = payload if payload is not None else {}

    # Catch-all REQUEST_ARRIVED → WAKE; check first so subsequent
    # branches do not need to repeat the rule per source state.
    if event is LifecycleEvent.REQUEST_ARRIVED:
        return LifecycleState.WAKE

    # FORCE_SLEEP: two-hop route via DROWSY so the DROWSY-edge teardown / drain
    # always runs before entering the consolidation window.
    # * + FORCE_SLEEP -> DROWSY (any non-SLEEP state)
    # DROWSY + FORCE_SLEEP -> SLEEP (bypasses idle / sleep_eligible guards)
    if event is LifecycleEvent.FORCE_SLEEP:
        if state is LifecycleState.DROWSY:
            return LifecycleState.SLEEP
        if state in (LifecycleState.SLEEP, LifecycleState.HIBERNATION):
            # Already at or past SLEEP — no-op (SLEEP is the target).
            return None
        # WAKE or any other non-SLEEP state → hop to DROWSY first.
        return LifecycleState.DROWSY

    if state is LifecycleState.WAKE:
        if event is LifecycleEvent.IDLE_5MIN:
            return LifecycleState.DROWSY
        return None

    if state is LifecycleState.DROWSY:
        if event is LifecycleEvent.HEARTBEAT_REFRESH:
            return LifecycleState.WAKE
        if event is LifecycleEvent.IDLE_30MIN and payload.get("sleep_eligible"):
            return LifecycleState.SLEEP
        return None

    if state is LifecycleState.SLEEP:
        if event is LifecycleEvent.SLEEP_CYCLE_DONE and payload.get("still_idle"):
            return LifecycleState.HIBERNATION
        return None

    if state is LifecycleState.HIBERNATION:
        if event is LifecycleEvent.WAKE_SIGNAL:
            return LifecycleState.WAKE
        # HIBERNATION_GRACE_EXPIRED is a future-phase trigger that
        # currently has no destination — kept as a known no-op so
        # the dispatcher does not raise on it.
        return None

    return None  # unreachable; defensive against future state additions


# ---------------------------------------------------------------------------
# File-lock context manager — separate file per advisor recommendation
# ---------------------------------------------------------------------------

@contextmanager
def _lifecycle_lock(lock_path: Path) -> Iterator[int]:
    """Acquire `fcntl.flock(LOCK_EX | LOCK_NB)` on a sibling lock file.

    Raises `LifecycleStateLocked` if the lock is held by another
    process. The lock file persists across releases — it is the
    "named-mutex" handle, not the data. The data file
    `lifecycle_state.json` is atomically replaced separately and
    therefore must NOT carry the lock (os.replace swaps the inode).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise LifecycleStateLocked(
                    f"another process holds {lock_path}"
                ) from exc
            raise
        try:
            yield fd
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                # Best effort — the close below releases the lock
                # whether or not the explicit unlock succeeded.
                pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# State machine class
# ---------------------------------------------------------------------------

class LifecycleStateMachine:
    """Side-effecting wrapper around `compute_transition`.

    Owns:
    - `lifecycle_state.json` reads + writes (single-writer enforced).
    - Event log emission (`state_transition`, `shadow_run_warning`).
    - `shadow_run` flag (default False since; True is a transition-test escape hatch).

    Construction is cheap; the lock is acquired only inside
    `dispatch`. Tests can drive transitions either via `dispatch`
    (full pipeline) or via `compute_transition` (pure-function
    coverage).
    """

    def __init__(
        self,
        state_path: Path | None = None,
        event_log: LifecycleEventLog | None = None,
        lock_path: Path | None = None,
        shadow_run: bool = False,
        *,
        # Forward-reference string type-hint keeps s2_coordinator.py out of
        # this module's import graph (the actual call is duck-typed inside
        # async `dispatch`). Production daemon.main always injects a real
        # coordinator; tests inject a tmp-path-rooted instance.
        # No legacy direct-save fallback: calling dispatch on an LSM
        # constructed without a coordinator raises RuntimeError before any
        # disk IO — all current_state persistence flows through coordinator.
        coordinator: "S2Coordinator | None" = None,  # noqa: F821 — forward ref
    ) -> None:
        self._state_path = state_path if state_path is not None else LIFECYCLE_STATE_PATH
        self._event_log = event_log if event_log is not None else LifecycleEventLog()
        self._lock_path = lock_path if lock_path is not None else DEFAULT_LOCK_PATH
        self._shadow_run = shadow_run
        self._coordinator = coordinator

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    @property
    def shadow_run(self) -> bool:
        return self._shadow_run

    @property
    def current_state(self) -> LifecycleState:
        record = load_state(self._state_path)
        return LifecycleState(record["current_state"])

    def snapshot(self) -> LifecycleStateRecord:
        """Return the on-disk record (or default if absent)."""
        return load_state(self._state_path)

    # ------------------------------------------------------------------
    # Pure transition (no I/O) — re-exposed for callers using an instance
    # ------------------------------------------------------------------

    def compute_transition(
        self,
        state: LifecycleState,
        event: LifecycleEvent,
        payload: dict[str, Any] | None = None,
    ) -> LifecycleState | None:
        return compute_transition(state, event, payload)

    # ------------------------------------------------------------------
    # Dispatcher — single-writer, persists + logs
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        event: LifecycleEvent,
        *,
        reason: str | None = None,
        **payload: Any,
    ) -> LifecycleState:
        """Apply `event` to current state, route persistence through coordinator.

        `dispatch` is async — every call site MUST `await`. State-change
        persistence is delegated to `S2Coordinator.transition` which owns
        the asyncio.Lock + CAS + ring-buffer + on-disk save_state for the
        `current_state` field. `dispatch` retains responsibility ONLY for
        incidental bookkeeping fields (last_activity_ts, wrapper_event_seq,
        shadow_run) that do NOT touch current_state.

        `reason` is the snake-case identifier routed through the
        coordinator's `s2_transition_attempt` event body. Falls back to
        `event.value` when None. Raises `S2OscillationConflict` /
        `S2OscillationBlocked` as normal control flow per — call
        sites are expected to catch both with `pass`.
        """
        if self._coordinator is None:
            # No legacy direct-save fallback. Production daemons inject a
            # coordinator via daemon.main; tests inject via fixtures
            # (tmp_path-rooted state_path). Bare LSM + dispatch is a
            # programming error — fail loud here rather than silently mutate
            # state through an unlocked path.
            raise RuntimeError(
                "LifecycleStateMachine.dispatch requires a coordinator. "
                "Production callers in daemon.main inject one; tests should "
                "construct an S2Coordinator with state_path=tmp_path."
            )

        current_record = await asyncio.to_thread(load_state, self._state_path)
        # Local Python identifier is `from_state` rather than `current_state`
        # so all current_state assignments are centralized in s2_coordinator.py.
        # The persisted record key remains "current_state".
        from_state = LifecycleState(current_record["current_state"])
        payload_dict = dict(payload)
        target = compute_transition(from_state, event, payload_dict)

        # Incidental bookkeeping: HEARTBEAT_REFRESH / REQUEST_ARRIVED /
        # WAKE_SIGNAL advance last_activity_ts + wrapper_event_seq even when
        # state is unchanged. These advisory fields do NOT touch current_state;
        # dispatch persists them directly without acquiring `_lifecycle_lock`:
        # the coordinator's asyncio
        # Lock + atomic os.replace in save_state make torn writes impossible,
        # and holding fcntl.flock here would deadlock observers reading the
        # file mid-transition. The bookkeeping fields are advisory (last
        # writer wins is acceptable — already v1 behaviour for monotonic-ish
        # fields).
        if event in {
            LifecycleEvent.HEARTBEAT_REFRESH,
            LifecycleEvent.REQUEST_ARRIVED,
            LifecycleEvent.WAKE_SIGNAL,
        }:
            now_iso = _utc_now_iso()
            updated_record: LifecycleStateRecord = dict(current_record)  # type: ignore[assignment]
            updated_record["last_activity_ts"] = now_iso
            updated_record["wrapper_event_seq"] = (
                current_record.get("wrapper_event_seq", 0) + 1
            )
            updated_record["shadow_run"] = self._shadow_run
            if updated_record != current_record:
                await asyncio.to_thread(save_state, updated_record, self._state_path)
                # Refresh local view so the post-coordinator append sees the
                # bookkeeping changes (the coordinator will read disk again
                # inside its lock — that path is independent).
                current_record = updated_record

        if target is None or target == from_state:
            # No state change. Bookkeeping persisted above (when applicable).
            return from_state

        # Real state change: delegate to coordinator. May raise
        # S2OscillationConflict (CAS mismatch — disk state moved between
        # our load above and the coordinator's load inside its lock) or
        # S2OscillationBlocked (reverse direction within MIN_INTERVAL_SEC).
        # Both are normal control flow; daemon call sites catch them with
        # `pass`. The coordinator emits the matching event body on every
        # emit-worthy path.
        resolved_reason = reason if reason is not None else event.value
        new_state = await self._coordinator.transition(
            from_state, target, resolved_reason,
        )

        # Legacy lifecycle_event_log entry preserved AFTER successful
        # coordinator transition. The coordinator writes its own
        # `s2_transition_attempt` event into the store events table;
        # this event_log is a separate JSONL audit trail consumed by the
        # doctor + identity-audit surfaces. Both ledgers update on every
        # successful transition.
        self._event_log.append(
            {
                "event": "state_transition",
                "from": from_state.value,
                "to": new_state.value,
                "trigger": resolved_reason,
            }
        )

        # Shadow-run guard for HIBERNATION (legacy behaviour preserved
        # for test scaffolding; production daemons run shadow_run=False).
        if new_state is LifecycleState.HIBERNATION and self._shadow_run:
            self._event_log.append(
                {
                    "event": "shadow_run_warning",
                    "would_action": "hibernate_kill_process",
                    "blocked_by": "shadow_run=True",
                    "note": (
                        "shadow_run=True is a test-only legacy guard "
                        "preserved for transition tests; production "
                        "daemons run with shadow_run=False where this "
                        "branch never fires."
                    ),
                }
            )

        return new_state
