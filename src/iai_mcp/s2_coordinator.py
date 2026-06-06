"""S2 anti-oscillation coordinator.

Single asyncio.Lock + monotonic version counter + ring-buffer oscillation
detection. Owns the on-disk lifecycle-state write (the sole production
assignment site for the persisted state field lives here).

Closes the daemon_state / lifecycle_state read-modify-write race. All
lifecycle FSM transition sites
(wake-on-request, sleep-on-idle, hibernate-on-deep-idle, drowsy-on-mild-idle,
fsm_reconcile correction) funnel through `S2Coordinator.transition(...)`.

Design rationale:
    * Separate module from `lifecycle_state.py` — persistence vs. transition
      layer separation of concerns.
    * `asyncio.Lock` only — a sync-thread mutex primitive would deadlock the
      async event loop.
    * Monotonic in-memory `version` counter; NOT persisted. Restarts at 0 on
      daemon respawn; the on-disk record has its own `schema_version`
      unrelated to this.
    * Compare-and-swap on `from_state`: actual != expected raises
      `S2OscillationConflict` BEFORE attempting any write.
    * Ring-buffer N=8 (hard-coded) of recent transitions. Match if the new
      attempt is the reverse direction of a prior entry within
      `MIN_INTERVAL_SEC` (default 5.0s, env-var configurable).
    * Conflict / block exceptions are normal control flow — callers catch
      both explicitly per the dispatch pattern. The coordinator emits the
      matching event on every emit-worthy path.

Event contract:
    * `s2_transition_attempt` — EXACTLY ONE per `transition()` call.
    * `s2_oscillation_blocked` — emitted on every detected oscillation
      (whether enforced or dry-run).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iai_mcp.events import write_event
from iai_mcp.lifecycle_state import (
    LIFECYCLE_STATE_PATH,
    LifecycleState,
    LifecycleStateRecord,
    load_state,
    save_state,
)

# Default ring-buffer size. Hard-coded — NOT env-var configurable
# (N=8 covers the last few seconds of activity at typical FSM transition
# rates of 1-2 per minute baseline; changing it without also changing
# MIN_INTERVAL_SEC would create surprising detection-window semantics).
_RING_BUFFER_DEFAULT_SIZE: int = 8


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp; central so tests can monkey-patch.

    Mirrors `iai_mcp.lifecycle_state._utc_now_iso` verbatim so the
    `since_ts` field this coordinator writes is identical in shape to
    values produced by the persistence layer itself.
    """
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Exceptions — normal control flow
# ---------------------------------------------------------------------------


class S2OscillationConflict(RuntimeError):
    """Raised by `S2Coordinator.transition` when actual on-disk state != `from_state`.

    Carries the actual state on `.actual_state` so callers can decide
    whether their transition is still meaningful given the new ground
    truth. Subclasses `RuntimeError` so callers that only want to swallow
    "transition didn't go through" can catch one base; the specific
    subclass tells them why.
    """

    actual_state: LifecycleState
    attempted_from: LifecycleState
    attempted_to: LifecycleState

    def __init__(
        self,
        *,
        actual_state: LifecycleState,
        attempted_from: LifecycleState,
        attempted_to: LifecycleState,
    ) -> None:
        self.actual_state = actual_state
        self.attempted_from = attempted_from
        self.attempted_to = attempted_to
        super().__init__(
            f"S2 CAS conflict: tried "
            f"{attempted_from.value}->{attempted_to.value} but actual="
            f"{actual_state.value}"
        )


class S2OscillationBlocked(RuntimeError):
    """Raised by `S2Coordinator.transition` when the reverse of `(from_state, to_state)`
    was recorded inside the ring buffer within `MIN_INTERVAL_SEC`.

    Carries the prior transition record + interval on `.first_transition`,
    `.second_transition`, and `.interval_sec` attributes for caller
    logging. NOT raised in dry-run mode (the `s2_oscillation_blocked`
    event still fires with `dry_run_mode=True`, but the transition is
    allowed to proceed in dry-run mode).
    """

    first_transition: dict
    second_transition: dict
    interval_sec: float

    def __init__(
        self,
        *,
        first_transition: dict,
        second_transition: dict,
        interval_sec: float,
    ) -> None:
        self.first_transition = first_transition
        self.second_transition = second_transition
        self.interval_sec = interval_sec
        super().__init__(
            f"S2 oscillation blocked: reverse of "
            f"{first_transition['from_state']}->{first_transition['to_state']}"
            f" within {interval_sec:.3f}s (MIN_INTERVAL_SEC)"
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class S2Coordinator:
    """Serialize every lifecycle FSM transition through one asyncio.Lock.

    Construction is cheap: no IO, no daemon-socket touch, no event-store
    touch. The first `transition()` call inside the lock does load_state
    + the CAS check + ring-buffer check + save_state + version-increment
    + event-emit — all under the single `asyncio.Lock`, so concurrent
    callers from different coroutines queue on the lock and never
    interleave their reads / writes.

    Attributes
    ----------
    lock:
        `asyncio.Lock` instance; tests inspect it for type.
    version:
        Monotonic `int` counter, starts at 0; increments by +1 on every
        SUCCESSFUL transition (NOT on CAS conflict, NOT on enforced
        oscillation block). NOT persisted to disk — survives only the
        daemon process lifetime; respawn starts at 0 again.

    Private attributes
    ------------------
    `_store` MemoryStore | None — passed to `write_event`. May be None in
    tests that don't need event verification; emission failure is
    swallowed via try/except so that path stays safe.

    `_state_path` Path — defaults to LIFECYCLE_STATE_PATH; tests pass a
    `tmp_path / "lifecycle_state.json"`.

    `_min_interval_sec` float — oscillation detection window, injected
    via `_load_s2_config()`.

    `_dry_run` bool — when True, oscillation matches emit the block event
    with `dry_run_mode=True` but allow the transition to proceed.

    `_ring_buffer` deque[dict] — last N transition records (default N=8).
    Each entry: `{from_state, to_state, reason, ts_monotonic}`.
    """

    def __init__(
        self,
        *,
        store: Any,
        state_path: Path | None = None,
        legacy_path: Path | None = None,
        min_interval_sec: float = 5.0,
        dry_run: bool = False,
        ring_buffer_size: int = _RING_BUFFER_DEFAULT_SIZE,
    ) -> None:
        self._store = store
        self._state_path: Path = (
            state_path if state_path is not None else LIFECYCLE_STATE_PATH
        )
        # Optional legacy-mirror path for lock-step writes.  When set,
        # every successful transition also updates the legacy
        # .daemon-state.json fsm_state field so the drift detector does
        # not fire on normal operation.  Defaults to the production
        # location under ~/.iai-mcp/; pass an explicit path in tests.
        if legacy_path is not None:
            self._legacy_path: Path | None = legacy_path
        else:
            from iai_mcp.daemon_state import STATE_PATH as _LEGACY_STATE_PATH
            self._legacy_path = _LEGACY_STATE_PATH
        # asyncio.Lock — never a sync-thread mutex (would deadlock the
        # async event loop).
        self.lock: asyncio.Lock = asyncio.Lock()
        self.version: int = 0
        self._min_interval_sec: float = float(min_interval_sec)
        self._dry_run: bool = bool(dry_run)
        self._ring_buffer: deque[dict] = deque(maxlen=ring_buffer_size)

    async def transition(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
    ) -> LifecycleState:
        """Acquire lock; load_state; CAS-check; ring-buffer-check; persist; emit.

        Single serialisation point for every lifecycle FSM transition.

        Parameters
        ----------
        from_state:
            Expected current state (compare-and-swap reference). If the
            actual on-disk state doesn't match this, raises
            `S2OscillationConflict` BEFORE attempting any write.
        to_state:
            Desired new state.
        reason:
            Short snake-case identifier (e.g. `"wake_on_mcp_request"`,
            `"sleep_on_idle_30min"`, `"hibernate_on_deep_idle"`,
            `"fsm_reconcile_correction"`). Recorded in the
            `s2_transition_attempt` event body and in the ring buffer.

        Returns
        -------
        LifecycleState — the new state (`to_state`) on success.

        Raises
        ------
        S2OscillationConflict
            actual on-disk `current_state` != `from_state`.
        S2OscillationBlocked
            reverse of `(from_state, to_state)` was recorded within
            `MIN_INTERVAL_SEC` AND `dry_run` is False. In dry-run mode
            the block event still fires with `dry_run_mode=True` but
            the transition IS allowed to proceed.
        """
        # Snapshot version_before + monotonic clock BEFORE the lock so the
        # value emitted in events matches the value the caller observed
        # at call time. (Acquiring the lock can queue arbitrarily long if
        # another coroutine is mid-transition; reading these inside the
        # lock would race the event-body semantics against lock-wait
        # latency.)
        version_before = self.version
        now_mono = time.monotonic()

        async with self.lock:
            # 1. Load the persisted record. `load_state` returns a
            # default WAKE record if the file is absent or malformed
            # (lifecycle_state.py self-heal contract), so we always get
            # a valid LifecycleState back.
            rec: LifecycleStateRecord = load_state(self._state_path)
            actual_state = LifecycleState(rec["current_state"])

            # 2. CAS check. Mismatch raises BEFORE any write.
            # Emit exactly one s2_transition_attempt with succeeded=False
            # + conflict_reason="cas_mismatch", then raise.
            if actual_state != from_state:
                cas_body = {
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "version_before": version_before,
                    "version_after": version_before,
                    "succeeded": False,
                    "conflict_reason": "cas_mismatch",
                }
                try:
                    write_event(
                        self._store,
                        "s2_transition_attempt",
                        cas_body,
                        severity="info",
                    )
                except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                    # Event-store failure must never shadow load-bearing
                    # FSM semantics. Swallow and proceed to raise the CAS
                    # conflict — the caller's exception handler sees the
                    # real story.
                    pass
                raise S2OscillationConflict(
                    actual_state=actual_state,
                    attempted_from=from_state,
                    attempted_to=to_state,
                )

            # 3. Ring-buffer oscillation check. Walk the buffer newest-first;
            # match the reverse direction of the current attempt within
            # MIN_INTERVAL_SEC. Break on first match (the most recent
            # reverse hit is the one we care about for the first_transition
            # payload).
            oscillation_first: dict | None = None
            for entry in reversed(self._ring_buffer):
                if (
                    entry["from_state"] == to_state.value
                    and entry["to_state"] == from_state.value
                    and (now_mono - entry["ts_monotonic"]) < self._min_interval_sec
                ):
                    oscillation_first = entry
                    break

            if oscillation_first is not None:
                second_transition = {
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "ts_monotonic": now_mono,
                }
                interval_sec = now_mono - oscillation_first["ts_monotonic"]
                block_body = {
                    "first_transition": oscillation_first,
                    "second_transition": second_transition,
                    "interval_sec": interval_sec,
                    "dry_run_mode": self._dry_run,
                }
                # s2_oscillation_blocked event ALWAYS fires on detected
                # oscillation, whether enforced or dry-run.
                try:
                    write_event(
                        self._store,
                        "s2_oscillation_blocked",
                        block_body,
                        severity="warning",
                    )
                except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                    pass

                # EXACTLY ONE s2_transition_attempt per call: emit the
                # attempt event ONLY on the enforced (non-dry-run) branch;
                # on dry-run we fall through to the success path which
                # emits the single attempt with succeeded=True.
                if not self._dry_run:
                    block_attempt_body = {
                        "from_state": from_state.value,
                        "to_state": to_state.value,
                        "reason": reason,
                        "version_before": version_before,
                        "version_after": version_before,
                        "succeeded": False,
                        "conflict_reason": "oscillation_blocked",
                    }
                    try:
                        write_event(
                            self._store,
                            "s2_transition_attempt",
                            block_attempt_body,
                            severity="info",
                        )
                    except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                        pass
                    raise S2OscillationBlocked(
                        first_transition=oscillation_first,
                        second_transition=second_transition,
                        interval_sec=interval_sec,
                    )
                # Dry-run: fall through to step 4 (apply transition).
                # The s2_transition_attempt event fires there with
                # succeeded=True.

            # 4. Apply transition. This is the sole production assignment
            # site for the persisted current_state field.
            rec["current_state"] = to_state.value
            rec["since_ts"] = _utc_now_iso()
            save_state(rec, self._state_path)

            # 4a. Lock-step legacy mirror write.
            # Keep the legacy .daemon-state.json fsm_state in sync with
            # the canonical current_state so the drift detector never
            # fires on normal operation.  The mapping table lives in
            # fsm_reconcile._CANONICAL_TO_LEGACY (same table the
            # auto-corrector uses).  Failure is non-fatal — the per-tick
            # reconcile with auto_correct=True is still the safety net.
            if self._legacy_path is not None:
                try:
                    from iai_mcp.fsm_reconcile import _auto_correct_legacy
                    _auto_correct_legacy(self._legacy_path, to_state.value)
                except Exception:  # noqa: BLE001 -- mirror write is best-effort
                    pass

            # 5. Update in-memory bookkeeping: ring buffer + version.
            # Ring buffer keeps the SUCCEEDED transitions only; that is
            # what the oscillation detector compares against (we don't
            # want a rejected attempt to itself trigger a future block).
            self._ring_buffer.append(
                {
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "ts_monotonic": now_mono,
                }
            )
            self.version += 1
            version_after = self.version

            # 6. Emit the success attempt event (the single one per call
            # on the success path, or the single one on the dry-run-
            # oscillation path).
            success_body = {
                "from_state": from_state.value,
                "to_state": to_state.value,
                "reason": reason,
                "version_before": version_before,
                "version_after": version_after,
                "succeeded": True,
                "conflict_reason": None,
            }
            try:
                write_event(
                    self._store,
                    "s2_transition_attempt",
                    success_body,
                    severity="info",
                )
            except Exception:  # noqa: BLE001 -- event emit must never crash FSM
                pass

            return to_state

    async def set_crisis_mode(self, value: bool, reason: str) -> None:
        """Mutate persisted crisis_mode field under the S2 lock.

        Parallel to transition() but for the crisis_mode bool field. Does
        NOT mutate current_state — S2Coordinator is the sole writer of
        current_state.

        Acquires self.lock, loads the record, sets rec['crisis_mode'] =
        value, persists via save_state, increments self.version by +1.
        The caller emits the matching essential_variable_breach or
        crisis_recluster_pass event around this call — this method emits
        no event of its own.

        Parameters
        ----------
        value:
            New crisis_mode value (True on first breach, False on
            CRISIS_RECLUSTER completion).
        reason:
            Short snake-case reason (e.g. 'essential_variable_breach:rich_club_ratio',
            'crisis_recluster_complete'). Currently consumed only by
            caller logging; this method does not persist it (the field
            is bool-only). Signature parity with transition() keeps
            call-site greps clean.
        """
        # No CAS check: crisis_mode is bool, not an FSM state, and
        # set-to-same-value is idempotent (the persist still happens but
        # the on-disk content is unchanged). No ring-buffer oscillation
        # check: True->False->True flow is the expected behaviour for
        # every crisis cycle.
        async with self.lock:
            rec: LifecycleStateRecord = load_state(self._state_path)
            rec["crisis_mode"] = bool(value)
            save_state(rec, self._state_path)
            self.version += 1
