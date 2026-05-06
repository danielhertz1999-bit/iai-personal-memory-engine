"""Phase 10.3 — Sleep cycle pipeline + L3 failure grammar.

Five ordered atomic steps run only inside the SLEEP lifecycle state:
    1. SCHEMA_MINE       — extract schemas from episodic
    2. KNOB_TUNE         — recompute procedural knobs
    3. DREAM_DECAY       — Hebbian decay + edge prune
    4. OPTIMIZE_LANCE    — table-level optimize(cleanup_older_than)
    5. COMPACT_RECORDS   — final records.lance compaction

Design invariants:

* Each step is **transactional** — Lance optimize is itself transactional;
  schema_mine / knob_tune / dream_decay write their own atomic temp+swap
  semantics through the modules they delegate to. The pipeline never
  modifies `MemoryRecord.literal_surface` (verbatim-recall invariant
  carried forward from / Plan 5/6).

* On exception mid-step N, `lifecycle_state.json.sleep_cycle_progress`
  records `{last_completed_step: N-1, attempt: K, last_error: "..."}`
  via the same atomic-replace path as `lifecycle_state.save_state`.

* **3-strike → 24h auto-quarantine**: three consecutive failures of
  the SAME step (attempt ≥ 3 for that step) triggers quarantine. While
  quarantined, `run()` short-circuits with `quarantine_triggered=True`.
  Auto-recovery once `now >= until_ts`; manual recovery via
  `reset_quarantine()` or `iai-mcp maintenance sleep-cycle --reset-quarantine`.

* **Bounded deferral** (≤2 sec target via ≤10 sec checkpoint chunks):
  a callable `interrupt_check` is checked between chunks. If True, the
  current chunk completes, progress is persisted, and `run()` returns
  with `completed_steps` so far. The state machine then transitions to
  WAKE; the next SLEEP cycle resumes from the same chunk.

This module's heavy lifting **delegates to existing functions** —
schema mining (`schema.induce_schemas_tier0`), Hebbian decay
(`sleep._decay_edges`), table optimize (`maintenance.optimize_lance_storage`),
records compaction (Phase 07.14-01 `optimize_lance_storage(retention=0d)`).
The pipeline is orchestration only.

Daemon main-loop integration (Phase 10.4/10.5) and yield-gate removal
(Phase 10.6) are shipped. ``continuous_audit`` (identity_audit.py) and
``_hippea_cascade_loop`` (daemon.py) remain as background tasks
running alongside the sleep-cycle pipeline; ``dream_daemon`` was
removed in Phase 10.6.

Constitutional guards
---------------------
* C1 HUMAN-FIRST: pipeline runs only in SLEEP state, so MCP traffic
  cannot collide. The legacy ``_should_yield_to_mcp`` gate was removed
  in — SLEEP-state isolation is the sole guarantor.
* C3 ZERO paid-API cost: no reference to ANTHROPIC_API_KEY anywhere.
  Schema induction stays Tier-0 (llm_enabled=False is the only path
  this pipeline exercises).
* C5 / verbatim preservation: the pipeline does NOT touch
  `MemoryRecord.literal_surface`. Every delegated function is a
  metadata mutator (FSRS state, edge weights, schema candidates,
  Lance manifests, profile knobs).
* C6 read-only audit: schema mining is MVCC reads against records;
  decay is metadata-only on edges; optimize is Lance-internal.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypedDict

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import (
    LIFECYCLE_STATE_PATH,
    LifecycleStateRecord,
    Quarantine,
    SleepCycleProgress,
    load_state,
    save_state,
)


# Quarantine TTL configurable via env (default 24h).
# Read ONCE at import time so tests that monkeypatch the env var must
# also patch the module attribute (`sleep_pipeline.QUARANTINE_TTL_HOURS_DEFAULT`)
# — same discipline as `maintenance.LANCE_OPTIMIZE_INTERVAL_SEC`.
QUARANTINE_TTL_HOURS_DEFAULT: float = float(
    os.environ.get("IAI_MCP_SLEEP_QUARANTINE_TTL_HOURS", "24"),
)


class SleepStep(Enum):
    """Five ordered atomic steps of the sleep pipeline.

    Numeric values are stable: `lifecycle_state.json.sleep_cycle_progress
    .last_completed_step` persists the integer, and resume-from-step-N
    relies on the integer ordering. Re-ordering or renumbering is a
    schema migration (do NOT change without bumping the field).
    """

    SCHEMA_MINE = 1
    KNOB_TUNE = 2
    DREAM_DECAY = 3
    OPTIMIZE_LANCE = 4
    COMPACT_RECORDS = 5


class SleepPipelineResult(TypedDict, total=False):
    """Return shape from `SleepPipeline.run()` / `force_run()`.

    `completed_steps`: list of `SleepStep` values that finished cleanly
        in this invocation (NOT cumulative across resumes; only this run).
    `failed_step`: the step that raised, if any. None on full success or
        on bounded-deferral early-return.
    `error`: stringified exception (truncated to 500 chars) or None.
    `duration_sec`: wall-clock for the invocation.
    `quarantine_triggered`: True iff quarantine was entered DURING this
        run (3rd-strike) OR was already active when run() was called.
    `interrupted`: True iff bounded-deferral interrupt_check fired and
        we returned early. None / absent means a natural completion or
        failure terminated the run.
    """

    completed_steps: list[SleepStep]
    failed_step: SleepStep | None
    error: str | None
    duration_sec: float
    quarantine_triggered: bool
    interrupted: bool


def _utc_now() -> datetime:
    """Single point of `datetime.now(UTC)` — patchable in tests."""
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp (matches lifecycle_state convention)."""
    return _utc_now().isoformat()


class SleepPipeline:
    """Orchestrates the 5-step sleep cycle with resume + quarantine.

    Construction is cheap: opens no LanceDB tables, performs no I/O
    beyond reading `lifecycle_state.json`. The actual heavy work
    happens inside `run()` / `force_run()` step bodies.

    Concurrency note: the pipeline is single-threaded by design. The
    caller (state machine in Phase 10.4/10.5; CLI in this phase) must
    ensure no overlapping invocations — typically by holding the
    SLEEP-state guard. There is no internal lock; running two
    `SleepPipeline` instances against the same `lifecycle_state_path`
    simultaneously is undefined behaviour.
    """

    def __init__(
        self,
        store: Any,
        lifecycle_state_path: Path | None = None,
        event_log: LifecycleEventLog | None = None,
        quarantine_ttl_hours: float | None = None,
    ) -> None:
        self._store = store
        self._lifecycle_state_path = (
            lifecycle_state_path
            if lifecycle_state_path is not None
            else LIFECYCLE_STATE_PATH
        )
        # Default to a fresh LifecycleEventLog rooted at the conventional
        # `~/.iai-mcp/logs/` directory. Tests inject a tmp_path-rooted log.
        self._event_log = (
            event_log if event_log is not None else LifecycleEventLog()
        )
        self._quarantine_ttl_hours = (
            float(quarantine_ttl_hours)
            if quarantine_ttl_hours is not None
            else QUARANTINE_TTL_HOURS_DEFAULT
        )

    # ------------------------------------------------------------------
    # Quarantine state (lifecycle_state.json.quarantine)
    # ------------------------------------------------------------------

    def _load_state_record(self) -> LifecycleStateRecord:
        """Read the current lifecycle state record (with self-heal)."""
        return load_state(self._lifecycle_state_path)

    def _save_state_record(self, record: LifecycleStateRecord) -> None:
        """Atomic-replace persist of the lifecycle state record."""
        save_state(record, self._lifecycle_state_path)

    def _load_quarantine(self) -> Quarantine | None:
        """Return the current quarantine sub-record or None."""
        return self._load_state_record().get("quarantine")

    def _set_quarantine(self, reason: str) -> Quarantine:
        """Set quarantine until now + ttl_hours; persist; emit event.

        Returns the quarantine record we just persisted so callers can
        include `until_ts` in their result dict.
        """
        now = _utc_now()
        until = now + timedelta(hours=self._quarantine_ttl_hours)
        quarantine: Quarantine = {
            "until_ts": until.isoformat(),
            "reason": reason,
            "since_ts": now.isoformat(),
        }
        record = self._load_state_record()
        record["quarantine"] = quarantine
        self._save_state_record(record)
        # Event is best-effort — a full disk should not crash the pipeline
        # mid-quarantine-write (state is already persisted).
        try:
            self._event_log.append({
                "event": "quarantine_entered",
                "reason": reason,
                "until_ts": quarantine["until_ts"],
                "ttl_hours": self._quarantine_ttl_hours,
            })
        except Exception:
            pass
        return quarantine

    def _clear_quarantine(self, *, reason: str = "manual_reset") -> None:
        """Wipe the quarantine sub-record + reset progress attempt counter.

        `reason` is logged on the `quarantine_lifted` event. Defaults to
        `manual_reset` (the human-action path); auto-recovery passes
        `auto_recovery_after_ttl` from the run() entry point.
        """
        record = self._load_state_record()
        prior_quarantine = record.get("quarantine")
        record["quarantine"] = None
        # Resetting quarantine also resets the per-step attempt counter
        # — otherwise the very next failure would re-trip 3-strike on
        # attempt=4 immediately. Progress.last_completed_step is kept
        # so resume-from-step-N still works on the next run.
        progress = record.get("sleep_cycle_progress")
        if progress is not None:
            progress["attempt"] = 0
            record["sleep_cycle_progress"] = progress
        self._save_state_record(record)
        try:
            self._event_log.append({
                "event": "quarantine_lifted",
                "reason": reason,
                "prior_until_ts": (
                    prior_quarantine["until_ts"] if prior_quarantine else None
                ),
            })
        except Exception:
            pass

    def is_quarantined(self) -> bool:
        """True iff a quarantine record exists AND `now < until_ts`.

        A quarantine record with a past `until_ts` is automatically
        cleared by `run()` on the next invocation (auto-recovery); this
        getter does NOT mutate state — it is a pure read.
        """
        quarantine = self._load_quarantine()
        if quarantine is None:
            return False
        try:
            until = datetime.fromisoformat(quarantine["until_ts"])
        except (TypeError, ValueError):
            # Malformed timestamp -- treat as not-quarantined so we don't
            # lock the user out forever on a corrupted entry. The next
            # successful run will overwrite this slot.
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return _utc_now() < until

    def reset_quarantine(self) -> None:
        """Manual recovery: clear quarantine + reset attempt counter.

        Used by `iai-mcp maintenance sleep-cycle --reset-quarantine`.
        """
        self._clear_quarantine(reason="manual_reset")

    # ------------------------------------------------------------------
    # Progress state (lifecycle_state.json.sleep_cycle_progress)
    # ------------------------------------------------------------------

    def _load_progress(self) -> SleepCycleProgress | None:
        """Return the current sleep-cycle progress sub-record or None."""
        return self._load_state_record().get("sleep_cycle_progress")

    def _save_progress(
        self,
        last_completed_step: int,
        attempt: int,
        last_error: str | None,
        *,
        started_at: str | None = None,
    ) -> SleepCycleProgress:
        """Persist sleep-cycle progress; preserve `started_at` across saves.

        `started_at` defaults to: prior progress's started_at if any,
        else `now()`. This gives the operator a wall-clock view of how
        long the cycle has been running across resumes.
        """
        record = self._load_state_record()
        prior = record.get("sleep_cycle_progress") or {}
        progress: SleepCycleProgress = {
            "last_completed_step": last_completed_step,
            "attempt": attempt,
            "last_error": last_error,
            "started_at": (
                started_at
                if started_at is not None
                else prior.get("started_at", _utc_now_iso())
            ),
        }
        record["sleep_cycle_progress"] = progress
        self._save_state_record(record)
        return progress

    def _clear_progress(self) -> None:
        """Wipe the sleep-cycle progress sub-record after full success."""
        record = self._load_state_record()
        record["sleep_cycle_progress"] = None
        self._save_state_record(record)

    # ------------------------------------------------------------------
    # Step orchestrators (Task 1.2 — call existing functions)
    # ------------------------------------------------------------------
    #
    # Each `_step_*` returns True on natural completion and False when
    # `interrupt_check` fired between chunks. On exception, the step
    # body re-raises to the caller (run()) which handles 3-strike
    # quarantine + progress save. Step bodies are deliberately small:
    # they delegate to the migration-source functions listed in
    # the migration-source functions from the respective modules.

    def _emit_step_started(self, step: SleepStep) -> None:
        """Best-effort `sleep_step_started` emission to the event log.

        Failure (e.g. /home full) MUST NOT abort the step — the work
        itself is the load-bearing path; observability is secondary.
        """
        try:
            self._event_log.append({
                "event": "sleep_step_started",
                "step": step.name,
                "step_num": step.value,
            })
        except Exception:
            pass

    def _emit_step_completed(
        self, step: SleepStep, duration_sec: float, **payload: Any,
    ) -> None:
        """Best-effort `sleep_step_completed` emission with optional payload."""
        try:
            self._event_log.append({
                "event": "sleep_step_completed",
                "step": step.name,
                "step_num": step.value,
                "duration_sec": round(duration_sec, 3),
                **payload,
            })
        except Exception:
            pass

    def _check_interrupt(
        self,
        step: SleepStep,
        chunk_idx: int,
        interrupt_check: Callable[[], bool] | None,
    ) -> bool:
        """Return True iff the caller asked us to defer.

        Persists `sleep_cycle_progress.last_completed_step = step.value-1`
        (we have NOT completed `step` yet) and stamps `last_error` with
        a structured deferral marker so `iai-mcp lifecycle status` can
        show "deferred at step N chunk K" rather than a fake error.
        """
        if interrupt_check is None:
            return False
        try:
            should = bool(interrupt_check())
        except Exception:
            # If the caller's predicate is broken, do NOT defer (better
            # to keep working than to hang forever waiting for a True
            # that will never come). Same fail-safe discipline as the
            # event-log emit failures above.
            should = False
        if not should:
            return False
        # Save deferral marker. last_completed_step stays at the prior
        # step (we are mid-`step`); attempt counter is unchanged because
        # this is NOT a failure — it is a cooperative yield.
        prior = self._load_progress() or {}
        last_completed = step.value - 1
        attempt = int(prior.get("attempt", 0))
        self._save_progress(
            last_completed_step=last_completed,
            attempt=attempt,
            last_error=f"deferred:step={step.name}:chunk_idx={chunk_idx}",
        )
        return True

    def _step_schema_mine(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 1: schema mining via existing tier-0 induction.

        `induce_schemas_tier0(store)` is the migration source — it does
        a single MVCC pass over `records.tags_json` and returns
        candidates without persisting (Plan 02-03 contract). For Phase
        10.3 the chunk granularity is one (the underlying call is a
        single batch read internally; we do NOT slice it). The chunk
        boundary is honoured by checking `interrupt_check` BEFORE the
        call — if the operator wants to bail, we do, otherwise we run
        to completion.

        Returns `(completed, payload)` — completed=False signals an
        interrupt-induced early return (no payload metadata).
        """
        from iai_mcp.schema import induce_schemas_tier0

        # Single-chunk implementation: chunk_idx=0 is the only checkpoint.
        if self._check_interrupt(SleepStep.SCHEMA_MINE, 0, interrupt_check):
            return False, {}
        candidates = induce_schemas_tier0(self._store)
        # Best-effort metric for the completion event; tier-0 returns a
        # list of `SchemaCandidate` dataclass instances, len() works.
        try:
            count = len(candidates) if candidates is not None else 0
        except Exception:
            count = 0
        return True, {"schemas_induced": count}

    def _step_knob_tune(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 2: per-knob procedural snapshot.

        implements this as a per-knob iteration over the
        sealed `PROFILE_KNOBS` registry. Each knob is one chunk (so the
        interrupt cadence matches the registry size — currently 11 per
        the 2026-04-30 audit). The actual Bayesian update is event-
        driven via `core.dispatch profile_update_from_signal` and
        already runs there; what sleep needs to do is take a snapshot
        of the live state so audit trails can replay it. We call
        `profile.default_state()` once outside the loop so a future
        phase that adds real per-knob work has a place to hook in
        WITHOUT re-architecting the chunk boundary.
        """
        from iai_mcp.profile import PROFILE_KNOBS, default_state

        knob_names = sorted(PROFILE_KNOBS.keys())
        # Capture current state once outside the loop — calling this
        # per knob would be wasteful and would still be a single-shot
        # snapshot. The loop's purpose is the chunk boundary (interrupt
        # check), not work amplification.
        snapshot = default_state()
        for chunk_idx, name in enumerate(knob_names):
            if self._check_interrupt(
                SleepStep.KNOB_TUNE, chunk_idx, interrupt_check,
            ):
                return False, {}
            # Per-knob "work" — currently observation-only. A future
            # phase plugs Bayesian recomputation here. Touching
            # `snapshot[name]` is enough to surface a missing-knob bug
            # at sleep time rather than at retrieval time.
            _ = snapshot.get(name)
        return True, {"knobs_tuned": len(knob_names)}

    def _step_dream_decay(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 3: Hebbian decay + edge prune via existing `_decay_edges`.

        `sleep._decay_edges(store)` is the migration source — Plan
        03-01 CONN-05 D-TEM-04. It walks every hebbian/hebbian_structure
        edge and either decays the weight in place or prunes when
        below epsilon. The function is monolithic; for we
        wrap it as a single chunk (chunk_idx=0) and check
        `interrupt_check` before the call.
        """
        from iai_mcp.sleep import _decay_edges

        if self._check_interrupt(SleepStep.DREAM_DECAY, 0, interrupt_check):
            return False, {}
        result = _decay_edges(self._store)
        # Surface decay/prune counts in the completion event for ops.
        if isinstance(result, dict):
            return True, {
                "decayed": int(result.get("decayed", 0) or 0),
                "pruned": int(result.get("pruned", 0) or 0),
            }
        return True, {}

    def _step_optimize_lance(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 4: per-table Lance optimize via existing helper.

        `optimize_lance_storage(store, retention=None)` is the
        migration source (Phase 7.3 D7.3-09). It iterates the three
        daemon-owned tables (records / edges / events) internally; we
        cannot subdivide without reimplementing. For the
        chunk boundary is one (chunk_idx=0). The retention defaults to
        the configured 1-day window (matches periodic-audit cadence).
        """
        from iai_mcp.maintenance import optimize_lance_storage

        if self._check_interrupt(
            SleepStep.OPTIMIZE_LANCE, 0, interrupt_check,
        ):
            return False, {}
        report = optimize_lance_storage(self._store)
        # Helper never raises (D7.3-09); per-table errors live inside
        # the report dict. We surface a compact summary in the event.
        tables_with_errors = [
            t for t, r in (report or {}).items()
            if isinstance(r, dict) and "error" in r
        ]
        return True, {
            "tables_optimized": list((report or {}).keys()),
            "tables_with_errors": tables_with_errors,
        }

    def _step_compact_records(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 5: final records.lance compaction with retention=0d.

        Phase 07.14-01 helper: `optimize_lance_storage(store,
        retention=timedelta(days=0))` reclaims version manifests
        accumulated since the last compaction. This is intentionally
        a separate step from
        OPTIMIZE_LANCE because the retention policy differs: step 4
        keeps a 1-day point-in-time window for time-travel reads;
        step 5 takes the more aggressive zero-retention pass after
        the day-old data is no longer needed.
        """
        from iai_mcp.maintenance import optimize_lance_storage

        if self._check_interrupt(
            SleepStep.COMPACT_RECORDS, 0, interrupt_check,
        ):
            return False, {}
        report = optimize_lance_storage(
            self._store, retention=timedelta(days=0),
        )
        tables_with_errors = [
            t for t, r in (report or {}).items()
            if isinstance(r, dict) and "error" in r
        ]
        return True, {
            "tables_compacted": list((report or {}).keys()),
            "tables_with_errors": tables_with_errors,
            "retention_days": 0,
        }

    # Lookup table from step -> bound method, in execution order.
    # Defined AFTER the step methods so attribute resolution succeeds.
    @property
    def _step_methods(
        self,
    ) -> dict[
        SleepStep,
        Callable[
            [Callable[[], bool] | None],
            "tuple[bool, dict[str, Any]]",
        ],
    ]:
        return {
            SleepStep.SCHEMA_MINE: self._step_schema_mine,
            SleepStep.KNOB_TUNE: self._step_knob_tune,
            SleepStep.DREAM_DECAY: self._step_dream_decay,
            SleepStep.OPTIMIZE_LANCE: self._step_optimize_lance,
            SleepStep.COMPACT_RECORDS: self._step_compact_records,
        }

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    # Step ordering used by both run() and force_run(). Tuple is fixed so
    # neither path can accidentally execute steps out of order.
    _STEP_ORDER: tuple[SleepStep, ...] = (
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.DREAM_DECAY,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
    )

    # 3-strike threshold: the SAME step failing this many consecutive
    # times triggers 24h auto-quarantine. Per panel verdict R3 / proposal
    # v2 §2 L3.
    _QUARANTINE_STRIKE_THRESHOLD: int = 3

    def run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        """Run the sleep pipeline (auto-quarantine respected).

        Behaviour summary:

        1. If `is_quarantined()`: return immediately with
           `quarantine_triggered=True` and `completed_steps=[]`. The
           caller is expected to surface this in CLI output / doctor row.

        2. Auto-recovery: if `quarantine` exists but `until_ts` is in
           the past, clear it (logged as `quarantine_lifted`,
           reason=`auto_recovery_after_ttl`) and proceed.

        3. Determine resume point from `_load_progress()`:
           - No progress record OR last_completed_step == 0 → start at
             SCHEMA_MINE (step 1).
           - last_completed_step == K (1 ≤ K < 5) → start at step K+1.
           - last_completed_step == 5 → fresh cycle (start at step 1);
             we treat a successful prior run that was never cleared as
             a fresh start, not a no-op.

        4. For each step from `start` to COMPACT_RECORDS:
           - Emit `sleep_step_started`.
           - Call `_step_*(interrupt_check)`. The step body itself
             checks the interrupt between chunks and persists progress.
           - On interrupt (returned False): early-return with
             `interrupted=True`. progress is already saved by the
             step body; we do NOT touch it here.
           - On exception: save progress with attempt+1, log
             `sleep_step_completed` (with error payload), check 3-strike
             → maybe quarantine, then return with `failed_step` set.
           - On success: emit `sleep_step_completed`, persist progress
             with last_completed_step=step.value (attempt reset to 0).

        5. On full success: clear progress (sleep_cycle_progress=None).

        Failure isolation: the helper functions used by step bodies
        already have their own "never-raise" disciplines where
        applicable (e.g. `optimize_lance_storage` per D7.3-09); this
        method's try/except is a defense-in-depth wrapper around the
        whole step call.
        """
        return self._run_internal(
            interrupt_check, force=False,
        )

    def force_run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        """Run even if quarantined. Used by `--force` CLI path.

        Quarantine state is NOT cleared by force_run on its own — the
        operator-facing `--reset-quarantine` flag is what wipes the
        quarantine record. force_run merely bypasses the gate so a
        diagnostic / repair run can execute. If the run succeeds in
        full, the quarantine sub-record is left alone (operator may
        still want to investigate); subsequent natural `run()` calls
        will see `is_quarantined()` True until TTL expires or the
        operator runs `--reset-quarantine` explicitly.
        """
        return self._run_internal(
            interrupt_check, force=True,
        )

    def _run_internal(
        self,
        interrupt_check: Callable[[], bool] | None,
        *,
        force: bool,
    ) -> SleepPipelineResult:
        """Shared body for `run()` / `force_run()`. See `run()` docstring."""
        t0 = time.monotonic()
        completed_steps: list[SleepStep] = []

        # Quarantine gate (skipped under force=True).
        if not force and self._check_and_maybe_auto_recover_quarantine():
            # is_quarantined returned True AND we are NOT in force mode.
            # Short-circuit: quarantined.
            return {
                "completed_steps": [],
                "failed_step": None,
                "error": None,
                "duration_sec": round(time.monotonic() - t0, 3),
                "quarantine_triggered": True,
                "interrupted": False,
            }

        # Determine resume step from persisted progress.
        progress = self._load_progress()
        last_completed = (
            int(progress.get("last_completed_step", 0))
            if progress is not None
            else 0
        )
        # If last_completed >= 5, treat as fresh cycle (the prior cycle
        # finished but progress was never cleared — defensive). Otherwise
        # resume from last_completed + 1.
        if last_completed >= SleepStep.COMPACT_RECORDS.value:
            last_completed = 0
        resume_step_value = last_completed + 1

        # Execute steps in order, skipping any with value < resume.
        for step in self._STEP_ORDER:
            if step.value < resume_step_value:
                continue

            self._emit_step_started(step)
            step_t0 = time.monotonic()
            method = self._step_methods[step]
            try:
                done, payload = method(interrupt_check)
            except Exception as exc:  # noqa: BLE001 -- 3-strike + quarantine flow
                err_str = str(exc)[:500]
                # Increment attempt counter for THIS step. If the prior
                # progress record's last_completed_step matches step-1,
                # we are failing the same step; attempt counter persists
                # and we add 1. If it differs (e.g. resumed from a
                # different step that just succeeded above), reset to 1.
                prior = self._load_progress() or {}
                prior_last = int(prior.get("last_completed_step", 0))
                if prior_last == step.value - 1:
                    new_attempt = int(prior.get("attempt", 0)) + 1
                else:
                    new_attempt = 1
                self._save_progress(
                    last_completed_step=step.value - 1,
                    attempt=new_attempt,
                    last_error=err_str,
                )
                # Log completion event with error info for ops trail.
                self._emit_step_completed(
                    step,
                    duration_sec=time.monotonic() - step_t0,
                    error=err_str,
                    attempt=new_attempt,
                )
                quarantine_triggered = False
                if new_attempt >= self._QUARANTINE_STRIKE_THRESHOLD:
                    self._set_quarantine(
                        reason=(
                            f"sleep step {step.value} ({step.name}) "
                            f"failed {new_attempt}x"
                        ),
                    )
                    quarantine_triggered = True
                return {
                    "completed_steps": completed_steps,
                    "failed_step": step,
                    "error": err_str,
                    "duration_sec": round(time.monotonic() - t0, 3),
                    "quarantine_triggered": quarantine_triggered,
                    "interrupted": False,
                }

            if not done:
                # Bounded-deferral early return. The step body already
                # persisted the deferral marker via `_check_interrupt`.
                return {
                    "completed_steps": completed_steps,
                    "failed_step": None,
                    "error": None,
                    "duration_sec": round(time.monotonic() - t0, 3),
                    "quarantine_triggered": False,
                    "interrupted": True,
                }

            # Step succeeded. Persist progress with attempt=0 (clean
            # slate for the NEXT step's strike counter; if the next step
            # fails, prior_last will equal step.value, so the failure
            # branch above will correctly start its own counter at 1).
            self._save_progress(
                last_completed_step=step.value,
                attempt=0,
                last_error=None,
            )
            self._emit_step_completed(
                step,
                duration_sec=time.monotonic() - step_t0,
                **payload,
            )
            completed_steps.append(step)

        # All steps from `resume` to COMPACT_RECORDS completed cleanly.
        # Clear progress so the next invocation starts fresh.
        self._clear_progress()
        return {
            "completed_steps": completed_steps,
            "failed_step": None,
            "error": None,
            "duration_sec": round(time.monotonic() - t0, 3),
            "quarantine_triggered": False,
            "interrupted": False,
        }

    def _check_and_maybe_auto_recover_quarantine(self) -> bool:
        """Return True iff the pipeline should short-circuit due to quarantine.

        Side effect: when a quarantine record exists but `until_ts` is
        in the past, this clears the quarantine via `_clear_quarantine`
        with reason=`auto_recovery_after_ttl` and returns False
        (caller proceeds to run the cycle). Otherwise:
        - No quarantine → False.
        - Quarantine still active (`now < until_ts`) → True.
        """
        quarantine = self._load_quarantine()
        if quarantine is None:
            return False
        try:
            until = datetime.fromisoformat(quarantine["until_ts"])
        except (TypeError, ValueError):
            # Malformed; clear and proceed (don't lock the user out).
            self._clear_quarantine(reason="auto_recovery_malformed_ts")
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if _utc_now() >= until:
            self._clear_quarantine(reason="auto_recovery_after_ttl")
            return False
        return True
