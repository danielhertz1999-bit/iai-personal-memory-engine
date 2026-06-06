"""Sleep cycle pipeline.

Eight ordered atomic steps run only inside the SLEEP lifecycle state.
The steps are split into NREM (stabilization) and REM (pruning +
abstraction) phases. The NREM phase runs first; the REM phase runs
second; ``CRISIS_RECLUSTER`` is the last REM step and is a no-op
unless ``lifecycle_state.crisis_mode`` was raised by the cycle-start
EssentialVariableTracker hook.

NREM phase (stabilization):
    1. SCHEMA_MINE       — extract schemas from episodic
    2. KNOB_TUNE         — recompute procedural knobs
    3. OPTIMIZE_LANCE    — WAL checkpoint + VACUUM + hnswlib rebuild +
                           tombstone-drop sweep (SleepStep value 4)
    4. COMPACT_RECORDS   — no-op stub under Hippo; preserved for
                           crash-window resume token (SleepStep value 5)

REM phase (pruning + abstraction):
    5. DREAM_DECAY       — Hebbian decay + edge prune
    6. ERASURE_AGENT     — active-forgetting tombstone pass
    7. CLUSTER_REPLAY    — temporal-cluster Hebbian batch replay
    8. CRISIS_RECLUSTER  — emergency re-cluster
                           (conditional on crisis_mode=True)

* Each step is **transactional** — the compaction step acquires an
  EXCLUSIVE SQLite lock during VACUUM; schema_mine / knob_tune /
  dream_decay write their own atomic temp+swap semantics through the
  modules they delegate to. The pipeline never modifies
  `MemoryRecord.literal_surface` (verbatim-recall invariant).

* On exception mid-step N, `lifecycle_state.json.sleep_cycle_progress`
  records `{last_completed_index: idx(step)-1, attempt: K, last_error: "..."}`
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
(`sleep._decay_edges`), Hippo compaction (`maintenance.optimize_hippo_storage`).
The pipeline is orchestration only.

Guards
------
* Human-first: pipeline runs only in SLEEP state, so MCP traffic
  cannot collide. SLEEP-state isolation is the sole guarantor.
* Zero paid-API cost: no reference to the paid-API env key anywhere.
  Schema induction stays Tier-0 (llm_enabled=False is the only path
  this pipeline exercises).
* Verbatim preservation: the pipeline does NOT touch
  `MemoryRecord.literal_surface`. Every delegated function is a
  metadata mutator (FSRS state, edge weights, schema candidates,
  profile knobs).
* Read-only audit: schema mining is read-only on records;
  decay is metadata-only on edges; compaction is storage-internal.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypedDict

from iai_mcp.exceptions import (
    SleepCheckpointError,
    SleepPipelineError,
    SleepQuarantineError,
    SleepStepError,
    StoreError,
)

if TYPE_CHECKING:
    # These imports run only during static type checking — never at runtime.
    # Required for type hints on optional constructor parameters and internal
    # method signatures. The FSL stand-alone path passes None for both
    # lifecycle_state_machine and lifecycle_event_log.
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import (
        LifecycleStateRecord,
        Quarantine,
        SleepCycleProgress,
    )

logger = logging.getLogger(__name__)


# Quarantine TTL configurable via env (default 24h).
# Read ONCE at import time so tests that monkeypatch the env var must
# also patch the module attribute (`sleep_pipeline.QUARANTINE_TTL_HOURS_DEFAULT`).
QUARANTINE_TTL_HOURS_DEFAULT: float = float(
    os.environ.get("IAI_MCP_SLEEP_QUARANTINE_TTL_HOURS", "24"),
)


class SleepStep(Enum):
    """Ordered atomic steps of the sleep pipeline.

    Numeric values are LOAD-BEARING: legacy ``lifecycle_state.json
    .sleep_cycle_progress.last_completed_step`` payloads carry the
    integer and ``sleep_step_started`` / ``sleep_step_completed`` events
    reference them. Re-ordering or renumbering these values would break
    1700+ historical event consumers. New steps MUST be APPENDed with
    fresh numeric values — never renumber existing entries.

    Dispatch order is controlled by the ``_STEP_ORDER`` tuple (NOT by
    ``step.value``); resume math uses ``_STEP_ORDER.index(step)`` so
    APPEND-without-renumber inserts stay safe.
    """

    SCHEMA_MINE = 1
    KNOB_TUNE = 2
    DREAM_DECAY = 3
    OPTIMIZE_LANCE = 4
    COMPACT_RECORDS = 5
    # ERASURE_AGENT appended without renumbering prior steps.
    # Its numeric value (6) is NOT the same as its dispatch position.
    ERASURE_AGENT = 6
    # CLUSTER_REPLAY and CRISIS_RECLUSTER reserve REM-phase slots
    # without shifting any prior step.value (event-payload back-compat).
    CLUSTER_REPLAY = 7
    CRISIS_RECLUSTER = 8
    # RECONSOLIDATION appended between CLUSTER_REPLAY and CRISIS_RECLUSTER.
    # Dispatch position is controlled by _STEP_ORDER, NOT by step.value.
    RECONSOLIDATION = 9
    # USER_MODEL_UPDATE appended between RECONSOLIDATION and CRISIS_RECLUSTER.
    # Dispatch position controlled by _STEP_ORDER.
    USER_MODEL_UPDATE = 10
    # DMN_REFLECTION appended between USER_MODEL_UPDATE and CRISIS_RECLUSTER.
    # Dispatch position controlled by _STEP_ORDER.
    DMN_REFLECTION = 11
    # CLUSTER_SUMMARY appended at the end (new numeric value, no prior step
    # renumbered). Runs hebbian connected-component cluster summarisation +
    # consolidated_from edges + cluster LTP via the shared _process_cluster_summaries
    # helper. Dispatch position controlled by _STEP_ORDER (appended to end).
    CLUSTER_SUMMARY = 12
    # RECALL_INDEX_REBUILD appended after CLUSTER_SUMMARY (MEDIUM-2: final
    # topology step — runs AFTER the last REM edge mutation so it re-detects
    # mosaic communities from SQLite ground truth with the fully-settled edge
    # set, then stamps a fresh generation epoch + rebuild_timestamp + resets
    # the in-process dirty counter.  Dispatch position controlled by
    # _STEP_ORDER (always last so it sees all edge mutations from this cycle).
    RECALL_INDEX_REBUILD = 13


class SleepPhase(Enum):
    """REM/NREM bifurcation (dual-process).

    NREM = stabilization (no record/edge pruning).
    REM  = pruning + abstraction (DREAM_DECAY, ERASURE_AGENT, CLUSTER_REPLAY,
           and conditional CRISIS_RECLUSTER).

    Each cycle runs all NREM steps first, then all REM steps (see _STEP_ORDER).
    """

    NREM = "NREM"
    REM = "REM"


# Class-level mapping of each SleepStep to its NREM/REM phase.
# Keeps SleepStep values stable for event-payload back-compat.
# All current SleepStep members are explicitly listed; future inserts MUST
# update this dict (no sentinel default in production code).
STEP_PHASE: dict[SleepStep, SleepPhase] = {
    SleepStep.SCHEMA_MINE: SleepPhase.NREM,
    SleepStep.KNOB_TUNE: SleepPhase.NREM,
    SleepStep.OPTIMIZE_LANCE: SleepPhase.NREM,
    SleepStep.COMPACT_RECORDS: SleepPhase.NREM,
    SleepStep.DREAM_DECAY: SleepPhase.REM,
    SleepStep.ERASURE_AGENT: SleepPhase.REM,
    SleepStep.CLUSTER_REPLAY: SleepPhase.REM,
    SleepStep.RECONSOLIDATION: SleepPhase.REM,
    SleepStep.USER_MODEL_UPDATE: SleepPhase.REM,
    SleepStep.DMN_REFLECTION: SleepPhase.REM,
    SleepStep.CRISIS_RECLUSTER: SleepPhase.REM,
    # CLUSTER_SUMMARY runs after all REM edge mutations (post-CRISIS_RECLUSTER).
    SleepStep.CLUSTER_SUMMARY: SleepPhase.REM,
    # RECALL_INDEX_REBUILD is the final topology step: re-detects mosaic from
    # SQLite ground truth after all edge mutations and stamps the generation epoch.
    SleepStep.RECALL_INDEX_REBUILD: SleepPhase.REM,
}


# Production-correctness cap: cartesian-product pair generation on an
# outlier-large temporal cluster (e.g. 1000 records reviewed within the
# lookback window) would otherwise emit ~500k boost_edges entries in a
# single step. 100 = half of pattern_separation top_k * 2 precedent;
# clusters that exceed this contribute only their first 100 canonical pairs.
MAX_PAIRS_PER_CLUSTER: int = 100


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
    """Orchestrates the sleep cycle steps with resume + quarantine support.

    Construction is cheap: opens no storage tables, performs no I/O
    beyond reading ``lifecycle_state.json``. The actual heavy work
    happens inside ``run()`` / ``force_run()`` step bodies.

    Concurrency note: the pipeline is single-threaded by design. The
    caller must ensure no overlapping invocations — typically by holding
    the SLEEP-state guard. There is no internal lock; running two
    ``SleepPipeline`` instances against the same ``lifecycle_state_path``
    simultaneously is undefined behaviour.
    """

    def __init__(
        self,
        store: Any,
        lifecycle_state_path: Path | None = None,
        event_log: Any | None = None,
        quarantine_ttl_hours: float | None = None,
        s2_coordinator: Any | None = None,
        loop: Any | None = None,
        *,
        lifecycle_state_machine: Any | None = None,
        lifecycle_event_log: Any | None = None,
    ) -> None:
        self._store = store

        # lifecycle_state_path: when None, use the default ~/.iai-mcp path.
        # Resolved lazily by _get_state_path() to avoid module-level import.
        self._lifecycle_state_path: Path | None = lifecycle_state_path

        # event_log / lifecycle_event_log: both names accepted for back-compat.
        # Resolved lazily when None (FSL stand-alone path creates a default).
        self._lel: Any | None = lifecycle_event_log if lifecycle_event_log is not None else event_log

        self._quarantine_ttl_hours = (
            float(quarantine_ttl_hours)
            if quarantine_ttl_hours is not None
            else QUARANTINE_TTL_HOURS_DEFAULT
        )
        # Optional S2Coordinator + asyncio loop for routing crisis_mode
        # bool field mutations through the lock owned by S2.
        # When None, the hook + crisis_recluster path fall back to direct
        # save_state writes; daemon construction will inject the live coordinator.
        self._s2_coordinator = s2_coordinator
        self._loop = loop

    def _get_state_path(self) -> Path:
        """Return the resolved lifecycle_state.json path (lazy default)."""
        if self._lifecycle_state_path is not None:
            return self._lifecycle_state_path
        from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
        return LIFECYCLE_STATE_PATH

    def _get_event_log(self) -> Any:
        """Return the event log instance. Creates a default one lazily when None."""
        if self._lel is not None:
            return self._lel
        # FSL stand-alone: no event log. Create a fresh default one.
        from iai_mcp.lifecycle_event_log import LifecycleEventLog
        self._lel = LifecycleEventLog()
        return self._lel

    @property
    def _event_log(self) -> Any:
        """Backward-compat property so existing code using self._event_log still works."""
        return self._get_event_log()

    # ------------------------------------------------------------------
    # Quarantine state (lifecycle_state.json.quarantine)
    # ------------------------------------------------------------------

    def _load_state_record(self) -> Any:
        """Read the current lifecycle state record (with self-heal)."""
        from iai_mcp.lifecycle_state import load_state
        return load_state(self._get_state_path())

    def _save_state_record(self, record: Any) -> None:
        """Atomic-replace persist of the lifecycle state record."""
        from iai_mcp.lifecycle_state import save_state
        save_state(record, self._get_state_path())

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
        except (OSError, ValueError) as exc:
            logger.debug("best-effort quarantine_entered event failed: %s", exc)
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
        # attempt=4 immediately. Progress.last_completed_index is kept
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
        except (OSError, ValueError) as exc:
            logger.debug("best-effort quarantine_lifted event failed: %s", exc)

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

    def _load_progress(self) -> Any:
        """Return the current sleep-cycle progress sub-record or None.

        One-shot in-memory migration: legacy persisted records use the integer
        ``last_completed_step`` (a ``SleepStep.<NAME>.value``). Current code
        uses ``last_completed_index`` (a position into ``_STEP_ORDER``) so
        APPEND-without-renumber enum inserts remain safe for resume math.

        When a legacy file is read, the old ``step.value`` is resolved back to
        the enum member and its new position in ``_STEP_ORDER`` is looked up.
        The next ``_save_progress`` call writes the canonical
        ``last_completed_index`` key. The read path is side-effect-free.
        Subtraction-based mappings (``legacy - 1``) are FORBIDDEN because they
        silently regress the ERASURE_AGENT slot for the legacy crash-window
        value of 4 (OPTIMIZE_LANCE).
        """
        progress = self._load_state_record().get("sleep_cycle_progress")
        if progress is None:
            return None
        if (
            "last_completed_step" in progress
            and "last_completed_index" not in progress
        ):
            legacy = int(progress.pop("last_completed_step", 0))
            try:
                legacy_step = SleepStep(legacy)
                progress["last_completed_index"] = self._STEP_ORDER.index(
                    legacy_step,
                )
            except (ValueError, KeyError):
                # Unknown legacy value (corrupt file, far-future value past
                # the end of any known _STEP_ORDER, etc.) → fall back to a
                # fresh cycle on the next resume rather than crashing.
                progress["last_completed_index"] = -1
        return progress

    def _save_progress(
        self,
        last_completed_index: int,
        attempt: int,
        last_error: str | None,
        *,
        started_at: str | None = None,
    ) -> SleepCycleProgress:
        """Persist sleep-cycle progress; preserve `started_at` across saves.

        ``started_at`` defaults to: prior progress's started_at if any,
        else ``now()``. This gives the operator a wall-clock view of how
        long the cycle has been running across resumes.

        ``last_completed_index`` is a position into ``_STEP_ORDER`` (NOT an
        enum value). Reads via ``_load_progress`` apply a one-shot in-memory
        migration from the legacy ``last_completed_step`` (step.value) form,
        so writes always produce the canonical new shape.
        """
        record = self._load_state_record()
        prior = record.get("sleep_cycle_progress") or {}
        progress: dict = {
            "last_completed_index": last_completed_index,
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
    # Step orchestrators
    # ------------------------------------------------------------------
    #
    # Each `_step_*` returns True on natural completion and False when
    # `interrupt_check` fired between chunks. On exception, the step
    # body re-raises to the caller (run()) which handles 3-strike
    # quarantine + progress save. Step bodies are deliberately small:
    # they delegate to the underlying module functions for their core work.

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
        except (OSError, ValueError) as exc:
            logger.debug("best-effort sleep_step_started event failed: %s", exc)

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
        except (OSError, ValueError) as exc:
            logger.debug("best-effort sleep_step_completed event failed: %s", exc)

    def _check_interrupt(
        self,
        step: SleepStep,
        chunk_idx: int,
        interrupt_check: Callable[[], bool] | None,
    ) -> bool:
        """Return True iff the caller asked us to defer.

        Persists ``sleep_cycle_progress.last_completed_index =
        _STEP_ORDER.index(step) - 1`` (we have NOT completed ``step`` yet)
        and stamps ``last_error`` with a structured deferral marker so
        lifecycle status output can show "deferred at step N chunk K"
        rather than a fake error. Position-based math — using
        ``step.value - 1`` is the APPEND-without-renumber wrap-bug.
        """
        if interrupt_check is None:
            return False
        try:
            should = bool(interrupt_check())
        except Exception as exc:  # noqa: BLE001 -- caller predicate may raise anything
            # If the caller's predicate is broken, do NOT defer (better
            # to keep working than to hang forever waiting for a True
            # that will never come). Same fail-safe discipline as the
            # event-log emit failures above.
            logger.debug("interrupt_check predicate raised: %s", exc)
            should = False
        if not should:
            return False
        # Save deferral marker. last_completed_index stays at the prior
        # step (we are mid-`step`); attempt counter is unchanged because
        # this is NOT a failure — it is a cooperative yield.
        prior = self._load_progress() or {}
        last_completed_index = self._STEP_ORDER.index(step) - 1
        attempt = int(prior.get("attempt", 0))
        self._save_progress(
            last_completed_index=last_completed_index,
            attempt=attempt,
            last_error=f"deferred:step={step.name}:chunk_idx={chunk_idx}",
        )
        return True

    def _step_schema_mine(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 1: schema mining via tier-0 induction + tier-1 persistence.

        Two-phase step:
        1. ``induce_schemas_tier0(store)`` does a single MVCC pass over
           ``records.tags_json`` and returns candidates. The candidate count
           is reported as ``schemas_induced`` (existing key; candidate count).
        2. ``sleep._persist_tier1_schemas(store, ...)`` persists auto-status
           candidates (creating ``schema_instance_of`` edges) — the SAME
           helper that ``run_heavy_consolidation`` calls (single-source; no
           separate implementation). The persisted count is reported as
           ``schemas_persisted`` (new key; used by the pipeline-level
           ``cls_consolidation_run`` emit).

        Chunk granularity is one (both underlying calls are single batch
        reads; not sliced). The chunk boundary is honoured by checking
        ``interrupt_check`` BEFORE the call.

        Returns ``(completed, payload)`` — completed=False signals an
        interrupt-induced early return (no payload metadata).
        """
        from iai_mcp.schema import induce_schemas_tier0
        from iai_mcp.sleep import _persist_tier1_schemas

        # Single-chunk implementation: chunk_idx=0 is the only checkpoint.
        if self._check_interrupt(SleepStep.SCHEMA_MINE, 0, interrupt_check):
            return False, {}
        candidates = induce_schemas_tier0(self._store)
        # Best-effort metric for the completion event; tier-0 returns a
        # list of `SchemaCandidate` dataclass instances, len() works.
        try:
            count = len(candidates) if candidates is not None else 0
        except (TypeError, AttributeError) as exc:
            logger.debug("non-critical schema count failed: %s", exc)
            count = 0

        # Persist auto-status candidates via the shared helper (single-source).
        # _persist_tier1_schemas re-runs induce_schemas_tier1 (a thin
        # pass-through to tier0 that emits an llm_health event) and persists
        # auto candidates. budget/rate/llm_enabled use defaults compatible with
        # the tier0 path (induce_schemas_tier1 ignores them — retained for API
        # stability only). BudgetLedger requires the store instance.
        persisted = 0
        try:
            from iai_mcp.guard import BudgetLedger, RateLimitLedger
            _budget = BudgetLedger(self._store)
            _rate = RateLimitLedger(self._store)
            _persist_candidates, persisted = _persist_tier1_schemas(
                self._store, _budget, _rate, llm_enabled=False,
            )
        except Exception as exc:  # noqa: BLE001 -- persistence failure is non-fatal for this step
            logger.debug("non-critical schema persist in step failed: %s", exc)

        return True, {"schemas_induced": count, "schemas_persisted": persisted}

    def _step_knob_tune(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 2: per-knob procedural snapshot.

        Iterates over the sealed ``PROFILE_KNOBS`` registry; each knob is one
        chunk (interrupt cadence = registry size, currently 11). The actual
        Bayesian update is event-driven and runs elsewhere; sleep takes a
        snapshot of the live state so audit trails can replay it.
        ``profile.default_state()`` is called once outside the loop so future
        per-knob work can hook in without re-architecting the chunk boundary.
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

        # Soft_knobs auto-write: adjust monotropism multiplier based on
        # curiosity_bridge edge ratio (self-tuning).
        try:
            from iai_mcp.user_model import load as _load_um, save as _save_um
            tbl = self._store.db.open_table("edges")
            total_edges = tbl.count_rows()
            curiosity_count = tbl.count_rows("edge_type = 'curiosity_bridge'") if total_edges > 0 else 0
            curiosity_ratio = curiosity_count / max(total_edges, 1)
            if curiosity_ratio > 0.1 or curiosity_ratio < 0.02:
                um = _load_um()
                if curiosity_ratio > 0.1:
                    um.soft_knobs["monotropism"] = 1.5
                elif curiosity_ratio < 0.02:
                    um.soft_knobs["monotropism"] = 0.8
                _save_um(um)
        except (OSError, ValueError, RuntimeError, KeyError, StoreError) as exc:
            logger.debug("non-critical soft_knobs auto-write failed: %s", exc)

        # GABA k-annealing: compute annealed k for next retrieval cycle.
        try:
            from iai_mcp.gaba_annealing import compute_annealed_k, should_normalize
            cycle_count = self._cycle_counter
            annealed_k = compute_annealed_k(cycle_count)
            if should_normalize(cycle_count):
                logger.debug("GABA: k=%d at cycle %d, normalization due", annealed_k, cycle_count)
        except (ImportError, AttributeError, TypeError) as exc:
            logger.debug("GABA annealing skipped: %s", exc)

        return True, {"knobs_tuned": len(knob_names)}

    def _step_dream_decay(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 3: Hebbian decay + edge prune via ``_decay_edges``.

        ``sleep._decay_edges(store)`` walks every hebbian/hebbian_structure
        edge and either decays the weight in place or prunes when below
        epsilon. The function is monolithic; wrapped as a single chunk
        (chunk_idx=0), ``interrupt_check`` is checked before the call.
        """
        from iai_mcp.sleep import _decay_edges

        if self._check_interrupt(SleepStep.DREAM_DECAY, 0, interrupt_check):
            return False, {}
        # Meta-learning: read plasticity_gain from user_model
        # (faster decay in crisis, slower in stability).
        _plasticity = 1.0
        try:
            from iai_mcp.user_model import load as _load_um
            _um = _load_um()
            _plasticity = getattr(_um, "plasticity_gain", 1.0) or 1.0
        except (OSError, ValueError, RuntimeError, StoreError, AttributeError) as exc:
            logger.debug("non-critical plasticity_gain load failed: %s", exc)
        result = _decay_edges(self._store, plasticity_gain=_plasticity)
        # Surface decay/prune counts in the completion event for ops.
        if isinstance(result, dict):
            return True, {
                "decayed": int(result.get("decayed", 0) or 0),
                "pruned": int(result.get("pruned", 0) or 0),
            }
        return True, {}

    def _step_erasure_agent(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Active-forgetting tombstone pass.

        Eligibility predicate:
            centrality < threshold
            AND (last_reviewed IS NULL OR last_reviewed < now - window)
            AND created_at < now - age
            AND pinned = false
            AND never_decay = false
            AND tombstoned_at IS NULL

        Rows matching the predicate get ``tombstoned_at = now`` written
        into the records table; OPTIMIZE_LANCE physically drops them
        after ``cfg.tombstone_ttl_sec`` elapses. Pinned and never_decay
        are absolute carve-outs at this stage AND at drop time.

        Dry-run mode: when ``cfg.dry_run`` is True the eligibility set is
        still counted and the ``erasure_agent_pass`` event is still emitted,
        but no ``tombstoned_at`` write happens. Defaults to dry_run=True
        under pytest so accidental mass-tombstone in CI is impossible.

        Emits exactly one structured ``erasure_agent_pass`` event per
        invocation with body keys:
            count_quarantined, count_dropped, total_records_after,
            threshold_used, dry_run_mode
        ``count_dropped`` reports the most recent OPTIMIZE_LANCE drop pass;
        sourced via ``query_events(kind='erasure_optimize_drops', limit=1)``.
        On the first sleep cycle after deploy the field defaults to 0.

        Returns ``(True, {count_quarantined, dry_run})``.
        """
        # Single chunk_idx=0 boundary mirrors _step_dream_decay; the
        # eligibility + mutation pair is one atomic operation and
        # cannot be subdivided without reimplementing.
        if self._check_interrupt(
            SleepStep.ERASURE_AGENT, 0, interrupt_check,
        ):
            return False, {}

        # Config is re-read on every invocation so monkeypatch.setenv tests
        # see overrides without restarting the pipeline.
        from iai_mcp.daemon_config import _load_erasure_config
        # WAL: log intent before destructive erasure operation.
        try:
            from iai_mcp.sleep_wal import SleepWAL
            _wal = SleepWAL()
        except ImportError:
            _wal = None
        cfg = _load_erasure_config()
        threshold = cfg.centrality_threshold
        age_days = cfg.age_days
        window_days = cfg.retrieval_window_days
        dry_run = cfg.dry_run

        now = _utc_now()
        age_cutoff = now - timedelta(days=age_days)
        window_cutoff = now - timedelta(days=window_days)

        from iai_mcp.store import RECORDS_TABLE
        tbl = self._store.db.open_table(RECORDS_TABLE)

        # Eligibility predicate built as a SQL WHERE clause.
        # SQLite stores datetimes as UTC ISO-8601 strings (no TZ offset);
        # use plain string comparison — do NOT use TIMESTAMP '' literal
        # syntax (not supported by SQLite). Strip TZ offset with strftime.
        window_cutoff_str = window_cutoff.strftime("%Y-%m-%d %H:%M:%S")
        age_cutoff_str = age_cutoff.strftime("%Y-%m-%d %H:%M:%S")
        eligibility_where = (
            f"centrality < {threshold} "
            f"AND (last_reviewed IS NULL OR "
            f"last_reviewed < '{window_cutoff_str}') "
            f"AND created_at < '{age_cutoff_str}' "
            f"AND pinned = false "
            f"AND never_decay = false "
            f"AND tombstoned_at IS NULL"
        )

        # Count BEFORE mutate: after tbl.update the WHERE clause would
        # match zero rows (tombstoned_at would no longer be NULL). total_records_after
        # uses the unconditional count — tombstoning sets a column, not deletes rows.
        # Rows leave the table only in OPTIMIZE_LANCE after TTL.
        try:
            count_quarantined = int(tbl.count_rows(filter=eligibility_where))
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            # Defensive: a malformed predicate at runtime would lock the
            # whole sleep cycle. Surface zero quarantined, still emit
            # the event so ops can see the failure. The next pass picks
            # back up cleanly.
            logger.debug("erasure_agent count_rows failed: %s", exc)
            count_quarantined = 0
        total_records_after = int(tbl.count_rows())

        # Read the most recent OPTIMIZE_LANCE drop event to populate count_dropped.
        # Under NREM/REM ordering, OPTIMIZE_LANCE runs before ERASURE_AGENT in
        # the same cycle, so this read picks up THIS cycle's drop count.
        # On the first cycle after deploy no prior event exists and
        # count_dropped defaults to 0.
        from iai_mcp.events import query_events, write_event
        prior_drops = query_events(
            self._store, kind="erasure_optimize_drops", limit=1,
        )
        count_dropped = 0
        if prior_drops:
            prior_body = prior_drops[0].get("data") or {}
            count_dropped = int(prior_body.get("count_dropped", 0) or 0)

        # Tombstone mutation — guarded by dry_run. Counts above
        # are emitted in BOTH modes; only the column-write is skipped.
        if not dry_run and count_quarantined > 0:
            try:
                tbl.update(
                    where=eligibility_where,
                    values={"tombstoned_at": now},
                )
            except Exception as exc:  # noqa: BLE001 -- visibility over crash
                # If the mutation itself fails, surface the count we
                # WOULD have written in the event body (so ops see the
                # intended scope) and re-raise to let the standard
                # 3-strike machinery catch the underlying breakage on
                # the next pass.
                logger.error("erasure_agent tombstone mutation failed: %s", exc, exc_info=True)
                write_event(
                    self._store,
                    "erasure_agent_pass",
                    {
                        "count_quarantined": int(count_quarantined),
                        "count_dropped": int(count_dropped),
                        "total_records_after": int(total_records_after),
                        "threshold_used": float(threshold),
                        "dry_run_mode": bool(dry_run),
                        "mutation_error": str(exc)[:500],
                    },
                    severity="warning",
                )
                raise

        # Single structured event per invocation.
        write_event(
            self._store,
            "erasure_agent_pass",
            {
                "count_quarantined": int(count_quarantined),
                "count_dropped": int(count_dropped),
                "total_records_after": int(total_records_after),
                "threshold_used": float(threshold),
                "dry_run_mode": bool(dry_run),
            },
            severity="info",
        )

        return True, {
            "count_quarantined": int(count_quarantined),
            "dry_run": bool(dry_run),
        }

    def _step_compact_hippo(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step 4: Hippo compaction step — WAL checkpoint + VACUUM +
        hnswlib rebuild + atomic save — then tombstone-drop sweep.

        Calls ``optimize_hippo_storage(store)`` (PRAGMA
        wal_checkpoint(TRUNCATE) + VACUUM + hnswlib rebuild + atomic
        index save). The call is blocking — VACUUM holds an EXCLUSIVE
        SQLite lock for its full duration; MCP captures queue in the
        writer pool and drain at WAKE (sleep-cycle isolation guarantees
        no in-flight captures during NREM). Helper never raises; per-
        table failures land in the returned report dict.

        After the physical compaction the step performs the active-
        forgetting tombstone-drop sweep:
        1. Un-tombstone any rows that picked up ``pinned=True`` OR
           ``never_decay=True`` AFTER the tombstone was applied. Pin /
           never_decay is the absolute carve-out; a row tombstoned
           8 days ago that just got pinned today must be preserved.
           This branch runs FIRST so the subsequent drop predicate
           never matches protected rows.
        2. Delete rows whose ``tombstoned_at < now - cfg.tombstone_ttl_sec``.
        3. Emit a single ``erasure_optimize_drops`` event (storage-
           agnostic event kind — kept unchanged for ``_step_erasure_agent``
           consumer compatibility). ``_step_erasure_agent`` reads it in
           the same cycle's later REM phase to populate its own
           ``erasure_agent_pass.count_dropped`` field (NREM pos 2;
           ERASURE_AGENT is REM pos 5 — same cycle, NOT the next).

        After the tombstone-drop sweep emits a single ``hippo_compacted``
        event with ``{phase, per_table, total_elapsed_sec}`` keys
        (replaces the old ``lance_storage_optimized`` event).

        Ordering rule: ``count_rows`` is captured BEFORE the matching
        ``tbl.update`` / ``tbl.delete``. The WHERE clause would match
        zero rows post-mutation because the filtered columns change.
        """
        from iai_mcp.maintenance import optimize_hippo_storage

        if self._check_interrupt(
            SleepStep.OPTIMIZE_LANCE, 0, interrupt_check,
        ):
            return False, {}

        compact_t0 = time.monotonic()
        report = optimize_hippo_storage(self._store)
        # Helper never raises; per-table errors live inside the report dict.
        tables_with_errors = [
            t for t, r in (report or {}).items()
            if isinstance(r, dict) and "error" in r
        ]

        # Call-on-demand config load: fresh read of env vars each
        # invocation so monkeypatch tests work.
        from iai_mcp.daemon_config import _load_erasure_config
        cfg = _load_erasure_config()
        ttl_sec = cfg.tombstone_ttl_sec

        now = _utc_now()
        drop_cutoff = now - timedelta(seconds=ttl_sec)

        from iai_mcp.store import RECORDS_TABLE
        from iai_mcp.events import write_event

        # Un-tombstone protected rows FIRST. The pin / never_decay
        # check is the absolute carve-out; rows that picked up the
        # protection AFTER tombstoning get reset to NULL instead of
        # dropped. Order matters: this must happen before the drop
        # sweep so a row tombstoned 8 days ago that just got pinned
        # today is preserved.
        tbl = self._store.db.open_table(RECORDS_TABLE)
        untomb_where = (
            "tombstoned_at IS NOT NULL "
            "AND (pinned = true OR never_decay = true)"
        )
        try:
            count_untombstoned = int(tbl.count_rows(filter=untomb_where))
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("compact_hippo untombstone count failed: %s", exc)
            count_untombstoned = 0
        if count_untombstoned > 0:
            try:
                tbl.update(
                    where=untomb_where,
                    values={"tombstoned_at": None},
                )
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                # Defensive — surface zero on a malformed predicate so
                # the cycle doesn't crash; the next pass picks up.
                logger.debug("compact_hippo untombstone update failed: %s", exc)
                count_untombstoned = 0

        # Drop rows whose tombstone has aged past TTL. Re-open the
        # table so the un-tombstone result is visible to the WHERE
        # clause below (table handle snapshots the view at open time;
        # re-open picks up the un-tombstone write committed above).
        tbl = self._store.db.open_table(RECORDS_TABLE)
        drop_cutoff_str = drop_cutoff.strftime("%Y-%m-%d %H:%M:%S")
        drop_where = (
            "tombstoned_at IS NOT NULL "
            f"AND tombstoned_at < '{drop_cutoff_str}'"
        )
        try:
            count_dropped = int(tbl.count_rows(filter=drop_where))
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("compact_hippo drop count failed: %s", exc)
            count_dropped = 0
        if count_dropped > 0:
            try:
                tbl.delete(drop_where)
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                # Defensive — same logic as the un-tombstone branch.
                logger.debug("compact_hippo drop delete failed: %s", exc)
                count_dropped = 0

        # Emit the erasure_optimize_drops event so _step_erasure_agent
        # can read it in the same cycle's later REM phase and populate
        # its own erasure_agent_pass.count_dropped field.
        try:
            write_event(
                self._store,
                "erasure_optimize_drops",
                {
                    "count_dropped": int(count_dropped),
                    "count_untombstoned": int(count_untombstoned),
                    "ts": now.isoformat(),
                },
                severity="info",
            )
        except (OSError, ValueError, StoreError) as exc:
            # Event-emit is best-effort (a write failure here must NOT
            # roll back the physical drop that just succeeded). The
            # same cycle's erasure_agent_pass.count_dropped will be 0
            # in this edge case; ops can correlate via the standard
            # sleep_step_completed payload.
            logger.debug("best-effort erasure_optimize_drops event failed: %s", exc)

        elapsed = round(time.monotonic() - compact_t0, 3)
        try:
            write_event(
                self._store,
                "hippo_compacted",
                {
                    "phase": "sleep_cycle",
                    "per_table": report,
                    "total_elapsed_sec": elapsed,
                },
                severity="info",
            )
        except Exception:  # noqa: BLE001
            logger.debug("hippo_compacted event emit failed", exc_info=True)

        return True, {
            "tables_optimized": list((report or {}).keys()),
            "tables_with_errors": tables_with_errors,
            "count_dropped_by_erasure": int(count_dropped),
            "count_untombstoned_by_pin_override": int(count_untombstoned),
        }

    def _step_compact_records_noop(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Forward-compat stub preserving the SleepStep.COMPACT_RECORDS
        (=5) resume-token slot.

        Under Hippo, ``_step_compact_hippo`` already performs the full
        WAL checkpoint + VACUUM + hnswlib rebuild in one shot, so a
        second compaction pass is wasted I/O. This stub no-ops and
        returns immediately so the sleep cycle still completes the same
        number of phases (preserving crash-window resume tokens that
        reference enum value 5).
        """
        if self._check_interrupt(
            SleepStep.COMPACT_RECORDS, 0, interrupt_check,
        ):
            return False, {}
        return True, {"action": "noop_under_hippo"}

    def _step_optimize_lance(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Compatibility alias for the legacy step name."""
        return self._step_compact_hippo(interrupt_check)

    def _step_compact_records(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Compatibility alias for the legacy step name."""
        return self._step_compact_records_noop(interrupt_check)

    def _step_cluster_replay(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Cluster-replay temporal-coactivation Hebbian pass.

        Reads records whose ``last_reviewed`` falls within the last
        ``cluster_window_sec * 5`` seconds (5-window lookback). Groups them
        into windows of ``cluster_window_sec``. For each window with >= 2
        records, generates the cartesian product of intra-window pairs, capped
        at ``MAX_PAIRS_PER_CLUSTER`` canonical (sorted-tuple) pairs, and calls
        ``boost_edges`` with ``edge_type='hebbian_cluster_replay'`` and
        ``delta = cfg.cluster_replay_initial_weight``.

        Dry-run mode: when ``cfg.dry_run`` is True the cluster count and pair
        count are still calculated and the ``cluster_replay_pass`` event is
        still emitted, but the ``boost_edges`` call is skipped.

        Emits exactly one ``cluster_replay_pass`` event per invocation with
        body keys: clusters_replayed, total_edges_boosted, avg_cluster_size,
        window_sec, lookback_windows, max_pairs_per_cluster_applied, dry_run_mode.

        Returns ``(True, {"clusters_replayed": int, "dry_run": bool})``.
        """
        # Single chunk_idx=0 boundary: the query + grouping + per-cluster
        # boost_edges sequence is a single conceptual operation.
        if self._check_interrupt(
            SleepStep.CLUSTER_REPLAY, 0, interrupt_check,
        ):
            return False, {}

        # Config re-read on every invocation for monkeypatch.setenv test compat.
        from iai_mcp.daemon_config import _load_sleep_overhaul_config
        cfg = _load_sleep_overhaul_config()
        window_sec = cfg.cluster_window_sec
        delta = cfg.cluster_replay_initial_weight
        dry_run = cfg.dry_run
        lookback_windows = 5  # 5-window lookback

        from iai_mcp.events import write_event
        from iai_mcp.store import RECORDS_TABLE

        now = _utc_now()
        lookback_cutoff = now - timedelta(seconds=window_sec * lookback_windows)
        tbl = self._store.db.open_table(RECORDS_TABLE)

        # Read all records whose last_reviewed >= cutoff. Sort ascending
        # by last_reviewed so windowing is a single linear sweep.
        try:
            lookback_cutoff_str = lookback_cutoff.strftime("%Y-%m-%d %H:%M:%S")
            df = (
                tbl.search()
                .where(
                    f"last_reviewed >= '{lookback_cutoff_str}'"
                )
                .to_pandas()
            )
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            # Defensive: malformed timestamp / empty table -> emit
            # zero-cluster event below.
            logger.debug("cluster_replay query failed: %s", exc)
            df = None

        clusters: list[list[Any]] = []
        if df is not None and not df.empty and "last_reviewed" in df.columns:
            df_sorted = df.sort_values("last_reviewed").reset_index(drop=True)
            window_td = timedelta(seconds=window_sec)
            current_cluster: list[Any] = []
            current_window_end = None
            for _, row in df_sorted.iterrows():
                ts = row["last_reviewed"]
                try:
                    py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                except (TypeError, ValueError, AttributeError):
                    py = ts
                # Parse string timestamps returned by HippoDB (SQLite TEXT storage).
                if isinstance(py, str):
                    try:
                        py = datetime.fromisoformat(py)
                    except (TypeError, ValueError):
                        continue
                if getattr(py, "tzinfo", None) is None:
                    try:
                        py = py.replace(tzinfo=timezone.utc)
                    except (TypeError, ValueError, AttributeError):
                        continue
                if current_window_end is None or py > current_window_end:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [row["id"]]
                    current_window_end = py + window_td
                else:
                    current_cluster.append(row["id"])
            if current_cluster:
                clusters.append(current_cluster)

        # Filter to clusters of size >= 2; build canonical-pair list per cluster
        # capped at MAX_PAIRS_PER_CLUSTER. boost_edges canonicalises pairs to
        # sorted tuples, but the cap is applied BEFORE the call to avoid
        # ~500k-entry intermediates on outlier clusters.
        from itertools import combinations
        import uuid as _uuid

        replay_clusters = [c for c in clusters if len(c) >= 2]
        total_pairs = 0
        capped_count = 0
        all_pairs: list[tuple[Any, Any]] = []
        for c in replay_clusters:
            # Coerce raw record ids to UUID objects; tbl rows already
            # store UUIDs but the round-trip through pandas may return
            # strings.
            uids = [
                _uuid.UUID(str(x)) if not isinstance(x, _uuid.UUID) else x
                for x in c
            ]
            cluster_pairs = list(combinations(uids, 2))
            if len(cluster_pairs) > MAX_PAIRS_PER_CLUSTER:
                cluster_pairs = cluster_pairs[:MAX_PAIRS_PER_CLUSTER]
                capped_count += 1
            all_pairs.extend(cluster_pairs)
            total_pairs += len(cluster_pairs)

        # Mutation -- guarded by dry_run.
        edges_boosted = 0
        if all_pairs and not dry_run:
            try:
                result_map = self._store.boost_edges(
                    all_pairs,
                    delta=delta,
                    edge_type="hebbian_cluster_replay",
                )
                edges_boosted = len(result_map)
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                # Non-fatal at body level: surface in event, then re-raise
                # so the standard 3-strike machinery sees the failure.
                logger.error("cluster_replay boost_edges failed: %s", exc, exc_info=True)
                write_event(
                    self._store,
                    "cluster_replay_pass",
                    {
                        "clusters_replayed": len(replay_clusters),
                        "total_edges_boosted": 0,
                        "avg_cluster_size": (
                            sum(len(c) for c in replay_clusters)
                            / len(replay_clusters)
                            if replay_clusters else 0.0
                        ),
                        "window_sec": int(window_sec),
                        "lookback_windows": int(lookback_windows),
                        "max_pairs_per_cluster_applied": int(capped_count),
                        "dry_run_mode": False,
                        "mutation_error": str(exc)[:500],
                    },
                    severity="warning",
                )
                raise
        elif all_pairs and dry_run:
            # Report the count we WOULD have boosted.
            edges_boosted = total_pairs

        avg_cluster_size = (
            sum(len(c) for c in replay_clusters) / len(replay_clusters)
            if replay_clusters else 0.0
        )
        write_event(
            self._store,
            "cluster_replay_pass",
            {
                "clusters_replayed": int(len(replay_clusters)),
                "total_edges_boosted": int(edges_boosted),
                "avg_cluster_size": float(avg_cluster_size),
                "window_sec": int(window_sec),
                "lookback_windows": int(lookback_windows),
                "max_pairs_per_cluster_applied": int(capped_count),
                "dry_run_mode": bool(dry_run),
            },
            severity="info",
        )

        # Temporal trajectory replay: boost sequential pairs within each
        # temporal cluster (preserves causal ordering A→B→C, not just co-occurrence).
        seq_pairs_boosted = 0
        if replay_clusters and not dry_run:
            try:
                sequential_pairs: list[tuple[Any, Any]] = []
                for c in replay_clusters:
                    uids = [
                        _uuid.UUID(str(x)) if not isinstance(x, _uuid.UUID) else x
                        for x in c
                    ]
                    for i in range(len(uids) - 1):
                        sequential_pairs.append((uids[i], uids[i + 1]))
                if sequential_pairs:
                    self._store.boost_edges(
                        sequential_pairs,
                        delta=delta * 1.5,
                        edge_type="temporal_sequence",
                    )
                    seq_pairs_boosted = len(sequential_pairs)
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                logger.debug("non-critical temporal trajectory replay failed: %s", exc)

        return True, {
            "clusters_replayed": int(len(replay_clusters)),
            "sequential_pairs": int(seq_pairs_boosted),
            "dry_run": bool(dry_run),
        }

    def _step_reconsolidation(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Reconsolidation labile-window pass.

        Scans every record whose ``labile_until > now`` (stamped by
        ``memory_recall`` via ``reinforce_record(is_retrieval=True)``). For
        each labile record, optionally calls the Tier-1 LLM critic from
        ``iai_mcp.reconsolidation_critic`` (gated by ``cfg.reconsolidation_tier1``
        AND the guard ladder inside ``call_critic``). When the returned
        prediction_error is ``>= cfg.reconsolidation_error_threshold`` AND not
        dry-run:

        - Append ``{"reconsolidated_at": <iso8601>, "prediction_error": <float>}``
          to the record's provenance_json via ``store.append_provenance``.
        - Re-anchor FSRS stability via ``store.reinforce_record(rid)``
          (``is_retrieval=False`` — background re-anchor, not user retrieval,
          so it does NOT re-stamp labile_until and create a perpetual labile loop).

        Honours ``_check_interrupt`` between records (chunk_idx incremented per
        record so the bounded-deferral machine can suspend mid-scan).

        Dry-run mode: when ``cfg.dry_run`` is True the scan still runs and the
        ``reconsolidation_pass`` event is still emitted; the critic still runs
        (so operators can observe its rate); but the provenance update and FSRS
        re-anchor are skipped.

        Tier-1 disabled (``cfg.reconsolidation_tier1=False``, the default): the
        per-record critic loop is skipped entirely. ``critic_calls`` and
        ``records_reconsolidated`` are both zero; ``records_scanned`` still
        reports the labile-window inventory.

        Emits exactly one ``reconsolidation_pass`` event per invocation with
        body keys: records_scanned, records_reconsolidated, critic_calls,
        dry_run_mode.

        Returns ``(True, {"records_scanned": int,
        "records_reconsolidated": int, "dry_run": bool})``.
        """
        if self._check_interrupt(
            SleepStep.RECONSOLIDATION, 0, interrupt_check,
        ):
            return False, {}

        # Config re-read on every invocation so monkeypatch.setenv flips are observable.
        from iai_mcp.daemon_config import _load_reconsolidation_config
        cfg = _load_reconsolidation_config()

        from iai_mcp.events import write_event
        from iai_mcp.store import RECORDS_TABLE
        from iai_mcp.reconsolidation_critic import evaluate_batch_reconsolidation
        import uuid as _uuid

        now = _utc_now()
        tbl = self._store.db.open_table(RECORDS_TABLE)

        # Scan labile records via WHERE pushdown. Defensive:
        # malformed timestamp / empty table / missing column -> df=None
        # and we emit the zero-scan event below.
        try:
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            df = (
                tbl.search()
                .where(
                    f"labile_until > '{now_str}'"
                )
                .to_pandas()
            )
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("reconsolidation labile query failed: %s", exc)
            df = None

        records_scanned = 0 if df is None else int(len(df))
        records_reconsolidated = 0
        critic_calls = 0

        if (
            df is not None
            and not df.empty
            and cfg.reconsolidation_tier1
        ):
            # The per-record critic loop is replaced with a single batched call
            # honouring the "1 claude -p call per night" invariant.
            # evaluate_batch_reconsolidation caps the pool at MAX_RECORDS_PER_CALL
            # (100), spawns ONE subscription-billed subprocess, and returns a
            # {UUID: prediction_error} mapping. Records over the cap or absent
            # from the response default to Tier-0 (0.0 = no provenance update).

            # Collect (id, plaintext_surface) for every labile candidate.
            # literal_surface is AES-GCM encrypted in the raw row; round-trip
            # through store.get() to decrypt — matches the pattern used by
            # other step bodies needing plaintext.
            pool: list[tuple[_uuid.UUID, str]] = []
            for chunk_idx, (_, row) in enumerate(df.iterrows(), start=1):
                if self._check_interrupt(
                    SleepStep.RECONSOLIDATION,
                    chunk_idx,
                    interrupt_check,
                ):
                    return False, {}
                rid_str = row["id"]
                try:
                    rid = _uuid.UUID(str(rid_str))
                except (TypeError, ValueError):
                    continue
                rec = self._store.get(rid)
                if rec is None:
                    continue
                pool.append((rid, rec.literal_surface))

            # One subscription-billed subprocess call for the whole pool.
            try:
                errors_by_id = evaluate_batch_reconsolidation(
                    pool,
                    llm_enabled=True,
                )
            except Exception as exc:  # noqa: BLE001 -- critic must never raise into REM
                logger.debug("reconsolidation batch call raised: %s", exc)
                errors_by_id = {}

            # critic_calls accounts the subprocess invocation count (0 or 1),
            # not per-record. The event payload keeps the same key for
            # back-compat; semantics = "batched critic call".
            critic_calls = 1 if errors_by_id else 0

            for rid, err in errors_by_id.items():
                if err < float(cfg.reconsolidation_error_threshold):
                    continue
                if cfg.dry_run:
                    # Count the candidate but skip ALL mutation so
                    # shadow-deploy operators see the rate without
                    # changing any row.
                    records_reconsolidated += 1
                    continue
                try:
                    self._store.append_provenance(
                        rid,
                        {
                            "reconsolidated_at": now.isoformat(),
                            "prediction_error": float(err),
                        },
                    )
                    # FSRS re-anchor via the existing reinforce path.
                    # is_retrieval kept at its default False: reconsolidation
                    # is a background re-anchor, not a user retrieval --
                    # must NOT bump labile_until.
                    self._store.reinforce_record(rid)
                    records_reconsolidated += 1
                except (OSError, ValueError, RuntimeError, StoreError) as exc:
                    # Per-record write failure must not abort the whole
                    # pass; the event below still reflects the successful
                    # subset.
                    logger.debug("reconsolidation per-record write failed: %s", exc)

        # Single event emission per invocation.
        write_event(
            self._store,
            "reconsolidation_pass",
            {
                "records_scanned": int(records_scanned),
                "records_reconsolidated": int(records_reconsolidated),
                "critic_calls": int(critic_calls),
                "dry_run_mode": bool(cfg.dry_run),
            },
            severity="info",
        )

        return True, {
            "records_scanned": int(records_scanned),
            "records_reconsolidated": int(records_reconsolidated),
            "dry_run": bool(cfg.dry_run),
        }

    def _step_user_model_update(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """User-model REM aggregation pass.

        Calls ``UserModelAggregator`` to compute structured fields over the
        last ``window_days`` of events + records, persists the model unless
        ``dry_run``, emits exactly one ``user_model_aggregate_pass`` event per
        invocation with the four count summary keys + window_days + dry_run_mode.

        Dry-run mode: event STILL emits with ``dry_run_mode=True``; only the
        ``save()`` call is skipped.

        Returns ``(True, {"topics_count": int, "dry_run": bool})``.
        """
        if self._check_interrupt(
            SleepStep.USER_MODEL_UPDATE, 0, interrupt_check,
        ):
            return False, {}

        # Config re-read on every invocation for monkeypatch.setenv test compat.
        from iai_mcp.daemon_config import _load_user_model_config
        from iai_mcp.user_model import UserModelAggregator, save
        from iai_mcp.events import write_event

        cfg = _load_user_model_config()
        agg = UserModelAggregator()
        model = agg.aggregate(
            self._store, window_days=cfg.aggregation_window_days,
        )

        if not cfg.dry_run:
            try:
                save(model)
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                # Defensive: persistence failure must not abort the
                # cycle. Surface the error in the event payload so
                # ops can see it; still return success so subsequent
                # REM steps (CRISIS_RECLUSTER) run.
                logger.warning("user_model_update save failed: %s", exc, exc_info=True)
                write_event(
                    self._store,
                    "user_model_aggregate_pass",
                    {
                        "topics_count": int(len(model.top_recent_topics)),
                        "tools_count": int(len(model.tool_usage_freq)),
                        "hours_count": int(len(model.time_of_day_pattern)),
                        "projects_count": int(len(model.recent_projects)),
                        "window_days": int(cfg.aggregation_window_days),
                        "dry_run_mode": False,
                        "persist_error": str(exc)[:500],
                    },
                    severity="warning",
                )
                return True, {
                    "topics_count": int(len(model.top_recent_topics)),
                    "dry_run": False,
                    "persist_error": True,
                }

        write_event(
            self._store,
            "user_model_aggregate_pass",
            {
                "topics_count": int(len(model.top_recent_topics)),
                "tools_count": int(len(model.tool_usage_freq)),
                "hours_count": int(len(model.time_of_day_pattern)),
                "projects_count": int(len(model.recent_projects)),
                "window_days": int(cfg.aggregation_window_days),
                "dry_run_mode": bool(cfg.dry_run),
            },
            severity="info",
        )

        return True, {
            "topics_count": int(len(model.top_recent_topics)),
            "dry_run": bool(cfg.dry_run),
        }

    def _step_dmn_reflection(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """DMN Reflection + Meta-Analyst REM pass.

        Two coupled second-order observers (Buckner DMN + Von Foerster
        meta-observer):

        * ``MetaAnalyst.snapshot`` -> ``system_health_report`` event (gated by
          ``cfg.meta_analyst_enabled``).
        * ``ReflectionAgent.synthesize`` -> fresh semantic ``MemoryRecord``
          (gated by ``not cfg.dry_run``; pytest-aware: insert is skipped when
          ``PYTEST_CURRENT_TEST`` is set).

        Both calls are wrapped in a single outer try/except so neither side
        aborts the overall sleep cycle: this is a non-critical reflection step.
        The outer error path emits a ``dmn_reflection_pass`` event at
        severity=warning and returns ``(True, {...persist_error: True})`` so
        the dispatcher continues to CRISIS_RECLUSTER.

        Dry-run mode: the synthesised record is computed but the
        ``store.insert`` call is skipped; the meta-analyst snapshot still runs
        and the result dict carries ``dry_run_mode=True``.

        Honors ``_check_interrupt`` between the two sub-steps.

        Returns ``(True, {"meta_analyst_emitted": bool,
        "reflection_synthesized": bool, "dry_run_mode": bool})``.
        """
        # Lazy daemon import keeps sleep_pipeline cheap to import for tooling.
        from iai_mcp.daemon_config import _load_dmn_config
        from iai_mcp.dmn_reflection import MetaAnalyst, ReflectionAgent
        from iai_mcp.events import write_event

        meta_analyst_emitted = False
        reflection_synthesized = False
        try:
            cfg = _load_dmn_config()

            # === MetaAnalyst =========================================
            if cfg.meta_analyst_enabled:
                snapshot = MetaAnalyst().snapshot(
                    self._store, cfg.reflection_window_hours,
                )
                # Surface the active dry-run mode in the health report
                # so ops dashboards can distinguish a quiet pass from a
                # suppressed insert.
                snapshot["dry_run_mode"] = bool(cfg.dry_run)
                write_event(
                    self._store,
                    "system_health_report",
                    snapshot,
                    severity="info",
                )
                meta_analyst_emitted = True

            # Interrupt gate between the two sub-passes.
            if self._check_interrupt(
                SleepStep.DMN_REFLECTION, 0, interrupt_check,
            ):
                return False, {}

            # === ReflectionAgent =====================================
            synth_record = ReflectionAgent().synthesize(
                self._store, cfg.reflection_window_hours,
            )
            if not cfg.dry_run:
                self._store.insert(synth_record)
                reflection_synthesized = True

            return True, {
                "meta_analyst_emitted": meta_analyst_emitted,
                "reflection_synthesized": reflection_synthesized,
                "dry_run_mode": bool(cfg.dry_run),
            }
        except Exception as exc:  # noqa: BLE001 -- non-critical DMN pass
            # Non-critical pass: swallow the error, emit a warning-severity
            # event for ops visibility, return success so CRISIS_RECLUSTER
            # still runs. Best-effort event emission -- nested try guards
            # against a closed events table during shutdown.
            logger.warning("dmn_reflection step failed: %s", exc, exc_info=True)
            try:
                write_event(
                    self._store,
                    "dmn_reflection_pass",
                    {
                        "meta_analyst_emitted": meta_analyst_emitted,
                        "reflection_synthesized": reflection_synthesized,
                        "persist_error": str(exc)[:500],
                    },
                    severity="warning",
                )
            except (OSError, ValueError) as inner_exc:
                logger.debug("best-effort dmn_reflection_pass event failed: %s", inner_exc)
            return True, {
                "meta_analyst_emitted": meta_analyst_emitted,
                "reflection_synthesized": reflection_synthesized,
                "persist_error": True,
            }

    def _step_crisis_recluster(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Emergency re-cluster.

        Short-circuits to no-op when ``lifecycle_state.crisis_mode`` is False.
        When True:

        1. Read all records' community_id assignments.
        2. Count community sizes; sort ascending by member count.
        3. Drop the bottom CRISIS_DROP_QUARTILE fraction of communities;
           clear community_id (set to None) on records in dropped communities.
        4. Build a fresh MemoryGraph from records + edges.
        5. Run ``detect_communities(prior_mode="cold")`` on the rebuilt graph;
           reassign community_id on every record per the new partition
           (fresh UUID per label).
        6. Call ``s2_coordinator.set_crisis_mode(False, 'crisis_recluster_complete')``
           via the S2 path; fall back to direct save_state on miss.
           Emit exactly one ``crisis_recluster_pass`` event with the 6-key body.

        Dry-run mode: with ``cfg.dry_run=True`` the candidate-drop set is still
        computed and the event is still emitted, but no community_id mutation
        happens and crisis_mode is NOT cleared.

        Returns ``(True, {"communities_dropped": int, "dry_run": bool})``.
        """
        if self._check_interrupt(
            SleepStep.CRISIS_RECLUSTER, 0, interrupt_check,
        ):
            return False, {}

        # Read crisis_mode FIRST. No-op (skip without event) when False.
        state_rec = self._load_state_record()
        if not state_rec.get("crisis_mode", False):
            return True, {"communities_dropped": 0, "crisis_mode": False}

        from iai_mcp.daemon_config import _load_sleep_overhaul_config
        cfg = _load_sleep_overhaul_config()
        drop_quartile = cfg.crisis_drop_quartile
        dry_run = cfg.dry_run

        from iai_mcp.events import write_event
        from iai_mcp.store import RECORDS_TABLE
        tbl = self._store.db.open_table(RECORDS_TABLE)

        # Step 1+2: count communities by member size.
        try:
            df = tbl.search().to_pandas()
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("crisis_recluster records query failed: %s", exc)
            df = None

        communities_dropped = 0
        records_reassigned = 0
        new_community_count = 0
        modularity = 0.0
        backend = "flat"

        if df is not None and not df.empty and "community_id" in df.columns:
            non_null = df[df["community_id"].notna()]
            if not non_null.empty:
                sizes = (
                    non_null.groupby("community_id").size().sort_values()
                )
                total_communities = len(sizes)
                n_to_drop = int(total_communities * drop_quartile)
                drop_ids = list(sizes.index[:n_to_drop])
                communities_dropped = n_to_drop

                # Step 3: clear community_id on dropped-community records
                # (only outside dry-run).
                if drop_ids and not dry_run:
                    for cid in drop_ids:
                        try:
                            tbl.update(
                                where=f"community_id = '{str(cid)}'",
                                values={"community_id": None},
                            )
                        except (OSError, ValueError, RuntimeError, StoreError):
                            # Defensive: keep going; the event reflects
                            # actual reassignment count.
                            pass

                # Step 4+5: rebuild graph, run Leiden, reassign.
                if not dry_run:
                    # Re-open the table post-update so subsequent scans
                    # see the cleared ids.
                    tbl = self._store.db.open_table(RECORDS_TABLE)
                    try:
                        df2 = tbl.search().to_pandas()
                    except (OSError, ValueError, RuntimeError, StoreError):
                        df2 = df

                    try:
                        # crisis_recluster intentionally DISCARDS the prior partition
                        # (which is the broken topology that triggered crisis).
                        # prior_mode="cold" in detect_communities enforces this.
                        from iai_mcp.community import detect_communities
                        from iai_mcp.graph import MemoryGraph
                        from iai_mcp.store import EDGES_TABLE
                        import uuid as _uuid

                        g = MemoryGraph()
                        # Add nodes from records (skip rows with bad ids).
                        for _, row in df2.iterrows():
                            try:
                                rid = _uuid.UUID(str(row["id"]))
                                emb = row.get("embedding")
                                emb_list = (
                                    list(emb) if emb is not None else []
                                )
                                # MemoryGraph.add_node(node_id, community_id, embedding).
                                # Re-Leiden will reassign community_id, so pass None.
                                g.add_node(rid, None, emb_list)
                            except (ValueError, TypeError, AttributeError):
                                continue

                        # Walk edges table to populate the graph.
                        try:
                            edges_df = (
                                self._store.db.open_table(EDGES_TABLE)
                                .search()
                                .to_pandas()
                            )
                            for _, e in edges_df.iterrows():
                                try:
                                    src_u = _uuid.UUID(str(e["src"]))
                                    dst_u = _uuid.UUID(str(e["dst"]))
                                    g.add_edge(
                                        src_u, dst_u,
                                        weight=float(
                                            e.get("weight", 1.0) or 1.0
                                        ),
                                    )
                                except (ValueError, TypeError, KeyError):
                                    continue
                        except (OSError, ValueError, RuntimeError, StoreError) as exc:
                            logger.debug("crisis_recluster edges query failed: %s", exc)

                        # prior_mode="cold" discards the prior partition;
                        # the new CommunityAssignment carries fresh UUIDs
                        # per surviving community.
                        _assignment = detect_communities(
                            g, prior=None, prior_mode="cold"
                        )
                        modularity = float(_assignment.modularity)
                        backend = _assignment.backend
                        # Reconstruct the legacy (node_uuid -> int label)
                        # partition dict for the existing downstream code
                        # which iterates `partition.items()` to compute
                        # `new_community_count` + reassign records via
                        # the records table update loop.
                        _uuid_to_int: dict[_uuid.UUID, int] = {}
                        _next_int = 0
                        partition: dict[_uuid.UUID, int] = {}
                        for _node_uuid, _comm_uuid in _assignment.node_to_community.items():
                            if _comm_uuid not in _uuid_to_int:
                                _uuid_to_int[_comm_uuid] = _next_int
                                _next_int += 1
                            partition[_node_uuid] = _uuid_to_int[_comm_uuid]
                        # Stable UUID mapping: discard the prior CommunityAssignment
                        # (the broken topology that triggered crisis). Assign fresh
                        # UUIDs per label so consumers see a clean partition.
                        new_uuids: dict[int, str] = {}
                        for node, lbl in partition.items():
                            if lbl not in new_uuids:
                                new_uuids[lbl] = str(_uuid.uuid4())
                            new_cid = new_uuids[lbl]
                            try:
                                tbl.update(
                                    where=f"id = '{str(node)}'",
                                    values={"community_id": new_cid},
                                )
                                records_reassigned += 1
                            except (OSError, ValueError, RuntimeError, StoreError):
                                continue
                        new_community_count = len(new_uuids)
                    except Exception as exc:  # noqa: BLE001 -- Leiden/graph rebuild
                        # Leiden failure -- count the drop, leave
                        # reassignment at zero, surface in event below.
                        logger.warning("crisis_recluster Leiden rebuild failed: %s", exc, exc_info=True)

        # Step 6: clear crisis_mode (via S2Coordinator if available;
        # direct save_state as a fallback). Skipped under dry-run.
        if not dry_run:
            cleared = self._clear_crisis_mode_via_s2_or_fallback(
                reason="crisis_recluster_complete",
            )
            if not cleared:
                # Last-resort direct write so the system doesn't get
                # stuck in crisis_mode forever if both S2 and the save
                # path break.
                try:
                    rec = self._load_state_record()
                    rec["crisis_mode"] = False
                    self._save_state_record(rec)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("crisis_mode clear last-resort write failed: %s", exc)

        write_event(
            self._store,
            "crisis_recluster_pass",
            {
                "communities_dropped": int(communities_dropped),
                "records_reassigned": int(records_reassigned),
                "new_community_count": int(new_community_count),
                "modularity": float(modularity),
                "backend": str(backend),
                "dry_run_mode": bool(dry_run),
            },
            severity="warning" if communities_dropped > 0 else "info",
        )

        return True, {
            "communities_dropped": int(communities_dropped),
            "dry_run": bool(dry_run),
        }

    def _clear_crisis_mode_via_s2_or_fallback(self, *, reason: str) -> bool:
        """Helper: route the False crisis_mode write through the S2
        coordinator if one was injected at __init__; otherwise return
        False (caller does the direct save_state fallback).

        Returns True iff the coordinator path was taken AND completed
        without raising. False otherwise (and caller falls back).
        """
        s2 = getattr(self, "_s2_coordinator", None)
        loop = getattr(self, "_loop", None)
        if s2 is None:
            return False
        try:
            import asyncio
            coro = s2.set_crisis_mode(False, reason)
            if loop is not None and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                fut.result(timeout=5.0)
            else:
                asyncio.run(coro)
            return True
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("S2 clear_crisis_mode failed, falling back: %s", exc)
            return False

    def _set_crisis_mode_via_s2_or_fallback(
        self, *, value: bool, reason: str,
    ) -> bool:
        """Mirror of _clear_crisis_mode_via_s2_or_fallback for the True path.

        Returns True iff the coordinator path was taken AND completed
        without raising. Falls back to direct save_state on miss.
        """
        s2 = getattr(self, "_s2_coordinator", None)
        loop = getattr(self, "_loop", None)
        if s2 is not None:
            try:
                import asyncio
                coro = s2.set_crisis_mode(value, reason)
                if loop is not None and loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                    fut.result(timeout=5.0)
                else:
                    asyncio.run(coro)
                return True
            except (OSError, RuntimeError, TimeoutError) as exc:
                logger.debug("S2 set_crisis_mode failed, falling back: %s", exc)
        # Fallback: direct save_state.
        try:
            rec = self._load_state_record()
            rec["crisis_mode"] = bool(value)
            self._save_state_record(rec)
            return False
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("crisis_mode fallback save_state failed: %s", exc)
            return False

    def _step_cluster_summary(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step CLUSTER_SUMMARY: hebbian cluster summarisation + LTP.

        Runs the EXACT legacy connected-component cluster pass via the shared
        ``sleep._process_cluster_summaries`` helper (single-source — no
        clean-room reimplementation). Produces:
        - Semantic MemoryRecords with tier=="semantic" + tags ["semantic",
          "cls_summary"] for qualifying clusters (size >= CLUSTER_MIN_SIZE).
        - ``consolidated_from`` edges linking each summary to its source episodes.
        - Hebbian LTP boost (HEAVY_LTP_DELTA) on edges between co-cluster members.

        Does NOT re-run decay (DREAM_DECAY owns that) or schema induction
        (SCHEMA_MINE owns that). Single-chunk implementation; interrupt_check
        is checked at chunk_idx=0 before the call.

        Returns ``(completed, {"summaries_created": int})``.
        """
        from iai_mcp.sleep import _process_cluster_summaries

        if self._check_interrupt(SleepStep.CLUSTER_SUMMARY, 0, interrupt_check):
            return False, {}

        try:
            summaries_created = _process_cluster_summaries(self._store)
        except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
            logger.warning("cluster_summary step failed: %s", exc, exc_info=True)
            summaries_created = 0

        return True, {"summaries_created": summaries_created}

    def _step_recall_index_rebuild(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """Step RECALL_INDEX_REBUILD: MEDIUM-2 final-topology rebuild.

        Re-detects mosaic communities + rich-club from SQLite ground truth
        AFTER all REM edge mutations (DREAM_DECAY / ERASURE_AGENT /
        CLUSTER_REPLAY / RECONSOLIDATION / CLUSTER_SUMMARY) so the derived
        RecallIndex carries the fully-settled edge topology.

        Stamps a fresh generation epoch + rebuild_timestamp into the snapshot
        and RESETS the in-process record-mutation dirty counter to zero, so
        the O(1) freshness fuse measures age/delta from THIS rebuild.

        Uses the mosaic community detection path (community.py /
        lilli.graph) — not networkx, which is a dev-only dependency.

        On any error, returns (True, {"error": str}) so the pipeline step
        is marked completed (a failed topology snapshot is advisory; the
        Layer-1 path still answers correctly).
        """
        if self._check_interrupt(SleepStep.RECALL_INDEX_REBUILD, 0, interrupt_check):
            return False, {}

        try:
            from iai_mcp import runtime_graph_cache

            # Delegate to the shared rebuild helper.  This sees ALL edge
            # mutations from this nightly cycle, including edges the record-only
            # sync hook cannot observe.
            result = runtime_graph_cache._rebuild_and_save_rgc(self._store)
            return True, result

        except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
            logger.warning(
                "recall_index_rebuild step failed: %s", exc, exc_info=True,
            )
            return True, {"error": str(exc)[:200], "rebuilt": False}

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
            SleepStep.ERASURE_AGENT: self._step_erasure_agent,
            SleepStep.OPTIMIZE_LANCE: self._step_optimize_lance,
            SleepStep.COMPACT_RECORDS: self._step_compact_records,
            SleepStep.CLUSTER_REPLAY: self._step_cluster_replay,
            SleepStep.RECONSOLIDATION: self._step_reconsolidation,
            SleepStep.USER_MODEL_UPDATE: self._step_user_model_update,
            SleepStep.DMN_REFLECTION: self._step_dmn_reflection,
            SleepStep.CRISIS_RECLUSTER: self._step_crisis_recluster,
            SleepStep.CLUSTER_SUMMARY: self._step_cluster_summary,
            SleepStep.RECALL_INDEX_REBUILD: self._step_recall_index_rebuild,
        }

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    # Execution order for run() and force_run(). Tuple is fixed so neither
    # path can accidentally execute steps out of order. NREM phase runs fully
    # before REM phase. CRISIS_RECLUSTER is last (no-op when crisis_mode=False).
    # Resume math uses _STEP_ORDER.index(step), so APPENDed steps + reordered
    # tuple remain safe so long as every member is present here.
    _STEP_ORDER: tuple[SleepStep, ...] = (
        # NREM phase (stabilization)
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_LANCE,
        SleepStep.COMPACT_RECORDS,
        # REM phase (pruning + abstraction)
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        # CLUSTER_SUMMARY appended at the end — safe per resume math comment
        # above (APPENDed steps do not shift last_completed_index of prior
        # steps). Runs after DREAM_DECAY (decay-then-cluster ordering preserved)
        # and SCHEMA_MINE (schema-mine-before-emit ordering preserved).
        SleepStep.CLUSTER_SUMMARY,
        # RECALL_INDEX_REBUILD appended AFTER CLUSTER_SUMMARY (MEDIUM-2):
        # re-detects mosaic from SQLite ground truth after all REM edge
        # mutations, stamps a fresh generation epoch, resets the dirty counter.
        # Must be LAST so no subsequent step mutates edges after the rebuild.
        SleepStep.RECALL_INDEX_REBUILD,
    )

    # 3-strike threshold: the SAME step failing this many consecutive
    # times triggers 24h auto-quarantine.
    _QUARANTINE_STRIKE_THRESHOLD: int = 3

    def run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        """Run the sleep pipeline (auto-quarantine respected).

        Behaviour summary:

        1. If ``is_quarantined()``: return immediately with
           `quarantine_triggered=True` and `completed_steps=[]`. The
           caller is expected to surface this in CLI output / doctor row.

        2. Auto-recovery: if `quarantine` exists but `until_ts` is in
           the past, clear it (logged as `quarantine_lifted`,
           reason=`auto_recovery_after_ttl`) and proceed.

        3. Determine resume point from `_load_progress()`:
           - No progress record OR last_completed_index < 0 → start at
             SCHEMA_MINE (position 0).
           - last_completed_index == K (0 ≤ K < len(_STEP_ORDER)-1) →
             start at _STEP_ORDER[K+1].
           - last_completed_index == len(_STEP_ORDER)-1 → fresh cycle
             (start at position 0); we treat a successful prior run
             that was never cleared as a fresh start, not a no-op.

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
             with last_completed_index=_STEP_ORDER.index(step)
             (attempt reset to 0).

        5. On full success: clear progress (sleep_cycle_progress=None).

        Failure isolation: the helper functions used by step bodies
        already have their own "never-raise" disciplines where
        applicable (e.g. `optimize_hippo_storage`); this method's
        try/except is a defense-in-depth wrapper around the whole
        step call.
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
            return {
                "completed_steps": [],
                "failed_step": None,
                "error": None,
                "duration_sec": round(time.monotonic() - t0, 3),
                "quarantine_triggered": True,
                "interrupted": False,
            }

        # Essential-variable tracker: runs once per cycle invocation BEFORE
        # the step loop so crisis_mode is set before _step_crisis_recluster
        # runs later in the same cycle. Best-effort — must NOT abort the cycle.
        try:
            self._run_essential_variable_tracker_hook()
        except Exception as exc:  # noqa: BLE001 -- tracker is best-effort observer
            # Visible via the sleep_step_completed events that follow;
            # the tracker is a best-effort observer, not a hard gate.
            logger.warning("essential_variable_tracker hook failed: %s", exc, exc_info=True)

        # Determine resume step from persisted progress.
        # Index-based resume math: ``_load_progress`` applies a one-shot
        # migration from the legacy ``last_completed_step`` (step.value) key;
        # here we read the canonical ``last_completed_index`` (position into
        # ``_STEP_ORDER``). Default -1 = "no step has completed yet → start at 0".
        progress = self._load_progress()
        last_completed_index = (
            int(progress.get("last_completed_index", -1))
            if progress is not None
            else -1
        )
        # Position-based wrap-detection: compare against tuple length, not
        # against a specific enum value (the last step's value may not equal
        # the last tuple position after APPEND-without-renumber inserts).
        if last_completed_index >= len(self._STEP_ORDER) - 1:
            last_completed_index = -1
        resume_step_index = last_completed_index + 1

        # Accumulate per-step payloads so the clean-completion path can
        # emit a pipeline-level cls_consolidation_run event with the
        # correct aggregate values. Fresh dict per _run_internal call so
        # a resumed run that starts mid-pipeline only has the steps it
        # completed in this invocation; missing-key defaults are used for
        # the emit in that case (degraded event — acceptable transient).
        step_payloads: dict[SleepStep, dict] = {}

        # Execute steps in order, skipping any whose position < resume.
        for step in self._STEP_ORDER:
            if self._STEP_ORDER.index(step) < resume_step_index:
                continue

            self._emit_step_started(step)
            step_t0 = time.monotonic()
            method = self._step_methods[step]
            try:
                done, payload = method(interrupt_check)
            except Exception as exc:  # noqa: BLE001 -- 3-strike + quarantine flow
                logger.error("sleep step %s failed: %s", step.name, exc, exc_info=True)
                err_str = str(exc)[:500]
                # Increment attempt counter for THIS step. If the prior
                # progress record's last_completed_index matches
                # idx(step)-1, we are failing the same step; attempt
                # counter persists and we add 1. If it differs (e.g.
                # resumed from a different step that just succeeded
                # above), reset to 1.
                # Position-based strike check + persist.
                prior = self._load_progress() or {}
                prior_last_index = int(prior.get("last_completed_index", -1))
                step_idx = self._STEP_ORDER.index(step)
                if prior_last_index == step_idx - 1:
                    new_attempt = int(prior.get("attempt", 0)) + 1
                else:
                    new_attempt = 1
                self._save_progress(
                    last_completed_index=step_idx - 1,
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
            # fails, prior_last_index will equal idx(step), so the
            # failure branch above will correctly start its own counter
            # at 1).
            self._save_progress(
                last_completed_index=self._STEP_ORDER.index(step),
                attempt=0,
                last_error=None,
            )
            self._emit_step_completed(
                step,
                duration_sec=time.monotonic() - step_t0,
                **payload,
            )
            completed_steps.append(step)
            # Accumulate payload for the pipeline-level cls event.
            step_payloads[step] = payload

        # All steps completed cleanly. Emit one pipeline-level
        # cls_consolidation_run event aggregating the key step outputs.
        # Emitted ONLY on clean completion (not on interrupt/failure) so
        # the row represents a full cycle. On resume, step_payloads only
        # has the steps completed in THIS invocation; missing keys default
        # to 0 (degraded event — the learning outputs still landed in
        # their respective step_completed events).
        try:
            from iai_mcp.sleep import _emit_cls_consolidation_run

            _decay_payload = step_payloads.get(SleepStep.DREAM_DECAY, {})
            _schema_payload = step_payloads.get(SleepStep.SCHEMA_MINE, {})
            _cluster_payload = step_payloads.get(SleepStep.CLUSTER_SUMMARY, {})

            _emit_cls_consolidation_run(
                self._store,
                "system",
                summaries_created=int(_cluster_payload.get("summaries_created", 0)),
                decay_result={
                    "decayed": int(_decay_payload.get("decayed", 0)),
                    "pruned": int(_decay_payload.get("pruned", 0)),
                },
                schema_candidates=int(_schema_payload.get("schemas_induced", 0)),
                schemas_induced=int(_schema_payload.get("schemas_persisted", 0)),
            )
        except Exception as exc:  # noqa: BLE001 -- cls emit is best-effort introspection
            logger.debug("pipeline-level cls_consolidation_run emit failed: %s", exc)

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

    def _run_essential_variable_tracker_hook(self) -> None:
        """Build a TopologySnapshot, invoke EssentialVariableTracker,
        emit essential_variable_breach event(s), and set crisis_mode if
        any breach detected.

        Best-effort: any exception is swallowed by the caller in
        _run_internal. The crisis_mode mutation is routed through
        S2Coordinator.set_crisis_mode when the coordinator was injected
        at __init__; otherwise falls back to direct save_state.
        Idempotence: crisis_mode is set ONCE per cycle on the first
        breach; subsequent breaches in the same dict get reported in
        their own events but do NOT re-issue the set (the field is
        already True from breach #1).
        """
        from iai_mcp.daemon_config import _load_sleep_overhaul_config
        from iai_mcp.ashby_step import (
            EssentialVariableTracker,
            TopologySnapshot,
        )
        from iai_mcp.graph import MemoryGraph
        from iai_mcp.events import write_event
        from iai_mcp.store import RECORDS_TABLE, EDGES_TABLE

        cfg = _load_sleep_overhaul_config()
        dry_run = cfg.dry_run

        # Build a fresh MemoryGraph from records + edges. Same shape as
        # _step_crisis_recluster's rebuild path -- duplicated rather
        # than extracted-to-helper to keep this hook self-contained for
        # the fail-safe try/except in the caller.
        try:
            recs = (
                self._store.db.open_table(RECORDS_TABLE)
                .search().to_pandas()
            )
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("essential_variable_tracker records query failed: %s", exc)
            return
        if recs.empty:
            return

        import uuid as _uuid
        g = MemoryGraph()
        community_ids: set = set()
        # Collect embeddings per community for the optional
        # detect_hubness diagnostic emitted at the end of this hook
        # (gated on IAI_MCP_ORTHO_ENABLED=1). Build the dict in the same
        # iteration loop to avoid a second pass over records.
        _community_embeddings: dict[str, list[list[float]]] = {}
        for _, row in recs.iterrows():
            try:
                rid = _uuid.UUID(str(row["id"]))
                emb = row.get("embedding")
                emb_list = list(emb) if emb is not None else []
                # MemoryGraph.add_node(node_id, community_id, embedding)
                # -- 3 positional args; pass current community_id when
                # present so the snapshot reflects the existing
                # partition for community_count counting purposes.
                cid_raw = row.get("community_id")
                cid_uuid: _uuid.UUID | None
                if cid_raw is not None:
                    try:
                        cid_uuid = _uuid.UUID(str(cid_raw))
                        _cid_str = str(cid_uuid)
                        community_ids.add(_cid_str)
                        if emb_list:
                            _community_embeddings.setdefault(
                                _cid_str, []
                            ).append(emb_list)
                    except (ValueError, TypeError):
                        cid_uuid = None
                else:
                    cid_uuid = None
                g.add_node(rid, cid_uuid, emb_list)
            except (ValueError, TypeError, AttributeError):
                continue

        try:
            edges_df = (
                self._store.db.open_table(EDGES_TABLE).search().to_pandas()
            )
            for _, e in edges_df.iterrows():
                try:
                    src_u = _uuid.UUID(str(e["src"]))
                    dst_u = _uuid.UUID(str(e["dst"]))
                    g.add_edge(
                        src_u, dst_u,
                        weight=float(e.get("weight", 1.0) or 1.0),
                    )
                except (ValueError, TypeError, KeyError):
                    continue
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("essential_variable_tracker edges query failed: %s", exc)

        total_nodes = g.node_count()
        if total_nodes == 0:
            return

        try:
            rc_ratio = g.rich_club_coefficient()
        except (ValueError, RuntimeError, ZeroDivisionError) as exc:
            logger.debug("rich_club_coefficient failed: %s", exc)
            rc_ratio = 0.0
        nedges = sum(1 for _ in g.iter_edges_with_weight())
        edge_density = (
            (2.0 * nedges) / (total_nodes * (total_nodes - 1))
            if total_nodes >= 2 else 0.0
        )

        snapshot = TopologySnapshot(
            rich_club_ratio=float(rc_ratio),
            community_count=int(len(community_ids)),
            edge_density=float(edge_density),
            total_nodes=int(total_nodes),
        )
        tracker = EssentialVariableTracker(cfg)
        breaches = tracker.check(snapshot)

        # Set crisis_mode True ONCE per cycle (the first detected
        # breach is the trigger). Subsequent breaches in the same dict
        # accompany via additional events but do NOT re-issue the set
        # because the field is already True. Single-set-per-cycle contract.
        crisis_mode_already_set_this_cycle = False
        for var_name, breach in breaches.items():
            if breach is None:
                continue
            crisis_mode_set = False
            if not dry_run and not crisis_mode_already_set_this_cycle:
                self._set_crisis_mode_via_s2_or_fallback(
                    value=True,
                    reason=f"essential_variable_breach:{var_name}",
                )
                crisis_mode_already_set_this_cycle = True
                crisis_mode_set = True
            elif not dry_run and crisis_mode_already_set_this_cycle:
                # Second+ breach this cycle -- crisis_mode is already
                # True; don't re-set, but mark crisis_mode_set=True so
                # the event body reports correctly (the field IS True
                # because of breach #1).
                crisis_mode_set = True
            write_event(
                self._store,
                "essential_variable_breach",
                {
                    "variable_name": str(var_name),
                    "observed_value": float(breach.observed_value),
                    "threshold": float(breach.threshold),
                    "direction": str(breach.direction),
                    "total_nodes": int(total_nodes),
                    "crisis_mode_set": bool(crisis_mode_set),
                    "dry_run_mode": bool(dry_run),
                },
                severity="warning",
            )

        # detect_hubness diagnostic: read-only, non-blocking, fail-safe.
        # Emits at most ONE event per cycle on the largest community when
        # IAI_MCP_ORTHO_ENABLED=1. Cap at top-100 embeddings to bound the
        # O(n^2) cosine cost. Reuses `_community_embeddings` built during
        # the records iteration above, so no second pandas pass.
        if os.environ.get(
            "IAI_MCP_ORTHO_ENABLED", "",
        ).lower() in {"1", "true"}:
            try:
                from iai_mcp.pattern_separation import detect_hubness
                if _community_embeddings:
                    _largest_cid = max(
                        _community_embeddings,
                        key=lambda k: len(_community_embeddings[k]),
                    )
                    _largest = _community_embeddings[_largest_cid][:100]
                    if len(_largest) >= 2:
                        _hubness = detect_hubness(_largest, threshold=0.85)
                        write_event(
                            self._store,
                            "community_hubness_diagnostic",
                            {
                                "community_id": _largest_cid,
                                "mean_similarity": float(
                                    _hubness.get("mean_similarity", 0.0)
                                ),
                                "max_similarity": float(
                                    _hubness.get("max_similarity", 0.0)
                                ),
                                "is_hub": bool(_hubness.get("is_hub", False)),
                                "size": int(_hubness.get("size", 0)),
                            },
                            severity="info",
                        )
            except Exception as _hub_exc:  # noqa: BLE001 -- diagnostic MUST NOT crash sleep
                logger.debug(
                    "detect_hubness diagnostic skipped: %s",
                    str(_hub_exc)[:120],
                )

