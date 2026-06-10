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
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import (
        LifecycleStateRecord,
        Quarantine,
        SleepCycleProgress,
    )

logger = logging.getLogger(__name__)


QUARANTINE_TTL_HOURS_DEFAULT: float = float(
    os.environ.get("IAI_MCP_SLEEP_QUARANTINE_TTL_HOURS", "24"),
)


class SleepStep(Enum):

    SCHEMA_MINE = 1
    KNOB_TUNE = 2
    DREAM_DECAY = 3
    OPTIMIZE_LANCE = 4
    COMPACT_RECORDS = 5
    ERASURE_AGENT = 6
    CLUSTER_REPLAY = 7
    CRISIS_RECLUSTER = 8
    RECONSOLIDATION = 9
    USER_MODEL_UPDATE = 10
    DMN_REFLECTION = 11
    CLUSTER_SUMMARY = 12
    RECALL_INDEX_REBUILD = 13


class SleepPhase(Enum):

    NREM = "NREM"
    REM = "REM"


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
    SleepStep.CLUSTER_SUMMARY: SleepPhase.REM,
    SleepStep.RECALL_INDEX_REBUILD: SleepPhase.REM,
}


MAX_PAIRS_PER_CLUSTER: int = 100


class SleepPipelineResult(TypedDict, total=False):

    completed_steps: list[SleepStep]
    failed_step: SleepStep | None
    error: str | None
    duration_sec: float
    quarantine_triggered: bool
    interrupted: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class SleepPipeline:

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

        self._lifecycle_state_path: Path | None = lifecycle_state_path

        self._lel: Any | None = lifecycle_event_log if lifecycle_event_log is not None else event_log

        self._quarantine_ttl_hours = (
            float(quarantine_ttl_hours)
            if quarantine_ttl_hours is not None
            else QUARANTINE_TTL_HOURS_DEFAULT
        )
        self._s2_coordinator = s2_coordinator
        self._loop = loop

    def _get_state_path(self) -> Path:
        if self._lifecycle_state_path is not None:
            return self._lifecycle_state_path
        from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
        return LIFECYCLE_STATE_PATH

    def _get_event_log(self) -> Any:
        if self._lel is not None:
            return self._lel
        from iai_mcp.lifecycle_event_log import LifecycleEventLog
        self._lel = LifecycleEventLog()
        return self._lel

    @property
    def _event_log(self) -> Any:
        return self._get_event_log()


    def _load_state_record(self) -> Any:
        from iai_mcp.lifecycle_state import load_state
        return load_state(self._get_state_path())

    def _save_state_record(self, record: Any) -> None:
        from iai_mcp.lifecycle_state import save_state
        save_state(record, self._get_state_path())

    def _load_quarantine(self) -> Quarantine | None:
        return self._load_state_record().get("quarantine")

    def _set_quarantine(self, reason: str) -> Quarantine:
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
        record = self._load_state_record()
        prior_quarantine = record.get("quarantine")
        record["quarantine"] = None
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
        quarantine = self._load_quarantine()
        if quarantine is None:
            return False
        try:
            until = datetime.fromisoformat(quarantine["until_ts"])
        except (TypeError, ValueError):
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return _utc_now() < until

    def reset_quarantine(self) -> None:
        self._clear_quarantine(reason="manual_reset")


    def _load_progress(self) -> Any:
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
        record = self._load_state_record()
        record["sleep_cycle_progress"] = None
        self._save_state_record(record)


    def _emit_step_started(self, step: SleepStep) -> None:
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
        if interrupt_check is None:
            return False
        try:
            should = bool(interrupt_check())
        except Exception as exc:  # noqa: BLE001 -- caller predicate may raise anything
            logger.debug("interrupt_check predicate raised: %s", exc)
            should = False
        if not should:
            return False
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
        from iai_mcp.schema import induce_schemas_tier0
        from iai_mcp.sleep import _persist_tier1_schemas

        if self._check_interrupt(SleepStep.SCHEMA_MINE, 0, interrupt_check):
            return False, {}
        candidates = induce_schemas_tier0(self._store)
        try:
            count = len(candidates) if candidates is not None else 0
        except (TypeError, AttributeError) as exc:
            logger.debug("non-critical schema count failed: %s", exc)
            count = 0

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
        from iai_mcp.profile import PROFILE_KNOBS, default_state

        knob_names = sorted(PROFILE_KNOBS.keys())
        snapshot = default_state()
        for chunk_idx, name in enumerate(knob_names):
            if self._check_interrupt(
                SleepStep.KNOB_TUNE, chunk_idx, interrupt_check,
            ):
                return False, {}
            _ = snapshot.get(name)

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
        from iai_mcp.sleep import _decay_edges

        if self._check_interrupt(SleepStep.DREAM_DECAY, 0, interrupt_check):
            return False, {}
        _plasticity = 1.0
        try:
            from iai_mcp.user_model import load as _load_um
            _um = _load_um()
            _plasticity = getattr(_um, "plasticity_gain", 1.0) or 1.0
        except (OSError, ValueError, RuntimeError, StoreError, AttributeError) as exc:
            logger.debug("non-critical plasticity_gain load failed: %s", exc)
        result = _decay_edges(self._store, plasticity_gain=_plasticity)
        if isinstance(result, dict):
            return True, {
                "decayed": int(result.get("decayed", 0) or 0),
                "pruned": int(result.get("pruned", 0) or 0),
            }
        return True, {}

    def _step_erasure_agent(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        if self._check_interrupt(
            SleepStep.ERASURE_AGENT, 0, interrupt_check,
        ):
            return False, {}

        from iai_mcp.daemon_config import _load_erasure_config
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

        try:
            count_quarantined = int(tbl.count_rows(filter=eligibility_where))
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("erasure_agent count_rows failed: %s", exc)
            count_quarantined = 0
        total_records_after = int(tbl.count_rows())

        from iai_mcp.events import query_events, write_event
        prior_drops = query_events(
            self._store, kind="erasure_optimize_drops", limit=1,
        )
        count_dropped = 0
        if prior_drops:
            prior_body = prior_drops[0].get("data") or {}
            count_dropped = int(prior_body.get("count_dropped", 0) or 0)

        if not dry_run and count_quarantined > 0:
            try:
                tbl.update(
                    where=eligibility_where,
                    values={"tombstoned_at": now},
                )
            except Exception as exc:  # noqa: BLE001 -- visibility over crash
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
        from iai_mcp.maintenance import optimize_hippo_storage

        if self._check_interrupt(
            SleepStep.OPTIMIZE_LANCE, 0, interrupt_check,
        ):
            return False, {}

        compact_t0 = time.monotonic()
        report = optimize_hippo_storage(self._store)
        tables_with_errors = [
            t for t, r in (report or {}).items()
            if isinstance(r, dict) and "error" in r
        ]

        from iai_mcp.daemon_config import _load_erasure_config
        cfg = _load_erasure_config()
        ttl_sec = cfg.tombstone_ttl_sec

        now = _utc_now()
        drop_cutoff = now - timedelta(seconds=ttl_sec)

        from iai_mcp.store import RECORDS_TABLE
        from iai_mcp.events import write_event

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
                logger.debug("compact_hippo untombstone update failed: %s", exc)
                count_untombstoned = 0

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
                logger.debug("compact_hippo drop delete failed: %s", exc)
                count_dropped = 0

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
        if self._check_interrupt(
            SleepStep.COMPACT_RECORDS, 0, interrupt_check,
        ):
            return False, {}
        return True, {"action": "noop_under_hippo"}

    def _step_optimize_lance(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        return self._step_compact_hippo(interrupt_check)

    def _step_compact_records(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        return self._step_compact_records_noop(interrupt_check)

    def _step_cluster_replay(
        self, interrupt_check: Callable[[], bool] | None,
    ) -> tuple[bool, dict[str, Any]]:
        if self._check_interrupt(
            SleepStep.CLUSTER_REPLAY, 0, interrupt_check,
        ):
            return False, {}

        from iai_mcp.daemon_config import _load_sleep_overhaul_config
        cfg = _load_sleep_overhaul_config()
        window_sec = cfg.cluster_window_sec
        delta = cfg.cluster_replay_initial_weight
        dry_run = cfg.dry_run
        lookback_windows = 5

        from iai_mcp.events import write_event
        from iai_mcp.store import RECORDS_TABLE

        now = _utc_now()
        lookback_cutoff = now - timedelta(seconds=window_sec * lookback_windows)
        tbl = self._store.db.open_table(RECORDS_TABLE)

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

        from itertools import combinations
        import uuid as _uuid

        replay_clusters = [c for c in clusters if len(c) >= 2]
        total_pairs = 0
        capped_count = 0
        all_pairs: list[tuple[Any, Any]] = []
        for c in replay_clusters:
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
        if self._check_interrupt(
            SleepStep.RECONSOLIDATION, 0, interrupt_check,
        ):
            return False, {}

        from iai_mcp.daemon_config import _load_reconsolidation_config
        cfg = _load_reconsolidation_config()

        from iai_mcp.events import write_event
        from iai_mcp.store import RECORDS_TABLE
        from iai_mcp.reconsolidation_critic import evaluate_batch_reconsolidation
        import uuid as _uuid

        now = _utc_now()
        tbl = self._store.db.open_table(RECORDS_TABLE)

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

            try:
                errors_by_id = evaluate_batch_reconsolidation(
                    pool,
                    llm_enabled=True,
                )
            except Exception as exc:  # noqa: BLE001 -- critic must never raise into REM
                logger.debug("reconsolidation batch call raised: %s", exc)
                errors_by_id = {}

            critic_calls = 1 if errors_by_id else 0

            for rid, err in errors_by_id.items():
                if err < float(cfg.reconsolidation_error_threshold):
                    continue
                if cfg.dry_run:
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
                    self._store.reinforce_record(rid)
                    records_reconsolidated += 1
                except (OSError, ValueError, RuntimeError, StoreError) as exc:
                    logger.debug("reconsolidation per-record write failed: %s", exc)

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
        if self._check_interrupt(
            SleepStep.USER_MODEL_UPDATE, 0, interrupt_check,
        ):
            return False, {}

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
        from iai_mcp.daemon_config import _load_dmn_config
        from iai_mcp.dmn_reflection import MetaAnalyst, ReflectionAgent
        from iai_mcp.events import write_event

        meta_analyst_emitted = False
        reflection_synthesized = False
        try:
            cfg = _load_dmn_config()

            if cfg.meta_analyst_enabled:
                snapshot = MetaAnalyst().snapshot(
                    self._store, cfg.reflection_window_hours,
                )
                snapshot["dry_run_mode"] = bool(cfg.dry_run)
                write_event(
                    self._store,
                    "system_health_report",
                    snapshot,
                    severity="info",
                )
                meta_analyst_emitted = True

            if self._check_interrupt(
                SleepStep.DMN_REFLECTION, 0, interrupt_check,
            ):
                return False, {}

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
        if self._check_interrupt(
            SleepStep.CRISIS_RECLUSTER, 0, interrupt_check,
        ):
            return False, {}

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

                if drop_ids and not dry_run:
                    for cid in drop_ids:
                        try:
                            tbl.update(
                                where=f"community_id = '{str(cid)}'",
                                values={"community_id": None},
                            )
                        except (OSError, ValueError, RuntimeError, StoreError):
                            pass

                if not dry_run:
                    tbl = self._store.db.open_table(RECORDS_TABLE)
                    try:
                        df2 = tbl.search().to_pandas()
                    except (OSError, ValueError, RuntimeError, StoreError):
                        df2 = df

                    try:
                        from iai_mcp.community import detect_communities
                        from iai_mcp.graph import MemoryGraph
                        from iai_mcp.store import EDGES_TABLE
                        import uuid as _uuid

                        g = MemoryGraph()
                        for _, row in df2.iterrows():
                            try:
                                rid = _uuid.UUID(str(row["id"]))
                                emb = row.get("embedding")
                                emb_list = (
                                    list(emb) if emb is not None else []
                                )
                                g.add_node(rid, None, emb_list)
                            except (ValueError, TypeError, AttributeError):
                                continue

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

                        _assignment = detect_communities(
                            g, prior=None, prior_mode="cold"
                        )
                        modularity = float(_assignment.modularity)
                        backend = _assignment.backend
                        _uuid_to_int: dict[_uuid.UUID, int] = {}
                        _next_int = 0
                        partition: dict[_uuid.UUID, int] = {}
                        for _node_uuid, _comm_uuid in _assignment.node_to_community.items():
                            if _comm_uuid not in _uuid_to_int:
                                _uuid_to_int[_comm_uuid] = _next_int
                                _next_int += 1
                            partition[_node_uuid] = _uuid_to_int[_comm_uuid]
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
                        logger.warning("crisis_recluster Leiden rebuild failed: %s", exc, exc_info=True)

        if not dry_run:
            cleared = self._clear_crisis_mode_via_s2_or_fallback(
                reason="crisis_recluster_complete",
            )
            if not cleared:
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
        if self._check_interrupt(SleepStep.RECALL_INDEX_REBUILD, 0, interrupt_check):
            return False, {}

        try:
            from iai_mcp import runtime_graph_cache

            result = runtime_graph_cache._rebuild_and_save_rgc(self._store)
            return True, result

        except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
            logger.warning(
                "recall_index_rebuild step failed: %s", exc, exc_info=True,
            )
            return True, {"error": str(exc)[:200], "rebuilt": False}

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


    _STEP_ORDER: tuple[SleepStep, ...] = (
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
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
    )

    _QUARANTINE_STRIKE_THRESHOLD: int = 3

    def run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        return self._run_internal(
            interrupt_check, force=False,
        )

    def force_run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        return self._run_internal(
            interrupt_check, force=True,
        )

    def _run_internal(
        self,
        interrupt_check: Callable[[], bool] | None,
        *,
        force: bool,
    ) -> SleepPipelineResult:
        t0 = time.monotonic()
        completed_steps: list[SleepStep] = []

        if not force and self._check_and_maybe_auto_recover_quarantine():
            return {
                "completed_steps": [],
                "failed_step": None,
                "error": None,
                "duration_sec": round(time.monotonic() - t0, 3),
                "quarantine_triggered": True,
                "interrupted": False,
            }

        try:
            self._run_essential_variable_tracker_hook()
        except Exception as exc:  # noqa: BLE001 -- tracker is best-effort observer
            logger.warning("essential_variable_tracker hook failed: %s", exc, exc_info=True)

        progress = self._load_progress()
        last_completed_index = (
            int(progress.get("last_completed_index", -1))
            if progress is not None
            else -1
        )
        if last_completed_index >= len(self._STEP_ORDER) - 1:
            last_completed_index = -1
        resume_step_index = last_completed_index + 1

        step_payloads: dict[SleepStep, dict] = {}

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
                return {
                    "completed_steps": completed_steps,
                    "failed_step": None,
                    "error": None,
                    "duration_sec": round(time.monotonic() - t0, 3),
                    "quarantine_triggered": False,
                    "interrupted": True,
                }

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
            step_payloads[step] = payload

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
        quarantine = self._load_quarantine()
        if quarantine is None:
            return False
        try:
            until = datetime.fromisoformat(quarantine["until_ts"])
        except (TypeError, ValueError):
            self._clear_quarantine(reason="auto_recovery_malformed_ts")
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if _utc_now() >= until:
            self._clear_quarantine(reason="auto_recovery_after_ttl")
            return False
        return True

    def _run_essential_variable_tracker_hook(self) -> None:
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
        _community_embeddings: dict[str, list[list[float]]] = {}
        for _, row in recs.iterrows():
            try:
                rid = _uuid.UUID(str(row["id"]))
                emb = row.get("embedding")
                emb_list = list(emb) if emb is not None else []
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

