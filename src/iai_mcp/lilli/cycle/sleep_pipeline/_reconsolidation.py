from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_reconsolidation(
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

    now = self._now()
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
