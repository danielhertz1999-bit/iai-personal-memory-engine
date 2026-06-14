from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_dmn_reflection(
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
