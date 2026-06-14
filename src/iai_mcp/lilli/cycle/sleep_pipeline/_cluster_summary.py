from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_cluster_summary(
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
