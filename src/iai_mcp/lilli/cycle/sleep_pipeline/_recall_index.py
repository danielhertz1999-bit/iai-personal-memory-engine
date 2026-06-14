from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_recall_index_rebuild(
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
