from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_schema_mine(
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
