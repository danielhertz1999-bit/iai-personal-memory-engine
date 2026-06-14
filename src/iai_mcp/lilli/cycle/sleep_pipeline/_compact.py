from __future__ import annotations

from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep


def step_compact_records_noop(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.COMPACT_RECORDS, 0, interrupt_check,
    ):
        return False, {}
    return True, {"action": "noop_under_hippo"}


def step_compact_records(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    return self._step_compact_records_noop(interrupt_check)
