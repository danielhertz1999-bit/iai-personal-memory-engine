"""Back-compat shim -- sleep pipeline algorithms have moved to iai_mcp.lilli.cycle.sleep_pipeline.

The canonical SleepPipeline is FSL-extraction-ready: it does NOT import
lifecycle_state or lifecycle_event_log at module level. This shim wraps it
to auto-inject iai-mcp's lifecycle event log when callers use the legacy
import path.

Re-exports preserve the historical public API so the daemon and historical
event consumers (which reference SleepStep.value integers) keep working
without modification.
"""
from __future__ import annotations

from iai_mcp.lilli.cycle.sleep_pipeline import (
    MAX_PAIRS_PER_CLUSTER,
    QUARANTINE_TTL_HOURS_DEFAULT,
    STEP_PHASE,
    SleepPhase,
    SleepPipelineResult,
    SleepStep,
    SleepPipeline as _LilliSleepPipeline,
    _utc_now,
    _utc_now_iso,
)


class SleepPipeline(_LilliSleepPipeline):
    """Legacy SleepPipeline -- auto-injects iai-mcp lifecycle event log.

    Callers using ``from iai_mcp.sleep_pipeline import SleepPipeline``
    continue to get the lifecycle-aware constructor. FSL stand-alone code
    imports from ``iai_mcp.lilli.cycle.sleep_pipeline`` directly and passes
    ``lifecycle_event_log=None`` or ``event_log=None``.
    """

    def __init__(
        self,
        store: object,
        lifecycle_state_path: object | None = None,
        event_log: object | None = None,
        quarantine_ttl_hours: float | None = None,
        s2_coordinator: object | None = None,
        loop: object | None = None,
        *,
        lifecycle_state_machine: object | None = None,
        lifecycle_event_log: object | None = None,
    ) -> None:
        # Resolve event_log: prefer lifecycle_event_log (injection-style)
        # over event_log (original positional-style) for forward-compat.
        resolved_log = lifecycle_event_log if lifecycle_event_log is not None else event_log
        if resolved_log is None:
            # Lazy import: this shim is in iai_mcp proper (not lilli/cycle),
            # so importing lifecycle_event_log here is intentional.
            from iai_mcp.lifecycle_event_log import LifecycleEventLog
            resolved_log = LifecycleEventLog()
        super().__init__(
            store,
            lifecycle_state_path=lifecycle_state_path,
            lifecycle_event_log=resolved_log,
            quarantine_ttl_hours=quarantine_ttl_hours,
            s2_coordinator=s2_coordinator,
            loop=loop,
        )


__all__ = [
    "SleepPipeline",
    "SleepStep",
    "SleepPhase",
    "SleepPipelineResult",
    "STEP_PHASE",
    "QUARANTINE_TTL_HOURS_DEFAULT",
    "MAX_PAIRS_PER_CLUSTER",
    "_utc_now",
    "_utc_now_iso",
]
