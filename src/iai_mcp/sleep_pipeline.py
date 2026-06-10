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
        resolved_log = lifecycle_event_log if lifecycle_event_log is not None else event_log
        if resolved_log is None:
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
