"""Continuous S5 identity audit. Runs even when daemon is paused (C6).

Wraps `s5.detect_drift_anomaly` + `sigma.compute_and_emit` on a 1-hour cadence.
Both calls are SQLite WAL-mode reads, so this loop does NOT acquire the fcntl
exclusive lock. The daemon continues to observe its
own identity even when heavy consolidation is paused.

Hippo storage compaction (`optimize_hippo_storage`) is driven exclusively by the
canonical sleep pipeline's OPTIMIZE_LANCE step. It was previously also run here
on a separate cooldown; that redundant out-of-window path has been removed so
compaction happens only inside the canonical consolidation window.

The `_should_yield_to_mcp(socket)` gate was removed; the lifecycle state
machine handles SLEEP-state coexistence outside this loop.

Guards:
- S5 invariant audit does NOT acquire the process-wide exclusive lock.
- Zero paid-API cost: no reference to paid-API env var.
- Literal preservation -- no writes to MemoryRecord.literal_surface.

Exception handling: each underlying call is wrapped in its own try/except.
Failures are emitted as `identity_audit_error` events with a `stage`
discriminator ("s5" | "sigma") and the loop continues to the next tick.
The daemon must never die from an audit failure.
"""
from __future__ import annotations

import asyncio
import logging

from iai_mcp.events import write_event
from iai_mcp.s5 import detect_drift_anomaly
from iai_mcp.sigma import compute_and_emit

logger = logging.getLogger(__name__)

# 1-hour cadence -- same granularity as sigma snapshot + S5 audit in S4 pass.
AUDIT_INTERVAL_SEC: int = 60 * 60


async def continuous_audit(
    store,
    shutdown: asyncio.Event,
    *,
    interval_sec: float | None = None,
) -> None:
    """Loop until `shutdown` is set.

    On each tick: run S5 drift anomaly detection, then sigma topology
    snapshot. Both stages are independent: a failure in one does not abort
    the other. The interval sleep is implemented via
    `asyncio.wait_for(shutdown.wait(), timeout=interval_sec)` so shutdown is
    responsive within a fraction of a second rather than having to wait a
    full hour.

    Hippo storage compaction is driven exclusively by the sleep pipeline's
    OPTIMIZE_LANCE step (inside the canonical consolidation window). This loop
    is a pure-read audit loop — it does NOT call optimize_hippo_storage.

    When `interval_sec` is None we look up the current module-level
    `AUDIT_INTERVAL_SEC` at call time. This lets tests monkeypatch the
    constant before calling the function.

    Args:
        store: MemoryStore instance.
        shutdown: asyncio.Event that breaks the loop when set.
        interval_sec: optional override for the per-tick sleep. Tests use
            small values (e.g. 0.05) to drive the loop quickly.
    """
    while not shutdown.is_set():
        effective_interval: float = (
            float(interval_sec) if interval_sec is not None else float(AUDIT_INTERVAL_SEC)
        )
        # Stage 1: S5 drift anomaly detection (MVCC read).
        try:
            await asyncio.to_thread(detect_drift_anomaly, store, 5)
        except Exception as exc:  # noqa: BLE001 -- daemon must never die
            logger.warning("identity_audit_s5_failed", extra={"err": str(exc)[:200]})
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "identity_audit_error",
                    {"stage": "s5", "error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write failure is non-fatal
                # Even the event write failed -- swallow silently so the loop
                # can continue. Next tick gets a fresh chance.
                pass

        # Stage 2: sigma topology snapshot + emit (MVCC read).
        try:
            await asyncio.to_thread(compute_and_emit, store)
        except Exception as exc:  # noqa: BLE001 -- daemon must never die
            logger.warning("identity_audit_sigma_failed", extra={"err": str(exc)[:200]})
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "identity_audit_error",
                    {"stage": "sigma", "error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write failure is non-fatal
                pass

        # Shutdown-responsive sleep: return early if shutdown fires.
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=effective_interval)
            break  # shutdown fired mid-sleep
        except asyncio.TimeoutError:
            continue  # normal path: time for next audit tick
