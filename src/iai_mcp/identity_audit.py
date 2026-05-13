"""Continuous S5 identity audit. Runs even when daemon is paused.

Wraps `s5.detect_drift_anomaly` + `sigma.compute_and_emit` on a 1-hour cadence.
Both calls are MVCC reads (LanceDB handles concurrent readers natively), so
this loop does NOT acquire the fcntl exclusive lock. That is the C6 invariant:
the daemon continues to observe its own identity even when heavy consolidation
is paused.

addition : the same loop iteration also runs Lance
storage maintenance (`optimize_lance_storage`) on a configurable cadence
(default 1h via `LANCE_OPTIMIZE_INTERVAL_SEC`). The optimize body is gated
by a `time.monotonic()` cooldown against the configured interval; the
cooldown gate is silent when blocked (no event flooding).

REMOVED the `_should_yield_to_mcp(socket)`
HUMAN-FIRST gate. The lifecycle state machine + sleep_pipeline supersede
this design — periodic optimize runs unconditionally once the cooldown
passes; SLEEP-state coexistence is provided by the lifecycle predicate
that gates SLEEP entry on `sleep_eligible`. The `socket` kwarg has been
removed from `continuous_audit`'s signature.

Constitutional guard:
- C6: S5 invariant audit runs read-only (MVCC) and does NOT acquire the
  process-wide exclusive lock. Grep-guarded by
  tests/test_constitutional_guards.py (C6 = no lock module imported here).
- C3: ZERO paid-API cost. No reference to paid-API env var.
- C5: literal preservation -- no writes to MemoryRecord.literal_surface.
- Light daemon ops run concurrent with MCP via LanceDB MVCC; the audit
  path is exactly one such op.

Exception handling: each of the underlying calls is wrapped in its own
try/except. Failures are emitted as `identity_audit_error` events with a
`stage` discriminator ("s5" | "sigma") and the loop continues to the next
tick. The Lance optimize step uses a separate try/except path because its
helper already swallows per-table failures into the report dict ;
the outer guard there only protects against event-write failure. The
daemon must never die from an audit OR maintenance failure.
"""
from __future__ import annotations

import asyncio
import time

from iai_mcp import maintenance as _maintenance
from iai_mcp.events import write_event
from iai_mcp.maintenance import optimize_lance_storage
from iai_mcp.s5 import detect_drift_anomaly
from iai_mcp.sigma import compute_and_emit

# 1-hour cadence -- same granularity as sigma snapshot + S5 audit in S4 pass.
AUDIT_INTERVAL_SEC: int = 60 * 60

# R2: timestamp of the most recent successful periodic
# Lance optimize. Module-level mutable; the loop body declares
# `global _last_optimize_completed_at` to write. Ephemeral by design --
# daemon restart resets to 0.0 so the first periodic poll runs immediately
# (the startup wire-in in daemon.main() already handled the boot-time bloat
# collapse, so this just establishes the periodic cadence baseline).
#
# Mirrors 's _last_cascade_completed_at pattern in daemon.py
# exactly (/): time.monotonic not datetime.now so the
# cooldown is immune to clock skew + system suspend/resume.
_last_optimize_completed_at: float = 0.0


async def continuous_audit(
    store,
    shutdown: asyncio.Event,
    *,
    interval_sec: float | None = None,
) -> None:
    """Loop until `shutdown` is set.

    On each tick: run S5 drift anomaly detection, then sigma topology
    snapshot, then gated Lance storage optimize. All three
    are independent: a failure in any one stage does not abort the others.
    The interval sleep is implemented via `asyncio.wait_for(shutdown.wait(),
    timeout=interval_sec)` so shutdown is responsive within a fraction of a
    second rather than having to wait a full hour.

    When `interval_sec` is None we look up the current module-level
    `AUDIT_INTERVAL_SEC` at call time. This lets tests monkeypatch the
    constant before calling the function.

    REMOVED the `socket` kwarg + the
    `_should_yield_to_mcp(socket)` gate inside the periodic Lance
    optimize branch. SLEEP-state coexistence is now provided by the
    lifecycle state machine instead of an in-loop yield probe.

    Args:
        store: MemoryStore instance.
        shutdown: asyncio.Event that breaks the loop when set.
        interval_sec: optional override for the per-tick sleep. Tests use
            small values (e.g. 0.05) to drive the loop quickly.
    """
    # R2: explicit `global` so the assignment in the periodic body
    # updates module-level state, not a local binding. Mirrors the Pitfall 3
    # discipline from 's _hippea_cascade_loop.
    global _last_optimize_completed_at

    while not shutdown.is_set():
        effective_interval: float = (
            float(interval_sec) if interval_sec is not None else float(AUDIT_INTERVAL_SEC)
        )
        # Stage 1: S5 drift anomaly detection (MVCC read).
        try:
            await asyncio.to_thread(detect_drift_anomaly, store, 5)
        except Exception as exc:  # noqa: BLE001 -- daemon must never die
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "identity_audit_error",
                    {"stage": "s5", "error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:
                # Even the event write failed -- swallow silently so the loop
                # can continue. Next tick gets a fresh chance.
                pass

        # Stage 2: sigma topology snapshot + emit (MVCC read).
        try:
            await asyncio.to_thread(compute_and_emit, store)
        except Exception as exc:  # noqa: BLE001
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "identity_audit_error",
                    {"stage": "sigma", "error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:
                pass

        # Stage 3 (R2/R3): gated periodic Lance storage optimize.
        # Task 1.4 simplified: single gate
        # (interval cooldown). The MCP-active yield
        # gate via `_should_yield_to_mcp(socket)` was removed; the
        # lifecycle state machine handles SLEEP-state coexistence
        # outside this loop.
        try:
            # Access the module attribute at call time (not at import time)
            # so test fixtures can monkeypatch
            # `maintenance.LANCE_OPTIMIZE_INTERVAL_SEC` and observe the new
            # value without needing `importlib.reload(identity_audit)`.
            interval_sec_now = _maintenance.LANCE_OPTIMIZE_INTERVAL_SEC
            retention_sec_now = _maintenance.LANCE_OPTIMIZE_RETENTION_SEC
            elapsed_since_last = time.monotonic() - _last_optimize_completed_at
            if elapsed_since_last < interval_sec_now:
                # : silent skip -- no event. The cooldown gates
                # work, it does not consume a ledger slot.
                pass
            else:
                periodic_t0 = time.monotonic()
                try:
                    periodic_report = await asyncio.to_thread(
                        optimize_lance_storage, store,
                    )
                    try:
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "lance_storage_optimized",
                            {
                                "phase": "periodic",
                                "retention_days": (
                                    retention_sec_now / 86400.0
                                ),
                                "per_table": periodic_report,
                                "total_elapsed_sec": round(
                                    time.monotonic() - periodic_t0, 3,
                                ),
                            },
                            severity="info",
                        )
                    except Exception:
                        pass
                finally:
                    # : stamp completion timestamp regardless of
                    # success/exception so a failed optimize still gates
                    # the next run by LANCE_OPTIMIZE_INTERVAL_SEC.
                    _last_optimize_completed_at = time.monotonic()
        except Exception:
            # Outer defense-in-depth: a bug in the gate logic itself must
            # not crash the audit loop (C6 invariant: the daemon must
            # continue observing its own identity even when maintenance
            # work fails). Same discipline as the S5/sigma stages above.
            pass

        # Shutdown-responsive sleep: return early if shutdown fires.
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=effective_interval)
            break  # shutdown fired mid-sleep
        except asyncio.TimeoutError:
            continue  # normal path: time for next audit tick
