"""Cycle orchestrator helpers.

Thin wrappers exposed on Brain (via Brain.cycle) for callers to invoke
REM / SWS / consolidation passes without binding to the internal
SleepPipeline class.
"""
from __future__ import annotations

import logging
from typing import Any

from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

log = logging.getLogger(__name__)


def run_rem(brain: Any, store: Any, **kwargs: Any) -> dict:
    """Run the REM-phase steps via SleepPipeline.

    Args:
        brain: Brain instance (unused by this helper; passed for API symmetry
               so callers can use brain.cycle.run_rem(brain, store)).
        store: Storage backend (MemoryStore-like).
        **kwargs: Forwarded to SleepPipeline.run().

    Returns:
        SleepPipeline.run() result dict.
    """
    pipeline = SleepPipeline(store=store)
    return pipeline.run(**kwargs)


def run_sws(brain: Any, store: Any, **kwargs: Any) -> dict:
    """Run the SWS/NREM-phase steps via SleepPipeline.

    SWS (slow-wave sleep) and REM dispatch through the same SleepPipeline.run()
    entry point; the lifecycle state machine decides which phase executes.

    Args:
        brain: Brain instance (unused; passed for API symmetry).
        store: Storage backend (MemoryStore-like).
        **kwargs: Forwarded to SleepPipeline.run().

    Returns:
        SleepPipeline.run() result dict.
    """
    pipeline = SleepPipeline(store=store)
    return pipeline.run(**kwargs)


def run_consolidation(
    brain: Any,
    store: Any,
    hvs: list[bytes],
    tier: str = "bsc",
) -> bytes:
    """Run the consolidate operation via brain.ops.consolidation.consolidate.

    Args:
        brain: Brain instance — provides access to the consolidation op.
        store: Storage backend (unused directly; passed for API symmetry).
        hvs: List of hypervector byte payloads to consolidate.
        tier: HDC tier name ("bsc", "fhrr", or "sparse_vsa"). Default "bsc".

    Returns:
        Consolidated hypervector bytes.
    """
    return brain.ops.consolidation.consolidate(hvs, tier=tier)
