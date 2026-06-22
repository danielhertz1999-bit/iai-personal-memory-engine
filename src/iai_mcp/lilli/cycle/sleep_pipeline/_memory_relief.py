"""Force the process allocator to return freed pages to the OS.

Called at the dispatch-loop tail after a heavy sleep step has finished and
released its store/index locks. Three best-effort actions, in order:

  1. ``pa.default_memory_pool().release_unused()`` — hand idle arrow pool
     blocks back to the underlying allocator.
  2. ``gc.collect()`` — reclaim Python cycles holding large transients.
  3. On macOS, ``malloc_zone_pressure_relief(NULL, 0)`` — ask the system
     allocator to ``madvise`` freed pages back to the kernel immediately
     instead of holding them as resident memory until the next cycle re-grows
     the heap.

Resident-set size is read with the *current* RSS primitive
(``psutil.Process().memory_info().rss``), not peak RSS — peak is monotonic and
can never show a reclaim. Every step is wrapped fail-soft: the helper must
never raise on any platform, because it runs after a successful step whose
progress is already persisted.
"""
from __future__ import annotations

import gc
import logging
import platform
import time

logger = logging.getLogger(__name__)

# Lazily-resolved, one-time cache of the macOS system-allocator pressure-relief
# function. ``False`` means "resolution already failed; do not retry".
_zone_pressure_relief_fn = None  # type: ignore[var-annotated]


def _resolve_zone_pressure_relief():
    """Resolve ``malloc_zone_pressure_relief`` once, fail-soft.

    Mirrors the watchdog's lazy ``CDLL(find_library("c"))`` pattern. Returns the
    callable, or ``None`` if the symbol is unavailable (a future macOS, a
    sandbox, or a non-macOS host).
    """
    global _zone_pressure_relief_fn
    if _zone_pressure_relief_fn is not None:
        return _zone_pressure_relief_fn if _zone_pressure_relief_fn is not False else None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        fn = libc.malloc_zone_pressure_relief
        fn.argtypes = (ctypes.c_void_p, ctypes.c_size_t)  # (zone*, goal)
        fn.restype = ctypes.c_size_t  # bytes reclaimed
        _zone_pressure_relief_fn = fn
        return fn
    except Exception:  # noqa: BLE001 -- a missing symbol must never crash the cycle
        _zone_pressure_relief_fn = False
        return None


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 -- psutil flakiness must not crash the cycle
        return 0


def _step_memory_relief(label: str = "") -> dict:
    """Release idle allocator pages and report the RSS reclaim.

    Returns a telemetry dict with five numeric keys: ``rss_before_mb``,
    ``rss_after_mb``, ``rss_delta_mb`` (clamped at 0), ``zone_reclaimed_mb``,
    ``elapsed_ms``. Never raises.
    """
    t0 = time.monotonic()
    rss_before = _current_rss_bytes()

    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception as exc:  # noqa: BLE001 -- a pool API change must not crash the cycle
        logger.debug("release_unused failed for step %s: %s", label, exc)

    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001 -- collection must not crash the cycle
        logger.debug("gc.collect failed for step %s: %s", label, exc)

    zone_reclaimed = 0
    if platform.system() == "Darwin":
        fn = _resolve_zone_pressure_relief()
        if fn is not None:
            try:
                zone_reclaimed = int(fn(None, 0))
            except Exception as exc:  # noqa: BLE001 -- relief is advisory, never fatal
                logger.debug("zone pressure relief failed for step %s: %s", label, exc)
                zone_reclaimed = 0

    rss_after = _current_rss_bytes()
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    rss_before_mb = rss_before / 1e6
    rss_after_mb = rss_after / 1e6
    return {
        "rss_before_mb": round(rss_before_mb, 3),
        "rss_after_mb": round(rss_after_mb, 3),
        "rss_delta_mb": round(max(0.0, rss_before_mb - rss_after_mb), 3),
        "zone_reclaimed_mb": round(zone_reclaimed / 1e6, 3),
        "elapsed_ms": round(elapsed_ms, 3),
    }
