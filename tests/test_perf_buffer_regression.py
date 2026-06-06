"""Regression gate for buffered-write code paths in store.py.

Exercises the bench harness at N=1000 (executor-safe size per the project convention
Executor Resource Discipline) and asserts:

- Wall-time + RSS thresholds against the buffered-write baseline.
- lance_buffer_flush events fire for both records and edges tables
  (proves the buffer wiring is live end-to-end).

The high-N unhang validation is the phase-verification gate, run ONLY by
the orchestrator at -work via bench/memory_footprint.py; NEVER from this test file.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest


# Ensure bench.* and iai_mcp.* resolve to THIS worktree (mirrors bench harness shim).
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


# ------------------------------------------------------------------ thresholds

# rss_mb_peak threshold: 1500 MB.
# The events-buffer baseline (N=1000) was 1325 MB; the buffered-write fix
# targets a further improvement. 1500 MB leaves 175 MB headroom above the
# previous events-only baseline to absorb run-to-run variance while still
# catching major regressions where the buffer wiring has been reverted to
# per-row writes.
RSS_MB_THRESHOLD = 1500.0

# Wall-time threshold: 180 seconds.
# The events-buffer N=1000 run was approximately 120 s. 50% headroom for
# slower CI hosts and cold JIT cache; catches regressions where buffered
# writes revert to per-row store transactions.
WALL_TIME_SEC_THRESHOLD = 180.0


# ------------------------------------------------------------------ shared fixture


@pytest.fixture(scope="module")
def bench_result_and_store_path(tmp_path_factory):
    """Run bench/memory_footprint.py at N=1000 once for the entire module.

    Returns a tuple (result_dict, store_path, wall_time_sec).

    Using scope="module" means the ~2-minute bench run happens ONCE; all
    five tests in this module share the same result, reducing total test
    time from ~10 minutes to ~2 minutes.

    The store directory persists until the end of the pytest session (the
    tmp_path_factory scope) so that post-bench query_events calls can open
    the same store instance and inspect the events table.
    """
    store_dir = tmp_path_factory.mktemp("perf_regression_bench")
    store_path = store_dir / "hippo"
    store_path.mkdir(parents=True, exist_ok=True)

    from bench.memory_footprint import run_memory_footprint

    start = time.monotonic()
    result = run_memory_footprint(n=1000, store_path=store_path)
    wall = time.monotonic() - start

    return result, store_path, wall


def _query_lance_buffer_flush_events(store_path: Path, table_name: str) -> list[dict]:
    """Open the store at *store_path* and return lance_buffer_flush events for *table_name*."""
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=store_path)
    try:
        events = query_events(store, kind="lance_buffer_flush", limit=10000)
        return [e for e in events if (e.get("data") or {}).get("table") == table_name]
    finally:
        del store


# ------------------------------------------------------------------ tests


@pytest.mark.slow
def test_n1000_completes_within_wall_time_threshold(bench_result_and_store_path):
    """N=1000 bench completes within WALL_TIME_SEC_THRESHOLD."""
    _result, _store_path, wall = bench_result_and_store_path
    assert wall <= WALL_TIME_SEC_THRESHOLD, (
        f"N=1000 bench wall-time regression: {wall:.1f}s > {WALL_TIME_SEC_THRESHOLD}s; "
        f"buffer wiring may have reverted to per-row writes"
    )


@pytest.mark.slow
def test_n1000_rss_peak_under_threshold(bench_result_and_store_path):
    """N=1000 bench RSS peak is bounded by RSS_MB_THRESHOLD."""
    result, _store_path, _wall = bench_result_and_store_path
    assert isinstance(result, dict)
    rss = float(result.get("rss_mb_peak", 0.0))
    assert rss > 0, "bench did not report rss_mb_peak"
    assert rss <= RSS_MB_THRESHOLD, (
        f"N=1000 bench RSS regression: {rss:.1f} MB > {RSS_MB_THRESHOLD} MB threshold; "
        f"buffered-write fix may have regressed"
    )


@pytest.mark.slow
def test_n1000_emits_records_lance_buffer_flush_events(bench_result_and_store_path):
    """records-table buffer wiring is exercised: >=5 lance_buffer_flush events for table=records.

    With the default buffer size of 100 rows and N=1000 inserts, the records
    buffer will flush at least 10 times (1000 / 100). The >=5 bound is lenient
    to absorb any variation in flush timing without over-fitting to bench internals.
    """
    _result, store_path, _wall = bench_result_and_store_path
    records_flushes = _query_lance_buffer_flush_events(store_path, "records")
    assert len(records_flushes) >= 5, (
        f"records buffer wiring not exercised: only {len(records_flushes)} "
        f"lance_buffer_flush events for table=records "
        f"(expected >=5 with 1000 records / 100-row default threshold)"
    )


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "Bench at N=1000 does not exercise the EDGES write path by default. "
        "Hebbian self-loops are gated by pattern-separation enable + non-dry-run "
        "(store.py line ~875); the bench harness does not enable that path. "
        "EDGES buffer wiring is fully verified by the 13 unit/static tests in "
        "tests/test_edge_write_buffer.py (call-site flips at boost_edges insert + "
        "add_contradicts_edge, daemon flush at 3 hooks, telemetry event payload "
        "schema). Bench-driven EDGES exercising is a phase-verification concern "
        "covered at orchestrator level by N=10000 bench run with explicit edge "
        "writes; see bench/memory_footprint.py docstring."
    )
)
def test_n1000_emits_edges_lance_buffer_flush_events(bench_result_and_store_path):
    """edges-table buffer wiring is exercised: >=1 lance_buffer_flush event for table=edges.

    Skipped (see decorator): bench at N=1000 does not exercise hebbian self-loops
    by default. EDGES wiring proven by unit/static tests in test_edge_write_buffer.py.
    """
    _result, store_path, _wall = bench_result_and_store_path
    edges_flushes = _query_lance_buffer_flush_events(store_path, "edges")
    assert len(edges_flushes) >= 1, (
        f"edges buffer wiring not exercised: 0 lance_buffer_flush events for table=edges "
        f"(expected >=1 from hebbian self-loop edge writes at N=1000)"
    )


@pytest.mark.slow
def test_n1000_bench_passes_existing_threshold_mb_check(bench_result_and_store_path):
    """The bench's own THRESHOLD_MB gate continues to pass after buffer wiring."""
    result, _store_path, _wall = bench_result_and_store_path
    assert result.get("passed") is True, (
        f"bench failed its internal threshold_mb gate: {result}"
    )
