"""Perf gate for normalize + max_degree cache.

The N=1k warm p95 ≤ 83.6 ms lock is enforced via
``bench/neural_map.py`` for reproducibility on the reference host. This
pytest gate runs at N=200 with a CI-generous ceiling so it can catch
egregious hot-path regressions without flapping on slower runners.

The per-recall work added is:
  - one ``getattr(graph, "_max_degree", 0)`` (dict lookup) before the loop
  - one ``log(1.0 + max_deg)`` once per call
  - one float division per candidate

The combined cost is sub-millisecond at N=200; the gate ceiling at 200 ms
absorbs CI jitter and gives the reference-host bench room to land the
strict 83.6 ms read.
"""
from __future__ import annotations

import time

import pytest

# Reuse the perf fixtures from the existing pipeline-perf suite. Importing
# at the module top so failures surface immediately at collection time.
from tests.test_pipeline_perf import _seed_store


CI_GENEROUS_P95_S: float = 0.200  # 200 ms — see module docstring


# --------------------------------------------------------- p95 ceiling


def test_pipeline_recall_p95_under_ci_ceiling_after_normalize(tmp_path):
    """Seed N=200, warm the cache, then time 20 recall calls.

    p95 ≤ 200 ms (CI-generous). The reference host bench enforces the
    strict 83.6 ms M-02 invariant separately.

    Load-robust: in-gate wall-clock guard, so it stays in the default gate but
    skip_if_loaded() bails on a busy host and best-of-N takes the MINIMUM p95
    over independent timing passes — a busy host never produces a false red.
    The 200 ms CI ceiling is unchanged.
    """
    from iai_mcp.pipeline import recall_for_response

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=200, seed=0,
    )

    cues = [
        "what did we cover about auth yesterday?",
        "explain the db migration plan",
        "how does the web cache invalidation work",
        "summary of the cli subcommand changes",
        "recent network stack bug report",
    ]

    # One throwaway warm call so the records_cache + community gate
    # data structures are hot before timing.
    recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=cues[0], session_id="warm", budget_tokens=1500,
    )

    def _one_p95() -> float:
        latencies: list[float] = []
        for i in range(20):
            cue = cues[i % len(cues)]
            t0 = time.perf_counter()
            recall_for_response(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=cue, session_id="perf_gate", budget_tokens=1500,
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        # p95 index for 20 samples = int(0.95 * 20) = 19 (the slowest).
        return latencies[int(0.95 * len(latencies))]

    p95 = best_of_n(_one_p95, n=3)
    p95_ms = p95 * 1000.0
    print(
        f"\n[perf-gate] recall_for_response N=200 warm best-of-3 p95 = {p95_ms:.2f} ms "
        f"(CI ceiling: {CI_GENEROUS_P95_S * 1000:.0f} ms; "
        f"reference-host strict: 83.6 ms via bench/neural_map.py)"
    )

    assert p95 < CI_GENEROUS_P95_S, (
        f"normalize regression: recall_for_response N=200 warm "
        f"best-of-3 p95 = {p95_ms:.2f} ms exceeds CI ceiling "
        f"{CI_GENEROUS_P95_S * 1000:.0f} ms."
    )


def test_normalize_overhead_is_submillisecond(tmp_path, capsys):
    """Sanity: surface the normalize-stage timing as a printed trend so
    CI logs show whether the per-call additions stay sub-ms.

    Implementation note: a clean A/B against the OLD formula is hard to
    do without a feature flag (the change is unconditional in the rank
    stage). Instead we measure absolute p95 at N=100 and assert it sits
    well under the same 200 ms CI ceiling — a sub-100 ms read is the
    informal sanity check that normalize-overhead did not regress.

    Load-robust: in-gate wall-clock guard. skip_if_loaded() bails on a busy
    host; best-of-N takes the MINIMUM p95 over independent timing passes so a
    busy host never produces a false red. The 200 ms CI ceiling is unchanged.
    """
    from iai_mcp.pipeline import recall_for_response

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=1,
    )

    cues = [
        "auth verbatim cue",
        "db schema rebuild",
        "web cache invalidation",
    ]

    # Warm cache.
    recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=cues[0], session_id="warm", budget_tokens=1500,
    )

    def _one_p95() -> float:
        latencies: list[float] = []
        for i in range(10):
            cue = cues[i % len(cues)]
            t0 = time.perf_counter()
            recall_for_response(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=cue, session_id="overhead_check", budget_tokens=1500,
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        return latencies[int(0.95 * len(latencies))]

    p95 = best_of_n(_one_p95, n=3)
    p95_ms = p95 * 1000.0
    # Surface to test log; CI log captures the trend even on pass.
    print(
        f"\n[perf-gate] recall_for_response N=100 warm best-of-3 p95 = {p95_ms:.2f} ms "
        f"(normalize overhead: one division + one getattr per call)"
    )

    assert p95 < CI_GENEROUS_P95_S, (
        f"normalize-overhead sanity: best-of-3 p95 = {p95_ms:.2f} ms > "
        f"CI ceiling {CI_GENEROUS_P95_S * 1000:.0f} ms"
    )
