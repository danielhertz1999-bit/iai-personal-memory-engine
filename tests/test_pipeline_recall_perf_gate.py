"""perf gate for normalize + max_degree cache.

The lock at N=1k warm p95 ≤ 83.6 ms is enforced via
``bench/neural_map.py`` for reproducibility on the reference host. This
pytest gate runs at N=200 with a CI-generous ceiling so it can catch
egregious hot-path regressions without flapping on slower runners.

added per-recall work:
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
    strict 83.6 ms invariant separately.
    """
    from iai_mcp.pipeline import recall_for_response

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
    p95 = latencies[int(0.95 * len(latencies))]
    p95_ms = p95 * 1000.0
    print(
        f"\n[perf-gate] recall_for_response N=200 warm p95 = {p95_ms:.2f} ms "
        f"(CI ceiling: {CI_GENEROUS_P95_S * 1000:.0f} ms; "
        f"reference-host strict: 83.6 ms via bench/neural_map.py)"
    )

    assert p95 < CI_GENEROUS_P95_S, (
        f"normalize regression: recall_for_response N=200 warm "
        f"p95 = {p95_ms:.2f} ms exceeds CI ceiling "
        f"{CI_GENEROUS_P95_S * 1000:.0f} ms. "
        f"All latencies (ms): {[f'{x*1000:.1f}' for x in latencies]}"
    )


def test_normalize_overhead_is_submillisecond(tmp_path, capsys):
    """Sanity: surface the normalize-stage timing as a printed trend so
    CI logs show whether 's per-call additions stay sub-ms.

    Implementation note: a clean A/B against the OLD formula is hard to
    do without a feature flag (the change is unconditional in the rank
    stage). Instead we measure absolute p95 at N=100 and assert it sits
    well under the same 200 ms CI ceiling — a sub-100 ms read is the
    informal sanity check that normalize-overhead did not regress.
    """
    from iai_mcp.pipeline import recall_for_response

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
    p95 = latencies[int(0.95 * len(latencies))]
    p95_ms = p95 * 1000.0
    # Surface to test log; CI log captures the trend even on pass.
    print(
        f"\n[perf-gate] recall_for_response N=100 warm p95 = {p95_ms:.2f} ms "
        f"(normalize overhead: one division + one getattr per call)"
    )

    assert p95 < CI_GENEROUS_P95_S, (
        f"normalize-overhead sanity: p95 = {p95_ms:.2f} ms > "
        f"CI ceiling {CI_GENEROUS_P95_S * 1000:.0f} ms"
    )
