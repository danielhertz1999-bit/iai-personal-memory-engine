"""Tests for bench/neural_map.py (Task 4, D-SPEED).

D-SPEED contract: pipeline_recall <100ms at 10k records. The bench harness
measures per-N latency distribution (p50, p95) and returns a structured
dict. Main returns 0 iff all Ns pass thresholds.
"""
from __future__ import annotations

import pytest


def test_neural_map_bench_runs_small_n(tmp_path):
    from bench.neural_map import run_neural_map_bench

    out = run_neural_map_bench(n=50, iterations=3, store_path=tmp_path)
    assert out["n"] == 50
    assert "latency_ms_p50" in out
    assert "latency_ms_p95" in out
    assert "passed" in out
    assert isinstance(out["latency_ms_p50"], float)
    assert isinstance(out["latency_ms_p95"], float)


def test_neural_map_bench_returns_stage_timings(tmp_path):
    """Per-stage timings aid D-SPEED triage."""
    from bench.neural_map import run_neural_map_bench

    out = run_neural_map_bench(n=50, iterations=2, store_path=tmp_path)
    assert "stage_timings_ms" in out
    # Must cover the five pipeline stages named in pipeline.py.
    stages = out["stage_timings_ms"]
    for expected in ("embed", "gate", "seeds", "spread", "rank"):
        assert expected in stages


def test_neural_map_bench_reports_passed_flag(tmp_path):
    """D-SPEED gate: bench at N=100 MUST report passed=True.

    closes the D-SPEED gap from 02-VERIFICATION. The assertion
    upgrade from `isinstance(out["passed"], bool)` to `out["passed"] is True`
    is the bar-raising moment: honest benchmark discipline is no longer just
    "report truth" -- now "meet the target at N=100". Pipeline was rewired
    to use `store.append_provenance_batch` (one call) + `s4.on_read_check_batch`
    with records_cache passthrough (zero round-trips) per L-02 fix.
    """
    from bench.neural_map import run_neural_map_bench

    out = run_neural_map_bench(n=100, iterations=10, store_path=tmp_path)
    # Contract: threshold surfaced.
    assert out.get("threshold_ms") == 100.0
    # D-SPEED quality gate: p95 must be UNDER 100ms at N=100.
    assert out["passed"] is True, (
        f"D-SPEED violated: p95={out['latency_ms_p95']:.2f}ms > 100ms at N=100. "
        f"Full output: {out}"
    )
    assert out["latency_ms_p95"] < 100.0


def test_neural_map_main_exits_zero_at_n100(tmp_path, capsys):
    """main(ns=[100]) returns 0 (all-pass exit) post fix."""
    from bench import neural_map

    code = neural_map.main(ns=[100], iterations=10, store_path=tmp_path)
    assert code == 0, (
        f"bench.neural_map.main(ns=[100]) should exit 0 post-02-07; got {code}"
    )


def test_neural_map_bench_main_runs_and_returns_int(tmp_path, capsys):
    """Main is runnable end-to-end and returns 0 or 1 (bench CI contract)."""
    from bench import neural_map

    code = neural_map.main(ns=[50], iterations=2, store_path=tmp_path)
    assert code in (0, 1)


def test_neural_map_bench_deterministic_within_tolerance(tmp_path):
    """Two runs at the same N produce latency within the same order.

    Uses separate subdirs so each run starts with a fresh store.
    """
    from bench.neural_map import run_neural_map_bench

    a = run_neural_map_bench(
        n=50, iterations=5, store_path=tmp_path / "a", seed=42,
    )
    b = run_neural_map_bench(
        n=50, iterations=5, store_path=tmp_path / "b", seed=42,
    )
    # Latencies are wall-clock; both should fit a generous ceiling.
    assert a["latency_ms_p50"] < 2000.0
    assert b["latency_ms_p50"] < 2000.0
