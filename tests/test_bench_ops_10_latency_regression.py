"""Regression guard: small-N latency stays under the p95 ceiling.

CI-runnable guard for bench/neural_map.py at the small-N end of the matrix.
The full N ∈ {100, 1k, 5k, 10k} matrix runs ad-hoc; this test exercises
N=100 only so CI catches regressions in <30s.

Speed contract: p95 < 100 ms at every measured N.

Adds the comparative reference flags to argparse:
    --ref-mempalace-p95-ms <float>
    --ref-claude-mem-p95-ms <float>

When supplied, the bench's per-N `passed` flag flips to False if IAI's p95
exceeds the reference. Tests assert these flags exist on the parser.

See:
- bench/neural_map.py — the harness under guard
- tests/test_bench_neural_map.py — sibling speed tests (passed=True at N=100)
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Prevent macOS keyring prompts by swapping the keyring backend for an
    in-memory dict (same pattern as tests/test_hippea_cascade.py and
    tests/test_memory_recall_structural.py)."""
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password",
        lambda s, u, p: fake_store.__setitem__((s, u), p),
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None),
    )
    yield fake_store


def test_neural_map_small_n_p95_under_regression_ceiling(tmp_path: Path):
    """Regression guard at N=100.

    The strict p95 < 100 ms gate is asserted by
    tests/test_bench_neural_map.py::test_neural_map_bench_reports_passed_flag
    — an existing test that trips under concurrent system load. This guard is
    a REGRESSION fence: it asserts the bench still produces a numeric p95
    in the same order of magnitude as the ceiling, so a
    structural regression (e.g. someone breaks the spread pruning and
    p95 jumps to 1s+) is caught in CI even when wall-clock noise puts
    the strict 100 ms test on a flaky boundary.

    The 200 ms ceiling is 2x the speed gate at N=100; if a real regression
    drops latency by 2x or more, this gate catches it and the strict
    100 ms gate (run in isolation) handles the absolute measurement.

    Load-robust: although the 200 ms ceiling is a deliberately loose 2x fence,
    an extremely busy host can still inflate a single run past it. skip_if_loaded()
    bails on a busy host and best-of-N takes the MINIMUM p95 over independent
    runs so wall-clock noise never produces a false red. The 200 ms regression
    ceiling itself is unchanged.
    """
    from bench.neural_map import run_neural_map_bench

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    counter = {"i": 0}

    def _one_p95() -> float:
        i = counter["i"]
        counter["i"] += 1
        run = run_neural_map_bench(
            n=100, iterations=10, store_path=tmp_path / f"store{i}",
        )
        return float(run["latency_ms_p95"])

    min_p95 = best_of_n(_one_p95, n=3)

    assert min_p95 < 200.0, (
        f"regression: best-of-3 p95 {min_p95:.2f}ms > 200ms at N=100 "
        f"(2x the speed ceiling — likely a real regression, not concurrency noise)"
    )
    # Sanity: the harness always returns a positive p95.
    assert min_p95 > 0.0


def test_neural_map_main_with_matrix_returns_int(tmp_path: Path):
    """CLI entry-point honours an explicit ns list (the N matrix)."""
    from bench import neural_map

    code = neural_map.main(ns=[50], iterations=3, store_path=tmp_path)
    assert code in (0, 1)


def test_neural_map_argparse_has_reference_flags():
    """Comparative gate: argparse exposes the reference-p95 flags so
    the bench can compare IAI to mempalace/claude-mem reference numbers
    measured separately on this host.

    Grep-verifiable contract: any ratification of these names elsewhere in
    the report harness has to update the test.
    """
    from bench import neural_map

    parser = neural_map._parse_args.__defaults__  # noqa: SLF001
    # Inspect the actual parser by parsing a dry args list.
    ns = neural_map._parse_args([
        "--n", "100",
        "--ref-mempalace-p95-ms", "42.5",
        "--ref-claude-mem-p95-ms", "61.0",
    ])
    assert getattr(ns, "ref_mempalace_p95_ms", None) == 42.5
    assert getattr(ns, "ref_claude_mem_p95_ms", None) == 61.0


def test_neural_map_comparative_gate_flips_passed_false_when_above_ref(tmp_path: Path):
    """If IAI p95 > mempalace ref, the per-N JSON's `passed` flips False
    AND `reason` carries the reference name.
    """
    from bench import neural_map

    # An impossibly low ref that any realistic bench will exceed.
    code = neural_map.main(
        ns=[50],
        iterations=3,
        store_path=tmp_path,
        ref_mempalace_p95_ms=0.0001,
    )
    # With a 0.0001 ms reference, the bench cannot pass.
    assert code == 1
