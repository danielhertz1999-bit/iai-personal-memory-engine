""" regression guard: small-N RAM bench stays under threshold.

(D5-08) — CI-runnable guard for bench/memory_footprint.py. The
large-N target (RSS <= 300 MB at N=10k warm on 16+ GB machine) runs
ad-hoc from the published bench report; this test exercises the small-N path
(N=100-500 with a 64d embedding) so CI catches harness drift without
spinning up a 10k-record LanceDB table.

See:
- bench/memory_footprint.py — the harness under guard
- internal architecture spec
  Task 1 for the behavior contract
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_memory_footprint_small_n_under_threshold(tmp_path: Path):
    """Smoke: small-N run populates rss_mb_peak under a generous ceiling.

    The 300 MB large-N target is NOT asserted here — a fresh LanceDB +
    NetworkX graph at N=500 already allocates more than that on macOS
    when bge-m3 is loaded via embed import. This guard only asserts that
    the harness returns a plausible positive reading and respects the
    JSON schema the BENCH_REPORT consumes.
    """
    from bench.memory_footprint import run_memory_footprint

    out = run_memory_footprint(n=100, store_path=tmp_path / "store", dim=64)

    # Shape: every key promised in the module docstring is present.
    assert "n" in out
    assert "rss_mb_peak" in out
    assert "threshold_mb" in out
    assert "passed" in out
    assert "platform" in out

    # Values: rss is a real positive reading; threshold is the design target.
    assert out["n"] == 100
    assert isinstance(out["rss_mb_peak"], float)
    assert out["rss_mb_peak"] > 0.0
    assert out["threshold_mb"] == 300.0

    # Generous outer bound — catches a clearly broken reading (e.g. reporting
    # nanoseconds as MB). The tight 300 MB fence belongs to the large-N run.
    assert out["rss_mb_peak"] < 4000.0, (
        f"small-N RSS {out['rss_mb_peak']} MB suspicious"
    )


def test_memory_footprint_main_exits_int(tmp_path: Path):
    """CLI entry-point returns 0 or 1 (bench CI contract)."""
    from bench import memory_footprint

    code = memory_footprint.main(argv=["--n", "50", "--dim", "32"])
    assert code in (0, 1)


def test_memory_footprint_platform_units_documented(tmp_path: Path):
    """Harness records the platform it measured on — macOS bytes vs Linux KB
    is an correctness trap; the JSON output must carry the marker so
    downstream reports can reproduce the unit conversion.
    """
    from bench.memory_footprint import run_memory_footprint

    out = run_memory_footprint(n=50, store_path=tmp_path / "store2", dim=32)
    assert out["platform"] in ("darwin", "linux", "win32")
