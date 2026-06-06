"""Regression tests for the N-aware bench RSS threshold.

`bench/memory_footprint.py:_threshold_mb_for_n` scales the peak-RSS
budget with `log10(N)` above the N=1000 anchor so the bench self-check
stops false-failing at larger N once buffered writes are in play.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


# Ensure bench.* resolves to THIS worktree (mirrors bench harness shim).
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


def _t(n: int) -> float:
    from bench.memory_footprint import _threshold_mb_for_n

    return _threshold_mb_for_n(n)


def test_threshold_matches_legacy_anchor_at_n_1000():
    """At N=1000 the N-aware threshold must equal the legacy 1600 MB anchor."""
    assert _t(1000) == pytest.approx(1600.0)


def test_threshold_unchanged_at_or_below_anchor():
    """Below the N=1000 anchor, the threshold stays at the conservative 1600 MB."""
    assert _t(500) == pytest.approx(1600.0)
    assert _t(100) == pytest.approx(1600.0)
    assert _t(1) == pytest.approx(1600.0)


def test_threshold_grows_with_log_of_n():
    """Above N=1000 the threshold must scale with log10(N) at +25% per decade."""
    # Decade 1 (10x: 1k → 10k): +25% of 1600 = +400 → 2000 MB
    assert _t(10_000) == pytest.approx(2000.0, abs=0.5)
    # Decade 2 (100x: 1k → 100k): +50% → 2400 MB
    assert _t(100_000) == pytest.approx(2400.0, abs=0.5)


def test_threshold_strictly_monotonic():
    """A larger N never yields a smaller threshold."""
    sizes = [1, 100, 999, 1000, 1001, 5000, 10_000, 50_000, 100_000]
    thresholds = [_t(n) for n in sizes]
    for prev, cur in zip(thresholds, thresholds[1:]):
        assert cur >= prev, (
            f"threshold must be non-decreasing across N; got {prev} → {cur}"
        )


def test_threshold_covers_observed_n_10000_peak():
    """At N=10000 the threshold (≈2000 MB) must cover the observed 1862 MB peak from the -B closure run."""
    # Observed peak post-unhang at N=10000 was 1862 MB. The N-aware
    # threshold must comfortably cover that without false-flagging.
    observed_peak_n10k = 1862.0
    assert _t(10_000) >= observed_peak_n10k, (
        f"N-aware threshold at N=10000 ({_t(10_000)}) must cover the "
        f"observed peak ({observed_peak_n10k} MB)"
    )


def test_threshold_function_uses_log10():
    """The function uses log10 (decade scaling) — checked by ratio at N=10k vs N=1k."""
    ratio = (_t(10_000) - 1600.0) / 1600.0
    # 25% growth at the 1-decade mark.
    assert ratio == pytest.approx(0.25, abs=0.001)


def test_bench_json_output_uses_n_aware_threshold(tmp_path, monkeypatch):
    """The bench's JSON output's threshold_mb field reflects the N-aware value."""
    # We import the module and call run_memory_footprint directly with a
    # tiny N to avoid loading the full embedder / store stack just to
    # validate the output shape.
    pytest.importorskip("numpy")

    from bench.memory_footprint import (  # noqa: E402
        _threshold_mb_for_n,
        run_memory_footprint,
    )

    # N=50 → below anchor → threshold = THRESHOLD_MB = 1600.0.
    result = run_memory_footprint(n=50, store_path=tmp_path, skip_graph=True)
    assert result["threshold_mb"] == pytest.approx(_threshold_mb_for_n(50))
    assert result["n"] == 50
