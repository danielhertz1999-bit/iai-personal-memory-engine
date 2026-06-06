"""Ship-gate analyzer for contradiction_longitudinal_claude CSV.

RED-witness suite for `bench/analyze_efe_ab.py`. Synthetic CSV fixtures only;
no MemoryStore, no embedder. Exit-code assertions go through `subprocess.run`
on the script; per-seed math goes through direct imports.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYZER_PATH = REPO_ROOT / "bench" / "analyze_efe_ab.py"


# CSV column layout matches bench/contradiction_longitudinal_claude.py
# write_outputs after Task 1.
SAMPLE_HEADER = [
    "probe_id", "seed", "n_slice", "condition", "topic",
    "pipeline_rank", "cosine_rank",
    "pipeline_hit_at_k", "cosine_hit_at_k",
    "s4_contradiction_emitted", "anti_hits_count", "hint_kinds",
    "pipeline_top1_text",
    "route", "cue_hash",
]


def _write_csv(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    csv_path = results_dir / "contradiction_longitudinal_synth.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SAMPLE_HEADER)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in SAMPLE_HEADER}
            full.update(r)
            w.writerow(full)
    return csv_path


def _make_row(seed: int, route: str, rank: int, **kw: Any) -> dict[str, Any]:
    return {
        "probe_id": kw.get("probe_id", f"p-{seed}-{route}-{rank}"),
        "seed": str(seed),
        "n_slice": "0",
        "condition": "post_flip",
        "topic": "launch_date",
        "pipeline_rank": str(rank),
        "pipeline_hit_at_k": "1" if 0 < rank <= 10 else "0",
        "route": route,
        "cue_hash": kw.get("cue_hash", "deadbeef"),
    }


def _run_analyzer(results_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ANALYZER_PATH), str(results_dir)],
        capture_output=True,
        text=True,
    )


def _rows_for_seed_arm(seed: int, route: str, hits: int, misses: int) -> list[dict[str, Any]]:
    """Build N hit-rows + M miss-rows for a (seed, route) cell."""
    rows: list[dict[str, Any]] = []
    for i in range(hits):
        rows.append(_make_row(seed, route, rank=1 + (i % 10)))
    for i in range(misses):
        rows.append(_make_row(seed, route, rank=-1))
    return rows


# ---------------------------------------------------------------------------
# Test 1 — clear pass exits 0
# ---------------------------------------------------------------------------


def test_clear_pass_exits_zero(tmp_path: Path) -> None:
    """3 seeds, deltas ~+0.50 each → cross_seed_mean_delta > +0.10 → exit 0."""
    rows: list[dict[str, Any]] = []
    for seed in (13, 42, 137):
        rows += _rows_for_seed_arm(seed, "efe_real", hits=25, misses=5)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0, (
        f"expected exit 0, got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )

    summary_json = tmp_path / "results" / "EFE-AB-SUMMARY.json"
    summary_md = tmp_path / "results" / "EFE-AB-SUMMARY.md"
    assert summary_json.exists(), "EFE-AB-SUMMARY.json not written"
    assert summary_md.exists(), "EFE-AB-SUMMARY.md not written"

    summary = json.loads(summary_json.read_text())
    assert summary["ship_gate_hit"] is True
    assert summary["cross_seed_mean_delta"] > 0.10
    assert summary["threshold"] == pytest.approx(0.10)
    assert summary["n_seeds"] == 3


# ---------------------------------------------------------------------------
# Test 2 — clear fail exits 1
# ---------------------------------------------------------------------------


def test_clear_fail_exits_one(tmp_path: Path) -> None:
    """3 seeds, deltas ~+0.067 → cross_seed_mean_delta < +0.10 → exit 1."""
    rows: list[dict[str, Any]] = []
    for seed in (13, 42, 137):
        rows += _rows_for_seed_arm(seed, "efe_real", hits=12, misses=18)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 1, (
        f"expected exit 1, got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    summary = json.loads((tmp_path / "results" / "EFE-AB-SUMMARY.json").read_text())
    assert summary["ship_gate_hit"] is False
    assert summary["cross_seed_mean_delta"] < 0.10


# ---------------------------------------------------------------------------
# Test 3 — missing efe_real → setup error (exit 2)
# ---------------------------------------------------------------------------


def test_missing_efe_real_exits_two(tmp_path: Path) -> None:
    """No efe_real rows → exit 2, stderr names the missing arm."""
    rows = _rows_for_seed_arm(13, "efe_shadow", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 2, (
        f"expected exit 2 (setup error), got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    assert "efe_real" in proc.stderr, (
        f"stderr should name the missing arm; got {proc.stderr!r}"
    )


def test_missing_efe_shadow_exits_two(tmp_path: Path) -> None:
    """No efe_shadow rows → exit 2, symmetric to test_missing_efe_real_exits_two."""
    rows = _rows_for_seed_arm(13, "efe_real", hits=10, misses=20)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 2
    assert "efe_shadow" in proc.stderr


def test_no_csv_in_dir_exits_two(tmp_path: Path) -> None:
    """Empty results dir → exit 2 with no-CSV message."""
    (tmp_path / "results").mkdir()
    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 2
    assert "contradiction_longitudinal" in proc.stderr.lower() or "no" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Test 4 — per-seed aggregation (direct import, no subprocess)
# ---------------------------------------------------------------------------


def test_per_seed_aggregation() -> None:
    """3 seeds with deltas [0.05, 0.15, 0.20] → cross-seed mean = 0.133..."""
    # Make per-seed Rescue@10 land exactly on the target deltas by using 20-row
    # arms with hit counts that produce the chosen rates.
    #
    # delta=0.05: real=0.20 (4/20), shadow=0.15 (3/20)
    # delta=0.15: real=0.50 (10/20), shadow=0.35 (7/20)
    # delta=0.20: real=0.60 (12/20), shadow=0.40 (8/20)
    rows: list[dict[str, Any]] = []
    seed_to_counts = {
        13: ((4, 16), (3, 17)),   # delta 0.05
        42: ((10, 10), (7, 13)),  # delta 0.15
        137: ((12, 8), (8, 12)),  # delta 0.20
    }
    for seed, ((real_hit, real_miss), (shadow_hit, shadow_miss)) in seed_to_counts.items():
        rows += _rows_for_seed_arm(seed, "efe_real", hits=real_hit, misses=real_miss)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=shadow_hit, misses=shadow_miss)

    # Import the pure-Python pieces directly (no subprocess needed).
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from bench.analyze_efe_ab import (
            aggregate_across_seeds,
            compute_per_route_rescue_at_k,
        )
    finally:
        sys.path.pop(0)

    per_seed = compute_per_route_rescue_at_k(rows)
    agg = aggregate_across_seeds(per_seed)

    # Per-seed exact assertions
    assert agg["per_seed"]["13"]["delta"] == pytest.approx(0.05, abs=1e-9)
    assert agg["per_seed"]["42"]["delta"] == pytest.approx(0.15, abs=1e-9)
    assert agg["per_seed"]["137"]["delta"] == pytest.approx(0.20, abs=1e-9)
    # Cross-seed mean = (0.05 + 0.15 + 0.20) / 3 ≈ 0.1333...
    assert agg["cross_seed_mean_delta"] == pytest.approx(0.4 / 3.0, abs=1e-9)
    assert agg["ship_gate_hit"] is True  # 0.133 > 0.10


def test_per_seed_aggregation_exit_zero_on_threshold_boundary(tmp_path: Path) -> None:
    """Same fixtures as test_per_seed_aggregation — exit 0 (mean=0.133 > 0.10)."""
    rows: list[dict[str, Any]] = []
    seed_to_counts = {
        13: ((4, 16), (3, 17)),
        42: ((10, 10), (7, 13)),
        137: ((12, 8), (8, 12)),
    }
    for seed, ((rh, rm), (sh, sm)) in seed_to_counts.items():
        rows += _rows_for_seed_arm(seed, "efe_real", hits=rh, misses=rm)
        rows += _rows_for_seed_arm(seed, "efe_shadow", hits=sh, misses=sm)
    _write_csv(tmp_path, rows)
    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0, (
        f"expected exit 0 (mean=0.133 > 0.10); got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — efe_skip rows excluded
# ---------------------------------------------------------------------------


def test_efe_skip_rows_excluded(tmp_path: Path) -> None:
    """100 real + 100 shadow + 100 efe_skip → analyzer reports n_rows=200 attributable."""
    rows: list[dict[str, Any]] = []
    # Single seed for simplicity.
    rows += _rows_for_seed_arm(13, "efe_real", hits=60, misses=40)
    rows += _rows_for_seed_arm(13, "efe_shadow", hits=20, misses=80)
    rows += _rows_for_seed_arm(13, "efe_skip", hits=50, misses=50)
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0, (
        f"expected exit 0 with delta=0.40; got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    summary = json.loads((tmp_path / "results" / "EFE-AB-SUMMARY.json").read_text())
    assert summary["n_rows"] == 200, (
        f"efe_skip rows should be excluded; got n_rows={summary['n_rows']}"
    )
    # delta = 0.60 - 0.20 = 0.40
    assert summary["per_seed"]["13"]["delta"] == pytest.approx(0.40, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 6 — rank semantics: -1 and rank>10 count as miss
# ---------------------------------------------------------------------------


def test_pipeline_rank_zero_or_negative_or_overK_is_miss(tmp_path: Path) -> None:
    """Hits = rank in 1..10; misses = rank in {-1, 0, 11, 100, 'bad'}."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from bench.analyze_efe_ab import _is_hit_at_k
    finally:
        sys.path.pop(0)

    # rank=0 is NOT a hit (strict `0 < rank`).
    assert _is_hit_at_k("1") is True
    assert _is_hit_at_k("5") is True
    assert _is_hit_at_k("10") is True
    assert _is_hit_at_k("0") is False
    assert _is_hit_at_k("-1") is False
    assert _is_hit_at_k("11") is False
    assert _is_hit_at_k("100") is False
    assert _is_hit_at_k("not-a-number") is False
    assert _is_hit_at_k("") is False


# ---------------------------------------------------------------------------
# Test 7 — back-compat: rows with empty route (pre- CSVs) are dropped
# ---------------------------------------------------------------------------


def test_empty_route_rows_dropped(tmp_path: Path) -> None:
    """Rows with `route == ""` are unattributable and excluded from Rescue@10."""
    rows: list[dict[str, Any]] = []
    rows += _rows_for_seed_arm(13, "efe_real", hits=10, misses=10)
    rows += _rows_for_seed_arm(13, "efe_shadow", hits=2, misses=18)
    # 50 legacy rows with no route — must be dropped.
    for i in range(50):
        rows.append(_make_row(13, "", rank=1))
    _write_csv(tmp_path, rows)

    proc = _run_analyzer(tmp_path / "results")
    assert proc.returncode == 0  # delta = 0.5 - 0.1 = 0.4 → pass
    summary = json.loads((tmp_path / "results" / "EFE-AB-SUMMARY.json").read_text())
    assert summary["n_rows"] == 40, (
        f"empty-route rows should be excluded; got n_rows={summary['n_rows']}"
    )


# ---------------------------------------------------------------------------
# Test 8 — missing required column → exit 2
# ---------------------------------------------------------------------------


def test_missing_required_column_exits_two(tmp_path: Path) -> None:
    """Old-format CSV without `route` column → setup error exit 2."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    csv_path = results_dir / "contradiction_longitudinal_legacy.csv"
    legacy_header = [c for c in SAMPLE_HEADER if c not in ("route", "cue_hash")]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=legacy_header)
        w.writeheader()
        w.writerow({k: "x" for k in legacy_header})

    proc = _run_analyzer(results_dir)
    assert proc.returncode == 2
    assert "route" in proc.stderr or "missing" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Test 9 — --help works
# ---------------------------------------------------------------------------


def test_help_exits_zero() -> None:
    """`bench/analyze_efe_ab.py --help` returns 0 and prints usage."""
    proc = subprocess.run(
        [sys.executable, str(ANALYZER_PATH), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()
