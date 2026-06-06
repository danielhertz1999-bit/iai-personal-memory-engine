"""Parity gate: custom_leiden Rescue@10 within +/-0.02 of leidenalg baseline.

The release gate for the Leiden-replacement work.

The Risk row "Retrieval Rescue@10 regression > 0.02" defines the parity
contract as the GO/NO-GO gate for the Leiden-replacement work.
The baseline (efe_shadow route, the path that hit Rescue@10 = 1.000 per
`EFE-AB-SUMMARY.json`) USES leidenalg. The measurement must reproduce
1.000 +/-0.02 with custom_leiden in production.

These tests parse the bench output JSON that the post-process step produces
and enforce the parity gate as a permanent regression check. Future re-runs
of the bench (e.g., after changes) will fail this test if any seed
regresses by more than +/-0.02 from the baseline.

The hardcoded baseline values plus the citation below guard against both a
hidden regression and a comparison against the wrong baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Parity gate constants
# ---------------------------------------------------------------------------

# Repo root resolved relative to this test file so the test is invocation-
# directory-independent (works from.venv, from worktree, from CI runner).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# The custom_leiden bench output JSON path (Task 2 writes it here).
PARITY_JSON_PATH = (
    _REPO_ROOT
    / "bench"
    / "results"
    / "v7.1"
    / "iteration-0"
    / "contradiction_longitudinal_custom_leiden.json"
)

# baseline per-seed Rescue@10 values.
# Source: the EFE-AB-SUMMARY.json efe_shadow_rescue values — the route that
# drove the 1.000 hit-rate that must reproduce within +/-0.02 with
# custom_leiden.
V7_0_BASELINE_PER_SEED: dict[str, float] = {
    "13": 1.0,
    "42": 1.0,
    "137": 1.0,
}

# Parity tolerance from the Risk row.
PARITY_TOLERANCE = 0.02

# Cross-seed mean baseline (mean of the three 1.000 values).
V7_0_CROSS_SEED_MEAN_BASELINE = (
    sum(V7_0_BASELINE_PER_SEED.values()) / len(V7_0_BASELINE_PER_SEED)
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_data() -> dict:
    """Load the custom_leiden parity JSON.

    Schema (produced by post-processing of bench + analyze_efe_ab outputs):

        {
          "backend": "leiden-custom",
          "seeds": [13, 42, 137],
          "n_recalls": <int>,
          "per_seed": {
            "13": {"efe_real_rescue": <f>, "efe_shadow_rescue": <f>, "delta": <f>},
            "42": {...},
            "137": {...}
          },
          "cross_seed_mean_rescue": <f>,
          "cross_seed_mean_delta": <f>,
          "baseline_v7_0": { # baseline embedded for traceability
            "13": 1.0, "42": 1.0, "137": 1.0
          }
        }
    """
    if not PARITY_JSON_PATH.exists():
        pytest.skip(
            f"Parity JSON not yet produced. Run "
            f"`PYTHONPATH=src python bench/contradiction_longitudinal_claude.py "
            f"--scale honest --seeds 13 42 137 "
            f"--output-dir bench/results/v7.1/iteration-0/` then the post-process "
            f"pipeline (analyze_efe_ab.py + write contradiction_longitudinal_"
            f"custom_leiden.json). Missing: {PARITY_JSON_PATH}"
        )
    return json.loads(PARITY_JSON_PATH.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_parity_json_exists() -> None:
    """The result JSON file exists at the expected path.

    Marked slow because it requires bench/results to have been populated
    by running the mosaic benchmark first. Skip in fast test runs.
    """
    assert PARITY_JSON_PATH.exists(), (
        f"Parity JSON missing: {PARITY_JSON_PATH}. "
        f"Bench must be run before parity enforcement is meaningful."
    )


def test_parity_json_schema_matches_baseline(parity_data: dict) -> None:
    """Top-level keys include the required fields for parity comparison.

    Mirrors the baseline shape (`per_seed`, `cross_seed_mean_rescue`)
    plus our additions (`backend`, `seeds`, `n_recalls`).
    """
    required_keys = {
        "per_seed",
        "cross_seed_mean_rescue",
        "seeds",
        "n_recalls",
        "backend",
    }
    missing = required_keys - set(parity_data.keys())
    assert not missing, (
        f"Parity JSON missing required top-level keys: {sorted(missing)}. "
        f"Got: {sorted(parity_data.keys())}"
    )

    per_seed = parity_data["per_seed"]
    for seed_key in V7_0_BASELINE_PER_SEED:
        assert seed_key in per_seed, (
            f"per_seed missing seed {seed_key!r}. Got: {sorted(per_seed.keys())}"
        )
        seed_block = per_seed[seed_key]
        for required_field in ("efe_real_rescue", "efe_shadow_rescue", "delta"):
            assert required_field in seed_block, (
                f"per_seed[{seed_key!r}] missing {required_field!r}. "
                f"Got: {sorted(seed_block.keys())}"
            )


def test_backend_label_is_leiden_custom(parity_data: dict) -> None:
    """The bench ran the new backend, not a leftover leidenalg path.

    `community.detect_communities` is wired to `run_mosaic`, and the bench
    MUST exercise that path. This field is injected by the post-process so the
    test can verify the right code path was measured.
    """
    backend = parity_data.get("backend")
    assert backend == "leiden-custom", (
        f"Parity bench must run the custom_leiden backend, got: {backend!r}. "
        f"Either community.detect_communities was not wired correctly "
        f"or the post-process injected the wrong label."
    )


@pytest.mark.parametrize("seed_key", sorted(V7_0_BASELINE_PER_SEED.keys()))
def test_per_seed_rescue_within_002_of_baseline(
    parity_data: dict, seed_key: str
) -> None:
    """Each seed's Rescue@10 stays within +/-0.02 of the baseline.

    The parity contract is the GO/NO-GO gate. Baseline values are HARDCODED
    with an explicit citation to the canonical source.

    Baseline source: EFE-AB-SUMMARY.json
    """
    baseline = V7_0_BASELINE_PER_SEED[seed_key]
    seed_block = parity_data["per_seed"][seed_key]
    # The efe_shadow_rescue route is what produced the 1.000 baseline; the
    # measurement must reproduce that route's Rescue@10 within +/-0.02.
    measured = float(seed_block["efe_shadow_rescue"])
    delta = abs(measured - baseline)
    assert delta <= PARITY_TOLERANCE, (
        f"PARITY_GATE_FAILED seed={seed_key}: "
        f"efe_shadow_rescue={measured:.4f} vs baseline={baseline:.4f} "
        f"(|delta|={delta:.4f} > tolerance={PARITY_TOLERANCE:.4f}). "
        f"The parity check gates the Leiden-replacement work."
    )


def test_cross_seed_mean_rescue_within_002(parity_data: dict) -> None:
    """Cross-seed mean Rescue@10 stays within +/-0.02 of baseline mean.

    The cross-seed mean is the primary headline number from the parity
    summary; it must be >= 0.98 (i.e., 1.000 - 0.02). This guards the global
    release gate independently of per-seed variance.
    """
    measured = float(parity_data["cross_seed_mean_rescue"])
    delta = abs(measured - V7_0_CROSS_SEED_MEAN_BASELINE)
    assert delta <= PARITY_TOLERANCE, (
        f"PARITY_GATE_FAILED cross_seed: "
        f"mean Rescue@10={measured:.4f} vs baseline mean="
        f"{V7_0_CROSS_SEED_MEAN_BASELINE:.4f} "
        f"(|delta|={delta:.4f} > tolerance={PARITY_TOLERANCE:.4f}). "
        f"The parity check gates the Leiden-replacement work."
    )


def test_seeds_match_required_three(parity_data: dict) -> None:
    """Seeds list matches the canonical {13, 42, 137} parity baseline.

    At least 3 seeds are required for statistical validity; the
    baseline was measured on exactly these three so a re-measurement on
    different seeds would not be a fair comparison.
    """
    seeds_field = parity_data.get("seeds", [])
    # Accept either int or str list (different bench outputs format
    # seeds differently); coerce to str for comparison with our baseline.
    seeds_str = {str(s) for s in seeds_field}
    expected = set(V7_0_BASELINE_PER_SEED.keys())
    assert seeds_str == expected, (
        f"Seed list mismatch: got {sorted(seeds_str)}, expected "
        f"{sorted(expected)}. Re-run bench with --seeds 13 42 137 for "
        f"a fair comparison."
    )


def test_n_recalls_at_least_3000(parity_data: dict) -> None:
    """Total attributable recall count matches the honest scale.

    The bench --scale honest produces 3000 recalls
    (3 seeds * 1000 sessions * 2 slices * 500 probes/cell = 3000 per the
    bench corpus generator). Smaller counts indicate a misconfigured run
    (smoke/mvp scale) and invalidate the parity comparison.
    """
    n_recalls = int(parity_data.get("n_recalls", 0))
    assert n_recalls >= 3000, (
        f"Parity bench n_recalls={n_recalls} too small for honest-scale "
        f"comparison. Baseline was 3000 attributable recalls. "
        f"Re-run with --scale honest."
    )
