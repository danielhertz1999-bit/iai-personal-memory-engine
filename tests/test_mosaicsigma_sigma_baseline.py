"""Sigma-baseline oracle tests.

This file is the gatekeeper for tests/fixtures/sigma_baseline.json. It:

1. Locks the fixture by SHA256 self-check (any byte-level edit fails loudly).
2. Verifies per-fixture sanity bands derived from closed-form math
   (1998) and published literature (Humphries-Gurney 2008).
3. Enforces classify_regime round-trip equality across every fixture.

Closed-form derivation (1998 + Humphries-Gurney 2008):

  WS k=4 ring lattice, p=0:
    C(p=0) = 3(k-2)/(4(k-1)) = 0.5
    Cr ~ avg_degree/(N-1) = 4/(N-1)
    L ~ N/(2k) = N/8 (large-N limit)
    Lr ~ ln(N)/ln(4) ~ 0.72*ln(N)
    sigma ~ (C/Cr) / (L/Lr) ~ 0.72*ln(N)

  Predictions:
    N=10: sigma_pred ~ 1.66 (finite-N actual ~ 0.86) -> band 0.5<sigma<2.0
    N=2500: sigma_pred ~ 5.62 (comfortable margin) -> band sigma>5.0

  H-G 2008 Table 1 empirical anchors:
    karate (N=34): sigma ~ 4.18 -> band 3.5<sigma<5.0
    les_miserables (N=77): sigma ~ 6.14 -> band 5.5<sigma<7.0
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

# Skip the entire module when networkx is unavailable. iai_mcp.sigma imports
# networkx at module scope, so importorskip MUST run before any other import
# that could pull sigma.py in.
pytest.importorskip("networkx")

from iai_mcp.sigma import classify_regime  # noqa: E402


FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"

# Locked at fixture-freeze time. Regenerate alongside the JSON in the same
# atomic commit when the fixture content actually changes.
SIGMA_BASELINE_SHA256 = "79b90bd3e2b66515ba394a924500ebb6eda811bd367aaf624c8e327f805eddef"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _canonical_bytes_without_hash(doc: dict) -> bytes:
    """Re-serialize doc without sha256_self_check, exactly as the generator does."""
    inner = {k: v for k, v in doc.items() if k != "sha256_self_check"}
    return json.dumps(inner, sort_keys=True, indent=2).encode("utf-8")


# ---------------------------------------------------------------- SHA256 lock


def test_baseline_sha256_locked():
    """Fixture content is frozen; any drift fails this test loudly.

    Mirrors the SHA-pin idiom used by the embedder numeric-parity gate.
    """
    doc = _load_fixture()
    stored_hash = doc.get("sha256_self_check")
    assert stored_hash, "sha256_self_check key missing from fixture"

    recomputed = hashlib.sha256(_canonical_bytes_without_hash(doc)).hexdigest()
    assert recomputed == stored_hash, (
        f"baseline SHA mismatch: stored={stored_hash} recomputed={recomputed} "
        "-- fixture content has drifted from canonical-form serialization"
    )

    assert stored_hash == SIGMA_BASELINE_SHA256, (
        f"baseline SHA drift vs module constant: "
        f"file_hash={stored_hash} module_constant={SIGMA_BASELINE_SHA256} "
        "-- regenerate baseline and update SIGMA_BASELINE_SHA256 in the same commit"
    )


# ---------------------------------------------------------------- classify_regime invariants


def test_classify_regime_invariants_per_fixture():
    """Every fixture's `regime` field must equal classify_regime(n, sigma).

    Optional `live_n2000` with regime == "unavailable" is skipped because
    classify_regime is not invoked when sigma is None and the fixture is a
    missing-snapshot placeholder.
    """
    fixtures = _load_fixture()["fixtures"]
    for key, fx in fixtures.items():
        if fx.get("regime") == "unavailable":
            # Optional fixture, placeholder; nothing to cross-check.
            continue
        computed = classify_regime(int(fx["n"]), fx["sigma"])
        assert computed == fx["regime"], (
            f"regime drift on {key}: stored={fx['regime']!r} "
            f"computed={computed!r} (n={fx['n']}, sigma={fx['sigma']!r})"
        )


# ---------------------------------------------------------------- band assertions


def test_sigma_band_ring_lattice_tiny_10():
    """tiny_10_ws_k4: C > 0.4 AND 0.5 < sigma < 2.0.

    Closed-form: C(p=0) = 3(k-2)/(4(k-1)) = 0.5 exactly.
    Finite-N empirical sigma ~ 0.86 (below the 0.72*ln(10)=1.66 asymptote).
    """
    fx = _load_fixture()["fixtures"]["tiny_10_ws_k4"]
    assert fx["C"] > 0.4, f"tiny_10_ws_k4 C={fx['C']:.4f} expected > 0.4"
    assert 0.5 < fx["sigma"] < 2.0, (
        f"tiny_10_ws_k4 sigma={fx['sigma']:.4f} outside band 0.5<sigma<2.0 "
        "(closed-form-derived; matches Watts-Strogatz 1998 finite-N reality)"
    )


def test_sigma_band_strict_magnitude_ws_2500():
    """ws_2500_k4_p0: C > 0.4 AND sigma > 5.0 — strict magnitude sanity-band.

    Closed-form: sigma ~ 0.72 * ln(2500) ~ 5.62 (asymptote with margin
    above 5.0). Empirical sigma ~ 6.13 exceeds the asymptote.
    """
    fx = _load_fixture()["fixtures"]["ws_2500_k4_p0"]
    assert fx["C"] > 0.4, f"ws_2500_k4_p0 C={fx['C']:.4f} expected > 0.4"
    assert fx["sigma"] > 5.0, (
        f"ws_2500_k4_p0 sigma={fx['sigma']:.4f} expected > 5.0 "
        "(strict magnitude sanity-band; closed-form pred ~ 5.62 at N=2500)"
    )


def test_sigma_band_karate():
    """karate: 3.5 < sigma < 5.0 (Humphries-Gurney 2008 Table 1, sigma ~ 4.18)."""
    fx = _load_fixture()["fixtures"]["karate"]
    assert 3.5 < fx["sigma"] < 5.0, (
        f"karate sigma={fx['sigma']:.4f} outside band 3.5<sigma<5.0 "
        "(Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1 row: sigma ~ 4.18)"
    )


def test_sigma_band_les_miserables():
    """les_miserables: 5.5 < sigma < 7.0 (H-G 2008 Table 1, sigma ~ 6.14)."""
    fx = _load_fixture()["fixtures"]["les_miserables"]
    assert 5.5 < fx["sigma"] < 7.0, (
        f"les_miserables sigma={fx['sigma']:.4f} outside band 5.5<sigma<7.0 "
        "(Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1 row: sigma ~ 6.14)"
    )


@pytest.mark.parametrize("key", ["er_200", "er_500", "er_1000"])
def test_sigma_band_er(key: str):
    """Erdos-Renyi random baselines: 0.5 <= sigma <= 1.5 (by construction ~ 1)."""
    fx = _load_fixture()["fixtures"][key]
    assert 0.5 <= fx["sigma"] <= 1.5, (
        f"{key} sigma={fx['sigma']:.4f} outside random-reference band "
        "0.5<=sigma<=1.5 (sigma should be ~ 1 by construction for ER graphs)"
    )


def test_regime_direction_tiny_20():
    """tiny_20_ws_p010: C/Cr > 1.1 ONLY (regime-direction check).

    Status: informational-boundary. N=20 is too small for a meaningful
    small-world signature; the magnitude check is dropped and only the
    direction (clustering enrichment) is asserted.
    """
    fx = _load_fixture()["fixtures"]["tiny_20_ws_p010"]
    assert fx["status"] == "informational-boundary", (
        f"tiny_20_ws_p010 status={fx['status']!r} expected 'informational-boundary'"
    )
    assert fx["Cr"] > 0, f"tiny_20_ws_p010 Cr={fx['Cr']:.4f} must be > 0"
    ratio = fx["C"] / fx["Cr"]
    assert ratio > 1.1, (
        f"tiny_20_ws_p010 C/Cr={ratio:.4f} expected > 1.1 "
        "(regime-direction check; sigma magnitude is intentionally not gated at N=20)"
    )


@pytest.mark.literature
def test_literature_cross_check_humphries_gurney():
    """Cross-check sigma against Humphries-Gurney 2008 Table 1 published values.

    Karate row: sigma ~ 4.18
    Les Miserables row: sigma ~ 6.14

    Allowed ±20% band absorbs FP differences across BLAS implementations and
    the small fan-out (n_random=3) we use vs the larger fan-out in H-G 2008.
    """
    fixtures = _load_fixture()["fixtures"]

    # Karate (Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1 row: sigma ~ 4.18)
    karate_sigma = fixtures["karate"]["sigma"]
    karate_published = 4.18
    karate_drift = abs(karate_sigma - karate_published) / karate_published
    assert karate_drift <= 0.20, (
        f"karate sigma={karate_sigma:.4f} drift {karate_drift:.2%} > 20% "
        f"from H-G 2008 Table 1 published value {karate_published}"
    )

    # Les Miserables (Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1 row: sigma ~ 6.14)
    lesmis_sigma = fixtures["les_miserables"]["sigma"]
    lesmis_published = 6.14
    lesmis_drift = abs(lesmis_sigma - lesmis_published) / lesmis_published
    assert lesmis_drift <= 0.20, (
        f"les_miserables sigma={lesmis_sigma:.4f} drift {lesmis_drift:.2%} > 20% "
        f"from H-G 2008 Table 1 published value {lesmis_published}"
    )


def test_live_n2000_snapshot_loadable():
    """Optional live snapshot — skip cleanly when no snapshot is materialized.

    When a snapshot IS available, verify the npz arrays {indptr, indices,
    data, node_id_strs} are loadable and consistent with the fixture's
    declared node count.
    """
    fx = _load_fixture()["fixtures"]["live_n2000"]
    if fx.get("regime") == "unavailable" or fx.get("source") == "missing-snapshot":
        pytest.skip("live_n2000 snapshot unavailable (skip-on-missing posture)")

    snapshot_path = fx.get("snapshot_path")
    assert snapshot_path, "live_n2000 entry is non-missing but has no snapshot_path"
    snapshot_file = pathlib.Path(snapshot_path)
    assert snapshot_file.exists(), f"snapshot file missing: {snapshot_path}"

    # Lazy numpy import so the test stays importable in stripped-down envs.
    numpy = pytest.importorskip("numpy")
    payload = numpy.load(snapshot_file)
    required = {"indptr", "indices", "data", "node_id_strs"}
    missing = required - set(payload.files)
    assert not missing, f"snapshot missing arrays: {missing}"

    n = int(fx["n"])
    assert n > 0, "live_n2000 declares n>0 when snapshot is present"
    assert len(payload["node_id_strs"]) == n, (
        f"node_id_strs length {len(payload['node_id_strs'])} != fixture n={n}"
    )
    assert len(payload["indptr"]) == n + 1, (
        f"indptr length {len(payload['indptr'])} != n+1={n + 1}"
    )
