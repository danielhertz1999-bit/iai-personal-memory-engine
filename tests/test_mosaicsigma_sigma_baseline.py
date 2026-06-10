
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

pytest.importorskip("networkx")

from iai_mcp.sigma import classify_regime  # noqa: E402


FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"

SIGMA_BASELINE_SHA256 = "79b90bd3e2b66515ba394a924500ebb6eda811bd367aaf624c8e327f805eddef"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _canonical_bytes_without_hash(doc: dict) -> bytes:
    inner = {k: v for k, v in doc.items() if k != "sha256_self_check"}
    return json.dumps(inner, sort_keys=True, indent=2).encode("utf-8")


def test_baseline_sha256_locked():
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


def test_classify_regime_invariants_per_fixture():
    fixtures = _load_fixture()["fixtures"]
    for key, fx in fixtures.items():
        if fx.get("regime") == "unavailable":
            continue
        computed = classify_regime(int(fx["n"]), fx["sigma"])
        assert computed == fx["regime"], (
            f"regime drift on {key}: stored={fx['regime']!r} "
            f"computed={computed!r} (n={fx['n']}, sigma={fx['sigma']!r})"
        )


def test_sigma_band_ring_lattice_tiny_10():
    fx = _load_fixture()["fixtures"]["tiny_10_ws_k4"]
    assert fx["C"] > 0.4, f"tiny_10_ws_k4 C={fx['C']:.4f} expected > 0.4"
    assert 0.5 < fx["sigma"] < 2.0, (
        f"tiny_10_ws_k4 sigma={fx['sigma']:.4f} outside band 0.5<sigma<2.0 "
        "(closed-form-derived; matches Watts-Strogatz 1998 finite-N reality)"
    )


def test_sigma_band_strict_magnitude_ws_2500():
    fx = _load_fixture()["fixtures"]["ws_2500_k4_p0"]
    assert fx["C"] > 0.4, f"ws_2500_k4_p0 C={fx['C']:.4f} expected > 0.4"
    assert fx["sigma"] > 5.0, (
        f"ws_2500_k4_p0 sigma={fx['sigma']:.4f} expected > 5.0 "
        "(strict magnitude sanity-band; closed-form pred ~ 5.62 at N=2500)"
    )


def test_sigma_band_karate():
    fx = _load_fixture()["fixtures"]["karate"]
    assert 3.5 < fx["sigma"] < 5.0, (
        f"karate sigma={fx['sigma']:.4f} outside band 3.5<sigma<5.0 "
        "(Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1 row: sigma ~ 4.18)"
    )


def test_sigma_band_les_miserables():
    fx = _load_fixture()["fixtures"]["les_miserables"]
    assert 5.5 < fx["sigma"] < 7.0, (
        f"les_miserables sigma={fx['sigma']:.4f} outside band 5.5<sigma<7.0 "
        "(Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1 row: sigma ~ 6.14)"
    )


@pytest.mark.parametrize("key", ["er_200", "er_500", "er_1000"])
def test_sigma_band_er(key: str):
    fx = _load_fixture()["fixtures"][key]
    assert 0.5 <= fx["sigma"] <= 1.5, (
        f"{key} sigma={fx['sigma']:.4f} outside random-reference band "
        "0.5<=sigma<=1.5 (sigma should be ~ 1 by construction for ER graphs)"
    )


def test_regime_direction_tiny_20():
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
    fixtures = _load_fixture()["fixtures"]

    karate_sigma = fixtures["karate"]["sigma"]
    karate_published = 4.18
    karate_drift = abs(karate_sigma - karate_published) / karate_published
    assert karate_drift <= 0.20, (
        f"karate sigma={karate_sigma:.4f} drift {karate_drift:.2%} > 20% "
        f"from H-G 2008 Table 1 published value {karate_published}"
    )

    lesmis_sigma = fixtures["les_miserables"]["sigma"]
    lesmis_published = 6.14
    lesmis_drift = abs(lesmis_sigma - lesmis_published) / lesmis_published
    assert lesmis_drift <= 0.20, (
        f"les_miserables sigma={lesmis_sigma:.4f} drift {lesmis_drift:.2%} > 20% "
        f"from H-G 2008 Table 1 published value {lesmis_published}"
    )


def test_live_n2000_snapshot_loadable():
    fx = _load_fixture()["fixtures"]["live_n2000"]
    if fx.get("regime") == "unavailable" or fx.get("source") == "missing-snapshot":
        pytest.skip("live_n2000 snapshot unavailable (skip-on-missing posture)")

    snapshot_path = fx.get("snapshot_path")
    assert snapshot_path, "live_n2000 entry is non-missing but has no snapshot_path"
    snapshot_file = pathlib.Path(snapshot_path)
    assert snapshot_file.exists(), f"snapshot file missing: {snapshot_path}"

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
