
from __future__ import annotations

import json
import pathlib

import pytest

iai_mcp_native = pytest.importorskip("iai_mcp_native")

from iai_mcp_native import graph  # noqa: E402  (importorskip is above)


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)


def _load_gnm_baseline() -> dict[str, dict]:
    doc = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    baseline = doc.get("gnm_baseline")
    assert baseline, "gnm_baseline missing from sigma_baseline.json"
    return baseline


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def test_gnm_edge_count_equals_m():
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    assert len(u_list) == 400, f"len(u_list)={len(u_list)} != 400"
    assert len(v_list) == 400, f"len(v_list)={len(v_list)} != 400"


def test_gnm_no_self_loops():
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    bad = [(u, v) for u, v in zip(u_list, v_list) if u == v]
    assert not bad, f"found {len(bad)} self-loops, e.g. {bad[:3]}"


def test_gnm_no_duplicate_edges():
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    canon = {_canonical_edge(u, v) for u, v in zip(u_list, v_list)}
    assert len(canon) == 400, (
        f"only {len(canon)} unique edges after canonicalization; "
        "duplicates present in generator output"
    )


def test_gnm_node_range_valid():
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    n = 200
    for u, v in zip(u_list, v_list):
        assert 0 <= u < n, f"u={u} outside [0, {n})"
        assert 0 <= v < n, f"v={v} outside [0, {n})"


def test_gnm_deterministic_under_seed():
    first = graph.gnm_random_graph(200, 400, 42)
    second = graph.gnm_random_graph(200, 400, 42)
    assert first == second, (
        "two calls with seed=42 produced different edge lists — "
        "rustworkx-core seed plumbing broken or rustworkx-core upgrade "
        "silently changed RNG output"
    )


def test_gnm_frozen_edge_set_matches_baseline():
    baseline = _load_gnm_baseline()
    for key, entry in baseline.items():
        n = int(entry["n"])
        m = int(entry["m"])
        seed = int(entry["seed"])
        expected_u = [int(x) for x in entry["u_list"]]
        expected_v = [int(x) for x in entry["v_list"]]

        u_list, v_list = graph.gnm_random_graph(n, m, seed)
        assert list(u_list) == expected_u, (
            f"{key}: u_list drift — generator first 5 = {list(u_list)[:5]} "
            f"baseline first 5 = {expected_u[:5]}"
        )
        assert list(v_list) == expected_v, (
            f"{key}: v_list drift — generator first 5 = {list(v_list)[:5]} "
            f"baseline first 5 = {expected_v[:5]}"
        )


def test_gnm_m_too_large_raises():
    with pytest.raises(ValueError, match=r"exceeds n\*\(n-1\)/2"):
        graph.gnm_random_graph(5, 20, 42)


def test_gnm_zero_edges_returns_empty():
    u_list, v_list = graph.gnm_random_graph(10, 0, 42)
    assert list(u_list) == [], f"u_list={list(u_list)} expected []"
    assert list(v_list) == [], f"v_list={list(v_list)} expected []"


def test_gnm_n_zero_raises():
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        graph.gnm_random_graph(0, 0, 42)
