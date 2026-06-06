"""Parity tests for the Rust ``gnm_random_graph`` generator.

Eight assertions form the constitutional contract:

  1. Edge count equals ``m`` exactly.
  2. No self-loops.
  3. No duplicate undirected edges (after canonicalization to
     ``(min(u, v), max(u, v))``).
  4. Node ids live in ``[0, n)``.
  5. Deterministic under fixed seed — two calls with the same
     ``(n, m, seed)`` return identical edge lists.
  6. Frozen edge sets in ``tests/fixtures/sigma_baseline.json::gnm_baseline``
     match the generator's current output bit-for-bit. This is the
     anti-drift gate — an ``rustworkx-core`` patch upgrade that silently
     shifts the RNG output trips this test loudly.
  7. ``m > n * (n - 1) / 2`` raises ``ValueError``.
  8. ``m == 0`` yields empty edge lists with no allocation issues.

The generator EXPLICITLY does NOT chase ``networkx.gnm_random_graph``
bit-for-bit — the parity contract is the graph-property invariants
(1-4) plus determinism + our own canonical edge sets (6).
"""

from __future__ import annotations

import json
import pathlib

import pytest

# Skip the entire module when the native wheel isn't importable; the σ
# baseline tests already skip on missing networkx, but ``iai_mcp_native``
# is a separate Rust wheel that may not be installed in every CI shape.
iai_mcp_native = pytest.importorskip("iai_mcp_native")

from iai_mcp_native import graph  # noqa: E402 (importorskip is above)


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)


def _load_gnm_baseline() -> dict[str, dict]:
    doc = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    baseline = doc.get("gnm_baseline")
    assert baseline, "gnm_baseline missing from sigma_baseline.json"
    return baseline


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    """Return the undirected-canonical (min, max) form of an edge."""
    return (u, v) if u <= v else (v, u)


# -------------------------------------------------------------- property tests


def test_gnm_edge_count_equals_m():
    """Calling with (n=200, m=400, seed=42) yields exactly 400 edges."""
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    assert len(u_list) == 400, f"len(u_list)={len(u_list)} != 400"
    assert len(v_list) == 400, f"len(v_list)={len(v_list)} != 400"


def test_gnm_no_self_loops():
    """No (u, v) pair has u == v."""
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    bad = [(u, v) for u, v in zip(u_list, v_list) if u == v]
    assert not bad, f"found {len(bad)} self-loops, e.g. {bad[:3]}"


def test_gnm_no_duplicate_edges():
    """No two (u, v) pairs canonicalize to the same edge."""
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    canon = {_canonical_edge(u, v) for u, v in zip(u_list, v_list)}
    assert len(canon) == 400, (
        f"only {len(canon)} unique edges after canonicalization; "
        "duplicates present in generator output"
    )


def test_gnm_node_range_valid():
    """Every node id is in [0, n)."""
    u_list, v_list = graph.gnm_random_graph(200, 400, 42)
    n = 200
    for u, v in zip(u_list, v_list):
        assert 0 <= u < n, f"u={u} outside [0, {n})"
        assert 0 <= v < n, f"v={v} outside [0, {n})"


def test_gnm_deterministic_under_seed():
    """Two consecutive calls with the same seed return identical lists."""
    first = graph.gnm_random_graph(200, 400, 42)
    second = graph.gnm_random_graph(200, 400, 42)
    assert first == second, (
        "two calls with seed=42 produced different edge lists — "
        "rustworkx-core seed plumbing broken or rustworkx-core upgrade "
        "silently changed RNG output"
    )


# ---------------------------------------------------------------- anti-drift gate


def test_gnm_frozen_edge_set_matches_baseline():
    """Per-seed canonical edge sets match the frozen fixture.

    This is the constitutional gate — if ``rustworkx-core 0.17.x`` ever
    changes its RNG between patches, or if the workspace pin shifts
    silently, this test fails loud and the operator is forced to
    decide: re-freeze the baseline, or revert the upgrade.
    """
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


# ---------------------------------------------------------------- error paths


def test_gnm_m_too_large_raises():
    """m > n*(n-1)/2 raises ValueError (NOT silently capped)."""
    # n=5 max undirected edges = 5*4/2 = 10
    with pytest.raises(ValueError, match=r"exceeds n\*\(n-1\)/2"):
        graph.gnm_random_graph(5, 20, 42)


def test_gnm_zero_edges_returns_empty():
    """m=0 yields empty lists with no edge-count or RNG side effects."""
    u_list, v_list = graph.gnm_random_graph(10, 0, 42)
    assert list(u_list) == [], f"u_list={list(u_list)} expected []"
    assert list(v_list) == [], f"v_list={list(v_list)} expected []"


def test_gnm_n_zero_raises():
    """n=0 raises ValueError (matches rustworkx-core's InvalidInputError)."""
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        graph.gnm_random_graph(0, 0, 42)
