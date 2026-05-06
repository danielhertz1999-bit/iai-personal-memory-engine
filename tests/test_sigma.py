"""Plan 03-02 CONN-07 RED: sigma module unit tests.

Constitutional contract:
- D-SIGMA-01: sigma is None below SIGMA_N_FLOOR (=200) (Humphries-Gurney 2008).
- fast_sigma uses single-reference random graph; nx.sigma is FORBIDDEN
  (RESEARCH.md §Pitfall 1; >60s timeout at N=200).
- classify_regime is the four-cell truth table (D-SIGMA-02 / D-SIGMA-03).

Negative invariant: `src/iai_mcp/sigma.py` MUST NOT call `nx.sigma` or
`networkx.sigma` (verified by source-text scan).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest


# ---------------------------------------------------------------- module API


def test_sigma_module_exposes_constants_and_functions():
    """SIGMA_N_FLOOR=200 (D-SIGMA-01), SIGMA_MID_LIFE_THRESHOLD=500 (D-SIGMA-03)."""
    from iai_mcp import sigma

    assert sigma.SIGMA_N_FLOOR == 200
    assert sigma.SIGMA_MID_LIFE_THRESHOLD == 500
    assert callable(sigma.fast_sigma)
    assert callable(sigma.compute_sigma)
    assert callable(sigma.classify_regime)
    assert callable(sigma.compute_topology_snapshot)
    assert callable(sigma.compute_and_emit)


# ---------------------------------------------------------------- D-SIGMA-01 floor


def test_compute_sigma_returns_none_below_floor():
    """D-SIGMA-01: graphs with N<200 yield None (random baselines too noisy)."""
    from iai_mcp.sigma import compute_sigma

    g = nx.Graph()
    g.add_nodes_from(range(199))
    # add a few edges so the graph is non-trivial
    for i in range(10):
        g.add_edge(i, i + 1)
    assert compute_sigma(g) is None


# ---------------------------------------------------------------- fast_sigma sanity


def test_fast_sigma_small_world_above_one_at_n_250():
    """Watts-Strogatz p=0.1 at N=250 should give sigma > 1 (small-world).

    Per RESEARCH.md timing table the empirical value is around 9.65; we use a
    conservative >1 floor here to avoid being seed-fragile.
    """
    from iai_mcp.sigma import fast_sigma

    g = nx.connected_watts_strogatz_graph(250, k=6, p=0.1, seed=42)
    sigma_val, C, L, Cr, Lr = fast_sigma(g, n_random=3, seed=42)
    assert sigma_val > 1.0, f"expected sigma > 1, got {sigma_val:.3f}"
    assert C > 0
    assert L > 0
    assert Cr > 0
    assert Lr > 0


def test_fast_sigma_random_graph_near_one_at_n_250():
    """Erdos-Renyi G(n, m=750) at N=250 should give sigma ~ 1 (no small-worldness)."""
    from iai_mcp.sigma import fast_sigma

    g = nx.gnm_random_graph(250, 750, seed=42)
    sigma_val, _C, _L, _Cr, _Lr = fast_sigma(g, n_random=3, seed=43)
    # Random reference vs random target should be ~1; allow a generous band
    # because we only average over a few references.
    assert 0.5 < sigma_val < 1.5, f"expected sigma ~ 1, got {sigma_val:.3f}"


def test_fast_sigma_handles_disconnected_input():
    """Disconnected input: take largest CC; do not raise."""
    from iai_mcp.sigma import fast_sigma

    g = nx.connected_watts_strogatz_graph(220, k=6, p=0.1, seed=7)
    # Add 10 isolated nodes
    for k in range(220, 230):
        g.add_node(k)
    sigma_val, _C, _L, _Cr, _Lr = fast_sigma(g, n_random=2, seed=42)
    assert sigma_val > 0  # finite + positive (no crash on disconnected input)


# ---------------------------------------------------------------- regime truth table


def test_classify_regime_insufficient_data():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(50, None) == "insufficient_data"
    assert classify_regime(0, None) == "insufficient_data"


def test_classify_regime_developmental_n_lt_500_sigma_lt_1():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(300, 0.5) == "developmental"
    assert classify_regime(499, 0.99) == "developmental"


def test_classify_regime_mid_life_drift_n_ge_500_sigma_lt_1():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(500, 0.5) == "mid_life_drift"
    assert classify_regime(1000, 0.99) == "mid_life_drift"


def test_classify_regime_healthy_sigma_ge_1():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(300, 1.5) == "healthy"
    assert classify_regime(800, 5.0) == "healthy"
    assert classify_regime(200, 1.0) == "healthy"


# ---------------------------------------------------------------- negative: no nx.sigma


def test_sigma_module_does_not_call_nx_sigma():
    """RESEARCH.md §Pitfall 1: nx.sigma is forbidden (>60s timeout at N=200).

    Custom fast_sigma is the only allowed implementation in src/iai_mcp/sigma.py.
    """
    src = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "sigma.py"
    text = src.read_text(encoding="utf-8")
    # Allow the strings as documentation only inside docstrings/comments.
    # Hard-fail on actual calls.
    forbidden_calls = ["nx.sigma(", "networkx.sigma("]
    for needle in forbidden_calls:
        assert needle not in text, (
            f"sigma.py must NOT call {needle} -- use fast_sigma "
            f"(RESEARCH.md §Pitfall 1)"
        )
