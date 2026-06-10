from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("networkx")
pytest.importorskip("numpy")

import networkx as nx  # noqa: E402

from tests.conftest import _nx_graph_to_memory_graph  # noqa: E402

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

REGIME_GATE_MANDATORY_FIXTURES = [
    "karate",
    "les_miserables",
    "er_200",
    "er_500",
    "er_1000",
    "ws_2500_k4_p0",
]
REGIME_GATE_OPTIONAL_FIXTURES = ["live_n2000"]


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


def _build_nx_from_fixture(name: str, fixture: dict):
    if fixture.get("source") == "missing-snapshot":
        return None
    n = int(fixture["n"])
    edges = [(int(u), int(v)) for u, v in fixture["edges"] if int(u) != int(v)]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


def _fast_sigma_via_networkx_oracle(
    g_nx, *, n_random: int = 3, seed: int = 42
) -> float:
    from iai_mcp_native import graph as lilli_graph

    if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
        return float("nan")
    if not nx.is_connected(g_nx):
        largest = max(nx.connected_components(g_nx), key=len)
        g_nx = g_nx.subgraph(largest).copy()
    n = int(g_nx.number_of_nodes())
    m = int(g_nx.number_of_edges())
    if n < 2 or m == 0:
        return float("nan")

    C = float(nx.average_clustering(g_nx))
    L = float(nx.average_shortest_path_length(g_nx))

    Cs: list[float] = []
    Ls: list[float] = []
    for k in range(max(1, n_random)):
        u_list, v_list = lilli_graph.gnm_random_graph(n, m, seed=seed + k)
        gr_full = nx.Graph()
        gr_full.add_nodes_from(range(n))
        gr_full.add_edges_from(zip(u_list, v_list))
        if gr_full.number_of_edges() == 0:
            continue
        if not nx.is_connected(gr_full):
            largest = max(nx.connected_components(gr_full), key=len)
            gr = gr_full.subgraph(largest).copy()
        else:
            gr = gr_full
        if gr.number_of_nodes() < 2 or gr.number_of_edges() == 0:
            continue
        Cs.append(float(nx.average_clustering(gr)))
        Ls.append(float(nx.average_shortest_path_length(gr)))

    if not Cs or not Ls:
        return float("nan")
    Cr = sum(Cs) / len(Cs)
    Lr = sum(Ls) / len(Ls)
    if Cr <= 0 or Lr <= 0 or L <= 0:
        return float("nan")
    return (C / Cr) / (L / Lr)


@pytest.mark.parametrize(
    "fixture_name",
    REGIME_GATE_MANDATORY_FIXTURES + REGIME_GATE_OPTIONAL_FIXTURES,
)
def test_sigma_regime_matches_baseline_on_mandatory_6_plus_optional_live(
    fixture_name: str,
) -> None:
    from iai_mcp.sigma import classify_regime, fast_sigma

    fixtures = _load_fixtures()
    fx = fixtures[fixture_name]
    g_nx = _build_nx_from_fixture(fixture_name, fx)
    if g_nx is None:
        pytest.skip(
            f"{fixture_name}: missing-snapshot placeholder; optional fixture skipped"
        )
    mg = _nx_graph_to_memory_graph(g_nx)
    sigma_tuple = fast_sigma(mg, seed=42)
    sigma_ours = float(sigma_tuple[0])
    sigma_oracle_live = _fast_sigma_via_networkx_oracle(g_nx, n_random=3, seed=42)

    regime_ours = classify_regime(mg.node_count(), sigma_ours)
    regime_oracle = classify_regime(mg.node_count(), sigma_oracle_live)
    assert regime_ours == regime_oracle, (
        f"GRAPH-03 regime equality violation for {fixture_name}: "
        f"ours regime={regime_ours} sigma_ours={sigma_ours:.4f} "
        f"oracle regime={regime_oracle} sigma_oracle_live={sigma_oracle_live:.4f} "
        f"(fixture historical sigma was {fx.get('sigma')}; this is documented "
        f"as a Pcg64-vs-networkx-gnm divergence, not used as the gate oracle)."
    )


def test_sigma_regime_invariant_under_seed_variation() -> None:
    from iai_mcp.sigma import classify_regime, fast_sigma

    fixtures = _load_fixtures()
    fx = fixtures["karate"]
    g_nx = _build_nx_from_fixture("karate", fx)
    mg = _nx_graph_to_memory_graph(g_nx)

    seeds = [42, 43, 44, 45, 46]
    regimes = []
    for s in seeds:
        sigma_val = float(fast_sigma(mg, seed=s)[0])
        regimes.append(classify_regime(mg.node_count(), sigma_val))

    from collections import Counter

    counts = Counter(regimes)
    majority_regime, _ = counts.most_common(1)[0]
    swap_count = sum(1 for r in regimes if r != majority_regime)
    assert swap_count <= 1, (
        f"regime instability under seed variation: {swap_count}/5 seed values "
        f"swap regime. seeds={seeds} regimes={regimes} majority={majority_regime}"
    )


@pytest.mark.parametrize(
    "fixture_name",
    REGIME_GATE_MANDATORY_FIXTURES + REGIME_GATE_OPTIONAL_FIXTURES,
)
def test_sigma_value_within_baseline_tolerance(fixture_name: str) -> None:
    from iai_mcp.sigma import fast_sigma

    fixtures = _load_fixtures()
    fx = fixtures[fixture_name]
    g_nx = _build_nx_from_fixture(fixture_name, fx)
    if g_nx is None:
        pytest.skip(
            f"{fixture_name}: missing-snapshot placeholder; optional fixture skipped"
        )
    mg = _nx_graph_to_memory_graph(g_nx)
    sigma_tuple = fast_sigma(mg, seed=42)
    sigma_ours = float(sigma_tuple[0])
    sigma_oracle_live = _fast_sigma_via_networkx_oracle(g_nx, n_random=3, seed=42)

    tol = max(0.10 * abs(sigma_oracle_live), 0.01)
    delta = abs(sigma_ours - sigma_oracle_live)
    assert delta <= tol, (
        f"{fixture_name}: sigma drift {delta:.4f} exceeds tolerance "
        f"max(0.10*|oracle|, 0.01) = {tol:.4f}. "
        f"ours={sigma_ours:.4f} oracle_live={sigma_oracle_live:.4f}"
    )


def test_main_install_does_not_load_networkx() -> None:
    check_script = (
        "import sys; "
        "import iai_mcp; "  # noqa: F401
        "assert 'networkx' not in sys.modules, "
        "'iai_mcp top-level import triggered a networkx load -- "
        "GRAPH-01 eviction invariant violated'"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        "iai_mcp top-level import triggered a networkx load -- "
        "GRAPH-01 eviction invariant violated. networkx must remain in "
        "[dev] extras only and be lazy-imported by MemoryGraph.__init__. "
        f"stderr: {result.stderr.decode(errors='replace')[:500]}"
    )


def test_no_nx_references_in_src() -> None:
    result = subprocess.run(
        ["git", "grep", "-nE", "^(import|from) networkx", "src/"],
        check=False,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 1, (
        f"networkx import detected in src/ at module scope -- "
        f"eviction invariant violated. grep output: "
        f"{result.stdout.decode(errors='replace')[:500]}"
    )


def test_no_graph_nx_references_in_src() -> None:
    result = subprocess.run(
        ["git", "grep", "-nE", r"graph\._nx\.", "src/"],
        check=False,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode == 1:
        return
    matches = result.stdout.decode(errors="replace").splitlines()
    foreign = [
        line for line in matches if not line.startswith("src/iai_mcp/graph.py:")
    ]
    assert not foreign, (
        "graph._nx. access detected outside src/iai_mcp/graph.py -- "
        f"private-attribute leak. Foreign callsites: {foreign[:10]}"
    )


def test_no_graph_adj_references_in_src() -> None:
    result = subprocess.run(
        ["git", "grep", "-nE", r"graph\._adj\.", "src/"],
        check=False,
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode == 1:
        return
    matches = result.stdout.decode(errors="replace").splitlines()
    foreign = [
        line for line in matches if not line.startswith("src/iai_mcp/graph.py:")
    ]
    assert not foreign, (
        "graph._adj. access detected outside src/iai_mcp/graph.py -- "
        f"private-attribute leak. Foreign callsites: {foreign[:10]}"
    )
