"""Differential parity for iai_mcp_native.graph.average_clustering.

LOCAL average clustering coefficient implemented in
pure Rust. Per-node formula:

    c(v) = 2 * T(v) / (d(v) * (d(v) - 1)) for d(v) >= 2
    c(v) = 0 for d(v) < 2

Average over all nodes (arithmetic mean):

    C_avg = (1 / n_nodes) * sum_v c(v)

The implementation must be EQUAL to networkx.average_clustering on
the canonical reference fixtures within |delta| <= 1e-9, and FP-exact
on the two boundary fixtures (K5 -> 1.0, K_n with isolated node ->
known closed form).

Critical: the native function must NOT delegate to a global
transitivity coefficient (3 * triangles / connected_triples). The
two values diverge whenever the degree distribution is non-uniform
(Karate, Erdos-Renyi, all real graphs). A separate test enforces the
"no global transitivity import" invariant via grep.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest


def _native_available() -> bool:
    try:
        import iai_mcp_native  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)

pytest.importorskip("networkx")
pytest.importorskip("numpy")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_fixtures() -> dict:
    """Return the fixtures dict from the locked sigma_baseline.json."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


def _edges_to_csr(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical CSR (indptr, indices) over an undirected simple graph.

    Each edge (u, v) is added both directions (symmetric CSR). Self-loops
    are preserved as-is (the clustering kernel ignores self-edges by
    requiring v != u when counting neighbor pairs).
    Neighbor lists are sorted ascending by node id — required for the
    binary-search edge check inside the kernel.
    """
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in edges:
        if u == v:
            # Self-loops do not contribute to clustering; skip to match
            # networkx semantics on simple graphs.
            continue
        adj[u].add(v)
        adj[v].add(u)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indices_list: list[int] = []
    for u in range(n_nodes):
        sorted_nbrs = sorted(adj[u])
        indices_list.extend(sorted_nbrs)
        indptr[u + 1] = indptr[u] + len(sorted_nbrs)
    indices = np.asarray(indices_list, dtype=np.int64)
    return indptr, indices


def _nx_graph(n_nodes: int, edges: list[tuple[int, int]]) -> nx.Graph:
    """Build the networkx oracle graph (skip self-loops, undirected)."""
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((u, v) for u, v in edges if u != v)
    return g


def _ours(n_nodes: int, edges: list[tuple[int, int]]) -> float:
    """Call the native LOCAL average_clustering kernel."""
    from iai_mcp_native import graph
    indptr, indices = _edges_to_csr(n_nodes, edges)
    return float(graph.average_clustering(indptr, indices, n_nodes))


# ---------------------------------------------------------------- fixtures


# Reference fixtures: real graphs with non-degenerate clustering.
REFERENCE_FIXTURES = [
    "karate",
    "les_miserables",
    "er_200",
    "er_500",
    "er_1000",
    "tiny_10_ws_k4",
    "tiny_20_ws_p010",
    "ws_2500_k4_p0",
]


def test_local_clustering_matches_networkx_on_fixtures() -> None:
    """LOCAL average_clustering parity vs networkx oracle.

    |our - networkx| <= 1e-9 on every reference fixture. Tolerance is
    high (1e-9) because LOCAL
    clustering is a sum of small rationals over n_nodes -- FP drift is
    extremely low for the per-node formula `2 T / (d (d - 1))`.

    Uses the failure-list idiom -- collect all mismatches,
    fail once with the full table for triage.
    """
    fixtures = _load_fixtures()
    failures: list[tuple[str, float, float, float]] = []
    for name in REFERENCE_FIXTURES:
        f = fixtures[name]
        n = int(f["n"])
        edges = [(int(u), int(v)) for u, v in f["edges"]]
        g_nx = _nx_graph(n, edges)
        oracle = float(nx.average_clustering(g_nx))
        ours = _ours(n, edges)
        delta = abs(ours - oracle)
        if delta > 1e-9:
            failures.append((name, ours, oracle, delta))
    assert not failures, (
        f"{len(failures)} clustering parity failures:\n"
        + "\n".join(
            f"  {name}: ours={ours:.15g} oracle={oracle:.15g} |delta|={delta:.3e}"
            for name, ours, oracle, delta in failures
        )
    )


def test_global_transitivity_not_used_in_source() -> None:
    """clustering.rs must NOT import rustworkx_core::transitivity.

    Global transitivity is
    `3 * triangles / connected_triples` and computes a different value
    from LOCAL clustering on non-regular graphs.
    Substituting it would silently shift sigma and could flip
    classify_regime.

    grep returns exit code 0 on match (= constitutional violation) and
    exit code 1 on no-match (= the desired state). We assert returncode
    == 1 directly rather than relying on `subprocess.check_output`,
    which raises CalledProcessError on no-match and would have to be
    caught -- the explicit `check=False` + returncode comparison is
    clearer.
    """
    target_dir = REPO_ROOT / "rust" / "iai_mcp_graph_core" / "src"
    result = subprocess.run(
        ["grep", "-rE", "rustworkx_core::transitivity", str(target_dir)],
        check=False,
        capture_output=True,
    )
    # grep exit code: 0 = matches found (BAD), 1 = no matches (GOOD),
    # 2 = grep error (BAD). We want exit 1.
    assert result.returncode == 1, (
        "transitivity import found in graph_core source -- constitutional "
        "violation. grep output: "
        + result.stdout.decode(errors="replace")[:500]
    )


def test_sanity_band_ring_lattice_k4_nonzero_clustering() -> None:
    """WS(10, k=4, p=0) ring lattice has C = 3(k-2)/(4(k-1)) = 0.5.

    Watts & Strogatz 1998 (Nature 393:440-442) gave the closed form for
    a k-nearest-neighbor ring lattice with no rewiring (p=0). For k=4:
    C = 6 / 12 = 0.5 exactly. This is a sharper sanity check than the
    pre-review k=2 cycle (which is a tree-like cycle with C=0 -- a
    vacuous test that misses formula errors).
    """
    fixtures = _load_fixtures()
    f = fixtures["tiny_10_ws_k4"]
    n = int(f["n"])
    edges = [(int(u), int(v)) for u, v in f["edges"]]
    ours = _ours(n, edges)
    assert abs(ours - 0.5) <= 1e-9, (
        f"WS(10, k=4, p=0) sanity band: expected 0.5, got {ours!r}"
    )


def test_sanity_band_complete_graph_unity_clustering() -> None:
    """K5 (complete graph on 5 nodes): every node's neighbors are all
    pairwise connected so c(v) = 1.0 for every v -> mean = 1.0.

    FP-exact equality is required: any drift here means the formula is
    structurally wrong (off-by-2 numerator, off-by-1 denominator, etc).
    """
    n = 5
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    ours = _ours(n, edges)
    assert ours == 1.0, (
        f"K5 complete-graph clustering: expected exactly 1.0, got {ours!r}"
    )


def test_disconnected_graph_clustering_well_defined() -> None:
    """Two disjoint K3 triangles: each node has c(v) = 1.0
    (its 2 neighbors are connected), so the average over 6 nodes is 1.0.

    Confirms the average is taken over ALL nodes (not over connected
    components separately) and matches networkx semantics on
    disconnected graphs.
    """
    edges = [
        (0, 1), (0, 2), (1, 2),     # K3 #1
        (3, 4), (3, 5), (4, 5),     # K3 #2
    ]
    n = 6
    oracle = float(nx.average_clustering(_nx_graph(n, edges)))
    ours = _ours(n, edges)
    assert oracle == 1.0  # sanity-check the oracle
    assert ours == 1.0, (
        f"Disjoint K3s clustering: expected exactly 1.0, got {ours!r}"
    )


def test_isolated_and_degree_one_nodes_contribute_zero() -> None:
    """Degree-0 (isolated) and degree-1 (leaf) nodes contribute 0 to the
    sum per the c(v < 2) = 0 convention. Verified against networkx
    on a 5-node star + 1 isolate (n=6): center=1 isolate + 4 leaves
    + 1 totally isolated node = average over 6 nodes of zeros = 0.0.
    """
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]  # star with center=0
    n = 6  # node 5 is isolated
    oracle = float(nx.average_clustering(_nx_graph(n, edges)))
    ours = _ours(n, edges)
    assert oracle == 0.0
    assert ours == 0.0, (
        f"Star+isolate clustering: expected exactly 0.0, got {ours!r}"
    )
