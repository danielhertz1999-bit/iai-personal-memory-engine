"""Differential parity for MemoryGraph.rich_club_coefficient.

Pure Python + numpy reimplementation of van den Heuvel & 2011
(J Neurosci 31:15775-15786) rich-club phi(k). This test file is the
gatekeeper: every supported fixture must match the networkx oracle within
|delta| <= 1e-9.

Formula (un-normalized phi):
    phi(k) = 2 * E_{>k} / (N_{>k} * (N_{>k} - 1))

where
    E_{>k} = edges (u, v) with deg(u) > k AND deg(v) > k after self-loop strip,
    N_{>k} = count of nodes with deg > k.

The implementation MUST:
1. Strip self-loops on a copy of the edge list (preserve graph.py's prior
   ``G_for_rc = self._nx.copy(); G_for_rc.remove_edges_from(selfloop_edges)``
   semantics from the pre-rewrite implementation).
2. Rebuild the degree dict from the post-strip edge list seeded with every
   node returned by ``iter_nodes()`` at zero — isolates contribute degree 0
   to the distribution and must NOT be silently dropped.
3. Default k_threshold = ``int(numpy.percentile(deg_values, 90))`` (90th
   percentile of the degree distribution, matching the 10%-rich-club
   convention in the connectome literature).
4. Return 0.0 when ``n_gt_k < 2`` (denominator guard) or when the
   post-strip edge list is empty.

The networkx oracle uses ``rich_club_coefficient(G_no_selfloops,
normalized=False)`` at the same default k_threshold. ``rc.get(k_threshold,
0.0)`` matches the prior wrapper's behavior when k exceeds the highest k
networkx materializes (e.g., on regular ring lattices where no node
exceeds the 90th-percentile degree).
"""
from __future__ import annotations

import inspect
import json
import pathlib
from uuid import uuid4

import pytest

# Skip the entire module if networkx/numpy unavailable. MemoryGraph imports
# networkx at module scope, so importorskip MUST run before iai_mcp.graph.
pytest.importorskip("networkx")
pytest.importorskip("numpy")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from iai_mcp.graph import MemoryGraph  # noqa: E402


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)


def _load_fixtures() -> dict:
    """Return the fixtures dict from the locked sigma_baseline.json."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


def _build_memory_graph(n_nodes: int, edges: list[tuple[int, int]]):
    """Build a MemoryGraph with `n_nodes` integer-indexed nodes and `edges`.

    Returns (graph, uuid_list) so callers can address individual nodes by
    integer index. Mirrors the helper idiom in test_rich_club_self_loop_safe.
    """
    uuids = [uuid4() for _ in range(n_nodes)]
    g = MemoryGraph()
    for uid in uuids:
        g.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for u, v in edges:
        g.add_edge(uuids[u], uuids[v])
    return g, uuids


def _networkx_oracle(n_nodes: int, edges: list[tuple[int, int]]) -> float:
    """Oracle: networkx.rich_club_coefficient at the same default 90th-pct k.

    Mirrors the prior implementation's behavior verbatim:
    1. Build a graph on ``range(n_nodes)`` with the given edges.
    2. Strip self-loops on a copy.
    3. If post-strip is edgeless -> 0.0.
    4. Compute k_threshold = 90th percentile of post-strip degrees.
    5. Return ``rc.get(k_threshold, 0.0)`` (networkx truncates the dict at
       the highest k where the rich club is non-trivial).
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    g_strip = g.copy()
    g_strip.remove_edges_from(list(nx.selfloop_edges(g_strip)))
    if g_strip.number_of_edges() == 0:
        return 0.0
    degrees = [d for _, d in g_strip.degree()]
    k_threshold = int(np.percentile(degrees, 90))
    rc = nx.rich_club_coefficient(g_strip, normalized=False)
    return float(rc.get(k_threshold, 0.0))


# ------------------------------------------------------------------- fixtures


# Subset of sigma_baseline.json fixtures that ship with an edge list.
# `live_n2000` is the optional unavailable snapshot. The 8 below all carry
# `edges` arrays and exercise the rich-club code path on a mix of:
# - empirical small graphs (karate, les_miserables),
# - Erdos-Renyi random baselines (er_200, er_500, er_1000),
# - ring lattices where every node has identical degree
# so the 90th-percentile k exceeds any node's degree -> phi == 0.0
# (tiny_10_ws_k4, tiny_20_ws_p010, ws_2500_k4_p0).
RICH_CLUB_FIXTURE_KEYS = [
    "karate",
    "les_miserables",
    "er_200",
    "er_500",
    "er_1000",
    "tiny_10_ws_k4",
    "tiny_20_ws_p010",
    "ws_2500_k4_p0",
]


# --------------------------------------------------------- parity assertions


def test_rich_club_matches_networkx_on_fixtures() -> None:
    """Differential parity vs networkx on every shipped fixture.

    Failure list idiom: collect all drifts, raise at end with the full
    diagnostic. Tolerance |delta| <= 1e-9 — every fixture should match to
    floating-point exactness since both implementations evaluate the same
    formula on the same edge set.
    """
    fixtures = _load_fixtures()
    drifts: list[str] = []
    for key in RICH_CLUB_FIXTURE_KEYS:
        fx = fixtures[key]
        n = int(fx["n"])
        edges = [tuple(e) for e in fx["edges"]]
        oracle = _networkx_oracle(n, edges)
        g, _ = _build_memory_graph(n, edges)
        ours = g.rich_club_coefficient()
        delta = abs(oracle - ours)
        if delta > 1e-9:
            drifts.append(
                f"{key}: oracle={oracle:.12f} ours={ours:.12f} "
                f"|delta|={delta:.2e}"
            )
    assert not drifts, (
        "rich_club parity drifts vs networkx oracle:\n  "
        + "\n  ".join(drifts)
    )


def test_rich_club_self_loop_strip_preserved() -> None:
    """Self-loop strip behavior matches networkx oracle on a hand-built graph.

    Graph: 3 nodes, edges = [(0,0), (0,1), (1,2)]. After self-loop strip,
    the effective graph has 2 edges: (0,1) and (1,2). Both implementations
    must agree on the resulting phi(k=90th-percentile).
    """
    n, edges = 3, [(0, 0), (0, 1), (1, 2)]
    g, _ = _build_memory_graph(n, edges)
    ours = g.rich_club_coefficient()
    oracle = _networkx_oracle(n, edges)
    delta = abs(oracle - ours)
    assert delta <= 1e-9, (
        f"self-loop strip drift: ours={ours!r} oracle={oracle!r} "
        f"|delta|={delta:.2e}"
    )


def test_rich_club_isolated_nodes_included_in_degree_distribution() -> None:
    """Isolates contribute degree 0 to the distribution — must NOT be dropped.

    Construct a graph from the karate fixture and add 10 deliberately
    isolated nodes (no edges to the karate sub-graph). The networkx
    reference picks ``k_threshold`` from the FULL degree distribution
    (including the 10 zeros). If the implementation skips the
    ``iter_nodes()`` seed and accumulates degrees only from edges, the 10
    isolates are silently dropped -> the 90th percentile shifts up from
    8 to 9, parity FAILS.

    Empirically verified at n_iso=3 the bug does NOT surface (k90 stays
    at 9 either way); n_iso=10 is the smallest count where the seed
    omission shifts the threshold. Higher counts (20, 30, 50) keep the
    threshold pinned at the smaller value, confirming the test exercises
    the invariant the docstring of rich_club_coefficient claims.
    """
    fixtures = _load_fixtures()
    karate = fixtures["karate"]
    n_karate = int(karate["n"])
    karate_edges = [tuple(e) for e in karate["edges"]]
    n_isolates = 10  # smallest count that surfaces the iter_nodes-seed bug
    n_total = n_karate + n_isolates
    g, _ = _build_memory_graph(n_total, karate_edges)
    ours = g.rich_club_coefficient()
    oracle = _networkx_oracle(n_total, karate_edges)
    delta = abs(oracle - ours)
    assert delta <= 1e-9, (
        f"isolate-seed parity drift: ours={ours!r} oracle={oracle!r} "
        f"|delta|={delta:.2e} (with n_isolates={n_isolates}, the seeded "
        f"and unseeded percentile thresholds differ — bug surfaces here)"
    )


def test_rich_club_empty_after_strip_returns_zero() -> None:
    """All-self-loop graph: post-strip edgeless -> exactly 0.0.

    Equivalent to test_all_self_loop_returns_zero in
    test_rich_club_self_loop_safe.py — duplicated here so the
    mosaicsigma parity gate is self-contained.
    """
    g, _ = _build_memory_graph(3, [(0, 0), (1, 1), (2, 2)])
    assert g.rich_club_coefficient() == 0.0


def test_rich_club_explicit_k_threshold() -> None:
    """Explicit k_threshold matches networkx oracle at the same k.

    Karate fixture at k=3 (well below the default 90th-percentile of 9).
    The oracle is rc[3] directly; ours must match.
    """
    fixtures = _load_fixtures()
    karate = fixtures["karate"]
    n = int(karate["n"])
    edges = [tuple(e) for e in karate["edges"]]
    g, _ = _build_memory_graph(n, edges)
    ours = g.rich_club_coefficient(k_threshold=3)
    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(n))
    g_nx.add_edges_from(edges)
    g_strip = g_nx.copy()
    g_strip.remove_edges_from(list(nx.selfloop_edges(g_strip)))
    oracle = float(nx.rich_club_coefficient(g_strip, normalized=False).get(3, 0.0))
    delta = abs(oracle - ours)
    assert delta <= 1e-9, (
        f"explicit-k drift: ours={ours!r} oracle={oracle!r} |delta|={delta:.2e}"
    )


def test_rich_club_n_gt_k_under_two_returns_zero() -> None:
    """Denominator guard: only 1 hub above k_threshold -> 0.0.

    Star graph K_{1,4}: hub has degree 4, every leaf degree 1. The 90th
    percentile of [4, 1, 1, 1, 1] is 3.4 -> int(3) -> exactly 1 node
    exceeds k=3 (the hub). With n_gt_k == 1, the denominator n*(n-1) is
    zero — implementation returns 0.0 by the guard.
    """
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    g, _ = _build_memory_graph(5, edges)
    assert g.rich_club_coefficient() == 0.0


def test_rich_club_no_networkx_import_in_method() -> None:
    """Zero networkx references in the method body.

    inspect.getsource captures the def-block as the source text. After the
    pure-numpy reimplementation, neither `nx.` nor the word `networkx`
    (case-insensitive) may appear anywhere in the method definition —
    including docstrings, comments, or function calls.
    """
    src = inspect.getsource(MemoryGraph.rich_club_coefficient)
    assert "nx." not in src, (
        f"`nx.` reference leaked into rich_club_coefficient implementation:\n{src}"
    )
    assert "networkx" not in src.lower(), (
        "`networkx` token leaked into rich_club_coefficient implementation "
        f"(docstring/comment/call):\n{src}"
    )
