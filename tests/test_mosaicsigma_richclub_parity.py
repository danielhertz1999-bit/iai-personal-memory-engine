from __future__ import annotations

import inspect
import json
import pathlib
from uuid import uuid4

import pytest

pytest.importorskip("networkx")
pytest.importorskip("numpy")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from iai_mcp.graph import MemoryGraph  # noqa: E402


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


def _build_memory_graph(n_nodes: int, edges: list[tuple[int, int]]):
    uuids = [uuid4() for _ in range(n_nodes)]
    g = MemoryGraph()
    for uid in uuids:
        g.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for u, v in edges:
        g.add_edge(uuids[u], uuids[v])
    return g, uuids


def _networkx_oracle(n_nodes: int, edges: list[tuple[int, int]]) -> float:
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


def test_rich_club_matches_networkx_on_fixtures() -> None:
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
    fixtures = _load_fixtures()
    karate = fixtures["karate"]
    n_karate = int(karate["n"])
    karate_edges = [tuple(e) for e in karate["edges"]]
    n_isolates = 10
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
    g, _ = _build_memory_graph(3, [(0, 0), (1, 1), (2, 2)])
    assert g.rich_club_coefficient() == 0.0


def test_rich_club_explicit_k_threshold() -> None:
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
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    g, _ = _build_memory_graph(5, edges)
    assert g.rich_club_coefficient() == 0.0


def test_rich_club_no_networkx_import_in_method() -> None:
    src = inspect.getsource(MemoryGraph.rich_club_coefficient)
    assert "nx." not in src, (
        f"`nx.` reference leaked into rich_club_coefficient implementation:\n{src}"
    )
    assert "networkx" not in src.lower(), (
        "`networkx` token leaked into rich_club_coefficient implementation "
        f"(docstring/comment/call):\n{src}"
    )
