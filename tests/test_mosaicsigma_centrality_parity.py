from __future__ import annotations

import inspect
import json
import os
import pathlib
from uuid import UUID, uuid4

import pytest

pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("iai_mcp_native")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from iai_mcp import graph as graph_module  # noqa: E402
from iai_mcp.graph import MemoryGraph  # noqa: E402


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)
PERF_GATE_JSON_PATH = pathlib.Path(
    os.environ.get(
        "IAI_PERF_GATE_JSON",
        str(pathlib.Path(__file__).parent / "fixtures" / "perf_gate.json"),
    )
)


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


CENTRALITY_FIXTURE_KEYS = [
    "karate",
    "er_200",
    "er_500",
    "er_1000",
    "tiny_10_ws_k4",
    "tiny_20_ws_p010",
    "ws_2500_k4_p0",
]


def _build_memory_graph_from_fixture(
    n_nodes: int, edges: list[tuple[int, int]]
) -> tuple[MemoryGraph, list[UUID]]:
    node_ids = [UUID(int=i) for i in range(n_nodes)]
    mg = MemoryGraph()
    for nid in node_ids:
        mg.add_node(nid, community_id=None, embedding=[0.0] * 384)
    for u, v in edges:
        if u == v:
            continue
        mg.add_edge(node_ids[u], node_ids[v], weight=1.0)
    return mg, node_ids


def _networkx_oracle(
    n_nodes: int, edges: list[tuple[int, int]]
) -> dict[int, float]:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((u, v) for u, v in edges if u != v)
    return nx.betweenness_centrality(g, normalized=True, weight=None)


def test_betweenness_matches_networkx_on_fixtures() -> None:
    fixtures = _load_fixtures()
    drifts: list[str] = []
    for key in CENTRALITY_FIXTURE_KEYS:
        fx = fixtures[key]
        n = int(fx["n"])
        edges = [tuple(e) for e in fx["edges"]]
        oracle = _networkx_oracle(n, edges)
        mg, node_ids = _build_memory_graph_from_fixture(n, edges)
        ours = mg.centrality()
        oracle_vec = np.array(
            [oracle[i] for i in range(n)], dtype=np.float64
        )
        ours_vec = np.array(
            [ours[node_ids[i]] for i in range(n)], dtype=np.float64
        )
        try:
            np.testing.assert_allclose(
                ours_vec, oracle_vec, rtol=1e-7, atol=1e-12
            )
        except AssertionError as exc:
            max_abs = float(np.max(np.abs(ours_vec - oracle_vec)))
            drifts.append(
                f"{key} n={n}: max|delta|={max_abs:.3e} -- {exc.args[0][:200]}"
            )
    assert not drifts, (
        "Brandes parity drifts vs networkx unweighted oracle:\n  "
        + "\n  ".join(drifts)
    )


def test_hub_beats_leaves_on_star_graph() -> None:
    mg = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(4)]
    mg.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        mg.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        mg.add_edge(hub, leaf)
    c = mg.centrality()
    for leaf in leaves:
        assert c[hub] > c[leaf], (
            f"hub={c[hub]} should beat leaf={c[leaf]}"
        )


def test_cache_off_recomputes_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "off")
    mg = MemoryGraph()
    hub = uuid4()
    leaf = uuid4()
    mg.add_node(hub, community_id=None, embedding=[0.0] * 384)
    mg.add_node(leaf, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(hub, leaf)
    c1 = mg.centrality()
    c2 = mg.centrality()
    assert c1 == c2
    assert c1 is not c2, "cache=off should hand back a fresh dict"


def test_cache_on_returns_same_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "on")
    mg = MemoryGraph()
    a = uuid4()
    b = uuid4()
    mg.add_node(a, community_id=None, embedding=[0.0] * 384)
    mg.add_node(b, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)
    c1 = mg.centrality()
    c2 = mg.centrality()
    assert c1 is c2, "cache=on should return the same dict reference"


def test_cache_auto_invalidates_on_add_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "auto")
    monkeypatch.setattr(graph_module, "AUTO_CACHE_DEFAULT", "on")
    mg = MemoryGraph()
    a, b, c_node = uuid4(), uuid4(), uuid4()
    for nid in (a, b):
        mg.add_node(nid, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)
    first = mg.centrality()
    mg.add_node(c_node, community_id=None, embedding=[0.0] * 384)
    second = mg.centrality()
    assert first is not second, (
        "auto mode should recompute after add_node mutation; "
        f"first id={id(first)} second id={id(second)}"
    )


def test_empty_graph_centrality_returns_empty_dict() -> None:
    mg = MemoryGraph()
    c = mg.centrality()
    assert c == {}


def test_no_networkx_in_centrality_method() -> None:
    src = inspect.getsource(MemoryGraph.centrality)
    assert "nx." not in src, "centrality() must not call into networkx"
    assert "networkx" not in src.lower(), (
        "centrality() must not reference networkx"
    )


def test_centrality_dict_uses_node_arr_not_iter_nodes_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "off")

    mg = MemoryGraph()
    a, b = UUID(int=1), UUID(int=2)
    mg.add_node(a, community_id=None, embedding=[0.0] * 384)
    mg.add_node(b, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)

    fake_centrality = np.array([10.0, 20.0], dtype=np.float64)
    fake_node_arr = np.array([1, 0], dtype=np.int64)

    def fake_native(
        indptr: np.ndarray,
        indices: np.ndarray,
        n_nodes: int,
        normalized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        return fake_centrality, fake_node_arr

    from iai_mcp_native import graph as native_graph_module
    monkeypatch.setattr(
        native_graph_module, "betweenness_centrality", fake_native
    )

    result = mg.centrality()
    assert result[b] == 10.0, (
        f"node_arr[0]=1 maps to CSR row 1 (B); should yield 10.0; got {result[b]}"
    )
    assert result[a] == 20.0, (
        f"node_arr[1]=0 maps to CSR row 0 (A); should yield 20.0; got {result[a]}"
    )


def test_cache_auto_resolves_to_AUTO_CACHE_DEFAULT(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mg = MemoryGraph()
    a, b = uuid4(), uuid4()
    mg.add_node(a, community_id=None, embedding=[0.0] * 384)
    mg.add_node(b, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)

    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "auto")
    monkeypatch.setattr(graph_module, "AUTO_CACHE_DEFAULT", "off")
    c1 = mg.centrality()
    c2 = mg.centrality()
    assert c1 is not c2, "AUTO=off + env=auto should recompute"

    mg._centrality_cache = None
    mg._dirty_since_centrality = True

    monkeypatch.setattr(graph_module, "AUTO_CACHE_DEFAULT", "on")
    c3 = mg.centrality()
    c4 = mg.centrality()
    assert c3 is c4, "AUTO=on + env=auto should cache"

    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "off")
    c5 = mg.centrality()
    c6 = mg.centrality()
    assert c5 is not c6, "env=off must override AUTO=on"


def test_memory_graph_init_centrality_cache_attrs_present() -> None:
    mg = MemoryGraph()
    assert hasattr(mg, "_centrality_cache")
    assert hasattr(mg, "_dirty_since_centrality")
    assert mg._centrality_cache is None
    assert mg._dirty_since_centrality is True
    assert mg.centrality() == {}


def test_auto_cache_default_matches_perf_gate_decision() -> None:
    if not PERF_GATE_JSON_PATH.exists():
        pytest.skip(
            f"perf gate JSON not present at {PERF_GATE_JSON_PATH}; "
            "bench has not been run in this environment"
        )
    decision = json.loads(PERF_GATE_JSON_PATH.read_text(encoding="utf-8"))
    action = decision.get("action")
    assert action in ("drop", "keep"), (
        f"perf gate JSON action must be 'drop' or 'keep'; got {action!r}"
    )
    expected = "off" if action == "drop" else "on"
    actual = graph_module.AUTO_CACHE_DEFAULT
    assert actual == expected, (
        f"AUTO_CACHE_DEFAULT={actual!r} does not match perf-gate "
        f"action={action!r} (expected {expected!r}). Reason from JSON: "
        f"{decision.get('reason', '<missing>')}"
    )
