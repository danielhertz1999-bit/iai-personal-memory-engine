
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _purge_networkx_from_sys_modules():
    for mod in [
        m for m in list(sys.modules) if m == "networkx" or m.startswith("networkx.")
    ]:
        del sys.modules[mod]
    yield


def test_memorygraph_constructs_without_loading_networkx():
    from iai_mcp.graph import MemoryGraph

    g = MemoryGraph()
    assert "networkx" not in sys.modules
    assert hasattr(g, "_adj")
    assert not hasattr(g, "_nx")


def test_memorygraph_full_api_does_not_load_networkx():
    from iai_mcp.graph import MemoryGraph

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for n in (a, b, c):
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    assert "networkx" not in sys.modules

    g.add_edge(a, b, weight=0.5)
    g.add_edge(b, c, weight=0.7)
    assert "networkx" not in sys.modules

    assert g.has_node(a)
    assert g.node_count() == 3
    assert "networkx" not in sys.modules

    nodes = list(g.iter_nodes())
    assert len(nodes) == 3
    assert "networkx" not in sys.modules

    edges = list(g.iter_edges_with_weight())
    assert len(edges) == 2
    assert "networkx" not in sys.modules

    degs = list(g.degrees())
    assert len(degs) == 3
    assert "networkx" not in sys.modules

    indptr, indices, data = g.to_csr_arrays()
    assert len(indptr) == 4
    assert "networkx" not in sys.modules

    cen = g.centrality()
    assert b in cen
    assert "networkx" not in sys.modules

    g.rich_club_coefficient()
    assert "networkx" not in sys.modules

    g.two_hop_neighborhood([a], top_k=5)
    assert "networkx" not in sys.modules

    g.remove_node(a)
    assert g.node_count() == 2
    assert "networkx" not in sys.modules


def test_no_networkx_lazy_import_in_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "grep",
            "-E",
            "^(import|from) networkx",
            "src/iai_mcp/graph.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"Found networkx imports in src/iai_mcp/graph.py:\n{result.stdout}"
    )
