from __future__ import annotations

from uuid import uuid4

from iai_mcp.graph import MemoryGraph
from iai_mcp.richclub import rich_club_nodes

def test_rich_club_selects_top_10_percent() -> None:
    g = MemoryGraph()
    nodes = [uuid4() for _ in range(20)]
    for n in nodes:
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    for i in range(19):
        g.add_edge(nodes[i], nodes[i + 1])
    rc = rich_club_nodes(g, percent=0.10)
    assert len(rc) == 2

def test_rich_club_never_empty_on_nonempty_graph() -> None:
    g = MemoryGraph()
    nodes = [uuid4() for _ in range(5)]
    for n in nodes:
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    g.add_edge(nodes[0], nodes[1])
    rc = rich_club_nodes(g, percent=0.10)
    assert len(rc) >= 1

def test_rich_club_empty_graph_returns_empty() -> None:
    g = MemoryGraph()
    assert rich_club_nodes(g) == []

def test_rich_club_picks_highest_centrality_first() -> None:
    g = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(9)]
    g.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        g.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        g.add_edge(hub, leaf)
    rc = rich_club_nodes(g, percent=0.10)
    assert len(rc) == 1
    assert rc[0] == hub

def test_rich_club_custom_percent() -> None:
    g = MemoryGraph()
    nodes = [uuid4() for _ in range(10)]
    for n in nodes:
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    for i in range(9):
        g.add_edge(nodes[i], nodes[i + 1])
    rc = rich_club_nodes(g, percent=0.5)
    assert len(rc) == 5
