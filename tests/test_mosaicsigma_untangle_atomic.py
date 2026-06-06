"""Atomic untangle invariant — node-payload sidecar replaces _attrs/_nx double-store.

After the mosaicsigma untangle wave, MemoryGraph stores topology in self._nx and
ONLY the community-assignment scalar in self._attrs. All record-payload fields
(embedding, surface, centrality, tier, pinned, tags, language) live in the
self._node_payload sidecar keyed by str(uuid). get_embedding reads from the
sidecar — _attrs no longer carries an "embedding" key.

These tests pin the constitutional invariants:

  T1 set(graph._attrs[uuid].keys()) <= {"community_id"} for every node.
  T2 graph._node_payload[str(uuid)] carries "embedding" and the other payload
      fields written via set_node_payload.
  T3 get_embedding reads from the sidecar, not from _attrs.
  T4 Public API surface: iter_nodes / iter_edges_with_weight / to_csr_arrays /
      degrees / set_node_payload all callable, return the documented shapes.
  T5 set_node_payload is idempotent — calling twice with the same payload
      leaves a single sidecar entry without error.
  T6 get_centrality / get_payload read from the sidecar; absent → defaults.

The probe (test_mosaicsigma_untangle_probe.py) detected divergence before this
wave. The invariant here REPLACES the probe's pre-untangle expectation.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


# --------------------------------------------------------------------------- helpers


def _build_payload(seed: float) -> dict:
    """Synthetic 7-field record payload, matching retrieve.py's write shape."""
    return {
        "embedding": [seed] + [0.0] * 383,
        "surface": f"untangle-surface-{seed}",
        "centrality": 0.125 * seed,
        "tier": "episodic",
        "pinned": False,
        "tags": ["untangle"],
        "language": "en",
    }


@pytest.fixture
def graph_with_records() -> tuple[MemoryGraph, list[UUID]]:
    """5-node MemoryGraph with full sidecar payload per node."""
    graph = MemoryGraph()
    ids: list[UUID] = []
    for i in range(5):
        nid = uuid4()
        ids.append(nid)
        payload = _build_payload(seed=float(i + 1))
        graph.add_node(
            nid,
            community_id=None,
            embedding=list(payload["embedding"]),
        )
        graph.set_node_payload(nid, payload)
    # One edge so to_csr_arrays / degrees exercise non-trivial state.
    graph.add_edge(ids[0], ids[1], weight=0.75)
    graph.add_edge(ids[1], ids[2], weight=0.50)
    return graph, ids


# --------------------------------------------------------------------------- T1


def test_attrs_no_longer_carries_record_payload(graph_with_records) -> None:
    """T1: post-untangle _attrs[uuid] keys are a subset of {community_id}."""
    graph, ids = graph_with_records
    for nid in ids:
        attrs_keys = set(graph._attrs[nid].keys())
        assert "embedding" not in attrs_keys, (
            f"_attrs[{nid}] must NOT carry 'embedding' (sidecar lives in "
            f"_node_payload); got keys={attrs_keys}"
        )
        assert attrs_keys <= {"community_id"}, (
            f"_attrs[{nid}] keys must be a subset of {{community_id}}, "
            f"got {attrs_keys}"
        )


# --------------------------------------------------------------------------- T2


def test_sidecar_carries_embedding(graph_with_records) -> None:
    """T2: every node's embedding is recoverable via the sidecar."""
    graph, ids = graph_with_records
    for i, nid in enumerate(ids):
        sidecar = graph._node_payload[str(nid)]
        assert "embedding" in sidecar
        expected_first = float(i + 1)
        assert sidecar["embedding"][0] == pytest.approx(expected_first)
        # Other payload fields round-trip.
        assert sidecar["surface"] == f"untangle-surface-{expected_first}"
        assert sidecar["tier"] == "episodic"
        assert sidecar["language"] == "en"
        assert sidecar["pinned"] is False
        assert sidecar["tags"] == ["untangle"]


# --------------------------------------------------------------------------- T3


def test_get_embedding_reads_from_sidecar() -> None:
    """T3: get_embedding routes through _node_payload, not _attrs."""
    graph = MemoryGraph()
    nid = uuid4()
    real_emb = [0.7] + [0.0] * 383
    graph.add_node(nid, community_id=None, embedding=real_emb)
    # add_node already populates the sidecar; verify the read.
    assert graph.get_embedding(nid) == real_emb

    # Mutate _attrs to a stale value — should NOT affect get_embedding.
    graph._attrs[nid]["embedding"] = [0.0] * 384  # would-be-stale shadow
    assert graph.get_embedding(nid) == real_emb, (
        "get_embedding must read sidecar; _attrs writes have no effect"
    )

    # Update via set_node_payload — read must reflect the new value.
    new_emb = [0.1] * 384
    graph.set_node_payload(nid, {"embedding": new_emb})
    assert graph.get_embedding(nid) == new_emb

    # Unknown UUID returns None.
    assert graph.get_embedding(uuid4()) is None


# --------------------------------------------------------------------------- T4


def test_new_public_api_signatures(graph_with_records) -> None:
    """T4: iter_nodes / iter_edges_with_weight / to_csr_arrays / degrees exist."""
    graph, ids = graph_with_records

    assert callable(graph.iter_nodes)
    assert callable(graph.iter_edges_with_weight)
    assert callable(graph.to_csr_arrays)
    assert callable(graph.degrees)
    assert callable(graph.set_node_payload)
    assert callable(graph.get_centrality)
    assert callable(graph.get_payload)

    # iter_nodes yields UUIDs and lists out to len 5.
    listed_nodes = list(graph.iter_nodes())
    assert len(listed_nodes) == 5
    assert all(isinstance(u, UUID) for u in listed_nodes)
    assert set(listed_nodes) == set(ids)

    # iter_edges_with_weight yields (UUID, UUID, float) tuples.
    edges = list(graph.iter_edges_with_weight())
    assert len(edges) == 2
    for u, v, w in edges:
        assert isinstance(u, UUID)
        assert isinstance(v, UUID)
        assert isinstance(w, float)

    # to_csr_arrays returns three numpy arrays of the documented dtypes.
    indptr, indices, data = graph.to_csr_arrays()
    assert isinstance(indptr, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert isinstance(data, np.ndarray)
    assert indptr.dtype == np.int64
    assert indices.dtype == np.int64
    assert data.dtype == np.float64
    # indptr length is n_nodes + 1 for CSR encoding.
    assert len(indptr) == 6  # 5 nodes + 1

    # degrees yields (UUID, int) pairs, count matches node count.
    deg_pairs = list(graph.degrees())
    assert len(deg_pairs) == 5
    for nid, deg in deg_pairs:
        assert isinstance(nid, UUID)
        assert isinstance(deg, int)


# --------------------------------------------------------------------------- T5


def test_set_node_payload_idempotent() -> None:
    """T5: repeated set_node_payload with the same payload is a no-op merge."""
    graph = MemoryGraph()
    nid = uuid4()
    graph.add_node(nid, community_id=None, embedding=[0.0] * 384)

    payload = {"embedding": [1.0] * 384, "surface": "idempotent"}
    graph.set_node_payload(nid, payload)
    graph.set_node_payload(nid, payload)

    assert str(nid) in graph._node_payload
    assert graph._node_payload[str(nid)]["surface"] == "idempotent"
    # Sidecar entry remains a single dict — not a list / multi-value.
    assert isinstance(graph._node_payload[str(nid)], dict)


# --------------------------------------------------------------------------- T6


def test_get_centrality_reads_from_sidecar() -> None:
    """T6a: get_centrality returns the sidecar centrality, or 0.0 if absent."""
    graph = MemoryGraph()
    nid = uuid4()
    graph.add_node(nid, community_id=None, embedding=[0.0] * 384)
    # No centrality written yet — default is 0.0.
    assert graph.get_centrality(nid) == 0.0
    # Write via set_node_payload.
    graph.set_node_payload(nid, {"centrality": 0.875})
    assert graph.get_centrality(nid) == pytest.approx(0.875)
    # Accept stringified UUID as well.
    assert graph.get_centrality(str(nid)) == pytest.approx(0.875)
    # Unknown id → 0.0 (default).
    assert graph.get_centrality(uuid4()) == 0.0


def test_get_payload_reads_from_sidecar(graph_with_records) -> None:
    """T6b: get_payload returns the full sidecar dict, or an empty dict."""
    graph, ids = graph_with_records
    payload = graph.get_payload(ids[0])
    assert isinstance(payload, dict)
    assert payload["surface"].startswith("untangle-surface-")
    assert "embedding" in payload
    # Unknown id → empty dict.
    assert graph.get_payload(uuid4()) == {}
