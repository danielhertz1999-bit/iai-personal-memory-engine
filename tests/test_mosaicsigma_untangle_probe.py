"""Post-untangle invariant pin for the node-payload sidecar.

The untangle wave collapsed the prior dual storage of the per-node embedding
(once mirrored across ``MemoryGraph._attrs[uuid]['embedding']`` and the
graph-backend node-attribute dict) into a single
``MemoryGraph._node_payload`` sidecar keyed by ``str(uuid)``. After the wave:

  * ``_attrs[uuid]`` carries ONLY ``community_id`` (subset of {community_id}).
  * Record payload (embedding / surface / centrality / tier / pinned / tags /
    language) lives ONLY in ``_node_payload[str(uuid)]``.
  * ``get_embedding`` reads from the sidecar, never from ``_attrs``.

This file pins those invariants. The atomic invariant test file
(``tests/test_mosaicsigma_untangle_atomic.py``) carries the full surface;
this probe stays as the cheap pre-flight check that survives across waves.

Cheap: pure stdlib + ``iai_mcp.graph.MemoryGraph``. No Hippo / embedder / Lance
imports — wall-clock budget < 5 s.
"""
from __future__ import annotations

import pathlib
from uuid import UUID, uuid4

from iai_mcp.graph import MemoryGraph


# --------------------------------------------------------------------------- helpers


def _make_payload(seed: float) -> dict:
    """Build a minimal node payload mirroring retrieve.py's update() dict."""
    return {
        "embedding": [seed] + [0.0] * 383,
        "surface": f"probe-surface-{seed}",
        "centrality": 0.0,
        "tier": "episodic",
        "pinned": False,
        "tags": ["probe"],
        "language": "en",
    }


def _drive_cached_path_sidecar_write(
    graph: MemoryGraph, node_id: UUID, payload: dict
) -> None:
    """Replay the cached-path sidecar write pattern (post-untangle).

    Step 1: ``graph.add_node(UUID, None, embedding=...)`` populates the
    sidecar via the canonical write path inside ``MemoryGraph.add_node``.
    Step 2: ``graph.set_node_payload(uuid, {...})`` writes the full 7-field
    payload into ``_node_payload`` — the post-untangle canonical sidecar.
    """
    graph.add_node(
        node_id,
        community_id=None,
        embedding=list(payload.get("embedding") or []),
    )
    graph.set_node_payload(
        node_id,
        {
            "embedding": list(payload.get("embedding") or []),
            "surface": payload.get("surface", ""),
            "centrality": float(payload.get("centrality") or 0.0),
            "tier": payload.get("tier", "episodic"),
            "pinned": bool(payload.get("pinned", False)),
            "tags": list(payload.get("tags") or []),
            "language": str(payload.get("language", "en") or "en"),
        },
    )


def _drive_miss_rebuild_path_sidecar_write(
    graph: MemoryGraph, node_id: UUID, payload: dict
) -> None:
    """Replay the MISS-rebuild path sidecar write pattern (post-untangle).

    Same shape as the cached path but called from the records-table iteration
    branch (when the cached payload is absent or stale).
    """
    embedding = list(payload.get("embedding") or [])
    graph.add_node(
        node_id,
        community_id=None,
        embedding=embedding,
    )
    graph.set_node_payload(
        node_id,
        {
            "embedding": list(embedding),
            "surface": str(payload.get("surface", "")),
            "centrality": float(payload.get("centrality") or 0.0),
            "tier": str(payload.get("tier", "episodic")),
            "pinned": bool(payload.get("pinned", False)),
            "tags": list(payload.get("tags") or []),
            "language": str(payload.get("language", "en") or "en"),
        },
    )


def _assert_embedding_in_sidecar(
    graph: MemoryGraph, node_id: UUID, source_lines: str
) -> None:
    """Assert the sidecar carries the embedding and ``get_embedding`` returns it.

    On mismatch, emit a diagnostic naming the source line range, the UUID,
    and both values side-by-side via ``repr()`` for byte-level comparison.
    """
    sidecar_emb = graph._node_payload[str(node_id)]["embedding"]
    api_emb = graph.get_embedding(node_id)
    assert api_emb == sidecar_emb, (
        f"sidecar-vs-API drift at retrieve.py {source_lines}; "
        f"uuid={node_id}; "
        f"_node_payload[str(uuid)]['embedding']={sidecar_emb!r}; "
        f"get_embedding(uuid)={api_emb!r}"
    )


# ---------------------------------------------- test 1: cached-path sidecar


def test_cached_path_embedding_field_in_sidecar() -> None:
    """Cached path — sidecar carries embedding; get_embedding round-trips.

    Builds a MemoryGraph, replays the cached-path sidecar write, and verifies
    the embedding lands in ``_node_payload`` and is read back via the public
    ``get_embedding`` API.
    """
    graph = MemoryGraph()
    node_id = uuid4()
    payload = _make_payload(seed=1.0)

    _drive_cached_path_sidecar_write(graph, node_id, payload)

    _assert_embedding_in_sidecar(graph, node_id, source_lines="lines 770-785")

    # Sanity: the graph carries the node (UUID accepted), _attrs is keyed by
    # raw UUID, and the sidecar is keyed by the str(UUID) label form.
    assert graph.has_node(node_id)
    assert node_id in graph._attrs
    assert str(node_id) in graph._node_payload


# ---------------------------------------- test 2: MISS-rebuild-path sidecar


def test_miss_rebuild_path_embedding_field_in_sidecar() -> None:
    """MISS-rebuild path — sidecar carries embedding; get_embedding round-trips."""
    graph = MemoryGraph()
    node_id = uuid4()
    payload = _make_payload(seed=2.0)

    _drive_miss_rebuild_path_sidecar_write(graph, node_id, payload)

    _assert_embedding_in_sidecar(graph, node_id, source_lines="lines 865-879")

    assert graph.has_node(node_id)
    assert node_id in graph._attrs
    assert str(node_id) in graph._node_payload


# ------------------------------------ test 3: post-untangle _attrs invariant


def test_attrs_post_untangle_carries_only_community_id() -> None:
    """Post-untangle invariant: ``_attrs[uuid]`` keys ⊆ {community_id}.

    After the wave, ``graph.add_node`` writes the embedding into the
    ``_node_payload`` sidecar (graph.py — sidecar is keyed by ``str(uuid)``).
    ``_attrs[uuid]`` carries ONLY ``community_id``. If this test fails, the
    untangle has regressed and either ``add_node`` or downstream consumers
    have started writing the old payload shape back into ``_attrs``.
    """
    graph = MemoryGraph()
    node_id = uuid4()
    embedding = [0.5] + [0.0] * 383

    graph.add_node(node_id, community_id=None, embedding=embedding)

    assert node_id in graph._attrs, (
        f"expected uuid={node_id} present in _attrs after add_node; "
        f"keys={list(graph._attrs.keys())}"
    )
    attrs_keys = set(graph._attrs[node_id].keys())
    assert "embedding" not in attrs_keys, (
        f"_attrs[{node_id}] must NOT carry 'embedding' post-untangle; "
        f"actual keys={attrs_keys}"
    )
    assert attrs_keys <= {"community_id"}, (
        f"_attrs[{node_id}] keys must ⊆ {{community_id}} post-untangle; "
        f"actual={attrs_keys}"
    )
    # Sidecar carries the embedding instead.
    assert str(node_id) in graph._node_payload
    assert graph._node_payload[str(node_id)].get("embedding") == embedding


# ------------------------------------ test 4: hardest-downstream files exist


def test_downstream_hardest_test_files_present() -> None:
    """Manifest check: the two hardest downstream tests must exist as files.

    The atomic untangle work will refactor these tests; a missing file
    means a rename / accidental deletion has happened between waves and
    must be resolved before the untangle proceeds.
    """
    sync_path = pathlib.Path("tests/test_graph_node_payload_sync.py")
    native_recall_path = pathlib.Path("tests/test_graph_native_recall.py")

    assert sync_path.exists(), (
        f"expected hardest downstream test file present: {sync_path} "
        "(refactored by the upcoming atomic untangle wave)"
    )
    assert native_recall_path.exists(), (
        f"expected hardest downstream test file present: {native_recall_path} "
        "(refactored by the upcoming atomic untangle wave)"
    )
