"""Spawn-context worker that owns the heavy graph rebuild allocations.

Receives compact tuples (records + edges) from the parent over a Pipe, builds
a MemoryGraph, runs community detection + rich-club ranking, and streams a
compact result back. Exits when the protocol completes — the OS reclaims the
worker's address space so the fat per-cycle allocations leave zero residue in
the parent daemon.

Constitutional invariant — AES-key fence:
The worker NEVER opens the storage backend and NEVER sees the encryption key.
The module's top-level import surface is empty by design (only
`from __future__ import annotations`). All compute-side imports are performed
unconditionally at the top of `_worker_entry`, BEFORE the recv loop and BEFORE
any branch on message kind, so the import-surface guard test can drive the
abort path and still observe the full late-import closure.

Adding any top-level or late import of `iai_mcp.hippo`, `iai_mcp.store`,
`iai_mcp.daemon`, or `iai_mcp.crypto` here breaks the fence and must be caught
by `tests/test_runtime_graph_cache_worker.py::test_worker_module_does_not_import_aes_surface`.

TEST-ONLY env-var affordance:
    IAI_MCP_RGC_WORKER_CRASH_AFTER_N_NODES — if set to a non-negative integer
    N, the worker exits with code 1 after receiving N node chunks (or
    immediately if N == 0). Used exclusively by the worker-crash chaos test
    to verify fail-fast disposition and last-good snapshot preservation; the
    daemon never sets this in production.
"""
from __future__ import annotations


_ASSIGN_CHUNK = 2000


def _worker_entry(conn) -> None:
    """Recv-then-compute-then-stream over `conn`.

    Hard contract: every late import below MUST execute unconditionally on
    function entry, before the recv loop. The first message handled is the
    abort sentinel `("abort", None)` which returns immediately AFTER the late
    imports have fired — this is both a real cancellation path and the
    mechanism that makes the import-surface guard test exercise the full
    module closure.

    Recognized message kinds:
      - `("abort", None)` — clean early exit (no compute, no result stream).
      - `("nodes", [(id_str, embedding_blob), ...])` — accumulate node tuples.
      - `("nodes_end", None)` — sentinel; no action.
      - `("edges", [(src_str, dst_str, weight), ...])` — accumulate edges.
      - `("edges_end", None)` — break the recv loop and proceed to compute.

    Result stream (parent reassembles into a CommunityAssignment):
      - `("community_table", [(comm_uuid_bytes, centroid_bytes_or_None), ...])`
      - repeated `("assign", [(node_uuid_bytes, comm_idx_int), ...])` of size
        `<= _ASSIGN_CHUNK`, then one `("assign_end", None)`
      - `("backend", str)` — streamed verbatim from `assignment.backend`
      - `("top_communities", [uuid_bytes, ...])`
      - `("mid_regions", [(comm_uuid_bytes, [node_uuid_bytes, ...]), ...])`
      - `("rich_club", [uuid_bytes, ...])`
      - `("max_degree", int)`
      - `("done", None)`
    """
    # Late imports — execute FIRST, unconditionally. These are the only
    # transitive surface the worker is allowed to load. Anything that pulls
    # in storage, encryption, or the daemon must be rejected here.
    import os
    import sys
    import numpy as np
    from uuid import UUID
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.community import detect_communities
    from iai_mcp.richclub import rich_club_nodes

    # TEST-ONLY: opt-in crash-after-N-node-chunks hook for the chaos test.
    crash_after_env = os.environ.get("IAI_MCP_RGC_WORKER_CRASH_AFTER_N_NODES")
    try:
        crash_after = int(crash_after_env) if crash_after_env is not None else None
    except ValueError:
        crash_after = None
    node_chunks_received = 0

    nodes_buf: list = []
    edges_buf: list = []

    try:
        # Recv loop.
        while True:
            envelope = conn.recv()
            if not (isinstance(envelope, tuple) and len(envelope) == 2):
                raise RuntimeError(f"malformed envelope: {envelope!r}")
            kind, payload = envelope
            if kind == "abort":
                return
            elif kind == "nodes":
                for id_str, emb_blob in payload:
                    emb = np.frombuffer(emb_blob, dtype=np.float32).tolist()
                    nodes_buf.append((UUID(id_str), None, emb, {}))
                node_chunks_received += 1
                if crash_after is not None and node_chunks_received >= crash_after:
                    # TEST-ONLY: simulate a mid-stream crash. Skip the
                    # ack/result loop and exit non-zero so the parent
                    # observes broken-pipe / pipe-EOF + nonzero-exit.
                    sys.stderr.write(
                        f"rgc_worker test-hook crash after "
                        f"{node_chunks_received} node chunks\n"
                    )
                    sys.exit(1)
            elif kind == "nodes_end":
                continue
            elif kind == "edges":
                for src_str, dst_str, weight in payload:
                    edges_buf.append((
                        UUID(src_str),
                        UUID(dst_str),
                        float(weight),
                        "hebbian",
                    ))
            elif kind == "edges_end":
                break
            else:
                raise RuntimeError(f"unknown chunk kind: {kind!r}")

        # Compute.
        graph = MemoryGraph()
        graph.clear_and_rebuild(nodes_buf, edges_buf)
        assignment = detect_communities(graph, prior=None, prior_mode="cold")
        rc = rich_club_nodes(graph)
        max_degree = max((d for _, d in graph.degrees()), default=0)

        # Result stream — community table first so the parent can index assigns.
        # Deterministic order: sort communities by UUID bytes.
        unique_comms = sorted(set(assignment.node_to_community.values()), key=lambda u: u.bytes)
        comm_index = {cu: i for i, cu in enumerate(unique_comms)}
        community_table_payload = []
        for cu in unique_comms:
            cent = assignment.community_centroids.get(cu)
            if cent is None or len(cent) == 0:
                cent_bytes = None
            else:
                cent_bytes = np.asarray(cent, dtype=np.float32).tobytes()
            community_table_payload.append((cu.bytes, cent_bytes))
        conn.send(("community_table", community_table_payload))

        # Assign stream — sorted by node UUID bytes for determinism.
        sorted_nodes = sorted(
            assignment.node_to_community.items(), key=lambda kv: kv[0].bytes
        )
        chunk: list = []
        for node_uuid, comm_uuid in sorted_nodes:
            chunk.append((node_uuid.bytes, comm_index[comm_uuid]))
            if len(chunk) >= _ASSIGN_CHUNK:
                conn.send(("assign", chunk))
                chunk = []
        if chunk:
            conn.send(("assign", chunk))
        conn.send(("assign_end", None))

        # Backend, top, mid, rich-club, max_degree, done.
        conn.send(("backend", str(assignment.backend)))
        conn.send((
            "top_communities",
            [u.bytes for u in assignment.top_communities],
        ))
        mid_payload = [
            (comm.bytes, [m.bytes for m in members])
            for comm, members in assignment.mid_regions.items()
        ]
        conn.send(("mid_regions", mid_payload))
        conn.send(("rich_club", [u.bytes for u in rc]))
        conn.send(("max_degree", int(max_degree)))
        conn.send(("done", None))
    except RuntimeError:
        # Protocol violation — re-raise so the in-process driver tests see it.
        # The subprocess wrapper below converts this into ("error", ...) + exit(1).
        try:
            import traceback
            err = traceback.format_exc(limit=2)[:500]
            conn.send(("error", err))
        except Exception:  # noqa: BLE001
            pass
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", repr(exc)[:500]))
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("not directly invocable")
