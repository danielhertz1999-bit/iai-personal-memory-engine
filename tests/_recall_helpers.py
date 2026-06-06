"""Shared seeding + structural-cache helpers for daemon-independent recall tests.

Lifted out of the daemon-down recall gate so BOTH the in-process gate and a
real-subprocess gate can build the SAME on-disk structural cache (the structural
cache is built ON DISK via build_runtime_graph + runtime_graph_cache.save, NOT
via a monkeypatch — that is what makes it reusable from a subprocess that runs
the real recall path).

Layout (hub-sensitive gold, mirrors the bounded-assembler parity fixture):
- UUID(1): direct-ANN gold (collinear with cue).
- UUID(2): hub node with an edge to an in-pool record.
- UUID(3): seed collinear with cue; never_merge=True so pattern separation does
           NOT merge it with UUID(1) — both must exist as separate records.
- UUID(4): intermediate node at cosine ~0.4 to cue (inside ANN top-K).
- UUID(5): two-hop gold at cosine ~0.02 — OUTSIDE ANN top-K; reachable ONLY via
           the 2-hop spread UUID(3)->UUID(4)->UUID(5).
- UUID(100..149): degree-boost helpers for UUID(5) — REAL in-pool records
           (never_merge=True) with vectors orthogonal to the cue so they are
           OUTSIDE ANN top-K but give UUID(5) ~52 hebbian edges (high deg_norm).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import numpy as np

# Imported by the test module so the gold builder IS the same hub-sensitive
# scheme the Layer-1 tests use.
from test_store import _make  # noqa: E402

EMBED_DIM = 384
RNG_SEED = 20260601

# UUID integers for the hub-sensitive gold records.
# UUID(5) is the two-hop gold — reachable only via 2-hop spread.
UUID_HUB = UUID(int=2)
UUID_SEED = UUID(int=3)
UUID_INTER = UUID(int=4)
UUID_TWO_HOP = UUID(int=5)

# The structural-only gold's literal surface (asserted PRESENT in the gate).
UUID_TWO_HOP_SURFACE = "User reference gold doc 5"


def _random_vec(seed: int) -> list[float]:
    """Return a normalized mean-ZERO 384-d vector (standard_normal).

    Using standard_normal (mean-zero) is critical for the structural geometry:
    uniform[0,1] fillers projected onto a mean-zero cue have cosine ~N(0, 0.05),
    which makes the k=200 ANN cutoff ~0, and UUID(5) at cosine=0.02 falls BELOW
    the cutoff at n_filler=700 (ensuring 2-hop spread is load-bearing).
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _deterministic_vec(seed: int = 12345) -> list[float]:
    """Return a normalized mean-ZERO 384-d vector (standard_normal).

    Using standard_normal is critical for geometric correctness — see _random_vec.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_gold_record(i: int, vec: list[float]):
    """Build a gold record with a deterministic stable UUID(int=i)."""
    from iai_mcp.types import MemoryRecord
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=f"User reference gold doc {i}",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _populate_store(store, cue_vec: list[float], n_filler: int = 300) -> None:
    """Populate store with n_filler records + hub-sensitive gold records.

    The cue_vec is the fixed vector the fake warm embedder always returns.
    """
    from iai_mcp.types import MemoryRecord as _MR

    rng = np.random.default_rng(RNG_SEED)
    cue_arr = np.asarray(cue_vec, dtype=np.float32)
    cue_arr /= np.linalg.norm(cue_arr)

    # Filler records: standard_normal so fillers cluster at cosine ~0 to the
    # mean-zero cue. This ensures the k=200 ANN cutoff is ~0.03 (at n_filler=700),
    # placing UUID(5) at cosine=0.02 outside ANN top-200 — proving 2-hop is load-bearing.
    for i in range(n_filler):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User filler record {i}", vec=v.tolist()))

    # UUID(1): direct ANN hit — collinear with cue.
    store.insert(_make_gold_record(1, cue_arr.tolist()))

    # UUID(2): hub node — random vector, high degree via an edge to an in-pool record.
    hub_rng = np.random.default_rng(44444)
    hub_vec = hub_rng.random(EMBED_DIM).astype(np.float32)
    hub_vec /= np.linalg.norm(hub_vec)
    store.insert(_make_gold_record(2, hub_vec.tolist()))
    store.boost_edges([(UUID(int=2), UUID(int=1))], edge_type="hebbian", delta=[3.0])

    # UUID(3): seed collinear with cue, never_merge=True to survive pattern separation.
    # Must be in ANN top-K so 2-hop from UUID(3)->UUID(4) fires.
    seed3_rec = _MR(
        id=UUID(int=3), tier="episodic",
        literal_surface="User reference gold doc 3",
        aaak_index="", embedding=cue_arr.tolist(), community_id=None,
        centrality=0.0, detail_level=2, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
        provenance=[], created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), tags=[], language="en",
    )
    store.insert(seed3_rec)

    # UUID(4): intermediate node at cosine ~0.4 to cue (inside ANN top-K).
    inter_rng = np.random.default_rng(55555)
    inter_noise = inter_rng.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_arr) * cue_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_vec = 0.4 * cue_arr + 0.9165 * inter_noise
    inter_vec /= np.linalg.norm(inter_vec)
    store.insert(_make_gold_record(4, inter_vec.tolist()))

    # UUID(5): two-hop gold — cosine ~0.02, outside ANN top-K.
    two_hop_rng = np.random.default_rng(66666)
    noise = two_hop_rng.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_arr) * cue_arr
    noise /= np.linalg.norm(noise)
    target_cos = 0.02
    orth_mag = float(np.sqrt(max(0.0, 1.0 - target_cos**2)))
    two_hop_vec = target_cos * cue_arr + orth_mag * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))

    # Edges: the 2-hop chain.
    store.boost_edges([(UUID(int=3), UUID(int=4))], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(UUID(int=4), UUID(int=5))], edge_type="hebbian", delta=[5.0])

    # UUID(5) degree-boost helpers: UUID(100..149) — REAL records with vectors
    # ORTHOGONAL to the cue so they are OUTSIDE ANN top-K and do NOT compete for
    # result slots. UUID(5) is boosted to all 50, giving it global degree ~52.
    rng_boost = np.random.default_rng(77001)
    for boost_i in range(50):
        raw_v = rng_boost.random(EMBED_DIM).astype(np.float32)
        raw_v -= np.dot(raw_v, cue_arr) * cue_arr
        raw_v /= np.linalg.norm(raw_v)
        boost_rec = _MR(
            id=UUID(int=100 + boost_i), tier="episodic",
            literal_surface=f"User boost helper {boost_i}",
            aaak_index="", embedding=raw_v.tolist(), community_id=None,
            centrality=0.0, detail_level=2, pinned=False, stability=0.0,
            difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
            provenance=[], created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc), tags=[], language="en",
        )
        store.insert(boost_rec)
        store.boost_edges(
            [(UUID(int=5), UUID(int=100 + boost_i))], edge_type="hebbian", delta=[2.0]
        )


def _prime_structural_cache(store) -> None:
    """OFF-PATH CACHE PRIME: build_runtime_graph + save UNTIMED, ON DISK.

    Builds the on-disk runtime-graph cache (mosaic communities + rich-club) so a
    real subprocess that runs the daemon-independent recall path reuses it. Must
    be called OUTSIDE any timed window (it's the off-path warm step).
    """
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, assignment, rc)
