"""Plan 03-01 CONN-05 D-TEM-04: structure-edge Hebbian LTP.

Mirrors content-edge Hebbian (retrieve.reinforce_edges -> store.boost_edges
with edge_type="hebbian"). Co-retrieval of two records whose structure_hv
hypervectors are sufficiently similar (Hamming similarity >= 0.7 by default)
strengthens a "hebbian_structure" edge between them. FSRS decay on the new
edge type is identical to the content-edge formula in sleep._decay_edges.

Constitutional fit:
- D-TEM-04: Hebbian LTP on structure edges. Autopoiesis applied to structure;
  the brain reinforces structural co-occurrence the same way it reinforces
  content co-occurrence in Phase 1.
- Flat layout (PATTERNS.md): no `connectome/` subpackage. Module path is
  src/iai_mcp/hebbian_structure.py.
- Same shape as retrieve.reinforce_edges -- pairwise iterate, compute
  similarity, call store.boost_edges with edge_type="hebbian_structure".

Public API:
- STRUCTURAL_SIMILARITY_THRESHOLD: pairs above this fire LTP (default 0.7).
- structural_similarity(a, b): 1 - hamming_distance(a, b) / D in [0, 1].
- strengthen_structure_edge(store, src_id, dst_id, gain=1.0): boost the
  structure edge between two records.
- co_retrieval_trigger(store, hits): pairwise scan of co-retrieved hits;
  fire strengthen_structure_edge for every pair above the threshold.
"""
from __future__ import annotations

from itertools import combinations
from uuid import UUID

import numpy as np

from iai_mcp.store import MemoryStore
from iai_mcp.types import STRUCTURE_HV_DIM


# D-TEM-04 default trigger (per plan Task 2b behavior contract):
# co-retrieval LTP fires when structural similarity >= 0.7 (Hamming distance
# fraction <= 0.3). Tunable later via the profile registry if a knob is added.
STRUCTURAL_SIMILARITY_THRESHOLD: float = 0.7


def structural_similarity(a: bytes, b: bytes) -> float:
    """Return 1 - hamming_distance(a, b) / STRUCTURE_HV_DIM in [0.0, 1.0].

    Empty / unequal-length / corrupt inputs return 0.0 (graceful degradation).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    # popcount of XOR -> hamming distance in bits.
    xor = np.bitwise_xor(aa, bb)
    # numpy >= 2.x has np.bitwise_count; fall back to unpackbits sum on older.
    try:
        ham_bits = int(np.bitwise_count(xor).sum())
    except AttributeError:
        ham_bits = int(np.unpackbits(xor).sum())
    return 1.0 - (ham_bits / STRUCTURE_HV_DIM)


def strengthen_structure_edge(
    store: MemoryStore,
    src_id: UUID,
    dst_id: UUID,
    gain: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Plan 03-01 D-TEM-04: structure-edge LTP via store.boost_edges.

    Returns the new weights dict (same shape as retrieve.reinforce_edges'
    underlying call). Mirrors content-edge LTP shape so downstream code
    (events, audit, decay sweep) treats structure edges identically.
    """
    return store.boost_edges(
        [(src_id, dst_id)],
        delta=float(gain),
        edge_type="hebbian_structure",
    )


def co_retrieval_trigger(
    store: MemoryStore,
    hits,
    *,
    threshold: float = STRUCTURAL_SIMILARITY_THRESHOLD,
    gain: float = 1.0,
) -> int:
    """Pairwise scan of co-retrieved hits; fire strengthen_structure_edge
    for each pair whose structural_similarity >= threshold.

    `hits` may be a list of MemoryHit (record_id only -- structure_hv is
    fetched lazily from store.get) OR a list of MemoryRecord (faster path,
    structure_hv read directly).

    Returns the number of structure edges strengthened. A structurally-
    isolated co-retrieved set returns 0 -- this is expected (means no two
    hits shared structure to reinforce).
    """
    # Materialise (id, structure_hv) tuples once.
    pairs: list[tuple[UUID, bytes]] = []
    for h in hits:
        rec_id = getattr(h, "record_id", None) or getattr(h, "id", None)
        if rec_id is None:
            continue
        hv = getattr(h, "structure_hv", None)
        if hv is None:
            rec = store.get(rec_id)
            if rec is None:
                continue
            hv = rec.structure_hv
        pairs.append((rec_id, hv or b""))

    fired = 0
    for (a_id, a_hv), (b_id, b_hv) in combinations(pairs, 2):
        if structural_similarity(a_hv, b_hv) >= threshold:
            try:
                strengthen_structure_edge(store, a_id, b_id, gain=gain)
                fired += 1
            except Exception:
                # Diagnostic only -- never block the pipeline on edge failure.
                continue
    return fired
