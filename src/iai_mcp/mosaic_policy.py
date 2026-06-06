"""MOSAIC: Memory-Oriented Sparse Aggregative Identification of Communities.

Policy layer -- isolated from the Numba-jitted core kernels in `mosaic.py`.

Policy layer -- isolated from the Numba-jitted core kernels in `mosaic.py`
so the Numba-accelerated core stays purely algorithmic and any
post-processing decision can be revised without re-jitting the kernels.

This module owns three policy questions that are NOT part of the
algorithm proper:

  - `compute_singleton_ratio`: fraction of size-1 communities; cheap
    unit-test-able shape signal used by the gamma tuner's
    hard-constraint check.
  - `compute_modularity_classical`: Newman 2006 classical modularity
    oracle for the calibration sweep. NOT used at runtime; only by
    `scripts/calibrate_cpm_floor.py` + sanity tests. Mathematically
    equivalent to CPM-Q at gamma=1.0 (RBConfigurationVertexPartition
    collapses to Newman when the resolution multiplier is unity).
  - `all_communities_connected`: scipy.sparse.csgraph oracle — every
    community induces a connected subgraph.
  - `should_fall_back_to_flat`: hyper-fragmentation guard combining
    CPM-Q floor (calibrated) with singleton ratio + community-count
    ceiling.

CPM-Q is gamma-dependent and NOT comparable to classical-Q at the 0.2
cutoff. `community.MODULARITY_FLOOR=0.2` was calibrated against
`ModularityVertexPartition`, NOT `CPMVertexPartition`. The CPM floor is
re-calibrated from a 15-row sweep:
  fixtures = {karate, football, lfr_n2000}
  gamma_grid = {0.5, 0.75, 1.0, 1.5, 2.0}
  floor = 5th-percentile cpm_q among partitions passing
          classical_q >= 0.2 AND singleton_ratio < 0.3.
Reproducible via `PYTHONPATH=src python scripts/calibrate_cpm_floor.py`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as _csgraph

from iai_mcp.mosaic import compute_modularity_cpm

#
# Calibrated empirically: 15-row sweep across
# {karate, football, lfr_n2000} x gamma in {0.5, 0.75, 1.0, 1.5, 2.0}.
# Floor = 5th-percentile cpm_q among 10 "good" partitions
# (classical_q >= 0.2 AND singleton_ratio < 0.3).
#
# Re-run: `PYTHONPATH=src python scripts/calibrate_cpm_floor.py`.
#
# CPM-Q is gamma-dependent and NOT comparable to classical-Q at 0.2;
# `community.MODULARITY_FLOOR=0.2` was calibrated for
# ModularityVertexPartition; this floor is calibrated for
# CPMVertexPartition. The full sweep table is in
# `scripts/calibrate_cpm_floor.py`.
CPM_MODULARITY_FLOOR: float = 0.1338


def compute_modularity_classical(
    csr: scipy.sparse.csr_matrix, partition: np.ndarray
) -> float:
    """Newman 2006 classical modularity (NOT CPM); calibration-sweep oracle.

      Q = (1 / 2m) * Σ_C [w_in(C) - (sigma_tot[C]^2 / 2m)]

    Mathematically equivalent to RB-Configuration modularity at
    `gamma=1.0` (the RB resolution multiplier collapses to unity, leaving
    the canonical Newman form). Implementation reuses
    `compute_modularity_cpm` with `gamma=1.0` to avoid re-implementing a
    formula that already lands at NMI parity with leidenalg's
    `ModularityVertexPartition`.

    NOT used at runtime; only by `scripts/calibrate_cpm_floor.py` +
    sanity tests for the calibration sweep.

    Args:
      csr: CSR matrix of the (undirected) graph.
      partition: int64 array; `partition[i]` is the community of node i.

    Returns:
      Classical Newman Q in [-0.5, 1.0).
    """
    if csr.nnz == 0 or partition.size == 0:
        return 0.0
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    # sigma_tot indexing requires contiguous community labels [0..k-1];
    # compact the partition first.
    unique = np.unique(partition)
    relabel = np.full(int(unique.max()) + 1 if unique.size else 1, -1, dtype=np.int64)
    for new_idx, lbl in enumerate(unique):
        relabel[int(lbl)] = new_idx
    compact = relabel[partition].astype(np.int64)
    k = int(unique.size)
    # Inline sigma_tot accumulation (Numba kernel imported at module
    # load adds 1-2s JIT cost; we avoid the dependency cycle here).
    sigma = np.zeros(k, dtype=np.float64)
    n = compact.shape[0]
    for i in range(n):
        s = 0.0
        for off in range(int(indptr[i]), int(indptr[i + 1])):
            s += float(data[off])
        sigma[int(compact[i])] += s
    return float(compute_modularity_cpm(indptr, indices, data, compact, sigma, 1.0))


def compute_singleton_ratio(partition: np.ndarray) -> float:
    """Fraction of nodes whose community has size 1.

    Used by the multi-objective gamma tuner's hard-constraint check
    (`singleton_ratio < 0.30`) and by `should_fall_back_to_flat`. Empty
    partition -> 0.0 (no nodes, nothing to count).
    """
    if partition.size == 0:
        return 0.0
    _, counts = np.unique(partition, return_counts=True)
    singletons = int((counts == 1).sum())
    return float(singletons) / float(partition.size)


def all_communities_connected(
    csr: scipy.sparse.csr_matrix, partition: np.ndarray
) -> bool:
    """Oracle: every community induces a connected subgraph.

    Hard target: every community induces a connected subgraph. Used by
    the multi-objective gamma tuner's hard-constraint check.

    Implementation: for each distinct community label, build the induced
    subgraph slice and call `scipy.sparse.csgraph.connected_components`.
    O(K * E) where K is the number of communities and E is the edge
    count of the largest community. Cheap enough for the per-candidate
    tuner check on graphs up to N=5000.

    Args:
      csr: CSR matrix of the (undirected) graph.
      partition: int64 array; `partition[i]` is the community of node i.

    Returns:
      True iff every community's induced subgraph is connected.
    """
    if partition.size == 0:
        return True
    for lbl in np.unique(partition):
        mask = partition == lbl
        if mask.sum() <= 1:
            continue  # singleton is trivially connected
        sub = csr[mask][:, mask]
        n_comp, _ = _csgraph.connected_components(sub, directed=False)
        if n_comp > 1:
            return False
    return True


def should_fall_back_to_flat(
    modularity: float,
    singleton_ratio: float,
    n_communities: int,
    n: int,
) -> bool:
    """Hyper-fragmentation guard for CPM-partitioned memory graphs.

    Returns True iff ANY of:
      - modularity < `CPM_MODULARITY_FLOOR` (calibrated; NOT 0.2)
      - singleton_ratio > 0.30
      - n_communities > n // 5 (hyper-fragmentation regression bound)

    The first criterion uses the calibrated `CPM_MODULARITY_FLOOR`, NOT
    `community.MODULARITY_FLOOR=0.2` which was calibrated for
    `ModularityVertexPartition` (see module docstring).

    Args:
      modularity: CPM-Q of the candidate partition.
      singleton_ratio: fraction of nodes in size-1 communities.
      n_communities: number of distinct community labels.
      n: total node count.

    Returns:
      True iff the policy decides the partition should be discarded in
      favour of `_flat_assignment`.
    """
    # CPM_MODULARITY_FLOOR is the CPM-calibrated floor; NOT
    # community.MODULARITY_FLOOR=0.2 which was calibrated for
    # ModularityVertexPartition (see module docstring).
    if modularity < CPM_MODULARITY_FLOOR:
        return True
    if singleton_ratio > 0.30:
        return True
    if n > 0 and n_communities > (n // 5):
        return True
    return False


