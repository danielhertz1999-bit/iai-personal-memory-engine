"""Task 2 -- CPM-Q floor calibration sweep.

One-shot calibration script. Runs ONE Leiden pass on each (fixture, gamma)
combination, records CPM-Q + classical-Q + singleton-ratio + community count,
and computes the 5th-percentile CPM-Q among "good" partitions (classical-Q
>= 0.2 AND singleton-ratio < 0.3).

This is NOT a test -- it produces the empirical value that is hard-coded into
`src/iai_mcp/mosaic_policy.py` as `CPM_MODULARITY_FLOOR`. Re-run only
when fixtures change OR an auditor wants to verify the empirical floor.

Output:
  - JSON sweep table printed to stdout (15 rows: 3 fixtures x 5 gammas)
  - Final `CPM_MODULARITY_FLOOR` value printed at the end

Usage:
  PYTHONPATH=src python scripts/calibrate_cpm_floor.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import numpy as np

# Repo path discovery -- this script is always run from the worktree root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iai_mcp.mosaic import (  # noqa: E402
    _njit_local_move,
    _njit_refine,
    _split_disconnected_communities,
    build_csr_sanitized,
    compute_modularity_cpm,
    compute_sigma_tot,
)
from iai_mcp.mosaic_lineage import LineageTracker  # noqa: E402
from iai_mcp.graph import MemoryGraph  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "leiden"
GAMMA_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_karate() -> MemoryGraph:
    data = json.loads((FIXTURES_DIR / "karate_club.json").read_text())
    g = MemoryGraph()
    karate_ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes = [uuid5(karate_ns, f"karate-{i}") for i in range(data["n"])]
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g


def _load_football() -> MemoryGraph:
    data = json.loads((FIXTURES_DIR / "football.json").read_text())
    g = MemoryGraph()
    football_ns = UUID("87654321-4321-8765-4321-876543218765")
    nodes = [uuid5(football_ns, f"football-{i}") for i in range(data["n"])]
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g


def _build_lfr_n2000() -> MemoryGraph:
    """3-community planted graph, N=2000. Slightly denser intra than the N=5000
    regression test (intra_p=0.04) to keep good-partition counts robust."""
    n = 2000
    intra_p = 0.04
    inter_p = 0.001
    rng = random.Random(13)
    third = n // 3
    groups = [
        list(range(0, third)),
        list(range(third, 2 * third)),
        list(range(2 * third, n)),
    ]
    uuids = [uuid4() for _ in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for grp in groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if rng.random() < intra_p:
                    g.add_edge(uuids[grp[i]], uuids[grp[j]], weight=1.0)
    for gi in range(3):
        for gj in range(gi + 1, 3):
            for i in groups[gi]:
                for j in groups[gj]:
                    if rng.random() < inter_p:
                        g.add_edge(uuids[i], uuids[j], weight=0.1)
    return g


def _compute_singleton_ratio_inline(partition: np.ndarray) -> float:
    """Standalone helper (mirrors `compute_singleton_ratio` we will land in
    `mosaic_policy.py` -- inlined here so this script does not depend
    on the policy module being already updated)."""
    if partition.size == 0:
        return 0.0
    _, counts = np.unique(partition, return_counts=True)
    singletons = int((counts == 1).sum())
    return float(singletons) / float(partition.size)


def _run_one_leiden_pass_inline(
    graph: MemoryGraph, gamma: float, seed: int = 42
):
    """ONE-SHOT inline Leiden pass for calibration.

    Pattern mirrors `_run_one_leiden_pass` we will add to `mosaic.py`
    in Task 3 (work on copies, run LM + refinement, return refined +
    sigma_tot_refined + CPM-Q). Plan-B kernel signature: precompute
    visit_order outside @njit.
    """
    csr, _order, _idx_map = build_csr_sanitized(graph)
    if csr.nnz == 0:
        return None, None, None, None
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1

    partition = np.arange(n, dtype=np.int64)
    sigma_tot = compute_sigma_tot(indptr, indices, data, partition, n)

    # Plan-B: visit_order computed OUTSIDE @njit.
    rng_lm = np.random.Generator(np.random.PCG64(seed))
    visit_lm = rng_lm.permutation(n).astype(np.int64)
    _njit_local_move(
        indptr, indices, data, partition, sigma_tot, gamma, visit_lm, 20
    )

    # Defensive split.
    import scipy.sparse
    curr_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))
    partition, sigma_tot, _ = _split_disconnected_communities(
        curr_csr, partition, sigma_tot, {}, LineageTracker()
    )

    # Refinement.
    refined = np.arange(n, dtype=np.int64)
    sigma_refined = compute_sigma_tot(indptr, indices, data, refined, n)
    rng_ref = np.random.Generator(np.random.PCG64(seed + 1))
    visit_ref = rng_ref.permutation(n).astype(np.int64)
    _njit_refine(
        indptr, indices, data, partition, refined, sigma_refined,
        gamma, visit_ref, 1,
    )

    cpm_q = float(compute_modularity_cpm(
        indptr, indices, data, refined, sigma_refined, gamma,
    ))
    classical_q = float(compute_modularity_cpm(
        indptr, indices, data, refined, sigma_refined, 1.0,
    ))
    s_ratio = _compute_singleton_ratio_inline(refined)
    n_comm = int(len(np.unique(refined)))
    return cpm_q, classical_q, s_ratio, n_comm


def main() -> None:
    calibration: list[dict] = []
    fixtures = [
        ("karate", _load_karate),
        ("football", _load_football),
        ("lfr_n2000", _build_lfr_n2000),
    ]
    for fixture_name, loader in fixtures:
        print(f"[CALIBRATE] loading {fixture_name}...", file=sys.stderr)
        graph = loader()
        for gamma in GAMMA_GRID:
            cpm_q, classical_q, s_ratio, n_comm = _run_one_leiden_pass_inline(
                graph, gamma, seed=42
            )
            row = {
                "fixture": fixture_name,
                "gamma": gamma,
                "cpm_q": cpm_q,
                "classical_q": classical_q,
                "singleton_ratio": s_ratio,
                "n_communities": n_comm,
            }
            calibration.append(row)
            print(
                f"  gamma={gamma:.2f}  cpm_q={cpm_q:.4f}  "
                f"classical_q={classical_q:.4f}  s_ratio={s_ratio:.4f}  "
                f"n_comm={n_comm}",
                file=sys.stderr,
            )

    # Good partitions: classical Q >= 0.2 AND singleton-ratio < 0.3.
    good = [
        d for d in calibration
        if d["classical_q"] >= 0.2 and d["singleton_ratio"] < 0.3
    ]
    if len(good) < 3:
        print(
            f"\n[ERROR] Calibration produced too few good partitions: "
            f"{len(good)} (need >= 3). Expand fixture set.",
            file=sys.stderr,
        )
        sys.exit(2)
    cpm_floor = float(np.percentile([d["cpm_q"] for d in good], 5))
    print("\n[CALIBRATION SWEEP TABLE]", file=sys.stderr)
    print(json.dumps(calibration, indent=2))
    print(
        f"\n[GOOD partitions]: {len(good)}/{len(calibration)} (passed "
        f"classical Q >= 0.2 AND singleton-ratio < 0.3)",
        file=sys.stderr,
    )
    print(
        f"[GOOD cpm_q values]: "
        f"{sorted([d['cpm_q'] for d in good])}",
        file=sys.stderr,
    )
    print(f"\nCPM_MODULARITY_FLOOR = {cpm_floor:.4f}", file=sys.stderr)
    print(f"CPM_MODULARITY_FLOOR = {cpm_floor:.4f}")
    if not (0.05 <= cpm_floor <= 0.30):
        print(
            f"\n[WARN] cpm_floor={cpm_floor:.4f} outside sanity bounds "
            f"[0.05, 0.30]. Investigate before committing.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
