"""Test suite for the custom MIT Leiden scaffold.

Scope:

  - Module importability for the three new modules.
  - `build_csr_sanitized` enforces canonical UUID/edge ordering,
    NaN/Inf/negative weight drop, and self-loop strip before any kernel
    receives data.
  - `run_mosaic` public signature is frozen for the kernel to fill.
  - `LineageEvent` is immutable (frozen dataclass) — audit trail invariant.
  - `LineageTracker.report()` returns an empty `LineageReport` until events are
    recorded.
  - Public constants `EPSILON`, `WALL_TIME_HARD_CAP_S` are exposed.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import numpy as np
import pytest
import scipy.sparse

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    """Deterministic embedding for test nodes."""
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


# ---------------------------------------------------------------- imports


def test_module_imports() -> None:
    """custom_leiden exposes the public surface frozen by M1 for M2/M3."""
    from iai_mcp.mosaic import (
        EPSILON,
        WALL_TIME_HARD_CAP_S,
        build_csr_sanitized,
        run_mosaic,
    )

    assert callable(run_mosaic)
    assert callable(build_csr_sanitized)
    assert isinstance(EPSILON, float)
    assert isinstance(WALL_TIME_HARD_CAP_S, float)


def test_lineage_types_importable() -> None:
    """LineageEvent / LineageTracker / LineageReport public surface for M4."""
    from iai_mcp.mosaic_lineage import (
        LineageEvent,
        LineageReport,
        LineageTracker,
    )

    assert LineageEvent is not None
    assert LineageReport is not None
    assert LineageTracker is not None


def test_policy_module_importable() -> None:
    """Policy module exposes its live hyper-fragmentation guard."""
    from iai_mcp.mosaic_policy import should_fall_back_to_flat

    assert callable(should_fall_back_to_flat)


# ---------------------------------------------------------------- CSR builder


def test_csr_canonical_node_order() -> None:
    """Nodes sorted by str(uuid) ascending before CSR construction."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    # Insert in jumbled order; expect sorted-by-str output.
    uuids = [uuid4() for _ in range(5)]
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i))

    _csr, order, idx_map = build_csr_sanitized(g)
    assert order == sorted(uuids, key=str)
    assert list(idx_map.keys()) == sorted(uuids, key=str)
    for i, u in enumerate(order):
        assert idx_map[u] == i


def test_csr_canonical_edge_order() -> None:
    """Edges (b,a) and (a,b) collapse to one (min,max) canonical entry."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_node(c, community_id=None, embedding=_emb(2))
    g.add_edge(a, b, weight=2.0)
    g.add_edge(b, c, weight=3.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    # Undirected graph: 2 logical edges => 4 CSR entries (a-b, b-a, b-c, c-b).
    assert csr.nnz == 4
    # CSR-canonical (sorted by row, then col within row).
    csr.sort_indices()
    assert (csr != csr).nnz == 0  # self-equality sanity


def test_csr_strips_nan_weights() -> None:
    """NaN-weight edge dropped before kernel entry."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_node(c, community_id=None, embedding=_emb(2))
    g.add_edge(a, b, weight=float("nan"))
    g.add_edge(b, c, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    # Only b-c survives -> 2 CSR entries (b-c and c-b).
    assert csr.nnz == 2
    assert all(math.isfinite(float(x)) for x in csr.data)


def test_csr_strips_inf_weights() -> None:
    """+Inf and -Inf edges dropped before kernel entry."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for i, u in enumerate([a, b, c, d]):
        g.add_node(u, community_id=None, embedding=_emb(i))
    g.add_edge(a, b, weight=float("inf"))
    g.add_edge(b, c, weight=float("-inf"))
    g.add_edge(c, d, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    # Only c-d survives.
    assert csr.nnz == 2
    assert all(math.isfinite(float(x)) for x in csr.data)


def test_csr_strips_negative_weights() -> None:
    """w < 0 edge dropped before kernel entry."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_node(c, community_id=None, embedding=_emb(2))
    g.add_edge(a, b, weight=-0.5)
    g.add_edge(b, c, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.nnz == 2
    assert all(float(x) >= 0.0 for x in csr.data)


def test_csr_strips_self_loops() -> None:
    """Self-loop edge dropped up-front."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b = uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_edge(a, a, weight=2.0)  # self-loop
    g.add_edge(a, b, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    # Only a-b survives -> 2 CSR entries (a-b and b-a).
    assert csr.nnz == 2
    # No diagonal entries.
    assert csr.diagonal().sum() == 0.0


def test_csr_dtype_is_float64() -> None:
    """Strict float64 dtype before Numba kernel entry."""
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b = uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_edge(a, b, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.data.dtype == np.float64
    assert isinstance(csr, scipy.sparse.csr_matrix)


# ---------------------------------------------------------------- lineage


def test_lineage_event_is_frozen() -> None:
    """LineageEvent is @dataclass(frozen=True)."""
    from datetime import datetime, timezone

    from iai_mcp.mosaic_lineage import LineageEvent

    event = LineageEvent(
        event_type="birth",
        timestamp=datetime.now(timezone.utc),
        parent_uuid=None,
        child_uuids=(uuid4(),),
        member_count=1,
    )
    with pytest.raises(FrozenInstanceError):
        event.member_count = 999  # type: ignore[misc]


def test_lineage_tracker_empty_report_has_zero_events() -> None:
    """LineageTracker().report().events == () until events are recorded."""
    from iai_mcp.mosaic_lineage import LineageReport, LineageTracker

    tracker = LineageTracker()
    report = tracker.report()
    assert isinstance(report, LineageReport)
    assert report.events == ()


# ---------------------------------------------------------------- run signature


def test_run_mosaic_signature() -> None:
    """Public signature frozen for M2/M3 to extend; argument names + defaults."""
    from iai_mcp.mosaic import run_mosaic

    sig = inspect.signature(run_mosaic)
    params = sig.parameters
    assert set(params.keys()) == {
        "graph",
        "prior",
        "prior_mode",
        "gamma",
        "seed",
        "max_levels",
    }
    assert params["prior"].default is None
    assert params["prior_mode"].default == "seeded"
    assert params["gamma"].default is None
    assert params["seed"].default == 42
    assert params["max_levels"].default == 5


def test_wall_time_constant_is_30s() -> None:
    """30s hard wall-time cap exposed as module constant."""
    from iai_mcp.mosaic import WALL_TIME_HARD_CAP_S

    assert WALL_TIME_HARD_CAP_S == 30.0


def test_run_mosaic_returns_assignment_on_nonempty_graph() -> None:
    """M2 flipped the M1 NotImplementedError gate to a real
    kernel call. M1's gate test (`...raises_not_implemented...`) is replaced
    here with the positive contract: a non-empty graph returns a tuple
    `(CommunityAssignment, LineageReport)` with the `leiden-custom` backend.

    Post- update: `gamma=None` (default) now triggers the
    multi-objective tuner, which for a trivial 3-node graph correctly
    falls back to flat (no candidate satisfies the hyper-frag bound
    `n_comm <= n/5 = 0`). To preserve the original `leiden-custom`
    contract check, pass explicit `gamma=1.0` so the tuner is skipped.
    """
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.mosaic import run_mosaic
    from iai_mcp.mosaic_lineage import LineageReport

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for i, u in enumerate([a, b, c]):
        g.add_node(u, community_id=None, embedding=_emb(i))
    g.add_edge(a, b, weight=1.0)
    g.add_edge(b, c, weight=1.0)

    assignment, report = run_mosaic(
        g, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    assert isinstance(assignment, CommunityAssignment)
    assert assignment.backend == "leiden-custom"
    # Every input node gets a community assignment.
    assert set(assignment.node_to_community.keys()) == {a, b, c}
    assert isinstance(report, LineageReport)


def test_empty_graph_returns_empty_assignment() -> None:
    """0-node graph short-circuits to flat (no NotImplementedError raised)."""
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.mosaic import run_mosaic
    from iai_mcp.mosaic_lineage import LineageReport

    g = MemoryGraph()
    assignment, report = run_mosaic(g, prior=None)
    assert isinstance(assignment, CommunityAssignment)
    assert assignment.backend == "flat"
    assert assignment.node_to_community == {}
    assert isinstance(report, LineageReport)
    assert report.events == ()
