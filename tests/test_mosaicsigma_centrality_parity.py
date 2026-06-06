"""Differential parity for iai_mcp_native.graph.betweenness_centrality.

UNWEIGHTED BFS-Brandes via rustworkx-core 0.17. The crate API signature
does not accept a weight map; the prior networkx-weighted-Brandes
semantic (``weight='weight'`` on Hebbian edge strengths) is
intentionally dropped at the source-of-truth crate boundary. Every test
in this module compares against ``networkx.betweenness_centrality(g,
normalized=True, weight=None)`` — i.e. networkx's UNWEIGHTED Brandes —
not the legacy weighted oracle.

Beyond the parity matrix, the module guards five behavioural invariants
of the Python wrapper:

  - Hub-vs-leaf dominance on a 5-node star (carries forward from the
    pre-cutover test in tests/test_graph.py).
  - Cache flag wiring: ``IAI_MCP_CENTRALITY_CACHE`` explicit ``on``/``off``
    overrides plus ``auto`` resolution against the module-level
    ``AUTO_CACHE_DEFAULT`` constant.
  - Cache invalidation on graph mutation (auto-dirty flip on add_node).
  - The empty-graph fast path (returns ``{}``).
  - The Python wrapper consumes the Rust-returned ``node_arr`` explicitly
    via ``zip(node_arr, centrality_arr)`` — a monkey-patched native
    function that returns a reversed ``node_arr`` is the regression-guard
    against silent zip-with-iter_nodes() drift.
  - The ``MemoryGraph.__init__`` attribute-initialisation guard so the
    first ``centrality()`` call on a fresh instance never raises
    ``AttributeError``.

The perf-gate-decision cross-check at the bottom (``test_auto_cache_default
_matches_perf_gate_decision``) is added by the perf-gate bench step and
asserts the committed ``AUTO_CACHE_DEFAULT`` matches the JSON ``action``
field. It skips gracefully when the JSON file is absent.
"""
from __future__ import annotations

import inspect
import json
import os
import pathlib
from uuid import UUID, uuid4

import pytest

# Skip the whole module when the native wheel or networkx/numpy is missing.
pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("iai_mcp_native")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from iai_mcp import graph as graph_module  # noqa: E402
from iai_mcp.graph import MemoryGraph  # noqa: E402


FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)
PERF_GATE_JSON_PATH = (
    pathlib.Path(__file__).parent.parent
    / ".planning"
    / "phases"
    / "50-mosaicsigma-networkx-custom-rust-pyo3-graph-engine-lilli-gra"
    / "50-10-perf-gate.json"
)


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


# Fixtures that ship with an `edges` array. live_n2000 carries an empty
# edge list (it's a metadata-only snapshot pointer), so the differential
# parity gate runs against the seven remaining keys: empirical small
# graphs (karate), Erdos-Renyi random baselines, and tiny / larger
# lattices.
CENTRALITY_FIXTURE_KEYS = [
    "karate",
    "er_200",
    "er_500",
    "er_1000",
    "tiny_10_ws_k4",
    "tiny_20_ws_p010",
    "ws_2500_k4_p0",
]


def _build_memory_graph_from_fixture(
    n_nodes: int, edges: list[tuple[int, int]]
) -> tuple[MemoryGraph, list[UUID]]:
    """Build a MemoryGraph + return the parallel list of node UUIDs.

    Each fixture node int gets a deterministic UUID via
    ``UUID(int=i)`` so the test is reproducible across runs and the
    same UUIDs appear in both the MemoryGraph and the networkx oracle
    construction below.
    """
    node_ids = [UUID(int=i) for i in range(n_nodes)]
    mg = MemoryGraph()
    for nid in node_ids:
        mg.add_node(nid, community_id=None, embedding=[0.0] * 384)
    for u, v in edges:
        if u == v:
            continue
        mg.add_edge(node_ids[u], node_ids[v], weight=1.0)
    return mg, node_ids


def _networkx_oracle(
    n_nodes: int, edges: list[tuple[int, int]]
) -> dict[int, float]:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((u, v) for u, v in edges if u != v)
    # weight=None is the unweighted-Brandes path. We ship unweighted
    # by design (the upstream Rust API has no weight map); this parity
    # gate locks the new contract, not the legacy weighted one.
    return nx.betweenness_centrality(g, normalized=True, weight=None)


# ----------------------------------------------------- parity / invariant gates


def test_betweenness_matches_networkx_on_fixtures() -> None:
    """Differential parity against networkx unweighted Brandes.

    Failure-list idiom: every fixture is compared at rtol=1e-7 / atol=
    1e-12; drifts are accumulated and reported together at the end.
    Both sides evaluate BFS-based Brandes on the same edge set, so the
    expected delta is at FP-noise.
    """
    fixtures = _load_fixtures()
    drifts: list[str] = []
    for key in CENTRALITY_FIXTURE_KEYS:
        fx = fixtures[key]
        n = int(fx["n"])
        edges = [tuple(e) for e in fx["edges"]]
        oracle = _networkx_oracle(n, edges)
        mg, node_ids = _build_memory_graph_from_fixture(n, edges)
        ours = mg.centrality()
        # Compare per-node (oracle keyed by int, ours by UUID).
        # Sort both by the underlying int identifier so the vectors are
        # aligned before the assert_allclose call.
        oracle_vec = np.array(
            [oracle[i] for i in range(n)], dtype=np.float64
        )
        ours_vec = np.array(
            [ours[node_ids[i]] for i in range(n)], dtype=np.float64
        )
        try:
            np.testing.assert_allclose(
                ours_vec, oracle_vec, rtol=1e-7, atol=1e-12
            )
        except AssertionError as exc:
            max_abs = float(np.max(np.abs(ours_vec - oracle_vec)))
            drifts.append(
                f"{key} n={n}: max|delta|={max_abs:.3e} -- {exc.args[0][:200]}"
            )
    assert not drifts, (
        "Brandes parity drifts vs networkx unweighted oracle:\n  "
        + "\n  ".join(drifts)
    )


def test_hub_beats_leaves_on_star_graph() -> None:
    """Hub-vs-leaf dominance on a 5-node star (verbatim invariant).

    Hub + 4 leaves; every edge connects the hub to a leaf. Brandes
    assigns the hub all shortest-path-counting weight because every
    pair (leaf_i, leaf_j) traverses the hub exactly once. The leaves
    sit on no shortest paths between any two other nodes, so each
    leaf's centrality is exactly 0. The test ensures the Rust-routed
    centrality preserves the same topological invariant the legacy
    NetworkX path satisfied.
    """
    mg = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(4)]
    mg.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        mg.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        mg.add_edge(hub, leaf)
    c = mg.centrality()
    for leaf in leaves:
        assert c[hub] > c[leaf], (
            f"hub={c[hub]} should beat leaf={c[leaf]}"
        )


def test_cache_off_recomputes_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IAI_MCP_CENTRALITY_CACHE=off`` returns a fresh dict per call.

    The cache flag's contract is recompute-on-every-call when set
    explicitly to ``off`` — the regression guard is the dict identity
    check (``id``), not value equality. Two consecutive calls must
    yield equal values but distinct dict objects.
    """
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "off")
    mg = MemoryGraph()
    hub = uuid4()
    leaf = uuid4()
    mg.add_node(hub, community_id=None, embedding=[0.0] * 384)
    mg.add_node(leaf, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(hub, leaf)
    c1 = mg.centrality()
    c2 = mg.centrality()
    assert c1 == c2
    assert c1 is not c2, "cache=off should hand back a fresh dict"


def test_cache_on_returns_same_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IAI_MCP_CENTRALITY_CACHE=on`` returns the same dict object.

    The second call must hand back the *identical* object — the cache
    is keyed on the dirty flag, and a clean graph is allowed to return
    the cached reference verbatim.
    """
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "on")
    mg = MemoryGraph()
    a = uuid4()
    b = uuid4()
    mg.add_node(a, community_id=None, embedding=[0.0] * 384)
    mg.add_node(b, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)
    c1 = mg.centrality()
    c2 = mg.centrality()
    assert c1 is c2, "cache=on should return the same dict reference"


def test_cache_auto_invalidates_on_add_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """add_node flips the dirty flag — auto mode recomputes after a mutation.

    Force ``auto`` to resolve to ``on`` so the test is independent of
    the perf-gate-committed default; the dirty-flag flip is the
    invariant being verified, not the AUTO_CACHE_DEFAULT value.
    """
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "auto")
    monkeypatch.setattr(graph_module, "AUTO_CACHE_DEFAULT", "on")
    mg = MemoryGraph()
    a, b, c_node = uuid4(), uuid4(), uuid4()
    for nid in (a, b):
        mg.add_node(nid, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)
    first = mg.centrality()
    # Mutate: add_node must flip _dirty_since_centrality.
    mg.add_node(c_node, community_id=None, embedding=[0.0] * 384)
    second = mg.centrality()
    assert first is not second, (
        "auto mode should recompute after add_node mutation; "
        f"first id={id(first)} second id={id(second)}"
    )


def test_empty_graph_centrality_returns_empty_dict() -> None:
    """No nodes in the graph → centrality() returns ``{}``.

    The Rust path handles ``n_nodes == 0`` cleanly (returns an empty
    Vec); the Python wrapper must NOT special-case the no-edge path the
    legacy implementation used — Rust is the single source of truth.
    """
    mg = MemoryGraph()
    c = mg.centrality()
    assert c == {}


def test_no_networkx_in_centrality_method() -> None:
    """``centrality()`` body must not call networkx.

    Inspect the method source after the cutover. Any
    ``nx.`` or ``networkx`` substring would mean the legacy path leaked
    back in — the cutover's goal is to ship Rust-only centrality.
    """
    src = inspect.getsource(MemoryGraph.centrality)
    assert "nx." not in src, "centrality() must not call into networkx"
    assert "networkx" not in src.lower(), (
        "centrality() must not reference networkx"
    )


def test_centrality_dict_uses_node_arr_not_iter_nodes_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python wrapper consumes the Rust-returned ``node_arr`` explicitly.

    Monkey-patch ``_native.betweenness_centrality`` to hand back a
    reversed ``node_arr`` plus a known centrality vector. If the
    wrapper falls back to ``zip(self.iter_nodes(), centrality_arr)``
    instead of consuming the returned node_arr, the assertion below
    fires because the centrality values would land on the wrong UUIDs.

    The graph here has TWO nodes added in a specific order — the test
    deliberately constructs a topology where iter_nodes() order and
    the monkey-patched node_arr order are distinct, so any silent drift
    surfaces on the wrong-UUID assertion.
    """
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "off")

    mg = MemoryGraph()
    a, b = UUID(int=1), UUID(int=2)
    mg.add_node(a, community_id=None, embedding=[0.0] * 384)
    mg.add_node(b, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)

    # In CSR row order, sorted(str(uuid)) yields [a, b] because
    # UUID(int=1) < UUID(int=2) lexicographically. The CSR-row-to-UUID
    # mapping inside centrality() is sorted(_nx.nodes()).
    # Hand back a *reversed* node_arr so iter_nodes()-zipping would
    # mis-assign centrality 10.0 to A and 20.0 to B; the corrected
    # wrapper consuming node_arr puts 10.0 on row 1 (= B) and 20.0 on
    # row 0 (= A).
    fake_centrality = np.array([10.0, 20.0], dtype=np.float64)
    fake_node_arr = np.array([1, 0], dtype=np.int64)  # reversed

    def fake_native(
        indptr: np.ndarray,
        indices: np.ndarray,
        n_nodes: int,
        normalized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        return fake_centrality, fake_node_arr

    # Patch the imported _native symbol inside the graph module.
    from iai_mcp_native import graph as native_graph_module
    monkeypatch.setattr(
        native_graph_module, "betweenness_centrality", fake_native
    )

    result = mg.centrality()
    # CSR row order is sorted by str(uuid). UUID(int=1) < UUID(int=2)
    # so row 0 = A, row 1 = B. Reversed node_arr [1, 0] therefore puts
    # the first centrality value (10.0) on row 1 (B) and the second
    # (20.0) on row 0 (A).
    assert result[b] == 10.0, (
        f"node_arr[0]=1 maps to CSR row 1 (B); should yield 10.0; got {result[b]}"
    )
    assert result[a] == 20.0, (
        f"node_arr[1]=0 maps to CSR row 0 (A); should yield 20.0; got {result[a]}"
    )


def test_cache_auto_resolves_to_AUTO_CACHE_DEFAULT(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` env mode resolves to the module-level ``AUTO_CACHE_DEFAULT``.

    Verified in three stages:
      1. AUTO_CACHE_DEFAULT="off" + env=auto → recompute every call.
      2. AUTO_CACHE_DEFAULT="on" + env=auto → cache hit on second call.
      3. AUTO_CACHE_DEFAULT="on" + env=off → env wins; recompute.
    """
    mg = MemoryGraph()
    a, b = uuid4(), uuid4()
    mg.add_node(a, community_id=None, embedding=[0.0] * 384)
    mg.add_node(b, community_id=None, embedding=[0.0] * 384)
    mg.add_edge(a, b)

    # Stage 1: AUTO=off + env=auto → no caching.
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "auto")
    monkeypatch.setattr(graph_module, "AUTO_CACHE_DEFAULT", "off")
    c1 = mg.centrality()
    c2 = mg.centrality()
    assert c1 is not c2, "AUTO=off + env=auto should recompute"

    # Reset the dirty/cache state by clearing it manually — the
    # previous stage's calls under AUTO=off did not write the cache,
    # so we can flip AUTO=on cleanly.
    mg._centrality_cache = None
    mg._dirty_since_centrality = True

    # Stage 2: AUTO=on + env=auto → cache hit on second call.
    monkeypatch.setattr(graph_module, "AUTO_CACHE_DEFAULT", "on")
    c3 = mg.centrality()
    c4 = mg.centrality()
    assert c3 is c4, "AUTO=on + env=auto should cache"

    # Stage 3: env=off overrides AUTO=on.
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "off")
    c5 = mg.centrality()
    c6 = mg.centrality()
    assert c5 is not c6, "env=off must override AUTO=on"


def test_memory_graph_init_centrality_cache_attrs_present() -> None:
    """``MemoryGraph.__init__`` initialises the cache + dirty-flag attrs.

    Without the init step, the first ``centrality()`` call on a fresh
    instance would raise ``AttributeError: 'MemoryGraph' object has no
    attribute '_centrality_cache'`` — the regression-guard pins the
    default-dirty (True) start state so the first computation never
    reads a stale cache.
    """
    mg = MemoryGraph()
    assert hasattr(mg, "_centrality_cache")
    assert hasattr(mg, "_dirty_since_centrality")
    assert mg._centrality_cache is None
    assert mg._dirty_since_centrality is True
    # And the empty-graph centrality call must NOT raise.
    assert mg.centrality() == {}


# ------------------------------------------------------------- perf-gate gate


def test_auto_cache_default_matches_perf_gate_decision() -> None:
    """``AUTO_CACHE_DEFAULT`` must match the perf-gate ``action`` field.

    Skips gracefully when the JSON file is missing — the test runs in
    every environment that has the wheel installed, but the bench is
    orchestrator-only and CI environments may not have run it.
    """
    if not PERF_GATE_JSON_PATH.exists():
        pytest.skip(
            f"perf gate JSON not present at {PERF_GATE_JSON_PATH}; "
            "bench has not been run in this environment"
        )
    decision = json.loads(PERF_GATE_JSON_PATH.read_text(encoding="utf-8"))
    action = decision.get("action")
    assert action in ("drop", "keep"), (
        f"perf gate JSON action must be 'drop' or 'keep'; got {action!r}"
    )
    expected = "off" if action == "drop" else "on"
    actual = graph_module.AUTO_CACHE_DEFAULT
    assert actual == expected, (
        f"AUTO_CACHE_DEFAULT={actual!r} does not match perf-gate "
        f"action={action!r} (expected {expected!r}). Reason from JSON: "
        f"{decision.get('reason', '<missing>')}"
    )
