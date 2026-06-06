"""Differential harness skeleton for downstream parity tests.

Downstream waves use this module's strategy and helper functions to assert
numeric parity between the project's graph metrics and the networkx==3.x
oracle. This file ships only the wiring (strategy + helpers + one smoke
test); per-metric differential tests live in later test modules.

Posture:
- hypothesis + hypothesis-networkx are dev-only dependencies. When absent
  (release env without the [dev] extra), the module skips cleanly via
  importorskip and downstream tests that import from it also skip.
- networkx is always required (the iai_mcp graph stack depends on it).
"""

from __future__ import annotations

from typing import Callable, Iterable

import pytest

# Skip the entire module unless every required dep is importable. Order
# matters: networkx is the hardest requirement (iai_mcp.sigma pulls it in),
# hypothesis is the test runner, hypothesis_networkx provides the strategy.
pytest.importorskip("networkx")
pytest.importorskip("hypothesis")
pytest.importorskip("hypothesis_networkx")

import hypothesis  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis_networkx import graph_builder  # noqa: E402


# Bounded strategy: connected graphs in the 20-200 node range, no self-loops.
# Bounds chosen so the harness wall-clock stays cheap at this skeleton wave;
# downstream waves may rescale via their own strategy if needed.
_GRAPH_STRATEGY = graph_builder(
    min_nodes=20,
    max_nodes=200,
    connected=True,
    self_loops=False,
)


def differential_check(
    g: "nx.Graph",
    our_metric_fn: Callable[["nx.Graph"], float],
    oracle_metric_fn: Callable[["nx.Graph"], float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> None:
    """Compute our_metric_fn(g) vs oracle_metric_fn(g) and assert allclose.

    Raises AssertionError on numeric drift outside (rtol, atol). Used by
    downstream parity tests; this skeleton wave ships the helper only.
    """
    our_value = our_metric_fn(g)
    oracle_value = oracle_metric_fn(g)
    np.testing.assert_allclose(our_value, oracle_value, rtol=rtol, atol=atol)


def make_failure_list_pattern(
    g_iter: Iterable["nx.Graph"],
    our_fn: Callable[["nx.Graph"], float],
    oracle_fn: Callable[["nx.Graph"], float],
) -> list[tuple[int, float, str]]:
    """Enumerate inputs and accumulate per-graph drift diagnostics.

    Mirrors the failure-accumulation idiom used by the embedder numeric-parity
    gate (rust + python). Returns a list of (index, abs_drift, descriptor)
    tuples; downstream tests assert the list is empty with a diagnostic
    summary of the first 10 offenders.
    """
    failures: list[tuple[int, float, str]] = []
    for idx, g in enumerate(g_iter):
        try:
            our_value = float(our_fn(g))
            oracle_value = float(oracle_fn(g))
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            failures.append((idx, float("inf"), f"raised={type(exc).__name__}: {exc!s}"))
            continue
        drift = abs(our_value - oracle_value)
        if drift > 0:
            descriptor = (
                f"n={g.number_of_nodes()} m={g.number_of_edges()} "
                f"ours={our_value!r} oracle={oracle_value!r}"
            )
            failures.append((idx, drift, descriptor))
    return failures


@given(g=_GRAPH_STRATEGY)
@settings(deadline=None, max_examples=20)
def test_harness_imports_and_strategy_yields_a_graph(g: "nx.Graph"):
    """Smoke test — strategy yields connected graphs with positive node count."""
    assert g.number_of_nodes() > 0, "graph_builder returned an empty graph"
    assert nx.is_connected(g), (
        f"strategy promised connected=True but got disconnected graph "
        f"(n={g.number_of_nodes()}, m={g.number_of_edges()})"
    )


__all__ = [
    "differential_check",
    "make_failure_list_pattern",
    "test_harness_imports_and_strategy_yields_a_graph",
]
