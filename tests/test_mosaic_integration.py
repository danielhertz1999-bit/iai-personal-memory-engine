"""Integration tests for the production wiring.

Custom MIT Leiden replaces the `leidenalg` path inside
`community.detect_communities`. These tests pin the contract:

  - `detect_communities` is the public entrypoint and now routes
    through `run_mosaic` under the hood.
  - The new `prior_mode` kwarg threads the two-mode invocation
    distinction (`"cold"` for crisis_recluster, `"seeded"` for normal
    recall paths).
  - `CommunityAssignment` carries the `lineage_report` event log
    instead of the heuristic cosine-rotation policy.
  - The mid-N modularity guard uses `CPM_MODULARITY_FLOOR` from the
    `mosaic_policy` module, NOT the legacy `MODULARITY_FLOOR=0.2` -- CPM-Q
    is gamma-dependent and not comparable to classical-Q at 0.2.
  - The three integration call sites (`sleep_pipeline.py`,
    `retrieve.py`, `sigma.py`) pass the right `prior_mode` value.
"""
from __future__ import annotations

import inspect
import random
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.community import (
    CommunityAssignment,
    detect_communities,
)
from iai_mcp.mosaic_lineage import LineageReport
from iai_mcp.graph import MemoryGraph


REPO_ROOT = Path(__file__).resolve().parent.parent


def _random_emb(seed: int) -> list[float]:
    """Match the fixture pattern in `tests/test_community.py`."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(384)]


def _make_two_clique_graph(n_per_clique: int = 150) -> MemoryGraph:
    """Two cliques of N nodes each -- N=300 total, Leiden's bread and butter.

    Matches the existing pattern in `tests/test_community.py`.
    """
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(n_per_clique)]
    clique_b = [uuid4() for _ in range(n_per_clique)]
    for i, n in enumerate(clique_a):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i, n in enumerate(clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(10_000 + i))
    for i in range(n_per_clique):
        for j in range(i + 1, n_per_clique):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    return g


# ============================================================================
# community.py wireup tests
# ============================================================================


def test_detect_communities_uses_mosaic_backend() -> None:
    """contract: the new backend label is `'leiden-custom'`.

    The legacy values were `'leiden-igraph'` / `'leiden-networkx'`. Existing
    tests that grep `backend.startswith('leiden')` keep working because the
    new label still starts with `'leiden'`.
    """
    g = _make_two_clique_graph()
    a = detect_communities(g)
    assert a.backend == "leiden-custom"
    assert a.modularity >= 0.20


def test_detect_communities_accepts_prior_mode_seeded() -> None:
    """`prior_mode='seeded'` is a valid kwarg; no TypeError."""
    g = _make_two_clique_graph()
    a = detect_communities(g, prior=None, prior_mode="seeded")
    assert isinstance(a, CommunityAssignment)


def test_detect_communities_accepts_prior_mode_cold() -> None:
    """`prior_mode='cold'` discards the prior -- no prior UUID in the new
    assignment.

    crisis_recluster intentionally breaks continuity, so the new partition
    must NOT carry any community UUID that appeared in `prior`.
    """
    g = _make_two_clique_graph()
    first = detect_communities(g, prior=None, prior_mode="seeded")
    # Cold mode: prior UUIDs must NOT appear in the new partition.
    second = detect_communities(g, prior=first, prior_mode="cold")
    prior_uuids = set(first.node_to_community.values())
    new_uuids = set(second.node_to_community.values())
    assert prior_uuids.isdisjoint(new_uuids), (
        "cold mode must discard prior UUIDs; "
        f"overlap = {prior_uuids & new_uuids}"
    )


def test_detect_communities_default_prior_mode_is_seeded() -> None:
    """The default `prior_mode` kwarg must be `'seeded'` for backwards-compat
    with the existing `retrieve.py` / `sigma.py` call sites that don't pass
    the new argument."""
    sig = inspect.signature(detect_communities)
    assert "prior_mode" in sig.parameters
    assert sig.parameters["prior_mode"].default == "seeded"


def test_community_assignment_lineage_report_field_exists() -> None:
    """`CommunityAssignment` gains a `lineage_report: LineageReport | None`
    field with default `None` -- backwards-compat for existing constructors."""
    assert "lineage_report" in CommunityAssignment.__dataclass_fields__
    fld = CommunityAssignment.__dataclass_fields__["lineage_report"]
    # The default factory or default must produce None.
    # dataclass `field` default-vs-default_factory: we use `default=None`.
    assert fld.default is None


def test_lineage_report_populated_on_leiden_path() -> None:
    """For graphs that take the Leiden path (N >= SMALL_N_FLAT), the
    assignment carries a real `LineageReport`. Tracks the wiring of
    `LineageTracker.report()` into the runtime assignment."""
    g = _make_two_clique_graph()
    a = detect_communities(g)
    assert a.lineage_report is not None
    assert isinstance(a.lineage_report, LineageReport)


def test_lineage_report_empty_on_flat_fallback() -> None:
    """For graphs that take the flat path (N < SMALL_N_FLAT), the assignment
    STILL carries a `LineageReport` -- empty events, but the field is the
    same type. Type-stability for downstream consumers."""
    g = MemoryGraph()
    for i in range(50):
        g.add_node(uuid4(), community_id=None, embedding=_random_emb(i))
    a = detect_communities(g)
    assert a.backend == "flat"
    assert a.lineage_report is not None
    assert isinstance(a.lineage_report, LineageReport)


def test_aggregate_uses_pick_merge_survivor() -> None:
    """`_aggregate` surviving-UUID selection now goes through
    `lineage.pick_merge_survivor`.

    Source-grep witness. The old placeholder `min(uuids, key=str)` is
    replaced by the lineage tracker's age-aware policy.
    """
    src = (REPO_ROOT / "src" / "iai_mcp" / "mosaic.py").read_text()
    # `_aggregate` body must call `pick_merge_survivor`.
    assert "pick_merge_survivor" in src, (
        "_aggregate must call lineage.pick_merge_survivor(...) instead of the "
        "old placeholder survivor selection"
    )


def test_detect_communities_uses_cpm_floor_not_legacy_0_2() -> None:
    """The mid-N modularity guard must compare
    against `CPM_MODULARITY_FLOOR`, NOT the legacy
    `MODULARITY_FLOOR=0.2`. CPM-Q is gamma-dependent and not comparable to
    classical-Q at the historical 0.2 cutoff.

    Source-grep witness: `community.py` body must import
    `CPM_MODULARITY_FLOOR` from `iai_mcp.mosaic_policy`.
    """
    src = (REPO_ROOT / "src" / "iai_mcp" / "community.py").read_text()
    assert "from iai_mcp.mosaic_policy import" in src, (
        "community.py must import the CPM-calibrated floor"
    )
    assert "CPM_MODULARITY_FLOOR" in src, (
        "community.py must reference CPM_MODULARITY_FLOOR in the mid-N guard"
    )


def test_existing_community_tests_still_pass_smoke() -> None:
    """Regression smoke: run the existing `tests/test_community.py` suite
    against the new backend. Subprocess invocation so a hard pytest crash
    doesn't kill THIS test runner."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_community.py",
            "-x",
            "--no-header",
            "-q",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            "tests/test_community.py regression failed:\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )


# ============================================================================
# call-site update tests
# ============================================================================


def test_sleep_pipeline_crisis_mode_uses_prior_mode_cold() -> None:
    """crisis_recluster must call `detect_communities` with
    `prior_mode='cold'` to honour the invariant that the prior partition is
    the broken topology being discarded.

    Source-grep witness.
    """
    src = (REPO_ROOT / "src" / "iai_mcp" / "lilli" / "cycle" / "sleep_pipeline.py").read_text()
    assert 'prior_mode="cold"' in src, (
        "sleep_pipeline.py crisis_recluster must call detect_communities "
        "with prior_mode=\"cold\" to honour the discard-prior invariant"
    )


def test_sleep_pipeline_does_not_use_run_leiden_directly() -> None:
    """The migration replaces the direct `_run_leiden` import with a
    `detect_communities` call so the prior_mode kwarg can flow through."""
    src = (REPO_ROOT / "src" / "iai_mcp" / "sleep_pipeline.py").read_text()
    assert "from iai_mcp.community import _run_leiden" not in src, (
        "sleep_pipeline.py must NOT import _run_leiden directly; use "
        "detect_communities with prior_mode='cold' instead"
    )


def test_retrieve_uses_seeded_mode() -> None:
    """retrieve.py is a normal recall path -- must thread
    `prior_mode='seeded'` (the default, but explicit is the witness)."""
    src = (REPO_ROOT / "src" / "iai_mcp" / "retrieve.py").read_text()
    assert 'prior_mode="seeded"' in src, (
        "retrieve.py must call detect_communities with prior_mode=\"seeded\""
    )


def test_sigma_uses_seeded_mode() -> None:
    """sigma.py is a normal recall-side path -- must thread
    `prior_mode='seeded'`."""
    src = (REPO_ROOT / "src" / "iai_mcp" / "sigma.py").read_text()
    assert 'prior_mode="seeded"' in src, (
        "sigma.py must call detect_communities with prior_mode=\"seeded\""
    )


def test_retrieve_continuity_preserves_uuids_across_unchanged_graph() -> None:
    """retrieval continuity contract: an unchanged graph re-run with
    its own prior must yield the same community UUIDs for every node.

    This is the test that catches the `init_partitions` wiring -- without
    threading the prior through `run_mosaic`, fresh `uuid4()` calls
    inside `_build_assignment` would silently break continuity.
    """
    g = _make_two_clique_graph()
    first = detect_communities(g, prior=None, prior_mode="seeded")
    second = detect_communities(g, prior=first, prior_mode="seeded")
    # Every node's community UUID must be preserved across the re-run.
    for node, comm_first in first.node_to_community.items():
        assert second.node_to_community[node] == comm_first, (
            f"node {node} community drifted: {comm_first} -> "
            f"{second.node_to_community[node]}"
        )


def test_sigma_community_count_stable_across_calls() -> None:
    """Sigma's community-count signal must be stable across re-runs of the
    same graph (otherwise the topology regime classifier would flap)."""
    g = _make_two_clique_graph()
    first = detect_communities(g, prior=None, prior_mode="seeded")
    second = detect_communities(g, prior=first, prior_mode="seeded")
    assert (
        len(set(second.node_to_community.values()))
        == len(set(first.node_to_community.values()))
    ), "community count must be stable across re-runs of an unchanged graph"
