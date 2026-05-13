""" RED: S4 sigma event-emission tests.

Constitutional contract:
- Developmental (N<500, sigma<1) -> kind=sigma_observation phase=developmental
  AND a profile_updated event for the Hebbian rate boost.
- Mid-life drift (N>=500, sigma<1) -> kind=sigma_drift.
- Healthy (sigma>=1) -> kind=sigma_observation phase=healthy.
- Insufficient data (N<200) -> kind=sigma_observation phase=insufficient_data.

sigma is NEVER a routing decision -- the regime classifier writes events only.
"""
from __future__ import annotations

import networkx as nx
import pytest

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore


def _seed_synthetic_graph(monkeypatch, *, n_nodes: int, sigma_val: float) -> None:
    """Stub sigma.compute_topology_snapshot to return a controlled snapshot."""
    from iai_mcp import sigma as sigma_mod

    def _fake_snapshot(graph):  # noqa: ARG001
        return {
            "C": 0.5,
            "L": 2.0,
            "sigma": sigma_val,
            "community_count": 3,
            "rich_club_ratio": 0.1,
            "N": n_nodes,
            "regime": sigma_mod.classify_regime(n_nodes, sigma_val),
        }

    monkeypatch.setattr(sigma_mod, "compute_topology_snapshot", _fake_snapshot)
    # Stub build_runtime_graph so we don't need a real LanceDB graph.
    from iai_mcp import retrieve

    def _fake_build(_store):
        g = nx.Graph()
        for i in range(n_nodes):
            g.add_node(i)
        return g, None, []

    monkeypatch.setattr(retrieve, "build_runtime_graph", _fake_build)


def test_compute_and_emit_developmental_phase_emits_sigma_observation(
    tmp_path, monkeypatch,
):
    """N=300, sigma=0.5 -> kind=sigma_observation, phase=developmental."""
    from iai_mcp import sigma as sigma_mod

    store = MemoryStore(path=tmp_path)
    _seed_synthetic_graph(monkeypatch, n_nodes=300, sigma_val=0.5)
    snap = sigma_mod.compute_and_emit(store)

    assert snap["regime"] == "developmental"
    events = query_events(store, kind="sigma_observation", limit=10)
    assert any(e["data"].get("phase") == "developmental" for e in events), (
        "developmental phase must emit kind=sigma_observation phase=developmental"
    )


def test_compute_and_emit_developmental_bumps_hebbian_rate(
    tmp_path, monkeypatch,
):
    """Developmental phase ALSO emits a profile_updated event for hebbian_rate."""
    from iai_mcp import sigma as sigma_mod

    store = MemoryStore(path=tmp_path)
    _seed_synthetic_graph(monkeypatch, n_nodes=300, sigma_val=0.5)
    sigma_mod.compute_and_emit(store)

    profile_events = query_events(store, kind="profile_updated", limit=10)
    hebbian_events = [
        e for e in profile_events if "hebbian" in str(e["data"].get("knob", "")).lower()
    ]
    assert hebbian_events, (
        "developmental phase must bump Hebbian rate via profile_updated"
    )


def test_compute_and_emit_mid_life_drift_emits_sigma_drift(
    tmp_path, monkeypatch,
):
    """N=600, sigma=0.5 -> kind=sigma_drift (S4 event)."""
    from iai_mcp import sigma as sigma_mod

    store = MemoryStore(path=tmp_path)
    _seed_synthetic_graph(monkeypatch, n_nodes=600, sigma_val=0.5)
    snap = sigma_mod.compute_and_emit(store)

    assert snap["regime"] == "mid_life_drift"
    events = query_events(store, kind="sigma_drift", limit=10)
    assert events, "mid-life drift must emit kind=sigma_drift"
    assert events[0]["data"]["sigma"] < 1.0


def test_compute_and_emit_healthy_emits_sigma_observation_healthy(
    tmp_path, monkeypatch,
):
    """sigma>=1 (any N>=floor) -> kind=sigma_observation phase=healthy."""
    from iai_mcp import sigma as sigma_mod

    store = MemoryStore(path=tmp_path)
    _seed_synthetic_graph(monkeypatch, n_nodes=300, sigma_val=2.5)
    snap = sigma_mod.compute_and_emit(store)

    assert snap["regime"] == "healthy"
    events = query_events(store, kind="sigma_observation", limit=10)
    assert any(e["data"].get("phase") == "healthy" for e in events)


def test_compute_and_emit_insufficient_data_below_floor(
    tmp_path, monkeypatch,
):
    """N<floor -> kind=sigma_observation phase=insufficient_data, no drift event."""
    from iai_mcp import sigma as sigma_mod

    store = MemoryStore(path=tmp_path)
    _seed_synthetic_graph(monkeypatch, n_nodes=50, sigma_val=None)
    snap = sigma_mod.compute_and_emit(store)

    assert snap["regime"] == "insufficient_data"
    drift_events = query_events(store, kind="sigma_drift", limit=10)
    assert not drift_events, "insufficient_data must NOT emit sigma_drift"
    obs_events = query_events(store, kind="sigma_observation", limit=10)
    assert any(e["data"].get("phase") == "insufficient_data" for e in obs_events)


def test_s4_run_offline_pass_calls_sigma_compute_and_emit(
    tmp_path, monkeypatch,
):
    """s4.run_offline_pass must invoke sigma.compute_and_emit."""
    from iai_mcp import s4

    called = {"n": 0}

    def _fake_emit(store):  # noqa: ARG001
        called["n"] += 1
        return {"regime": "healthy", "sigma": 1.5, "N": 250}

    monkeypatch.setattr("iai_mcp.sigma.compute_and_emit", _fake_emit)
    store = MemoryStore(path=tmp_path)
    out = s4.run_offline_pass(store)

    assert called["n"] == 1
    assert "sigma" in out
    assert out["sigma"]["regime"] == "healthy"


def test_s4_run_offline_pass_does_not_crash_on_sigma_failure(
    tmp_path, monkeypatch,
):
    """sigma raises -> run_offline_pass emits s4_error, does not propagate."""
    from iai_mcp import s4

    def _boom(_store):
        raise RuntimeError("synthetic sigma boom")

    monkeypatch.setattr("iai_mcp.sigma.compute_and_emit", _boom)
    store = MemoryStore(path=tmp_path)
    out = s4.run_offline_pass(store)

    assert "sigma" in out
    assert "error" in out["sigma"]
    err_events = query_events(store, kind="s4_error", limit=10)
    assert err_events, "failure must surface as kind=s4_error"
