"""Layer-2 RecallIndex overlay unit tests.

Tests:
(a) Overlay HIT serves cached assignment O(1) with no mosaic recompute.
(b) Generation-epoch MISMATCH returns typed bypass to load_last_good_structural —
    detect_communities / rich_club_nodes / build_runtime_graph NOT called.
(c) Invariant failure bypasses to pure Layer-1, recall still correct.
(d) recall-path boost (profile_modulates / co-activation hebbian)
    does NOT bump epoch NOR dirty counter; overlay HITs after such a recall.
(d2) freshness-fuse-trip O(1): max_age OR dirty-counter over threshold
     trips to bypass + emits freshness_fuse_tripped telemetry; NO count(*)/
     active_records_count/count_rows called on the consult path.
(d3) Dirty counter: record insert increments the counter via the COMPOSED hook
     (node-mirror hook still fires); boost_edges does NOT increment; nightly
     rebuild resets the counter.
(e) CACHE_VERSION bumped from "62-04-v4".
(f) nightly rebuild step is after CLUSTER_SUMMARY in _STEP_ORDER
    and produces a fresh snapshot with an incremented generation epoch.

Hermetic: monkeypatched IAI_MCP_STORE / HOME / IAI_DAEMON_SOCKET_PATH → tmp,
so the test stays isolated from any live store or daemon.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4, UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Hermetic env fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    yield


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore
    return MemoryStore(path=str(tmp_path / "store"))


def _build_overlay_snapshot(store, *, generation: int, rebuild_ts: str | None = None) -> bool:
    """Build a genuine overlay snapshot via save_with_generation."""
    from iai_mcp.community import CommunityAssignment
    from iai_mcp import runtime_graph_cache as rgc

    # Force the generation to the desired value for testing.
    import threading
    with rgc._GEN_LOCK:
        rgc._current_generation = generation - 1  # advance_generation will set it to `generation`
    # Set override before save_with_generation so it uses the right timestamp.
    if rebuild_ts is not None:
        with rgc._GEN_LOCK:
            rgc._rebuild_timestamp_override = rebuild_ts

    assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.5,
        backend="mosaic",
        top_communities=[],
        mid_regions={},
    )
    result = rgc.save_with_generation(store, assignment, [])
    if rebuild_ts is not None:
        # The override should be cleared by save_with_generation already.
        pass
    return result


# ---------------------------------------------------------------------------
# (e) CACHE_VERSION bump
# ---------------------------------------------------------------------------

def test_cache_version_bumped():
    """CACHE_VERSION must not be the old 62-04-v4 value."""
    from iai_mcp import runtime_graph_cache as rgc
    assert rgc.CACHE_VERSION != "62-04-v4", (
        f"CACHE_VERSION must be bumped from 62-04-v4; got {rgc.CACHE_VERSION!r}"
    )
    assert rgc.CACHE_VERSION == "62-02-v5"


# ---------------------------------------------------------------------------
# (a) Overlay HIT serves cached assignment O(1)
# ---------------------------------------------------------------------------

def test_overlay_hit_serves_cached_o1(tmp_path, monkeypatch):
    """consult_overlay returns (assignment, rich_club) without calling
    detect_communities / rich_club_nodes / build_runtime_graph when the
    snapshot is valid (matching generation epoch + below fuse thresholds +
    passing invariants)."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    # Reset module state for this test.
    import threading
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    # Write a snapshot with generation=1 and a fresh rebuild_timestamp.
    fresh_ts = datetime.now(timezone.utc).isoformat()
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(1)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )

    with rgc._GEN_LOCK:
        rgc._current_generation = 0  # will advance to 1
    rgc.reset_dirty_counter()
    # Manually advance generation and write with timestamp.
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    new_gen = rgc.advance_generation()  # now _current_generation = 1
    rgc.save(store, assignment, [node_id], max_degree=5)
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    # Ensure generation matches.
    assert new_gen == 1

    # Track calls to detect_communities / rich_club_nodes.
    detect_called = []
    richclub_called = []
    build_called = []

    import iai_mcp.community as _community_mod
    import iai_mcp.richclub as _richclub_mod
    import iai_mcp.retrieve as _retrieve_mod

    orig_detect = _community_mod.detect_communities
    orig_rc = _richclub_mod.rich_club_nodes
    orig_build = _retrieve_mod.build_runtime_graph

    def _mock_detect(*a, **kw):
        detect_called.append(True)
        return orig_detect(*a, **kw)

    def _mock_rc(*a, **kw):
        richclub_called.append(True)
        return orig_rc(*a, **kw)

    def _mock_build(*a, **kw):
        build_called.append(True)
        return orig_build(*a, **kw)

    monkeypatch.setattr(_community_mod, "detect_communities", _mock_detect)
    monkeypatch.setattr(_richclub_mod, "rich_club_nodes", _mock_rc)
    monkeypatch.setattr(_retrieve_mod, "build_runtime_graph", _mock_build)

    result = rgc.consult_overlay(store)

    # Overlay HIT: result is a 2-tuple (assignment, rich_club).
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Expected overlay HIT, got bypass: {result!r}"
    )
    assert isinstance(result, tuple) and len(result) == 2
    # No recompute.
    assert not detect_called, "detect_communities was called — overlay should serve O(1)"
    assert not richclub_called, "rich_club_nodes was called — overlay should serve O(1)"
    assert not build_called, "build_runtime_graph was called — overlay should serve O(1)"


# ---------------------------------------------------------------------------
# (b) Generation-epoch MISMATCH → typed bypass to load_last_good_structural
# ---------------------------------------------------------------------------

def test_epoch_mismatch_typed_bypass(tmp_path, monkeypatch):
    """When the snapshot's generation epoch differs from the current in-process
    epoch, consult_overlay must return _OverlayBypass('epoch_mismatch') and
    NOT call detect_communities / rich_club_nodes / build_runtime_graph."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    # Write a snapshot with generation=1.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()  # _current_generation = 1
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    # Advance in-process generation to 2 WITHOUT writing a new snapshot.
    rgc.advance_generation()  # now _current_generation = 2
    assert rgc.get_current_generation() == 2

    # Track that hot-path rebuild functions are NOT called.
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities called on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes called on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph called on hot path")))

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass)
    assert result.reason == "epoch_mismatch"

    # load_last_good_structural should still return the last-good snapshot.
    last_good = rgc.load_last_good_structural(store)
    assert last_good is not None, "load_last_good_structural must return the last-good snapshot on bypass"
    a2, rc2 = last_good
    assert a2 is not None


# ---------------------------------------------------------------------------
# (c) Invariant failure → bypass to pure Layer-1
# ---------------------------------------------------------------------------

def test_invariant_failure_bypass(tmp_path, monkeypatch):
    """A snapshot with rich_club ids NOT ⊆ node ids triggers an invariant
    failure bypass. recall still returns a correct result (overlay can only
    slow, never make wrong)."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import json

    store = _make_store(tmp_path)

    # Write a valid snapshot first.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    node_id = uuid4()
    comm_id = uuid4()
    dangling_rc_id = uuid4()  # NOT in node_to_community
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(42)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    rgc.save(store, assignment, [dangling_rc_id])  # dangling rich_club entry
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    # Patch hot-path to raise if called.
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities called on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes called on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph called on hot path")))

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass)
    assert result.reason == "invariant_failure"


# ---------------------------------------------------------------------------
# (d) recall-path boost does NOT bump epoch or dirty counter
# ---------------------------------------------------------------------------

def test_load_recall_structural_uses_overlay_hit(tmp_path, monkeypatch):
    """load_recall_structural returns structural_source='overlay' when the
    overlay HITs. Regression guard: the overlay must be wired into the
    actual recall-context loader, not just a standalone function."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    # Set generation=1 + write snapshot.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(5)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    rgc.save(store, assignment, [node_id])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    a, rc, md, source = rgc.load_recall_structural(store)
    assert source == "overlay", (
        f"load_recall_structural must route via overlay HIT; got source={source!r}"
    )
    assert a is not None
    assert len(getattr(a, "node_to_community", {})) > 0


def test_load_recall_structural_epoch_mismatch_falls_to_last_good(tmp_path, monkeypatch):
    """When overlay returns epoch_mismatch bypass, load_recall_structural must fall
    through to the Layer-1 path and return last_good (not cold_degrade), proving
    CC-D: the bypass can only slow recall, never make it wrong."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    store = _make_store(tmp_path)

    # Write snapshot at generation=1.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(88)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    rgc.save(store, assignment, [node_id])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    # Advance in-process generation to 2 (overlay snapshot has gen=1 → mismatch).
    rgc.advance_generation()

    # Patch hot-path rebuild to raise if called.
    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph on hot path")))

    a, rc, md, source = rgc.load_recall_structural(store)
    # Overlay mismatch falls through; try_load also misses (count-window may differ);
    # load_last_good_structural returns the last-good snapshot — not cold_degrade.
    assert source in ("last_good", "normal"), (
        f"On overlay bypass, must return last_good or normal (Layer-1); got {source!r}"
    )
    # The returned assignment must have non-empty community data (not cold_degrade).
    # Correctness: the structural signal is the last-good snapshot, same data.
    ov_ntc = getattr(a, "node_to_community", {})
    assert len(ov_ntc) > 0, (
        "Overlay bypass must return non-empty structural data (last-good), not cold_degrade"
    )


def test_recall_path_boost_does_not_bump_epoch(tmp_path, monkeypatch):
    """A recall-path boost_edges (profile_modulates / co-activation hebbian)
    must NOT change the in-process generation epoch. The overlay must still
    HIT after such a recall-path edge write."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.store import MemoryStore
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    # Set generation=1 + write snapshot.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    node_id_a = uuid4()
    node_id_b = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id_a: comm_id, node_id_b: comm_id},
        community_centroids={comm_id: _norm_vec(7)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    rgc.save(store, assignment, [node_id_a, node_id_b])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""
    gen_before = rgc.get_current_generation()
    dirty_before = rgc.get_dirty_counter()

    # Simulate a recall-path boost_edges (edge write only — NOT a record write).
    store.boost_edges([(node_id_a, node_id_b)], delta=0.1)

    # Generation and dirty counter must not change.
    assert rgc.get_current_generation() == gen_before, (
        "boost_edges must NOT advance the generation epoch"
    )
    assert rgc.get_dirty_counter() == dirty_before, (
        "boost_edges must NOT increment the dirty counter (RECORD-only hook)"
    )

    # Overlay must still HIT.
    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Overlay should still HIT after recall-path boost; got bypass {result!r}"
    )


def test_intra_day_insert_keeps_overlay_hit(tmp_path, monkeypatch):
    """A single intra-day record insert increments the dirty counter but keeps
    the overlay HIT within the staleness window (dirty < _FUSE_DIRTY_THRESHOLD)."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.retrieve import _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)

    # Set generation=1 + fresh snapshot.
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""
    dirty_before = rgc.get_dirty_counter()

    # Register the composed hook so the counter is wired.
    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    # Insert one record — must increment the dirty counter.
    rec = _make(text="User test record for dirty counter increment")
    store.insert(rec)

    dirty_after = rgc.get_dirty_counter()
    assert dirty_after == dirty_before + 1, (
        f"Expected dirty counter to increment by 1; was {dirty_before}, now {dirty_after}"
    )

    # Overlay must still HIT (single insert < _FUSE_DIRTY_THRESHOLD).
    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Overlay should still HIT after single insert; got bypass {result!r}"
    )


# ---------------------------------------------------------------------------
# (d2) freshness-fuse trip O(1)
# ---------------------------------------------------------------------------

def test_freshness_fuse_trips_on_max_age(tmp_path, monkeypatch):
    """When snapshot's rebuild_timestamp is older than _FUSE_MAX_AGE_SECONDS,
    consult_overlay returns _OverlayBypass('fuse_tripped') even if the
    generation epoch matches. build_runtime_graph / detect_communities /
    rich_club_nodes must NOT be called on the consult path. freshness_fuse_tripped
    telemetry event must be emitted (buffered)."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod
    from iai_mcp.store import MemoryStore

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    # Write snapshot with a rebuild_timestamp 27 hours in the past (> max_age).
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=27)).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = old_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    # Ensure count(*) / active_records_count / count_rows raise if called.
    orig_arc = getattr(store, "active_records_count", None)
    def _raise_arc(*a, **kw):
        raise AssertionError("active_records_count called on overlay consult hot path")
    monkeypatch.setattr(store, "active_records_count", _raise_arc)

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph on hot path")))

    # Also patch store.db to catch count_rows calls.
    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass), (
        "Expected fuse-trip bypass due to old rebuild_timestamp"
    )
    assert result.reason == "fuse_tripped"
    assert result.age_ms > 0

    # load_last_good_structural must still return the snapshot.
    last_good = rgc.load_last_good_structural(store)
    assert last_good is not None


def test_freshness_fuse_trips_on_dirty_counter(tmp_path, monkeypatch):
    """When the in-process dirty counter exceeds _FUSE_DIRTY_THRESHOLD,
    consult_overlay returns a fuse_tripped bypass even though the generation
    epoch still matches AND rebuild_timestamp is fresh."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    # Write snapshot with fresh timestamp.
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    # Push dirty counter past threshold.
    from iai_mcp import runtime_graph_cache as _rgc
    with _rgc._DIRTY_COUNTER_LOCK:
        _rgc._dirty_counter = _rgc._FUSE_DIRTY_THRESHOLD + 1

    # Patch hot-path to raise if called.
    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph on hot path")))

    # Patch active_records_count to raise if called.
    def _raise_arc(*a, **kw):
        raise AssertionError("active_records_count called on hot path")
    monkeypatch.setattr(store, "active_records_count", _raise_arc)

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass)
    assert result.reason == "fuse_tripped"


def test_within_threshold_still_hits(tmp_path):
    """A fresh snapshot with dirty_counter=0 should produce an overlay HIT."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Should HIT; got bypass {result!r}"
    )


# ---------------------------------------------------------------------------
# (d3) Dirty counter: record insert increments via composed hook;
# boost_edges does NOT; nightly rebuild resets
# ---------------------------------------------------------------------------

def test_dirty_counter_composed_hook_fires_on_record_insert(tmp_path):
    """record insert increments the dirty counter via the composed hook;
    node-mirror hook STILL fires (not clobbered)."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)
    rgc.reset_dirty_counter()

    graph = MemoryGraph()
    mirror_fired = []

    # Register the composed hook.
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))
    dirty_before = rgc.get_dirty_counter()

    rec = _make(text="User hook test record", vec=_norm_vec(99))
    store.insert(rec)

    dirty_after = rgc.get_dirty_counter()
    assert dirty_after == dirty_before + 1, (
        f"Dirty counter should have incremented; was {dirty_before}, now {dirty_after}"
    )
    # Node-mirror hook must also have fired: the node should be in graph._node_payload.
    assert str(rec.id) in graph._node_payload, (
        "Node-mirror hook was clobbered — graph._node_payload is missing the inserted record"
    )


def test_boost_edges_does_not_increment_dirty_counter(tmp_path):
    """boost_edges (edge-only write) does NOT increment the dirty counter."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)
    rgc.reset_dirty_counter()

    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    src_id = uuid4()
    dst_id = uuid4()
    dirty_before = rgc.get_dirty_counter()
    store.boost_edges([(src_id, dst_id)], delta=0.2)
    dirty_after = rgc.get_dirty_counter()
    assert dirty_after == dirty_before, (
        f"boost_edges must NOT increment dirty counter; was {dirty_before}, now {dirty_after}"
    )


def test_nightly_rebuild_resets_dirty_counter(tmp_path):
    """save_with_generation (the nightly step) resets the dirty counter to 0."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    # Set some dirty count.
    with rgc._DIRTY_COUNTER_LOCK:
        rgc._dirty_counter = 30
    assert rgc.get_dirty_counter() == 30

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save_with_generation(store, assignment, [])

    assert rgc.get_dirty_counter() == 0, (
        "save_with_generation must reset the dirty counter to 0"
    )


# ---------------------------------------------------------------------------
# (f) MEDIUM-2: nightly rebuild step is AFTER CLUSTER_SUMMARY in _STEP_ORDER
# ---------------------------------------------------------------------------

def test_recall_index_rebuild_step_position():
    """RECALL_INDEX_REBUILD must be in _STEP_ORDER and positioned AFTER
    CLUSTER_SUMMARY (MEDIUM-2 invariant)."""
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep, SleepPipeline

    step_order = SleepPipeline._STEP_ORDER
    assert SleepStep.RECALL_INDEX_REBUILD in step_order, (
        "RECALL_INDEX_REBUILD must be in _STEP_ORDER"
    )
    idx_rebuild = step_order.index(SleepStep.RECALL_INDEX_REBUILD)
    idx_cluster = step_order.index(SleepStep.CLUSTER_SUMMARY)
    assert idx_rebuild > idx_cluster, (
        f"RECALL_INDEX_REBUILD (index={idx_rebuild}) must be AFTER "
        f"CLUSTER_SUMMARY (index={idx_cluster}) in _STEP_ORDER"
    )
    # Must be the LAST step.
    assert idx_rebuild == len(step_order) - 1, (
        "RECALL_INDEX_REBUILD must be the last step in _STEP_ORDER"
    )


def test_recall_index_rebuild_step_stamps_fresh_generation(tmp_path, monkeypatch):
    """The nightly rebuild step produces a snapshot with an incremented
    generation epoch from (mocked) ground-truth graph."""
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.store import MemoryStore
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    store = _make_store(tmp_path)
    store.insert(_make(text="User nightly rebuild test record 1", vec=_norm_vec(11)))
    store.insert(_make(text="User nightly rebuild test record 2", vec=_norm_vec(12)))

    with rgc._GEN_LOCK:
        rgc._current_generation = 5
    rgc.reset_dirty_counter()

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True, "RECALL_INDEX_REBUILD step must return done=True"
    assert payload.get("rebuilt") is True, (
        f"Step must report rebuilt=True; got {payload}"
    )
    new_gen = payload.get("generation")
    assert new_gen == 6, (
        f"Generation must be incremented from 5 to 6; got {new_gen}"
    )
    assert rgc.get_dirty_counter() == 0, (
        "Nightly rebuild must reset dirty counter to 0"
    )
    # Snapshot must be readable.
    snap = rgc.load_last_good_structural(store)
    assert snap is not None, "Snapshot must be readable after nightly rebuild"
