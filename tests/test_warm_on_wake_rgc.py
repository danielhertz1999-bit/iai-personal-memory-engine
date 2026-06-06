"""RED-by-design hermetic tests for the two structural-cache cold-paths after wake.

TWO CASES:

CASE A — DROWSY-edge re-wake (primary path):
  After WAKE -> DROWSY (rgc invalidated) -> WAKE with NO sleep in between,
  the first daemon-side structural read lands on cold_degrade / last_good.
  The fix (not yet shipped) will kick a non-blocking background rebuild on the
  DROWSY edge, expose a completion Event, and return overlay/normal on the
  next load_recall_structural.

CASE B — interrupted-SLEEP re-warm via the wake-hook rebuild-if-cold helper:
  After an interrupted SLEEP cycle (RECALL_INDEX_REBUILD was skipped), the
  cache is cold.  The fix will provide a module-level sync helper that detects
  cold and rebuilds.  This is the ONLY automated guard on the wake-hook
  rebuild-if-cold code path.

The three seam symbols the background-rebuild fix MUST implement:

  1. ``runtime_graph_cache.rebuild_ready``
     A module-level ``threading.Event`` set when the DROWSY-edge background
     rebuild finishes (mirrors ``preload_ready`` in runtime_graph_cache).

  2. ``daemon._kick_drowsy_rgc_rebuild(store)``
     Module-level importable, NON-BLOCKING trigger for the DROWSY-edge primary
     path (mirrors ``_run_drowsy_drain``).  Returns immediately; the background
     worker sets ``rebuild_ready`` in a ``finally``.

  3. ``daemon._wake_hook_rebuild_if_cold(store)``
     Module-level importable SYNC helper for the wake-hook rebuild-if-cold path.
     Reads ``load_recall_structural(store)``; when
     ``structural_source in {"cold_degrade", "last_good"}`` calls
     ``runtime_graph_cache._rebuild_and_save_rgc(store)``.

CONTRACT for the background worker:
  The worker MUST invoke the rebuild via the MODULE ATTRIBUTE
  ``runtime_graph_cache._rebuild_and_save_rgc(store)`` (NOT a name bound at
  import time) so the test's monkeypatch gate reaches the worker thread.

CONTRACT for the wake-hook helper:
  The helper must SKIP (no rebuild, no generation advance) when the cache is
  already overlay/normal — avoid rebuilding on a warm cache and extending the
  exclusive-lock window unnecessarily.

Hermetic: tmp store root, IAI_MCP_STORE env, generic data, no live-daemon touch.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make


# ---------------------------------------------------------------------------
# Hermetic env fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path: Path):
    """Redirect HOME + IAI_MCP_STORE + IAI_DAEMON_SOCKET_PATH to tmp."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    yield


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    from iai_mcp.types import EMBED_DIM
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _build_thick_store(tmp_path: Path):
    """Build a store thick enough that a real rebuild yields overlay/normal.

    Inserts enough records so that detect_communities + rich_club_nodes can
    produce a non-empty assignment.  The caller asserts overlay/normal before
    invalidating to confirm the fixture is thick enough.
    """
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store_root = tmp_path / "store"
    store = MemoryStore(str(store_root))

    # Insert 15 records with distinct normalised embeddings.
    for i in range(15):
        store.insert(_make(text=f"User record {i}", vec=_norm_vec(i + 100)))
    flush_record_buffer(store)
    return store, store_root


def _warm_store_via_rebuild(store) -> None:
    """Drive the real nightly rebuild step to produce a warm overlay snapshot.

    Uses the exact body the fix will factor into _rebuild_and_save_rgc.
    After this call, load_recall_structural should return overlay/normal —
    proving the fixture is thick enough before any invalidation.
    """
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)
    assert done is True, f"Rebuild step must return done=True; got done={done}, payload={payload}"


# ---------------------------------------------------------------------------
# CASE A: DROWSY-edge re-wake — primary cold path
# ---------------------------------------------------------------------------

def test_drowsy_rewake_cold_then_rebuild_ready(monkeypatch, tmp_path):
    """Structural cache is cold after DROWSY invalidate -> re-wake; rebuild restores it.

    Today the test fails at the getattr-guard (seam symbols absent) — confirming
    RED.  Once the background-rebuild seam ships, the test turns GREEN.
    """
    from iai_mcp import runtime_graph_cache

    store, _ = _build_thick_store(tmp_path)

    # Step 1: Build a WARM snapshot via the real rebuild body.
    _warm_store_via_rebuild(store)

    # Step 2: Assert warm baseline (proves fixture thickness).
    _, _, _, src = runtime_graph_cache.load_recall_structural(store)
    assert src in ("overlay", "normal"), (
        f"Warm baseline must be overlay/normal before invalidation; got {src!r}. "
        "Fixture may be too thin — add more records/edges."
    )

    # Step 3: Simulate the DROWSY edge: invalidate (path.unlink()) — exactly what
    # the wake sequence does when pending embeds exist.
    runtime_graph_cache.invalidate(store)

    # Assert cold precondition: after invalidate the cache is cold.
    _, _, _, src_cold = runtime_graph_cache.load_recall_structural(store)
    assert src_cold in ("cold_degrade", "last_good"), (
        f"After invalidate the cache must be cold; got {src_cold!r}."
    )

    # -------------------------------------------------------------------------
    # Seam guard: all three symbols must exist before the non-blocking test.
    # getattr-guard each; pytest.fail (RED, not skip) if any is absent.
    # -------------------------------------------------------------------------
    kick_fn = getattr(runtime_graph_cache, "_rebuild_and_save_rgc", None)
    kick_daemon = getattr(
        __import__("iai_mcp.daemon", fromlist=["_kick_drowsy_rgc_rebuild"]),
        "_kick_drowsy_rgc_rebuild",
        None,
    )
    rebuild_ready_event = getattr(runtime_graph_cache, "rebuild_ready", None)

    if kick_fn is None:
        pytest.fail(
            "seam not yet implemented: runtime_graph_cache._rebuild_and_save_rgc"
        )
    if kick_daemon is None:
        pytest.fail(
            "seam not yet implemented: daemon._kick_drowsy_rgc_rebuild"
        )
    if rebuild_ready_event is None:
        pytest.fail(
            "seam not yet implemented: runtime_graph_cache.rebuild_ready (threading.Event)"
        )

    # -------------------------------------------------------------------------
    # Step 4: DETERMINISTIC non-blocking assertion via test-held gate.
    #
    # Monkeypatch runtime_graph_cache._rebuild_and_save_rgc to insert a gate so
    # "returned-but-not-yet-done" is provable without any time.sleep() race.
    #
    # CONTRACT: the background worker MUST call the rebuild via the MODULE
    # ATTRIBUTE runtime_graph_cache._rebuild_and_save_rgc(store), NOT a name
    # bound at import time, so this monkeypatch reaches the worker.
    # -------------------------------------------------------------------------
    gate = threading.Event()
    real_rebuild = runtime_graph_cache._rebuild_and_save_rgc  # type: ignore[attr-defined]

    def _gated_rebuild(s):
        gate.wait(timeout=10)
        return real_rebuild(s)

    monkeypatch.setattr(runtime_graph_cache, "_rebuild_and_save_rgc", _gated_rebuild)

    # Reset the completion event (it may have been set by the prior warm call).
    rebuild_ready_event.clear()  # type: ignore[union-attr]

    # Kick the non-blocking DROWSY-edge rebuild.
    import iai_mcp.daemon as _daemon_mod
    _daemon_mod._kick_drowsy_rgc_rebuild(store)  # type: ignore[attr-defined]

    # Assert it returned immediately while the worker is still gated.
    assert not rebuild_ready_event.is_set(), (  # type: ignore[union-attr]
        "rebuild_ready must NOT be set while the worker is blocked on the gate "
        "(the kick must be non-blocking — flag-not-gate design)."
    )

    # Release the gate and wait for the real rebuild to complete.
    gate.set()
    assert rebuild_ready_event.wait(timeout=10), (  # type: ignore[union-attr]
        "rebuild_ready must be set after the background rebuild completes."
    )

    # Step 5: RED assertion — warm after rebuild (GREEN once seam ships).
    _, _, _, src_after = runtime_graph_cache.load_recall_structural(store)
    assert src_after in ("overlay", "normal"), (
        f"After DROWSY-edge rebuild, structural_source must be overlay/normal; "
        f"got {src_after!r}."
    )


# ---------------------------------------------------------------------------
# CASE B: interrupted-SLEEP re-warm via the wake-hook rebuild-if-cold helper
# ---------------------------------------------------------------------------

def test_wake_hook_rebuilds_cold_cache(monkeypatch, tmp_path):
    """The wake-hook rebuild-if-cold helper rebuilds a cold cache.

    Simulates an interrupted SLEEP cycle where the topology rebuild was skipped
    and the cache is cold.  Today the test fails at the getattr-guard
    (_wake_hook_rebuild_if_cold absent) — confirming RED.  Once the helper
    ships, the test turns GREEN.

    This is the ONLY automated guard on the wake-hook rebuild-if-cold code path.
    """
    from iai_mcp import runtime_graph_cache

    store, _ = _build_thick_store(tmp_path)

    # Build a warm snapshot first (proves the fixture is thick enough).
    _warm_store_via_rebuild(store)

    # Simulate interrupted cycle: invalidate so the cache is cold.
    runtime_graph_cache.invalidate(store)

    # Assert cold precondition.
    _, _, _, src_cold = runtime_graph_cache.load_recall_structural(store)
    assert src_cold in ("cold_degrade", "last_good"), (
        f"After invalidate, cache must be cold; got {src_cold!r}."
    )

    # Seam guard.
    import iai_mcp.daemon as _daemon_mod
    helper = getattr(_daemon_mod, "_wake_hook_rebuild_if_cold", None)
    if helper is None:
        pytest.fail(
            "seam not yet implemented: daemon._wake_hook_rebuild_if_cold"
        )

    # RED assertion: helper detects cold and rebuilds (GREEN once seam ships).
    _daemon_mod._wake_hook_rebuild_if_cold(store)  # type: ignore[attr-defined]

    _, _, _, src_after = runtime_graph_cache.load_recall_structural(store)
    assert src_after in ("overlay", "normal"), (
        f"After wake-hook rebuild-if-cold, structural_source must be overlay/normal; "
        f"got {src_after!r}."
    )


def test_wake_hook_skips_when_warm(monkeypatch, tmp_path):
    """Wake-hook rebuild-if-cold helper skips rebuild on a warm cache.

    The helper must NOT rebuild when the cache is already overlay/normal —
    rebuilding unnecessarily extends the exclusive-lock window.
    Today the test fails at the getattr-guard — confirming RED.
    Once the helper ships, the skip-when-warm invariant must hold.
    """
    from iai_mcp import runtime_graph_cache

    store, _ = _build_thick_store(tmp_path)

    # Build a warm snapshot.
    _warm_store_via_rebuild(store)

    # Assert warm baseline.
    _, _, _, src_warm = runtime_graph_cache.load_recall_structural(store)
    assert src_warm in ("overlay", "normal"), (
        f"Warm baseline must be overlay/normal; got {src_warm!r}."
    )

    # Capture current generation before calling the helper.
    gen_before = runtime_graph_cache.get_current_generation()

    # Seam guard.
    import iai_mcp.daemon as _daemon_mod
    helper = getattr(_daemon_mod, "_wake_hook_rebuild_if_cold", None)
    if helper is None:
        pytest.fail(
            "seam not yet implemented: daemon._wake_hook_rebuild_if_cold"
        )

    # RED assertion: helper skips on a warm cache (GREEN once seam ships).
    _daemon_mod._wake_hook_rebuild_if_cold(store)  # type: ignore[attr-defined]

    # Must still be warm and generation must NOT have advanced.
    _, _, _, src_after = runtime_graph_cache.load_recall_structural(store)
    assert src_after in ("overlay", "normal"), (
        f"Cache must remain warm after helper called on warm cache; got {src_after!r}."
    )
    gen_after = runtime_graph_cache.get_current_generation()
    assert gen_after == gen_before, (
        f"Helper must NOT advance the generation when cache is already warm; "
        f"before={gen_before}, after={gen_after}."
    )
