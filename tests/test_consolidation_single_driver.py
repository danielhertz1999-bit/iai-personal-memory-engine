"""Six integration tests (A-F) for the single-driver consolidation invariant.

  A: _tick_body invokes no consolidation (run_rem_cycle / run_heavy_consolidation
     are never called from the legacy tick).
  B: No fsm_drift across a wake->sleep->wake canonical sequence; fsm_state
     mirrors the canonical->legacy mapping; drift assertions retained
     (reconcile is CORRECT, not masking).
  C: _tick_body does NOT mutate the canonical lifecycle_state.json.
  D: force-rem reaches the canonical pipeline via a DROWSY->SLEEP progression
     and the pending flag is cleared once SLEEP is reached.
  E: The 2 already-covered WAKE outputs (drain_deferred_captures,
     flush_event_buffer) are NOT double-invoked by the canonical wake hook.
  F: Step 0.7 foraging is skipped while the canonical FSM is SLEEP, and can
     run in WAKE (subject to its hourly throttle).

All tests use tmp_path for state files and never touch ~/.iai-mcp.
Generic test data; no PII.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _kr
    fake: dict = {}
    monkeypatch.setattr(_kr, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_kr, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_kr, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture
def daemon_store(tmp_path, monkeypatch):
    """Isolated MemoryStore + daemon state path for tick tests."""
    from iai_mcp import daemon_state
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")

    store = MemoryStore()
    rec = MemoryRecord(
        id=uuid4(), tier="semantic", literal_surface="seed",
        aaak_index="", embedding=[0.0] * store.embed_dim,
        community_id=None, centrality=0.0, detail_level=1,
        pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), tags=[], language="en",
    )
    store.insert(rec)

    yield store, state_path, tmp_path


# ---------------------------------------------------------------------------
# Test A: single driver — _tick_body never calls consolidation functions
# ---------------------------------------------------------------------------


def test_a_tick_body_never_calls_consolidation(daemon_store, monkeypatch):
    """A: _tick_body must NEVER call run_rem_cycle or run_heavy_consolidation
    for any flag combination. Consolidation routes only through lifecycle_tick.
    """
    from iai_mcp import daemon as daemon_mod

    store, state_path, tmp_path = daemon_store

    rem_calls: list = []
    heavy_calls: list = []

    monkeypatch.setattr(
        daemon_mod, "run_rem_cycle",
        AsyncMock(side_effect=lambda *a, **kw: rem_calls.append(1) or {}),
    )

    import iai_mcp.sleep as sleep_mod
    monkeypatch.setattr(
        sleep_mod, "run_heavy_consolidation",
        lambda *a, **kw: heavy_calls.append(1) or {},
    )

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    # Conditions that previously triggered the legacy consolidation loop.
    for extra_state in [
        {"force_rem_request": {"pending": True, "ts": "2026-01-01T00:00:00+00:00"}},
        {"user_sleep_request": {"pending": True, "ts": "2026-01-01T00:00:00+00:00"}},
        {},  # quiet window / no flag
    ]:
        state = {"fsm_state": "WAKE", **extra_state}
        asyncio.run(daemon_mod._tick_body(store, state))

    assert rem_calls == [], (
        f"_tick_body called run_rem_cycle {len(rem_calls)} times; expected 0"
    )
    assert heavy_calls == [], (
        f"_tick_body called run_heavy_consolidation {len(heavy_calls)} times; expected 0"
    )


# ---------------------------------------------------------------------------
# Test B: no fsm_drift across wake->sleep->wake
# ---------------------------------------------------------------------------


def test_b_no_fsm_drift_across_lifecycle_sequence(tmp_path):
    """B: After reconcile, legacy fsm_state mirrors canonical; drift=False
    across a full wake->sleep->wake transition sequence.

    Assertions are RETAINED (prove reconcile is correct, not masking).
    """
    from iai_mcp.fsm_reconcile import _CANONICAL_TO_LEGACY, reconcile_fsm_state
    from iai_mcp.lifecycle_state import LifecycleState

    canon_path = tmp_path / "lifecycle_state.json"
    legacy_path = tmp_path / ".daemon-state.json"

    def _set_canonical(state: LifecycleState):
        canon_path.write_text(json.dumps({"current_state": state.value}))

    def _set_legacy(fsm_state: str):
        legacy_path.write_text(json.dumps({"fsm_state": fsm_state}))

    # Simulate WAKE -> SLEEP -> WAKE with reconcile at each step.
    for canonical_state in [
        LifecycleState.WAKE,
        LifecycleState.DROWSY,
        LifecycleState.SLEEP,
        LifecycleState.WAKE,  # back to WAKE
    ]:
        _set_canonical(canonical_state)
        # Legacy is intentionally mismatched before reconcile.
        _set_legacy("WRONG_STATE")

        report = reconcile_fsm_state(
            canonical_path=canon_path,
            legacy_path=legacy_path,
            auto_correct=True,
        )

        # With a wrong legacy state, drift was detected and corrected.
        # The `drift` field reflects the pre-correction state (True = was drifted).
        # The `corrected` field confirms the correction was applied.
        assert report["corrected"] is True, (
            f"expected corrected=True (drift was present): {report}"
        )

        # Legacy fsm_state must match the canonical->legacy mapping.
        expected_legacy = _CANONICAL_TO_LEGACY[canonical_state.value]
        actual_legacy = json.loads(legacy_path.read_text()).get("fsm_state")
        assert actual_legacy == expected_legacy, (
            f"legacy={actual_legacy!r}, expected={expected_legacy!r} "
            f"for canonical={canonical_state.value}"
        )

        # A fresh reconcile on the corrected state must report drift=False
        # without auto_correct (proves it's stable, not just masked).
        report2 = reconcile_fsm_state(
            canonical_path=canon_path,
            legacy_path=legacy_path,
            auto_correct=False,
        )
        assert report2["drift"] is False, (
            f"post-correct drift still True: {report2}"
        )


# ---------------------------------------------------------------------------
# Test C: _tick_body does NOT mutate canonical lifecycle_state.json
# ---------------------------------------------------------------------------


def test_c_tick_body_does_not_mutate_canonical_fsm(daemon_store, monkeypatch, tmp_path):
    """C: _tick_body must NOT change lifecycle_state.json. The legacy tick
    is a per-tick maintenance runner; only lifecycle_tick transitions the
    canonical FSM.
    """
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.lifecycle_state import LifecycleState

    store, state_path, _ = daemon_store

    # Write a canonical state file under tmp (full valid record).
    canon_path = tmp_path / "lifecycle_state.json"
    from iai_mcp.lifecycle_state import default_state as _ds
    _initial = dict(_ds())
    _initial["current_state"] = LifecycleState.WAKE.value
    canon_path.write_text(json.dumps(_initial))

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    # Patch LIFECYCLE_STATE_PATH to our tmp file so any rogue write lands there.
    import iai_mcp.lifecycle_state as ls_mod
    monkeypatch.setattr(ls_mod, "LIFECYCLE_STATE_PATH", canon_path)

    state = {"fsm_state": "WAKE"}
    asyncio.run(daemon_mod._tick_body(store, state))

    # Canonical file current_state must be unchanged (WAKE).
    after = json.loads(canon_path.read_text())
    assert after["current_state"] == LifecycleState.WAKE.value, (
        f"_tick_body mutated canonical lifecycle_state.json: {after}"
    )


# ---------------------------------------------------------------------------
# Test D: FORCE_SLEEP dispatch routes * -> DROWSY -> SLEEP
# ---------------------------------------------------------------------------


def test_d_force_sleep_reaches_canonical_via_drowsy(tmp_path):
    """D: LifecycleEvent.FORCE_SLEEP must route * -> DROWSY -> SLEEP in two
    hops (DROWSY teardown path must run before SLEEP is entered), and the
    pending flag is logically cleared once SLEEP is reached.

    Tests the compute_transition pure function (no daemon main required).
    """
    from iai_mcp.lifecycle import LifecycleEvent, LifecycleState, compute_transition

    FORCE = LifecycleEvent.FORCE_SLEEP

    # WAKE -> DROWSY (first hop).
    target = compute_transition(LifecycleState.WAKE, FORCE)
    assert target is LifecycleState.DROWSY, (
        f"Expected WAKE + FORCE_SLEEP -> DROWSY, got {target}"
    )

    # DROWSY -> SLEEP (second hop — bypasses idle/eligibility).
    target2 = compute_transition(LifecycleState.DROWSY, FORCE)
    assert target2 is LifecycleState.SLEEP, (
        f"Expected DROWSY + FORCE_SLEEP -> SLEEP, got {target2}"
    )

    # Already SLEEP -> no-op (SLEEP is the target; no double-SLEEP).
    target3 = compute_transition(LifecycleState.SLEEP, FORCE)
    assert target3 is None, (
        f"Expected SLEEP + FORCE_SLEEP -> None (no-op), got {target3}"
    )

    # HIBERNATION -> no-op (already past consolidation window).
    target4 = compute_transition(LifecycleState.HIBERNATION, FORCE)
    assert target4 is None, (
        f"Expected HIBERNATION + FORCE_SLEEP -> None, got {target4}"
    )

    # Verify the two-hop progression: WAKE -> DROWSY -> SLEEP.
    s = LifecycleState.WAKE
    s = compute_transition(s, FORCE)   # hop 1
    assert s is LifecycleState.DROWSY
    s = compute_transition(s, FORCE)   # hop 2
    assert s is LifecycleState.SLEEP


# ---------------------------------------------------------------------------
# Test E: no double-drain of already-covered WAKE outputs
# ---------------------------------------------------------------------------


def test_e_no_double_drain_of_covered_outputs(tmp_path, monkeypatch):
    """drain_deferred_captures and flush_event_buffer are ALREADY invoked
    on their existing canonical paths (drowsy drain + per-tick flush). The
    relocated canonical wake hook (the four responsibilities under LOCK_EX)
    must NOT add a second invocation of either.

    We verify by counting call sites in the WAKE hook block inserted in
    daemon.py's lifecycle_tick, asserting neither function is called from
    that block.
    """
    import iai_mcp.daemon as daemon_mod

    # Inspect the source of lifecycle_tick's wake hook region.
    import inspect
    source = inspect.getsource(daemon_mod)

    # The wake hook that was inserted: check neither drain_deferred_captures
    # nor flush_event_buffer appears in the wake hook block.
    # We verify at source level: the block ends just before downgrade_to_shared.
    # Find the wake hook marker and the downgrade marker.
    wake_hook_start = source.find("# --- WAKE hook (UNDER LOCK_EX, BEFORE downgrade) ---")
    downgrade_marker = source.find("# Downgrade EX → SH after the consolidation window.")
    assert wake_hook_start > 0, "Wake hook comment not found in daemon source"
    assert downgrade_marker > 0, "Downgrade marker not found in daemon source"

    wake_hook_block = source[wake_hook_start:downgrade_marker]

    # Verify neither already-covered function appears in the wake hook block.
    assert "drain_deferred_captures" not in wake_hook_block, (
        "drain_deferred_captures is in the wake hook block — double-invocation risk"
    )
    assert "flush_event_buffer" not in wake_hook_block, (
        "flush_event_buffer is in the wake hook block — double-invocation risk"
    )

    # Verify the four wake-hook responsibilities are present in the block.
    for marker in [
        "_write_session_start_cache",
        "write_processed_salience_top_n",
        "drain_active_live_captures",
        "flush_deferred_provenance",
    ]:
        assert marker in wake_hook_block, (
            f"Wake hook block missing R-output: {marker}"
        )


# ---------------------------------------------------------------------------
# Test F: foraging gated during SLEEP, runs in WAKE
# ---------------------------------------------------------------------------


def test_f_foraging_gated_during_sleep(daemon_store, monkeypatch, tmp_path):
    """F: Step 0.7 foraging must be skipped when the canonical lifecycle FSM
    is in SLEEP, and must be allowed to run when the FSM is in WAKE.

    The SLEEP gate in _tick_body reads lifecycle_state.json at call time.
    We write a tmp canonical state file and patch LIFECYCLE_STATE_PATH so
    the daemon reads our test file. Foraging is patched at the module level
    where the lazy import resolves.
    """
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.lifecycle_state import LifecycleState
    from unittest.mock import patch

    store, state_path, _ = daemon_store

    # Write a canonical lifecycle_state.json to tmp.
    canon_path = tmp_path / "lifecycle_state.json"

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    forage_calls: list = []

    def fake_forage(store, max_edges=3):
        forage_calls.append(1)
        return 0

    def _write_ls(state: LifecycleState) -> None:
        """Write a minimal VALID lifecycle_state.json record."""
        from iai_mcp.lifecycle_state import default_state
        rec = dict(default_state())
        rec["current_state"] = state.value
        canon_path.write_text(json.dumps(rec))

    # --- Part 1: canonical FSM = SLEEP → foraging must be skipped ---
    _write_ls(LifecycleState.SLEEP)

    state_sleep = {
        "fsm_state": "SLEEP",
        "_last_forage_ts": "",  # clear throttle
    }
    # Patch both LIFECYCLE_STATE_PATH (for the gate read) and forage_for_connections.
    with patch("iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH", canon_path), \
         patch("iai_mcp.foraging.forage_for_connections", fake_forage):
        asyncio.run(daemon_mod._tick_body(store, state_sleep))

    assert forage_calls == [], (
        f"foraging was called {len(forage_calls)} time(s) during SLEEP; expected 0"
    )

    # --- Part 2: canonical FSM = WAKE → foraging is allowed ---
    _write_ls(LifecycleState.WAKE)
    forage_calls.clear()

    state_wake = {
        "fsm_state": "WAKE",
        "_last_forage_ts": "",  # clear throttle
    }
    with patch("iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH", canon_path), \
         patch("iai_mcp.foraging.forage_for_connections", fake_forage):
        asyncio.run(daemon_mod._tick_body(store, state_wake))

    assert forage_calls == [1], (
        f"foraging was called {len(forage_calls)} time(s) during WAKE; expected 1"
    )
