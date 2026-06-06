"""S2 Coordination regression suite.

Scope:
  - S2Coordinator class + asyncio.Lock + version counter + ring buffer.
  - s2_transition_attempt 7-field event body.
  - s2_oscillation_blocked event with 4 keys on rapid back-and-forth
    within MIN_INTERVAL_SEC.
  - dry-run permits the conflict and emits the block event with
    dry_run_mode=true.

That all production transition sites route through the coordinator is
verified by a static grep check rather than a runtime behavioural test —
it's a structural invariant of the daemon wiring. Env-var validation is
verified by a fail-loud smoke test.

Tests use tmp_path-rooted lifecycle_state.json (seeded via
save_state(default_state(),...)) so the production ~/.iai-mcp state file
is NEVER touched. The patch target for write_event is
`iai_mcp.s2_coordinator.write_event` — the symbol re-imported into the
coordinator's namespace at module load.
"""
# Standard-library imports first so optional iai_mcp.* imports below fail
# loud with a clear ImportError if the package layout changes.
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp.lifecycle_state import (
    LifecycleState,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.s2_coordinator import (
    S2Coordinator,
    S2OscillationBlocked,
    S2OscillationConflict,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers (— mock the event-emit, drive coordinator directly
# via tmp_path-rooted state file). All five tests share these fixtures so
# the suite stays under ~280 lines including docstrings.
# ---------------------------------------------------------------------------


class _CapturedEvents:
    """In-memory replacement for iai_mcp.events.write_event.

    Records every (kind, data, severity) tuple so tests assert on the
    event ledger without store IO. The patch target is
    `iai_mcp.s2_coordinator.write_event` (NOT `iai_mcp.events.write_event`)
    because the coordinator does `from iai_mcp.events import write_event`
    at module load, so the bound name inside the coordinator's namespace
    is what `transition` calls.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, store, kind, data, *, severity=None, **_kw):
        # Copy the body dict so later mutations by the coordinator (e.g.
        # ring-buffer entry reuse) can't retroactively alter what the
        # test asserted on.
        self.events.append(
            {"kind": kind, "data": dict(data), "severity": severity}
        )
        return None

    def by_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]


@pytest.fixture
def captured_events():
    """Patch the write_event symbol re-imported into s2_coordinator.

    The patch target is `iai_mcp.s2_coordinator.write_event` (NOT
    `iai_mcp.events.write_event`) because the coordinator imports
    `from iai_mcp.events import write_event` at module load, so the
    bound name inside the coordinator's namespace is what `transition`
    actually calls. Patching the original `iai_mcp.events.write_event`
    would NOT intercept calls inside the coordinator.
    """
    cap = _CapturedEvents()
    with patch("iai_mcp.s2_coordinator.write_event", cap):
        yield cap


@pytest.fixture
def seeded_state(tmp_path: Path) -> Path:
    """Return a tmp_path-rooted lifecycle_state.json seeded to WAKE.

    Tests that want a different initial state call
    `_reseed_state(state_path, LifecycleState.SLEEP)`.
    """
    state_path = tmp_path / "lifecycle_state.json"
    save_state(default_state(), state_path)  # default_state starts at WAKE
    return state_path


def _reseed_state(state_path: Path, target: LifecycleState) -> None:
    """Overwrite the seeded state file with a specific initial state.

    Used by tests 3/4 that need to start from SLEEP (so the first
    transition goes SLEEP->WAKE, exercising the reverse-direction
    oscillation-detection logic).
    """
    rec = default_state()
    rec["current_state"] = target.value
    save_state(rec, state_path)


def _make_coord(
    state_path: Path,
    *,
    min_interval_sec: float = 5.0,
    dry_run: bool = False,
) -> S2Coordinator:
    """Construct a wired S2Coordinator pointed at the tmp_path state file.

    `store=None` is intentional — the captured_events fixture intercepts
    every write_event call before the coordinator can touch the store,
    so the None never propagates to store IO.
    """
    return S2Coordinator(
        store=None,
        state_path=state_path,
        min_interval_sec=min_interval_sec,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Test 1 — single-transition happy path
# ---------------------------------------------------------------------------


def test_single_transition_happy_path_emits_attempt_event(
    captured_events, seeded_state
):
    """A single successful transition writes one s2_transition_attempt event with 7 fields.

    Exercises the success path end-to-end: CAS match → ring-buffer
    miss → save_state → version++ → emit. Locks the 7-field shape of
    the success-branch event body.
    """
    coord = _make_coord(seeded_state)

    new = asyncio.run(
        coord.transition(
            LifecycleState.WAKE,
            LifecycleState.DROWSY,
            "drowsy_on_idle_5min",
        )
    )

    assert new == LifecycleState.DROWSY
    assert coord.version == 1

    # Disk persisted to the tmp_path file (production ~/.iai-mcp untouched).
    rec = load_state(seeded_state)
    assert rec["current_state"] == "DROWSY"

    # Event ledger — exactly one attempt, no block.
    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 1
    body = attempts[0]["data"]
    assert set(body.keys()) == {
        "from_state",
        "to_state",
        "reason",
        "version_before",
        "version_after",
        "succeeded",
        "conflict_reason",
    }
    assert body["from_state"] == "WAKE"
    assert body["to_state"] == "DROWSY"
    assert body["reason"] == "drowsy_on_idle_5min"
    assert body["version_before"] == 0
    assert body["version_after"] == 1
    assert body["succeeded"] is True
    assert body["conflict_reason"] is None
    assert captured_events.by_kind("s2_oscillation_blocked") == []


# ---------------------------------------------------------------------------
# Test 2 — behavioural: two concurrent transitions, one wins one raises
# ---------------------------------------------------------------------------


def test_concurrent_transitions_one_wins_one_raises_conflict(
    captured_events, seeded_state
):
    """Two concurrent WAKE-origin transitions serialise; one wins, one raises CAS conflict.

    Both tasks claim `from_state=WAKE` and target different ends. The
    asyncio.Lock inside `transition` serialises them: the first writes
    WAKE→{winner}; the second (now under the lock) sees actual={winner}
    !=WAKE and raises S2OscillationConflict. Order is non-deterministic
    so assertions are symmetric.
    """
    coord = _make_coord(seeded_state)

    async def race():
        return await asyncio.gather(
            coord.transition(
                LifecycleState.WAKE, LifecycleState.DROWSY, "race_a"
            ),
            coord.transition(
                LifecycleState.WAKE, LifecycleState.SLEEP, "race_b"
            ),
            return_exceptions=True,
        )

    results = asyncio.run(race())

    # Exactly one LifecycleState success + exactly one S2OscillationConflict.
    successes = [r for r in results if isinstance(r, LifecycleState)]
    conflicts = [r for r in results if isinstance(r, S2OscillationConflict)]
    assert len(successes) == 1, results
    assert len(conflicts) == 1, results

    winner = successes[0]
    assert conflicts[0].actual_state == winner, (
        f"conflict.actual_state {conflicts[0].actual_state} should "
        f"equal winner {winner}"
    )
    assert conflicts[0].attempted_from == LifecycleState.WAKE
    assert coord.version == 1

    rec = load_state(seeded_state)
    assert rec["current_state"] == winner.value

    # Two s2_transition_attempt events: one success, one cas_mismatch.
    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 2
    succeeded_flags = sorted(a["data"]["succeeded"] for a in attempts)
    assert succeeded_flags == [False, True]
    rejected = [a for a in attempts if a["data"]["succeeded"] is False]
    assert rejected[0]["data"]["conflict_reason"] == "cas_mismatch"
    # No oscillation block — the conflict is CAS only.
    assert captured_events.by_kind("s2_oscillation_blocked") == []


# ---------------------------------------------------------------------------
# Test 3 — rapid oscillation raises S2OscillationBlocked
# ---------------------------------------------------------------------------


def test_rapid_oscillation_raises_blocked(captured_events, seeded_state):
    """A transition whose reverse just landed within MIN_INTERVAL raises S2OscillationBlocked.

    Uses min_interval_sec=60.0 so test-runtime variance never
    accidentally passes the time gate. Asserts state file shows the
    FIRST transition's destination (the blocked second write never
    landed) and that the rejected attempt is recorded with
    conflict_reason="oscillation_blocked".
    """
    _reseed_state(seeded_state, LifecycleState.SLEEP)
    coord = _make_coord(seeded_state, min_interval_sec=60.0)

    # First: SLEEP -> WAKE (succeeds, lands in ring buffer).
    asyncio.run(
        coord.transition(
            LifecycleState.SLEEP,
            LifecycleState.WAKE,
            "wake_on_signal_consumed",
        )
    )

    # Second (rapid reverse): WAKE -> SLEEP must raise.
    with pytest.raises(S2OscillationBlocked) as exc_info:
        asyncio.run(
            coord.transition(
                LifecycleState.WAKE,
                LifecycleState.SLEEP,
                "sleep_on_idle_30min",
            )
        )
    assert 0.0 <= exc_info.value.interval_sec < 60.0

    # First transition persisted; second blocked → state still WAKE.
    rec = load_state(seeded_state)
    assert rec["current_state"] == "WAKE"
    assert coord.version == 1  # only the first transition counted

    blocks = captured_events.by_kind("s2_oscillation_blocked")
    assert len(blocks) == 1
    body = blocks[0]["data"]
    assert set(body.keys()) == {
        "first_transition",
        "second_transition",
        "interval_sec",
        "dry_run_mode",
    }
    assert body["first_transition"]["from_state"] == "SLEEP"
    assert body["first_transition"]["to_state"] == "WAKE"
    assert body["second_transition"]["from_state"] == "WAKE"
    assert body["second_transition"]["to_state"] == "SLEEP"
    assert body["dry_run_mode"] is False
    assert 0.0 <= body["interval_sec"] < 60.0

    # Every attempt emits one s2_transition_attempt — the rejected
    # attempt's event carries conflict_reason="oscillation_blocked".
    attempts = captured_events.by_kind("s2_transition_attempt")
    rejected = [a for a in attempts if a["data"]["succeeded"] is False]
    assert any(
        a["data"]["conflict_reason"] == "oscillation_blocked"
        for a in rejected
    )


# ---------------------------------------------------------------------------
# Test 4 — dry-run permits the conflict, emits block event tagged true
# ---------------------------------------------------------------------------


def test_dry_run_permits_oscillation_emits_event(
    captured_events, seeded_state
):
    """Dry-run path emits the block event with dry_run_mode=true and DOES allow the transition through.

    Same trigger sequence as test 3 BUT coord constructed with
    dry_run=True; the second call returns normally; state file shows
    the SECOND transition's destination (it landed); block event
    fires with dry_run_mode=True; version reaches 2.
    """
    _reseed_state(seeded_state, LifecycleState.SLEEP)
    coord = _make_coord(seeded_state, min_interval_sec=60.0, dry_run=True)

    asyncio.run(
        coord.transition(
            LifecycleState.SLEEP,
            LifecycleState.WAKE,
            "wake_on_signal_consumed",
        )
    )
    # Second: should NOT raise — dry_run permits.
    new = asyncio.run(
        coord.transition(
            LifecycleState.WAKE,
            LifecycleState.SLEEP,
            "sleep_on_idle_30min",
        )
    )

    assert new == LifecycleState.SLEEP
    assert coord.version == 2  # both transitions counted

    rec = load_state(seeded_state)
    assert rec["current_state"] == "SLEEP"

    blocks = captured_events.by_kind("s2_oscillation_blocked")
    assert len(blocks) == 1
    assert blocks[0]["data"]["dry_run_mode"] is True

    # Dry-run path: per coordinator re-ordering (deviation in s2_coordinator.py),
    # exactly two s2_transition_attempt events fire (both succeeded=True).
    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 2
    assert all(a["data"]["succeeded"] is True for a in attempts)


# ---------------------------------------------------------------------------
# Test 5 — schema contract: locked key set + type discipline
# ---------------------------------------------------------------------------


def test_event_body_shape_contract(captured_events, seeded_state):
    """Schema: every event body has exactly the locked key set + type discipline.

    Deterministic 4-attempt script — chosen so attempts 1 and 3 walk
    through THREE distinct states (WAKE -> DROWSY -> SLEEP) so neither
    is the reverse of any prior successful transition:

      1. WAKE -> DROWSY (success, version 0->1, ring=[W->D])
      2. WAKE -> SLEEP (CAS conflict — actual=DROWSY, version stays 1,
                          NOT added to ring per coordinator contract)
      3. DROWSY -> SLEEP (success, version 1->2, ring=[W->D, D->S])
      4. SLEEP -> DROWSY (oscillation block — reverse of #3 within 60s,
                          version stays 2)

    Total events emitted: 4 × s2_transition_attempt + 1 × s2_oscillation_blocked.
    """
    coord = _make_coord(seeded_state, min_interval_sec=60.0)

    # Attempt 1: success — WAKE -> DROWSY.
    asyncio.run(
        coord.transition(LifecycleState.WAKE, LifecycleState.DROWSY, "r1")
    )
    # Attempt 2: CAS conflict (actual is now DROWSY, expected WAKE).
    with pytest.raises(S2OscillationConflict):
        asyncio.run(
            coord.transition(
                LifecycleState.WAKE, LifecycleState.SLEEP, "r2"
            )
        )
    # Attempt 3: success — DROWSY -> SLEEP (NOT reverse of #1).
    asyncio.run(
        coord.transition(LifecycleState.DROWSY, LifecycleState.SLEEP, "r3")
    )
    # Attempt 4: oscillation block — SLEEP -> DROWSY is reverse of #3.
    with pytest.raises(S2OscillationBlocked):
        asyncio.run(
            coord.transition(
                LifecycleState.SLEEP, LifecycleState.DROWSY, "r4"
            )
        )

    # 4 attempt events — deterministic count.
    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 4, [a["data"] for a in attempts]
    expected_attempt_keys = {
        "from_state",
        "to_state",
        "reason",
        "version_before",
        "version_after",
        "succeeded",
        "conflict_reason",
    }
    for a in attempts:
        body = a["data"]
        # `==` set equality (not subset) — the schema is locked.
        assert set(body.keys()) == expected_attempt_keys
        assert isinstance(body["from_state"], str)
        assert isinstance(body["to_state"], str)
        assert isinstance(body["reason"], str)
        assert isinstance(body["version_before"], int)
        assert isinstance(body["version_after"], int)
        assert isinstance(body["succeeded"], bool)
        assert body["conflict_reason"] is None or isinstance(
            body["conflict_reason"], str
        )

    # 1 block event — locked 4-key shape.
    blocks = captured_events.by_kind("s2_oscillation_blocked")
    assert len(blocks) == 1, [b["data"] for b in blocks]
    expected_block_keys = {
        "first_transition",
        "second_transition",
        "interval_sec",
        "dry_run_mode",
    }
    body = blocks[0]["data"]
    assert set(body.keys()) == expected_block_keys
    assert isinstance(body["first_transition"], dict)
    assert isinstance(body["second_transition"], dict)
    assert isinstance(body["interval_sec"], float)
    assert isinstance(body["dry_run_mode"], bool)
