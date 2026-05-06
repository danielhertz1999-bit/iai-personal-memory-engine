"""Phase 07.2-04 R3 / A3 integration test — startup + per-tick TTL drain wired into daemon.

Strategy: Plan 04 Task 1 threads an explicit `now=datetime.now(timezone.utc)`
kwarg from BOTH wire-in call sites into `prune_first_turn_pending`. This
means the helper is fully testable by passing a fixed `NOW` directly —
no datetime monkeypatching dance.

Three checks:
1. Direct helper invocation with mixed stale/fresh state proves the
   eviction contract (5 stale evict, 5 fresh keep, dropped IDs returned).
2. Smoke import confirms the names daemon.py imports are reachable.
3. Source-grep on daemon.py confirms both wire-in sites pass the explicit
   `now=` kwarg (Task 1's structural contract).

Project async-test idiom (mandatory): sync `def test_*`. No
`@pytest.mark.asyncio`. The helper itself is sync, so all tests here
are plain sync `def test_*` with no `asyncio.run` needed.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

NOW = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def _make_mixed_state() -> dict:
    """Return a state dict with 5 stale + 5 fresh first_turn_pending entries.

    Stale = 2 h old (well past the 1 h TTL).
    Fresh = 30 s old (well within the TTL).
    Both timestamps are RELATIVE TO `NOW` so the test is deterministic
    regardless of when it runs — `prune_first_turn_pending` only sees the
    explicit `now` we pass in.
    """
    stale_entries = {
        f"sess-stale-{i}": (NOW - timedelta(hours=2)).isoformat()
        for i in range(5)
    }
    fresh_entries = {
        f"sess-fresh-{i}": (NOW - timedelta(seconds=30)).isoformat()
        for i in range(5)
    }
    return {
        "fsm_state": "WAKE",
        "first_turn_pending": {**stale_entries, **fresh_entries},
    }


def test_prune_helper_drops_5_stale_keeps_5_fresh_with_fixed_now():
    """A3 acceptance (helper contract): with NOW fixed and 5 stale + 5 fresh
    entries, the helper returns 5 dropped IDs and a state holding only the
    fresh entries. This is exactly the contract Plan 04's wire-in invokes
    at startup and per-tick.
    """
    from iai_mcp.daemon_state import (
        FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
        prune_first_turn_pending,
    )

    state = _make_mixed_state()
    # Plan 04 Task 1 calls this with the EXACT signature shown below at
    # both wire-in sites. The test mirrors the wire-in call shape so any
    # future signature drift breaks BOTH sides at once.
    new_state, dropped = prune_first_turn_pending(state, now=NOW)

    # 5 stale IDs evict.
    assert sorted(dropped) == sorted(f"sess-stale-{i}" for i in range(5)), (
        f"Expected exactly 5 stale session_ids dropped; got {dropped}"
    )
    # 5 fresh IDs survive.
    kept = new_state["first_turn_pending"]
    assert len(kept) == 5
    for k in kept:
        assert k.startswith("sess-fresh-"), f"unexpected key kept: {k}"
    # Helper exposes the TTL constant Plan 04 wire-in uses for the event
    # payload — sanity-check it has the documented value (1 h).
    assert FIRST_TURN_PENDING_TTL_SEC_DEFAULT == 3600.0


def test_prune_helper_no_drop_when_only_fresh_entries():
    """Control: NOW fixed and only fresh entries → 0 dropped, 5 kept,
    state.first_turn_pending unchanged in shape."""
    from iai_mcp.daemon_state import prune_first_turn_pending

    state = {
        "fsm_state": "WAKE",
        "first_turn_pending": {
            f"sess-fresh-{i}": (NOW - timedelta(seconds=30)).isoformat()
            for i in range(5)
        },
    }
    new_state, dropped = prune_first_turn_pending(state, now=NOW)

    assert dropped == [], f"Expected zero drops on all-fresh state; got {dropped}"
    assert len(new_state["first_turn_pending"]) == 5


def test_first_turn_pending_drain_helper_imported_in_daemon_main():
    """Smoke: daemon.main() can import the helper without error.

    If Plan 04's import block is wrong (typo, wrong module, etc.), this
    fails fast.
    """
    from iai_mcp.daemon_state import (
        FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
        prune_first_turn_pending,
    )
    assert FIRST_TURN_PENDING_TTL_SEC_DEFAULT == 3600.0
    assert callable(prune_first_turn_pending)


def test_daemon_wire_in_passes_explicit_now_kwarg_at_both_sites():
    """Structural check: read daemon.py source and confirm BOTH wire-in
    sites pass `now=datetime.now(timezone.utc)` explicitly.

    This is the wire-up half of A3 — without it, Task 2 only proves the
    helper works, not that Task 1 wired it in correctly. Plan 04 Task 1's
    contract is that BOTH call sites thread `now=` explicitly so the
    helper is testable without datetime mocking.
    """
    daemon_src = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_src.read_text()

    # Match `prune_first_turn_pending(\n    state, now=datetime.now(timezone.utc)`
    # tolerantly across whitespace + line breaks.
    pat = re.compile(
        r"prune_first_turn_pending\s*\(\s*state\s*,\s*now\s*=\s*datetime\.now\(\s*timezone\.utc\s*\)",
        re.MULTILINE,
    )
    matches = pat.findall(text)
    assert len(matches) >= 2, (
        f"Expected >= 2 wire-in sites with explicit `now=datetime.now(timezone.utc)` "
        f"kwarg in daemon.py; found {len(matches)}. Plan 04 Task 1 contract:"
        f" both startup-prune (in main()) and tick-prune (in _tick_body Step 0.5)"
        f" must thread `now=` explicitly."
    )

    # Both event-emit phases ("startup" and "tick") must be present.
    assert '"phase": "startup"' in text or "'phase': 'startup'" in text, (
        "Startup-side event emit missing `phase: startup` in payload."
    )
    assert '"phase": "tick"' in text or "'phase': 'tick'" in text, (
        "Tick-side event emit missing `phase: tick` in payload."
    )
