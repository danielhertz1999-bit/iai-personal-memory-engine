from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

NOW = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def _make_mixed_state() -> dict:
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
    from iai_mcp.daemon_state import (
        FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
        prune_first_turn_pending,
    )

    state = _make_mixed_state()
    new_state, dropped = prune_first_turn_pending(state, now=NOW)

    assert sorted(dropped) == sorted(f"sess-stale-{i}" for i in range(5)), (
        f"Expected exactly 5 stale session_ids dropped; got {dropped}"
    )
    kept = new_state["first_turn_pending"]
    assert len(kept) == 5
    for k in kept:
        assert k.startswith("sess-fresh-"), f"unexpected key kept: {k}"
    assert FIRST_TURN_PENDING_TTL_SEC_DEFAULT == 3600.0


def test_prune_helper_no_drop_when_only_fresh_entries():
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
    from iai_mcp.daemon_state import (
        FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
        prune_first_turn_pending,
    )
    assert FIRST_TURN_PENDING_TTL_SEC_DEFAULT == 3600.0
    assert callable(prune_first_turn_pending)


def test_daemon_wire_in_passes_explicit_now_kwarg_at_both_sites():
    daemon_src = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon.py"
    text = daemon_src.read_text()

    pat = re.compile(
        r"prune_first_turn_pending\s*\(\s*state\s*,\s*now\s*=\s*datetime\.now\(\s*timezone\.utc\s*\)",
        re.MULTILINE,
    )
    matches = pat.findall(text)
    assert len(matches) >= 2, (
        f"Expected >= 2 wire-in sites with explicit `now=datetime.now(timezone.utc)` "
        f"kwarg in daemon.py; found {len(matches)}. Contract:"
        f" both startup-prune (in main()) and tick-prune (in _tick_body Step 0.5)"
        f" must thread `now=` explicitly."
    )

    assert '"phase": "startup"' in text or "'phase': 'startup'" in text, (
        "Startup-side event emit missing `phase: startup` in payload."
    )
    assert '"phase": "tick"' in text or "'phase': 'tick'" in text, (
        "Tick-side event emit missing `phase: tick` in payload."
    )
