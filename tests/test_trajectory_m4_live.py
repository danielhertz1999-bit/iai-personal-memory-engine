"""Task 2 Step 6: M4 profile-variance LIVE tests."""
from __future__ import annotations

import pytest

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.trajectory import m4_profile_variance_live


def test_m4_zero_on_empty_store(tmp_path):
    store = MemoryStore(path=tmp_path)
    assert m4_profile_variance_live(store) == 0.0


def test_m4_low_variance_on_stable_writes(tmp_path):
    """20 writes that converge near 0.5 -> low variance."""
    store = MemoryStore(path=tmp_path)
    for i in range(20):
        # Convergent series 0.49 -> 0.50 -> 0.51...
        new = 0.5 + (i % 3 - 1) * 0.01
        write_event(
            store, kind="profile_updated",
            data={"knob": "interest_boost", "old": 0.5, "new": new},
            severity="info",
        )
    val = m4_profile_variance_live(store, n_updates=20)
    assert val < 0.1


def test_m4_skips_non_numeric_knobs(tmp_path):
    """Bool/enum knobs do not contribute variance (skipped)."""
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="profile_updated",
        data={"knob": "masking_off", "old": True, "new": False},
        severity="info",
    )
    write_event(
        store, kind="profile_updated",
        data={"knob": "interest_boost", "old": 0.0, "new": 1.0},
        severity="info",
    )
    write_event(
        store, kind="profile_updated",
        data={"knob": "interest_boost", "old": 1.0, "new": 0.0},
        severity="info",
    )
    val = m4_profile_variance_live(store)
    # Only interest_boost contributes; values [1.0, 0.0]; variance = 0.25.
    assert val == pytest.approx(0.25, abs=1e-6)
