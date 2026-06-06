"""FSM consolidation tests.

Verifies that fsm_reconcile auto-corrects legacy state to match canonical,
handles interrupted transitions, and crash-between-writes scenarios.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from iai_mcp.fsm_reconcile import reconcile_fsm_state


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path


def _write_canonical(state_dir: Path, state: str):
    p = state_dir / "lifecycle_state.json"
    p.write_text(json.dumps({"current_state": state, "updated_at": "2026-05-17T00:00:00Z"}))
    return p


def _write_legacy(state_dir: Path, fsm_state: str):
    p = state_dir / ".daemon-state.json"
    p.write_text(json.dumps({"fsm_state": fsm_state, "daemon_started_at": "2026-05-17T00:00:00Z"}))
    return p


class TestReconcileNoDrift:
    def test_both_wake(self, state_dir):
        c = _write_canonical(state_dir, "WAKE")
        l = _write_legacy(state_dir, "WAKE")
        r = reconcile_fsm_state(c, l)
        assert r["drift"] is False
        assert r["canonical"] == "WAKE"
        assert r["legacy"] == "WAKE"

    def test_sleep_dreaming_is_equivalent(self, state_dir):
        c = _write_canonical(state_dir, "SLEEP")
        l = _write_legacy(state_dir, "DREAMING")
        r = reconcile_fsm_state(c, l)
        assert r["drift"] is False

    def test_drowsy_transitioning_is_equivalent(self, state_dir):
        c = _write_canonical(state_dir, "DROWSY")
        l = _write_legacy(state_dir, "TRANSITIONING")
        r = reconcile_fsm_state(c, l)
        assert r["drift"] is False

    def test_hibernation_accepts_any_legacy(self, state_dir):
        c = _write_canonical(state_dir, "HIBERNATION")
        l = _write_legacy(state_dir, "WAKE")
        r = reconcile_fsm_state(c, l)
        assert r["drift"] is False


class TestReconcileDrift:
    def test_wake_vs_sleep_is_drift(self, state_dir):
        c = _write_canonical(state_dir, "WAKE")
        l = _write_legacy(state_dir, "SLEEP")
        r = reconcile_fsm_state(c, l)
        assert r["drift"] is True

    def test_sleep_vs_wake_is_drift(self, state_dir):
        c = _write_canonical(state_dir, "SLEEP")
        l = _write_legacy(state_dir, "WAKE")
        r = reconcile_fsm_state(c, l)
        assert r["drift"] is True


class TestAutoCorrect:
    def test_corrects_legacy_on_drift(self, state_dir):
        c = _write_canonical(state_dir, "WAKE")
        l = _write_legacy(state_dir, "SLEEP")
        r = reconcile_fsm_state(c, l, auto_correct=True)
        assert r["drift"] is True
        assert r["corrected"] is True
        updated = json.loads(l.read_text())
        assert updated["fsm_state"] == "WAKE"
        assert "fsm_corrected_at" in updated

    def test_no_correction_when_no_drift(self, state_dir):
        c = _write_canonical(state_dir, "WAKE")
        l = _write_legacy(state_dir, "WAKE")
        r = reconcile_fsm_state(c, l, auto_correct=True)
        assert r["drift"] is False
        assert r["corrected"] is False

    def test_corrects_drowsy_to_transitioning(self, state_dir):
        c = _write_canonical(state_dir, "DROWSY")
        l = _write_legacy(state_dir, "WAKE")
        r = reconcile_fsm_state(c, l, auto_correct=True)
        assert r["corrected"] is True
        updated = json.loads(l.read_text())
        assert updated["fsm_state"] == "TRANSITIONING"

    def test_preserves_other_legacy_fields(self, state_dir):
        c = _write_canonical(state_dir, "SLEEP")
        l = state_dir / ".daemon-state.json"
        l.write_text(json.dumps({
            "fsm_state": "WAKE",
            "daemon_started_at": "2026-01-01T00:00:00Z",
            "custom_field": "preserved",
        }))
        r = reconcile_fsm_state(c, l, auto_correct=True)
        assert r["corrected"] is True
        updated = json.loads(l.read_text())
        assert updated["fsm_state"] == "SLEEP"
        assert updated["custom_field"] == "preserved"


class TestCrashRecovery:
    def test_missing_canonical_no_crash(self, state_dir):
        l = _write_legacy(state_dir, "WAKE")
        c = state_dir / "lifecycle_state.json"
        r = reconcile_fsm_state(c, l)
        assert r["canonical"] is None
        assert r["drift"] is False

    def test_missing_legacy_no_crash(self, state_dir):
        c = _write_canonical(state_dir, "WAKE")
        l = state_dir / ".daemon-state.json"
        r = reconcile_fsm_state(c, l)
        assert r["legacy"] is None
        assert r["drift"] is False

    def test_corrupt_canonical_no_crash(self, state_dir):
        c = state_dir / "lifecycle_state.json"
        c.write_text("not json{{{{")
        l = _write_legacy(state_dir, "WAKE")
        r = reconcile_fsm_state(c, l)
        assert r["canonical"] is None
        assert r["drift"] is False

    def test_corrupt_legacy_no_crash(self, state_dir):
        c = _write_canonical(state_dir, "WAKE")
        l = state_dir / ".daemon-state.json"
        l.write_text("corrupted bytes here")
        r = reconcile_fsm_state(c, l)
        assert r["legacy"] is None
        assert r["drift"] is False

    def test_both_missing(self, state_dir):
        c = state_dir / "lifecycle_state.json"
        l = state_dir / ".daemon-state.json"
        r = reconcile_fsm_state(c, l)
        assert r["canonical"] is None
        assert r["legacy"] is None
        assert r["drift"] is False
