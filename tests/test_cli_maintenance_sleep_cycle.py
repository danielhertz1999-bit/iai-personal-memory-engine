"""Task 1.5 -- CLI maintenance sleep-cycle tests.

Eight cases:
  1. test_happy_path_runs_pipeline_and_prints_progress
  2. test_quarantined_without_force_returns_nonzero_with_message
  3. test_force_runs_pipeline_when_quarantined
  4. test_reset_quarantine_clears_then_runs
  5. test_reset_quarantine_when_not_quarantined_no_op
  6. test_failure_returns_nonzero_with_error_in_stderr
  7. test_failure_after_3rd_strike_prints_quarantine_hint
  8. test_subparser_exposes_sleep_cycle_with_flags

All tests use stub `MemoryStore` + monkeypatched SleepPipeline methods —
no real store I/O.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iai_mcp.lifecycle_state import (
    default_state,
    load_state,
    save_state,
)
from iai_mcp.sleep_pipeline import SleepStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Construct argparse.Namespace with sleep-cycle defaults."""
    defaults = dict(
        force=False,
        reset_quarantine=False,
        store_path=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def iai_root(tmp_path, monkeypatch):
    """Sandbox HOME so LIFECYCLE_STATE_PATH points inside tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring"
    )
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase")
    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir()
    # Reload modules so they pick up the new HOME — LIFECYCLE_STATE_PATH
    # and STATE_PATH are module-scope captures.
    import importlib
    from iai_mcp import lifecycle_state as _ls
    from iai_mcp import cli as _cli
    importlib.reload(_ls)
    importlib.reload(_cli)
    yield iai_dir
    importlib.reload(_ls)
    importlib.reload(_cli)


def _patch_store_open(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace MemoryStore() with a MagicMock so the CLI can construct
    a 'store' without touching the real store / embedder.
    """
    fake_store = MagicMock()
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    return fake_store


def _patch_pipeline_steps_to_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace every _step_* method on SleepPipeline with a no-op so the
    real pipeline executes without doing real store work.
    """
    from iai_mcp.sleep_pipeline import SleepPipeline

    for step, method_name in [
        (SleepStep.SCHEMA_MINE, "_step_schema_mine"),
        (SleepStep.KNOB_TUNE, "_step_knob_tune"),
        (SleepStep.DREAM_DECAY, "_step_dream_decay"),
        (SleepStep.ERASURE_AGENT, "_step_erasure_agent"),
        (SleepStep.OPTIMIZE_LANCE, "_step_optimize_lance"),
        (SleepStep.COMPACT_RECORDS, "_step_compact_records"),
        # CLUSTER_REPLAY + CRISIS_RECLUSTER real bodies call write_event
        # against the MagicMock store and crash on store._key() not
        # returning 32 bytes; noop them here to keep the CLI happy-path /
        # failure-path tests focused on the CLI surface, not the step
        # internals (those have dedicated coverage in test_sleep_overhaul.py).
        (SleepStep.CLUSTER_REPLAY, "_step_cluster_replay"),
        (SleepStep.CRISIS_RECLUSTER, "_step_crisis_recluster"),
        # RECONSOLIDATION real body re-embeds via store on success
        # and writes a reconsolidation_pass event; MagicMock store crashes
        # the same way; noop here for the same reason.
        (SleepStep.RECONSOLIDATION, "_step_reconsolidation"),
        # USER_MODEL_UPDATE real body persists a UserModel JSON +
        # writes a user_model_aggregate_pass event; same MagicMock crash
        # surface; noop here.
        (SleepStep.USER_MODEL_UPDATE, "_step_user_model_update"),
        # DMN_REFLECTION real body calls MetaAnalyst.snapshot +
        # ReflectionAgent.synthesize and writes a system_health_report
        # event; same MagicMock crash surface; noop here so the CLI
        # tests stay focused on slot numbers / failure semantics rather
        # than the step internals (covered in test_dmn_meta.py).
        (SleepStep.DMN_REFLECTION, "_step_dmn_reflection"),
        # 62-02 appended CLUSTER_SUMMARY and RECALL_INDEX_REBUILD as the
        # final two REM steps (topology rebuild + generation-epoch stamp).
        # Real bodies call store.db / runtime_graph_cache which crash on
        # MagicMock; noop here so the CLI tests stay focused on slot
        # numbers and failure semantics.
        (SleepStep.CLUSTER_SUMMARY, "_step_cluster_summary"),
        (SleepStep.RECALL_INDEX_REBUILD, "_step_recall_index_rebuild"),
    ]:
        def _make_noop(s=step):
            def _impl(self, _interrupt_check):
                return True, {}
            return _impl

        monkeypatch.setattr(
            SleepPipeline, method_name, _make_noop(),
        )

    # _run_essential_variable_tracker_hook fires before the step loop.
    # It walks records + edges via the MagicMock store and may crash on
    # .to_pandas() or .open_table(). Replace with a no-op so the CLI tests
    # stay focused on slot numbers / failure semantics and don't drag the
    # tracker into scope.
    monkeypatch.setattr(
        SleepPipeline,
        "_run_essential_variable_tracker_hook",
        lambda self: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_runs_pipeline_and_prints_progress(
    iai_root, monkeypatch, capsys,
):
    """sleep-cycle with no flags + no quarantine -> exit 0, 13 step lines.

    62-02 appended CLUSTER_SUMMARY (12) and RECALL_INDEX_REBUILD (13) as
    the final two REM steps after CRISIS_RECLUSTER. Full 13-step order:
    schema_mine, knob_tune, optimize_lance, compact_records, dream_decay,
    erasure_agent, cluster_replay, reconsolidation, user_model_update,
    dmn_reflection, crisis_recluster, cluster_summary, recall_index_rebuild.
    """
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sleep cycle started." in out
    assert "[1/13] schema_mine" in out
    assert "[2/13] knob_tune" in out
    assert "[3/13] optimize_lance" in out
    assert "[4/13] compact_records" in out
    assert "[5/13] dream_decay" in out
    assert "[6/13] erasure_agent" in out
    assert "[7/13] cluster_replay" in out
    assert "[8/13] reconsolidation" in out
    assert "[9/13] user_model_update" in out
    assert "[10/13] dmn_reflection" in out
    assert "[11/13] crisis_recluster" in out
    assert "[12/13] cluster_summary" in out
    assert "[13/13] recall_index_rebuild" in out
    assert "Sleep cycle complete" in out


def test_quarantined_without_force_returns_nonzero_with_message(
    iai_root, monkeypatch, capsys,
):
    """Active quarantine + no --force -> exit 1, hint in stderr."""
    _patch_store_open(monkeypatch)
    # Seed an active quarantine in the lifecycle_state.json that the
    # reloaded module now points at.
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "test stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, LIFECYCLE_STATE_PATH)

    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 1
    captured = capsys.readouterr()
    assert "quarantined" in captured.err.lower()
    assert "test stuck" in captured.err
    assert "--force" in captured.err
    assert "--reset-quarantine" in captured.err


def test_force_runs_pipeline_when_quarantined(
    iai_root, monkeypatch, capsys,
):
    """--force bypasses quarantine and runs all 8 steps."""
    _patch_store_open(monkeypatch)
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "test stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, LIFECYCLE_STATE_PATH)

    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args(force=True))
    assert rc == 0
    out = capsys.readouterr().out
    # 62-02: 13 steps total; RECALL_INDEX_REBUILD is now the last slot.
    assert "[13/13] recall_index_rebuild" in out
    assert "Sleep cycle complete" in out

    # force_run leaves quarantine record alone.
    record_after = load_state(LIFECYCLE_STATE_PATH)
    assert record_after["quarantine"] is not None


def test_reset_quarantine_clears_then_runs(
    iai_root, monkeypatch, capsys,
):
    """--reset-quarantine wipes quarantine first, then runs normally."""
    _patch_store_open(monkeypatch)
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, LIFECYCLE_STATE_PATH)

    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args(reset_quarantine=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Quarantine cleared." in out
    assert "Sleep cycle complete" in out

    record_after = load_state(LIFECYCLE_STATE_PATH)
    assert record_after["quarantine"] is None


def test_reset_quarantine_when_not_quarantined_no_op(
    iai_root, monkeypatch, capsys,
):
    """--reset-quarantine when no quarantine -> friendly no-op message."""
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args(reset_quarantine=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Quarantine not active" in out
    assert "Sleep cycle complete" in out


def test_failure_returns_nonzero_with_error_in_stderr(
    iai_root, monkeypatch, capsys,
):
    """A step exception -> exit 1, FAILED line in stderr."""
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    # Patch one specific step to raise.
    from iai_mcp.sleep_pipeline import SleepPipeline

    def _raiser(self, _interrupt_check):
        raise RuntimeError("synthetic optimize failure")

    monkeypatch.setattr(
        SleepPipeline, "_step_optimize_lance", _raiser,
    )

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 1
    captured = capsys.readouterr()
    # 62-02: 13 steps total. OPTIMIZE_LANCE is still slot 3 (NREM position 3).
    # The first 2 steps print to stdout (completed_steps),
    # then FAILED on stderr at slot 3/13.
    assert "[1/13] schema_mine" in captured.out
    assert "[2/13] knob_tune" in captured.out
    assert "[3/13] optimize_lance ... FAILED" in captured.err
    assert "synthetic optimize failure" in captured.err


def test_failure_after_3rd_strike_prints_quarantine_hint(
    iai_root, monkeypatch, capsys,
):
    """3rd consecutive same-step failure -> exit 1 + quarantine hint."""
    _patch_store_open(monkeypatch)
    _patch_pipeline_steps_to_noop(monkeypatch)

    from iai_mcp.sleep_pipeline import SleepPipeline

    def _raiser(self, _interrupt_check):
        raise RuntimeError("boom")

    monkeypatch.setattr(SleepPipeline, "_step_dream_decay", _raiser)

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    cmd_maintenance_sleep_cycle(_make_args())  # attempt=1
    cmd_maintenance_sleep_cycle(_make_args())  # attempt=2
    capsys.readouterr()  # discard accumulated output

    rc = cmd_maintenance_sleep_cycle(_make_args())  # attempt=3 -> quarantine
    assert rc == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err
    assert "quarantined for 24h" in captured.err
    assert "--reset-quarantine" in captured.err


def test_subparser_exposes_sleep_cycle_with_flags():
    """`iai-mcp maintenance sleep-cycle --force --reset-quarantine` parses."""
    from iai_mcp.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "maintenance", "sleep-cycle",
        "--force", "--reset-quarantine",
    ])
    assert args.force is True
    assert args.reset_quarantine is True
    # Defaults for store-path.
    assert args.store_path is None
    assert args.maintenance_cmd == "sleep-cycle"


def test_subparser_defaults_force_false_reset_false():
    """Default flag values: both False."""
    from iai_mcp.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["maintenance", "sleep-cycle"])
    assert args.force is False
    assert args.reset_quarantine is False


def test_store_open_failure_returns_2(
    iai_root, monkeypatch, capsys,
):
    """MemoryStore() raising -> CLI exits 2 with stderr message."""

    def _broken_store(path=None, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", _broken_store,
    )

    from iai_mcp.cli import cmd_maintenance_sleep_cycle

    rc = cmd_maintenance_sleep_cycle(_make_args())
    assert rc == 2
    err = capsys.readouterr().err
    assert "could not open MemoryStore" in err
    assert "disk full" in err
