"""Tests verifying the sleep pipeline cleanup for Hippo storage.

Exercises:
- _maybe_self_heal_version_pileup is deleted (no more LanceDB version-pileup path)
- _step_optimize_lance renamed to _step_compact_hippo
- _step_compact_records replaced with _step_compact_records_noop (resume-token stub)
- SleepStep enum values OPTIMIZE_LANCE=4 / COMPACT_RECORDS=5 are frozen
- _step_compact_hippo calls optimize_hippo_storage (not optimize_lance_storage)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iai_mcp.sleep_pipeline import SleepPipeline, SleepStep


# ---------------------------------------------------------------------------
# self-heal predicate machinery is gone
# ---------------------------------------------------------------------------

class TestSelfHealDeleted:
    def test_maybe_self_heal_version_pileup_does_not_exist(self) -> None:
        """_maybe_self_heal_version_pileup must not exist on SleepPipeline.

        The LanceDB version-pileup scenario cannot occur under Hippo
        (SQLite has no MVCC version manifests). The entire code path is
        removed.
        """
        assert not hasattr(SleepPipeline, "_maybe_self_heal_version_pileup"), (
            "_maybe_self_heal_version_pileup still present — should have been deleted"
        )


# ---------------------------------------------------------------------------
# enum values frozen
# ---------------------------------------------------------------------------

class TestEnumValuesFrozen:
    def test_sleep_step_enum_values_preserved(self) -> None:
        """SleepStep.OPTIMIZE_LANCE==4 and COMPACT_RECORDS==5 are frozen.

        Crash-window resume tokens reference these integer values. Renumbering
        would break resume-from-crash for in-flight sleep cycles.
        """
        assert SleepStep.OPTIMIZE_LANCE.value == 4, (
            f"OPTIMIZE_LANCE value changed: {SleepStep.OPTIMIZE_LANCE.value}"
        )
        assert SleepStep.COMPACT_RECORDS.value == 5, (
            f"COMPACT_RECORDS value changed: {SleepStep.COMPACT_RECORDS.value}"
        )


# ---------------------------------------------------------------------------
# method rename and stub
# ---------------------------------------------------------------------------

class TestMethodRename:
    def test_step_optimize_lance_renamed_to_step_compact_hippo(self) -> None:
        """_step_compact_hippo must exist. _step_optimize_lance may exist
        as a compatibility alias for monkeypatch support but is not the
        canonical implementation."""
        assert hasattr(SleepPipeline, "_step_compact_hippo"), (
            "_step_compact_hippo is missing"
        )

    def test_step_compact_records_is_resume_noop(self) -> None:
        """_step_compact_records_noop returns (True, {'action': 'noop_under_hippo'}).

        The method must not call optimize_hippo_storage — it is a pure
        no-op forward-compat stub that preserves the COMPACT_RECORDS(=5)
        resume-token slot without doing any I/O.
        """
        pipeline = SleepPipeline.__new__(SleepPipeline)
        # Patch optimize_hippo_storage to detect if it gets called
        with patch(
            "iai_mcp.maintenance.optimize_hippo_storage",
        ) as mock_opt:
            done, payload = pipeline._step_compact_records_noop(
                interrupt_check=None,
            )

        assert done is True, f"Expected done=True, got {done!r}"
        assert payload == {"action": "noop_under_hippo"}, (
            f"Unexpected noop payload: {payload!r}"
        )
        mock_opt.assert_not_called()


# ---------------------------------------------------------------------------
# _step_compact_hippo calls optimize_hippo_storage
# ---------------------------------------------------------------------------

class TestCompactHippoCallsOptimizeHippoStorage:
    def test_compact_hippo_step_calls_optimize_hippo_storage(self) -> None:
        """_step_compact_hippo must call optimize_hippo_storage (not
        optimize_lance_storage) and return (True, <dict>).

        Monkeypatches optimize_hippo_storage to return a minimal per-table
        report and verifies the call is made with self._store as the sole
        positional argument.
        """
        pipeline = SleepPipeline.__new__(SleepPipeline)

        # Minimal store mock: provides.db.open_table() returning a table
        # where count_rows() returns 0 and delete / update are no-ops.
        mock_tbl = MagicMock()
        mock_tbl.count_rows.return_value = 0

        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_tbl

        mock_store = MagicMock()
        mock_store.db = mock_db

        pipeline._store = mock_store

        fake_report = {"records": {"rows_before": 10, "rows_after": 10}}

        # Patch all inline imports used by _step_compact_hippo.
        # Inline imports resolve from their source module, not from
        # sleep_pipeline's namespace, so we patch at the source.
        with (
            patch(
                "iai_mcp.maintenance.optimize_hippo_storage",
                return_value=fake_report,
            ) as mock_opt,
            patch(
                "iai_mcp.daemon_config._load_erasure_config",
            ) as mock_cfg,
            patch("iai_mcp.events.write_event"),
        ):
            from iai_mcp.daemon_config import ErasureConfig
            mock_cfg.return_value = ErasureConfig(
                centrality_threshold=0.5,
                age_days=30,
                retrieval_window_days=7,
                tombstone_ttl_sec=86400,
                dry_run=False,
            )
            done, payload = pipeline._step_compact_hippo(interrupt_check=None)

        assert done is True, f"Expected done=True, got {done!r}"
        mock_opt.assert_called_once_with(mock_store)
