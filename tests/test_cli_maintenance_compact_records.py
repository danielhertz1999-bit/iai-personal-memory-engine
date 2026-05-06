"""Plan 07.14-01 tests: `iai-mcp maintenance compact-records`.

Eight cases:
  1. test_dry_run_prints_metrics_no_optimize_call
  2. test_apply_with_yes_runs_optimize
  3. test_preflight_refuses_when_daemon_alive
  4. test_preflight_skips_when_daemon_state_missing
  5. test_record_id_set_invariant_aborts_on_divergence
  6. test_audit_file_written_on_apply
  7. test_dry_run_no_audit_file
  8. test_yes_required_with_apply_in_non_tty

All tests use mocked `MemoryStore` + mocked `optimize_lance_storage` +
mocked `psutil` — zero real LanceDB I/O, zero real embedder load,
combined wall-clock target < 5s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace with default flag values, overridable."""
    defaults = dict(
        dry_run=False,
        apply=False,
        yes=False,
        store_path=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _patch_psutil_alive(
    monkeypatch: pytest.MonkeyPatch, *, pid: int, cmdline: list[str],
) -> None:
    """Make psutil.Process(pid).cmdline() return the given list.

    Mirrors the pattern in tests/test_doctor_checklist.py — we patch
    sys.modules["psutil"] so the function-scope `import psutil` inside
    `_maintenance_compact_preflight_daemon_alive` resolves to the mock.
    """
    fake_proc = MagicMock()
    fake_proc.cmdline.return_value = cmdline
    fake_psutil = MagicMock()
    fake_psutil.Process.return_value = fake_proc
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)


def _make_optimize_report(
    *, versions_before: int = 3, versions_after: int = 1,
    rows_before: int = 0, rows_after: int = 0,
) -> dict:
    """Construct an optimize_lance_storage-shaped report (3 tables)."""
    base = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "versions_before": versions_before,
        "versions_after": versions_after,
        "size_bytes_before": 0,
        "size_bytes_after": 0,
        "elapsed_sec": 0.0,
    }
    return {
        "records": dict(base),
        "edges": dict(base, versions_before=0, versions_after=0),
        "events": dict(base, versions_before=0, versions_after=0),
    }


def _make_fake_store(record_ids: list[str]) -> MagicMock:
    """Construct a MagicMock MemoryStore exposing tbl.count_rows() +
    tbl.to_pandas(columns=['id']) for the given record-id list.
    """
    fake_store = MagicMock()
    fake_tbl = MagicMock()
    fake_tbl.count_rows.return_value = len(record_ids)
    fake_df = MagicMock()
    fake_df.__getitem__.return_value.tolist.return_value = list(record_ids)
    fake_tbl.to_pandas.return_value = fake_df
    fake_store.db.open_table.return_value = fake_tbl
    return fake_store


# ---------------------------------------------------------------------------
# Fixture: HOME-isolated IAI root with records.lance skeleton
# ---------------------------------------------------------------------------


@pytest.fixture
def iai_root(tmp_path, monkeypatch):
    """Sandbox HOME → tmp_path; pre-create
    `~/.iai-mcp/lancedb/records.lance` skeleton with `_versions/` subdir
    holding 3 fake manifests so the size/version walk has data to
    measure.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring"
    )
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase")
    try:
        import keyring.core
        keyring.core._keyring_backend = None
    except ImportError:
        pass

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir()
    records_lance = iai_dir / "lancedb" / "records.lance"
    records_lance.mkdir(parents=True)
    versions_dir = records_lance / "_versions"
    versions_dir.mkdir()
    for i in range(3):
        (versions_dir / f"{i:020d}.manifest").write_bytes(b"x" * 100)
    # Reload cli to pick up new HOME — STATE_PATH/LOCK_PATH/SOCKET_PATH are
    # module-scope Path.home() captures.
    import importlib
    from iai_mcp import cli as _cli
    importlib.reload(_cli)
    yield iai_dir
    importlib.reload(_cli)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dry_run_prints_metrics_no_optimize_call(iai_root, capsys):
    """--dry-run emits metrics-only JSON; mocked optimize never called."""
    from iai_mcp.cli import cmd_maintenance_compact_records
    with patch(
        "iai_mcp.maintenance.optimize_lance_storage"
    ) as mock_opt:
        rc = cmd_maintenance_compact_records(_make_args(dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "dry-run"
    assert "versions_count" in payload["metrics"]["pre"]
    assert "size_mb" in payload["metrics"]["pre"]
    assert "records_count" in payload["metrics"]["pre"]
    assert payload["metrics"]["post"] is None
    mock_opt.assert_not_called()


def test_apply_with_yes_runs_optimize(iai_root, monkeypatch, capsys):
    """Mocked optimize → `--apply --yes` calls it once with retention=0d."""
    from iai_mcp import cli as _cli

    fake_store = _make_fake_store(["id1", "id2", "id3", "id4", "id5"])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    mock_opt = MagicMock(return_value=_make_optimize_report(
        versions_before=3, versions_after=1,
        rows_before=5, rows_after=5,
    ))
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_lance_storage", mock_opt,
    )

    rc = _cli.cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 0
    assert mock_opt.call_count == 1
    _, kwargs = mock_opt.call_args
    assert kwargs["retention"] == timedelta(days=0)


def test_preflight_refuses_when_daemon_alive(iai_root, monkeypatch, capsys):
    """If daemon-state.json points at a live `iai_mcp.daemon` process,
    --apply --yes refuses with rc=1 + 'daemon running' in stderr.
    """
    state_path = iai_root / ".daemon-state.json"
    state_path.write_text(json.dumps({"daemon_pid": os.getpid()}))
    _patch_psutil_alive(
        monkeypatch, pid=os.getpid(),
        cmdline=["python", "-m", "iai_mcp.daemon"],
    )
    # os.kill(os.getpid(), 0) succeeds — process exists.

    from iai_mcp.cli import cmd_maintenance_compact_records
    with patch(
        "iai_mcp.maintenance.optimize_lance_storage"
    ) as mock_opt:
        rc = cmd_maintenance_compact_records(
            _make_args(apply=True, yes=True),
        )
    assert rc == 1
    err = capsys.readouterr().err
    assert "daemon running" in err
    mock_opt.assert_not_called()


def test_preflight_skips_when_daemon_state_missing(
    iai_root, monkeypatch, capsys,
):
    """No .daemon-state.json → preflight passes; optimize is called."""
    assert not (iai_root / ".daemon-state.json").exists()

    fake_store = _make_fake_store([])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    mock_opt = MagicMock(return_value=_make_optimize_report(
        versions_before=3, versions_after=1,
    ))
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_lance_storage", mock_opt,
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 0
    assert mock_opt.call_count == 1


def test_record_id_set_invariant_aborts_on_divergence(
    iai_root, monkeypatch, capsys,
):
    """Pre id-set has 3 ids; post id-set has 2. Abort + FAILED audit."""
    fake_store = _make_fake_store(["id1", "id2", "id3"])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_lance_storage",
        MagicMock(return_value=_make_optimize_report(
            versions_before=3, versions_after=1,
            rows_before=3, rows_after=2,
        )),
    )
    # Patch _maintenance_compact_metrics to return divergent id-sets across
    # its two invocations (pre, post).
    pre_set = {"id1", "id2", "id3"}
    post_set = {"id1", "id2"}
    metrics_seq = [
        {
            "versions_count": 3, "size_mb": 0.0,
            "records_count": 3, "record_id_set": pre_set,
        },
        {
            "versions_count": 1, "size_mb": 0.0,
            "records_count": 2, "record_id_set": post_set,
        },
    ]
    call_counter = {"n": 0}

    def _stub_metrics(*args, **kwargs):
        i = call_counter["n"]
        call_counter["n"] += 1
        return metrics_seq[min(i, 1)]

    monkeypatch.setattr(
        "iai_mcp.cli._maintenance_compact_metrics", _stub_metrics,
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "ABORT" in err
    assert "divergence" in err

    # FAILED audit file must exist.
    failed = list(iai_root.glob(".maintenance-compact-FAILED-*.json"))
    assert len(failed) == 1
    payload = json.loads(failed[0].read_text())
    assert payload["status"] == "aborted"
    assert payload["reason"] == "record_id_set divergence post-optimize"
    assert payload["missing_ids_count"] == 1


def test_audit_file_written_on_apply(iai_root, monkeypatch, capsys):
    """--apply --yes happy path → audit JSON with status=ok + pre/post."""
    fake_store = _make_fake_store(["id1", "id2"])
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore", lambda path=None, **kw: fake_store,
    )
    monkeypatch.setattr(
        "iai_mcp.maintenance.optimize_lance_storage",
        MagicMock(return_value=_make_optimize_report(
            versions_before=3, versions_after=1,
            rows_before=2, rows_after=2,
        )),
    )

    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=True),
    )
    assert rc == 0

    audits = list(iai_root.glob(".maintenance-compact-*.json"))
    audits = [a for a in audits if "FAILED" not in a.name]
    assert len(audits) == 1, (
        f"expected exactly 1 audit file, got {audits}"
    )
    payload = json.loads(audits[0].read_text())
    assert payload["status"] == "ok"
    assert "metrics_pre" in payload
    assert "metrics_post" in payload
    assert "elapsed_sec" in payload


def test_dry_run_no_audit_file(iai_root, capsys):
    """--dry-run never writes a `.maintenance-compact-*.json` file."""
    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(_make_args(dry_run=True))
    assert rc == 0
    audits = list(iai_root.glob(".maintenance-compact-*.json"))
    assert audits == []


def test_yes_required_with_apply_in_non_tty(iai_root, monkeypatch, capsys):
    """--apply on non-tty without --yes → exit 2, friendly hint."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from iai_mcp.cli import cmd_maintenance_compact_records
    rc = cmd_maintenance_compact_records(
        _make_args(apply=True, yes=False),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires --yes" in err
