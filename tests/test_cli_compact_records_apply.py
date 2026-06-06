"""`iai-mcp maintenance compact-hippo --apply` end-to-end test.

The sibling `tests/test_cli_maintenance_compact_records.py` mocks MemoryStore
and optimize_hippo_storage; this test uses a real seeded store so any
regression in the apply path (wrong arg types, bad metrics query, etc.)
is caught against live Hippo/SQLite.

Contract:
    (a) `_maintenance_compact_apply` returns 0 against a real seeded store.
    (b) Row count is unchanged after optimize (WAL checkpoint + VACUUM
        does not delete rows; verbatim-recall invariant holds).
    (c) The audit JSON at `<fake_home>/.iai-mcp/.maintenance-compact-<ts>.json`
        carries `status: "ok"`.
    (d) The pre/post `record_id_set` values agree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

# The canonical record factory lives in a sibling test file. Adding `tests/`
# to sys.path makes the helper importable without packaging.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from iai_mcp.store import MemoryStore
from test_capture_dedup_contract import _make_record


# --------------------------------------------------------------------------- fixtures
# Pattern copied verbatim from tests/test_capture_dedup_contract.py
# (`_isolated_keyring` autouse fixture is the project canon for tests touching
# encrypted records on the construction host where the real keyring is absent
# or hangs).


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


# --------------------------------------------------------------------------- contract


def test_apply_runs_to_completion_against_real_lancedb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end apply against a real MemoryStore (Hippo/SQLite backend).

    Verifies that _maintenance_compact_apply returns rc=0, writes an audit
    JSON with status "ok", and preserves the record-id invariant across
    optimize_hippo_storage (WAL checkpoint + VACUUM + hnswlib rebuild).

    The LanceDB version-count assertion (b) from the original test is
    replaced with a row-count invariant: SQLite has no MVCC, VACUUM does
    not delete rows, so count_before == count_after proves the verbatim-
    recall invariant holds after compaction.
    """
    # Patch Path.home() so the audit JSON lands inside tmp_path, not in the
    # operator's real ~/.iai-mcp/. The cli module re-imports Path.home() per
    # call so monkeypatching the class method is sufficient.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # IAI_MCP_STORE env overrides constructor path inside MemoryStore.__init__
    # (see store.py). Clear it so the `path=` kwarg is honoured.
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    store_path = tmp_path
    # Hippo storage directory (brain.sqlite3 lives here).
    hippo_dir = store_path / "hippo"

    # Seed 5 records then delete one to produce a non-trivial write history.
    store = MemoryStore(path=store_path)
    seeded_ids: set[str] = set()
    for i in range(5):
        rid = uuid4()
        store.insert(_make_record(rid, surface=f"compact-apply-fixture-{i}"))
        seeded_ids.add(str(rid))

    update_rid = uuid4()
    store.insert(_make_record(update_rid, surface="compact-apply-fixture-update"))
    seeded_ids.add(str(update_rid))
    store.delete(update_rid)
    seeded_ids.discard(str(update_rid))

    # Snapshot row count BEFORE apply.
    records_tbl = store.db.open_table("records")
    rows_before = records_tbl.count_rows()
    assert rows_before == len(seeded_ids), (
        f"row count before ({rows_before}) must match seeded id count "
        f"({len(seeded_ids)})"
    )

    # Drive the code path under test.
    from iai_mcp.cli import _maintenance_compact_apply

    rc = _maintenance_compact_apply(store_path, hippo_dir)

    # (a) Apply returned 0.
    assert rc == 0, f"apply exit code: {rc}"

    # (b) Row count preserved after optimize (verbatim-recall invariant).
    records_tbl_after = store.db.open_table("records")
    rows_after = records_tbl_after.count_rows()
    assert rows_after == rows_before, (
        f"row count changed after optimize: before={rows_before} after={rows_after}"
    )

    # (c) Audit file at <fake_home>/.iai-mcp/.maintenance-compact-<ts>.json
    # carries status: "ok". Glob the directory since the timestamp varies.
    audit_dir = fake_home / ".iai-mcp"
    audit_files = sorted(audit_dir.glob(".maintenance-compact-*.json"))
    assert audit_files, (
        f"expected audit file under {audit_dir}; found {list(audit_dir.iterdir())}"
    )
    # The FAILED branch writes `.maintenance-compact-FAILED-<ts>.json`; ensure
    # the OK branch matched.
    ok_files = [
        p for p in audit_files if "FAILED" not in p.name
    ]
    assert ok_files, (
        f"expected non-FAILED audit file; got {[p.name for p in audit_files]}"
    )
    payload = json.loads(ok_files[-1].read_text())
    assert payload["status"] == "ok", payload

    # (d) Pre/post record_id_set agree — cross-check post-state against seeded set.
    df_post = (
        records_tbl_after.search()
        .select(["id"])
        .to_pandas()
    )
    post_id_set = {str(x) for x in df_post["id"].tolist()}
    assert post_id_set == seeded_ids, (
        f"post id-set differs from seeded set; "
        f"missing={seeded_ids - post_id_set} extra={post_id_set - seeded_ids}"
    )
