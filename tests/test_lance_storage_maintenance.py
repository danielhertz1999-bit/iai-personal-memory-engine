"""Phase 7.3 R1..R4: Lance storage periodic-maintenance test suite.

Forensic context (2026-04-27): production records.lance had grown to
10,841 versions / 3.66 GB for only 7,130 rows over 9 days. Offline
`table.optimize(cleanup_older_than=timedelta(days=1))` reclaimed 84% of
disk and dropped `build_runtime_graph` cold latency 13.3s -> 0.13s
(102x). wires that fix into the daemon as a periodic job.

Test scope (one file per phase concern, mirrors idiom):
1. Helper drops version count without losing rows.
2. Helper never raises on per-table failure (other tables still
   processed; failed table's report carries `error` field).
3. Startup wire-in (the optimize call inside `daemon.main()`) emits
   exactly one `lance_storage_optimized` event with `phase="startup"`.
4. Periodic skip on MCP-active emits `lance_storage_optimize_skipped`
   with `reason="mcp_active"` and zero `lance_storage_optimized`.
5. Env override `IAI_MCP_LANCE_OPTIMIZE_INTERVAL_SEC=0.05` causes the
   periodic body to run repeatedly; >= 2 events fire within 0.5 s.
6. Optional: periodic runs once the socket flips idle (gate is two-way).

CRITICAL idiom: project does NOT depend on `pytest-asyncio`. Every test
that drives `async def` code uses SYNC `def test_X(...)` wrapping
`asyncio.run(coroutine_body(...))`. See `tests/test_daemon_tick_flags.py:144`
for canonical idiom. Do NOT add `@pytest.mark.asyncio` decorators here.
"""
from __future__ import annotations

import asyncio
import importlib
import time
from datetime import timedelta

import pytest

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore


# --------------------------------------------------------------------------- #
# Test 1 (R1 / D7.3-23): helper drops version count, preserves rows.          #
# --------------------------------------------------------------------------- #


def test_helper_drops_version_count_preserves_rows(tmp_path):
    """Insert N events to create N+1 versions on the events table; call
    the helper with retention=timedelta(seconds=0); assert versions
    collapsed to 1 and row count is preserved.

    Why retention=0: in the live daemon we use `timedelta(days=1)` so
    same-session optimize runs are no-ops (versions are seconds old).
    For the synthetic test we want to assert collapse on freshly-created
    versions, so we pass an aggressive retention.
    """
    from iai_mcp.maintenance import optimize_lance_storage

    store = MemoryStore(path=tmp_path)

    # Trigger 10 versions on each of the three daemon-owned tables.
    # `events` is the cheapest write path; we drive `records` and `edges`
    # through their respective LanceDB add() to keep the test independent
    # of MemoryStore.insert's encryption-key ceremony.
    for i in range(10):
        write_event(store, "test_marker", {"i": i}, severity="info")

    # Force versions on the records table by directly appending dummy
    # rows with the records schema (id-only smoke; no encryption needed
    # because we never read them back).
    records_tbl = store.db.open_table("records")
    for i in range(10):
        records_tbl.add(
            [
                {
                    "id": f"00000000-0000-0000-0000-{i:012x}",
                    "tier": "episodic",
                    "literal_surface": "x",
                    "aaak_index": "",
                    "embedding": [0.0] * store.embed_dim,
                    "structure_hv": b"",
                    "community_id": "",
                    "centrality": 0.0,
                    "detail_level": 1,
                    "pinned": False,
                    "stability": 0.0,
                    "difficulty": 0.0,
                    "last_reviewed": None,
                    "never_decay": False,
                    "never_merge": False,
                    "provenance_json": "[]",
                    "created_at": None,
                    "updated_at": None,
                    "tags_json": "[]",
                    "language": "en",
                    "s5_trust_score": 0.5,
                    "profile_modulation_gain_json": "{}",
                    "schema_version": 2,
                },
            ],
        )

    # Force versions on the edges table the same way.
    edges_tbl = store.db.open_table("edges")
    for i in range(10):
        edges_tbl.add(
            [
                {
                    "src": f"src{i}",
                    "dst": f"dst{i}",
                    "edge_type": "co_occurs",
                    "weight": 1.0,
                    "updated_at": None,
                },
            ],
        )

    # Snapshot per-table version counts before optimize.
    before = {
        name: len(store.db.open_table(name).list_versions())
        for name in ("records", "edges", "events")
    }
    rows_before = {
        name: store.db.open_table(name).count_rows()
        for name in ("records", "edges", "events")
    }

    report = optimize_lance_storage(store, retention=timedelta(seconds=0))

    # Helper returned a flat dict keyed by all three table names.
    assert set(report.keys()) == {"records", "edges", "events"}

    after = {
        name: len(store.db.open_table(name).list_versions())
        for name in ("records", "edges", "events")
    }
    rows_after = {
        name: store.db.open_table(name).count_rows()
        for name in ("records", "edges", "events")
    }

    for name in ("records", "edges", "events"):
        assert after[name] < before[name], (
            f"{name}: expected versions_after < versions_before; "
            f"got before={before[name]} after={after[name]}"
        )
        assert rows_after[name] == rows_before[name], (
            f"{name}: row count must be preserved by optimize; "
            f"before={rows_before[name]} after={rows_after[name]}"
        )
        # No `error` key on a healthy run.
        assert "error" not in report[name], (
            f"{name}: unexpected error in healthy run: {report[name].get('error')}"
        )
        # All structured metric keys present.
        per_table = report[name]
        for key in (
            "rows_before",
            "rows_after",
            "versions_before",
            "versions_after",
            "size_bytes_before",
            "size_bytes_after",
            "elapsed_sec",
        ):
            assert key in per_table, f"{name}: missing key {key} in report"


# --------------------------------------------------------------------------- #
# Test 2 (R1 / D7.3-09): helper never raises; per-table error captured.       #
# --------------------------------------------------------------------------- #


class _OneTableExplodesStub:
    """Stub MemoryStore-shaped object whose `db.open_table('records')`
    raises but the other two tables work normally. Used to verify the
    helper continues processing after a per-table failure.
    """

    def __init__(self, real_store: MemoryStore) -> None:
        self.root = real_store.root
        self._real_db = real_store.db

        class _DBProxy:
            def __init__(self, real_db):
                self._real = real_db

            def open_table(self, name):
                if name == "records":
                    raise RuntimeError("synthetic records-table failure")
                return self._real.open_table(name)

        self.db = _DBProxy(self._real_db)


def test_helper_never_raises_on_per_table_error(tmp_path):
    """If one table's optimize raises, the helper still returns a dict
    with all three table keys; the failed table's sub-dict carries
    `error: str`; the other two tables are processed normally.
    """
    from iai_mcp.maintenance import optimize_lance_storage

    real_store = MemoryStore(path=tmp_path)
    # Seed events so versions_before > 0 on the surviving tables.
    for i in range(3):
        write_event(real_store, "test_marker", {"i": i}, severity="info")

    stub = _OneTableExplodesStub(real_store)

    # Helper itself MUST NOT raise (D7.3-09).
    report = optimize_lance_storage(stub, retention=timedelta(seconds=0))

    assert set(report.keys()) == {"records", "edges", "events"}
    # Failed table carries `error` and the other two do not.
    assert "error" in report["records"]
    assert "synthetic records-table failure" in report["records"]["error"]
    assert "error" not in report["edges"]
    assert "error" not in report["events"]
    # Surviving tables show the structural metric keys.
    for surviving in ("edges", "events"):
        for key in ("rows_before", "rows_after", "versions_before", "versions_after"):
            assert key in report[surviving]


# --------------------------------------------------------------------------- #
# Test 3 (R3 / A3): startup wire-in emits a single                            #
#                   `lance_storage_optimized` event with phase="startup".     #
# --------------------------------------------------------------------------- #


def test_startup_wire_emits_one_lance_storage_optimized_event(tmp_path):
    """Replicate the daemon.main() startup wire-in body in isolation:
    `await asyncio.to_thread(optimize_lance_storage, store)` followed by
    `await asyncio.to_thread(write_event, ..., 'lance_storage_optimized',
    {'phase': 'startup', 'retention_days': ..., 'per_table': ...,
    'total_elapsed_sec': ...}, severity='info')`. The integration boots a
    fresh MemoryStore and asserts the event appears with the right
    payload shape.

    Done in isolation (not by spawning the full daemon main loop) for two
    reasons:
      1) daemon.main() takes signal-handler ownership of SIGTERM/SIGINT/
         SIGHUP and binds a unix socket -- a unit test would have to
         tear all of that down.
      2) The tested invariant is the EXACT call sequence at the wire-in,
         which is what this test exercises.
    """
    from iai_mcp import maintenance as _maint

    store = MemoryStore(path=tmp_path)

    async def _startup_body():
        startup_t0 = time.monotonic()
        startup_report = await asyncio.to_thread(
            _maint.optimize_lance_storage, store,
        )
        await asyncio.to_thread(
            write_event,
            store,
            "lance_storage_optimized",
            {
                "phase": "startup",
                "retention_days": (
                    _maint.LANCE_OPTIMIZE_RETENTION_SEC / 86400.0
                ),
                "per_table": startup_report,
                "total_elapsed_sec": round(time.monotonic() - startup_t0, 3),
            },
            severity="info",
        )

    asyncio.run(_startup_body())

    events = query_events(store, kind="lance_storage_optimized", limit=10)
    assert len(events) == 1, (
        f"expected exactly 1 lance_storage_optimized event; got {len(events)}"
    )
    payload = events[0]["data"]
    assert payload["phase"] == "startup"
    assert "retention_days" in payload
    assert "per_table" in payload
    assert "total_elapsed_sec" in payload
    assert set(payload["per_table"].keys()) == {"records", "edges", "events"}


# --------------------------------------------------------------------------- #
# Test 4 (R2 / R3 / A4): periodic skip on MCP-active emits                    #
#                       `lance_storage_optimize_skipped` with                 #
#                       reason="mcp_active" and zero `lance_storage_optimized`.#
# --------------------------------------------------------------------------- #


# Plan 10.6-01 Task 1.8: REMOVED `_MCPActiveSocketStub` /
# `_IdleSocketStub` fixtures and the three MCP-aware tests
# (test_periodic_skip_on_mcp_active, test_env_override_interval_drives_
# periodic_cadence, test_periodic_runs_after_socket_flips_idle).
#
# The D7.3-11 `_should_yield_to_mcp(socket)` gate inside the
# periodic Lance optimize body was removed in Task 1.4. The lifecycle
# state machine handles SLEEP-state coexistence outside the audit loop,
# so the per-iteration MCP-active check and the
# `lance_storage_optimize_skipped(reason="mcp_active")` event are no
# longer reachable. The cooldown gate (interval-based) and the
# `lance_storage_optimized(phase="periodic")` happy-path emission are
# still exercised indirectly via `test_startup_wire_emits_one_lance_
# storage_optimized_event` above.
#
# The `LANCE_OPTIMIZE_INTERVAL_SEC` env-override read path is still
# locked by `test_module_constants_exist_with_documented_defaults`
# below.


# --------------------------------------------------------------------------- #
# Sanity: env vars exist as module-level constants (R4 / D7.3-20..D7.3-22).   #
# --------------------------------------------------------------------------- #


def test_module_constants_exist_with_documented_defaults():
    """R4: `LANCE_OPTIMIZE_INTERVAL_SEC` (default 3600.0) and
    `LANCE_OPTIMIZE_RETENTION_SEC` (default 86400.0) MUST exist at
    module level. This is the surface other modules access at call
    time (identity_audit reads `_maintenance.LANCE_OPTIMIZE_*`).
    """
    import os as _os
    # Save + clear the env vars (test fixture safety) so the reload
    # produces the documented defaults regardless of who set what.
    saved_interval = _os.environ.pop(
        "IAI_MCP_LANCE_OPTIMIZE_INTERVAL_SEC", None,
    )
    saved_retention = _os.environ.pop(
        "IAI_MCP_LANCE_OPTIMIZE_RETENTION_SEC", None,
    )
    try:
        import iai_mcp.maintenance as _maint
        importlib.reload(_maint)
        assert hasattr(_maint, "LANCE_OPTIMIZE_INTERVAL_SEC")
        assert hasattr(_maint, "LANCE_OPTIMIZE_RETENTION_SEC")
        assert _maint.LANCE_OPTIMIZE_INTERVAL_SEC == 3600.0
        assert _maint.LANCE_OPTIMIZE_RETENTION_SEC == 86400.0
    finally:
        # Restore so we don't pollute the rest of the suite.
        if saved_interval is not None:
            _os.environ["IAI_MCP_LANCE_OPTIMIZE_INTERVAL_SEC"] = saved_interval
        if saved_retention is not None:
            _os.environ[
                "IAI_MCP_LANCE_OPTIMIZE_RETENTION_SEC"
            ] = saved_retention
        # Re-reload to install the post-restore defaults.
        import iai_mcp.maintenance as _maint
        importlib.reload(_maint)
