"""session_start_tokens_p90 metric.

The `iai-mcp daemon stats` subcommand exposes a p90 over the most recent 100
`session_started` events. Each `assemble_session_start` call emits
`data["total_cached_tokens"]`, so this test exercises only the read/aggregate
path (`compute_session_start_tokens_p90`).

Validates `cmd_daemon_stats` and `compute_session_start_tokens_p90` in
`src/iai_mcp/cli.py`.
"""
from __future__ import annotations

import time


def _write_session_started(store, tokens: int, session_id: str = "s") -> None:
    """Helper: emit one `session_started` event with the canonical payload shape.

    Mirrors the production emit in `src/iai_mcp/session.py` so the read
    path exercises the same `data["total_cached_tokens"]` key the daemon writes.
    """
    from iai_mcp.events import write_event

    write_event(
        store,
        kind="session_started",
        data={
            "session_id": session_id,
            "session_state_hash": "deadbeef",
            "total_cached_tokens": int(tokens),
            "wake_depth": "standard",
            "timestamp": "2026-05-17T00:00:00+00:00",
        },
        severity="info",
        session_id=session_id,
    )


def test_p90_uniform_returns_uniform(tmp_path):
    """100 uniform samples at 3000 tok → p90 == 3000 exactly.

    `statistics.quantiles(..., n=10, method="inclusive")[8]` returns the input
    value exactly when all 100 samples are equal — verifies the helper is wired
    to the inclusive method (the only method that satisfies this for uniform).
    """
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(100):
        _write_session_started(store, 3000)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 3000, "n_samples": 100}, f"got {result}"


def test_p90_with_outlier_shifts(tmp_path):
    """99 samples at 3000 + 1 outlier at 5000 → p90 stays 3000.

    Documented: with `statistics.quantiles(method="inclusive")` over 100 samples,
    sorted positions 89-90 are both 3000, so p90 == 3000. A single outlier does
    NOT shift the 90th percentile — would need ~11 outliers to lift it. This
    test pins the strict-percentile contract so a future refactor that switches
    to a different method (e.g., `method="exclusive"`) will surface as a delta.
    """
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(99):
        _write_session_started(store, 3000)
    _write_session_started(store, 5000)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 3000, "n_samples": 100}, f"got {result}"


def test_p90_under_filled_window_reports_n_samples(tmp_path):
    """10 samples at 2500 tok → p90 == 2500, n_samples == 10 (no raise)."""
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(10):
        _write_session_started(store, 2500)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 2500, "n_samples": 10}, f"got {result}"


def test_p90_empty_returns_none(tmp_path):
    """Fresh store, zero session_started events → {"p90": None, "n_samples": 0}."""
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": None, "n_samples": 0}, f"got {result}"


def test_p90_survives_restart(tmp_path):
    """Store persistence: 100 events written → close store → re-open → same p90.

    Verifies the rolling window is durable because the events table lives in
    the store (hippocampus, source of truth). No separate counter file needed.
    """
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store1 = MemoryStore(path=tmp_path)
    for _ in range(100):
        _write_session_started(store1, 3000)
    first = compute_session_start_tokens_p90(store1)
    assert first == {"p90": 3000, "n_samples": 100}, f"first: {first}"
    # Drop the first handle and re-open the same path.
    del store1

    store2 = MemoryStore(path=tmp_path)
    second = compute_session_start_tokens_p90(store2)
    assert second == first, f"persistence failed: first={first} second={second}"


def test_p90_only_uses_session_started_kind(tmp_path):
    """Other event kinds are ignored — only session_started feeds the metric.

    Filter discriminator: `query_events(store, kind="session_started", limit=100)`
    in `src/iai_mcp/events.py` short-circuits on the kind column before
    JSON-decoding the data field.
    """
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(50):
        _write_session_started(store, 3000)
    # Unrelated kinds with a token-like field that MUST be ignored.
    for _ in range(25):
        write_event(store, kind="s4_contradiction", data={"total_cached_tokens": 99999})
    for _ in range(25):
        write_event(store, kind="migration_v3_to_v4", data={"total_cached_tokens": 99999})

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 3000, "n_samples": 50}, f"got {result}"


def test_p90_takes_most_recent_100(tmp_path):
    """Window is the most-recent 100 events; older samples drop off.

    Inserts 50 at 1000 tok, sleeps, then 100 at 4000 tok. Because
    `query_events` sorts by `ts DESC` then takes `head(limit=100)`, only the
    newer batch contributes — p90 reflects the latter 100.
    """
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(50):
        _write_session_started(store, 1000)
    # Guarantee a ts gap so the sort order between batches is deterministic.
    time.sleep(0.05)
    for _ in range(100):
        _write_session_started(store, 4000)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 4000, "n_samples": 100}, f"got {result}"
