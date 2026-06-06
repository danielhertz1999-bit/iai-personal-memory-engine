"""Tests for the _hippo_health_check_on_boot boot integrity check.

Verifies:
1. Fresh store -> hippo_boot_health event with action="ok".
2. Corrupted records.hnsw -> hnsw access still returns result (no crash);
   action driven by _label_map parity, not raw hnswlib count.
3. _run_bounded_startup_optimize does NOT exist in daemon module.
4. _post_bind_full_optimize does NOT exist in daemon module.
5. IAI_MCP_SKIP_STARTUP_OPTIMIZE env-var handling does NOT exist in daemon module.

All tests use pytest ``tmp_path`` fixture — no ~/.iai-mcp/ touched.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from iai_mcp.daemon import _hippo_health_check_on_boot
from iai_mcp.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> MemoryStore:
    """Open a MemoryStore backed by a fresh temporary directory."""
    return MemoryStore(path=tmp_path)


# ---------------------------------------------------------------------------
# 1. Clean store -> ok
# ---------------------------------------------------------------------------


def test_hippo_health_check_on_clean_store(tmp_path: Path) -> None:
    """Fresh empty store: sqlite_count == hnsw_active_count == 0, action='ok'."""
    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    assert result["action"] == "ok", f"Expected ok, got: {result}"
    assert result["sqlite_count"] == 0
    assert result["hnsw_active_count"] == 0
    assert "hnsw_raw_count" in result


# ---------------------------------------------------------------------------
# 2. Corrupted records.hnsw -> function returns dict, no crash
# ---------------------------------------------------------------------------


def test_hippo_health_check_on_corrupted_hnsw(tmp_path: Path) -> None:
    """Corrupt records.hnsw before opening store.

    HippoDB.__init__ rebuilds the index from SQLite on load failure, so by
    the time _hippo_health_check_on_boot runs the index is valid. The check
    should still complete without raising. On an empty store the action is 'ok'.
    """
    # Write junk to the hnsw path before the store opens.
    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    hnsw_path.parent.mkdir(parents=True, exist_ok=True)
    hnsw_path.write_bytes(b"not a valid hnswlib index -- corrupted by test")

    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    # Function must return a dict (not raise).
    assert isinstance(result, dict)
    assert "action" in result
    # After HippoDB's own rebuild, counts should be consistent.
    assert result["action"] in ("ok", "divergence_at_boot", "sqlite_count_failed")


# ---------------------------------------------------------------------------
# 3. _run_bounded_startup_optimize is DELETED
# ---------------------------------------------------------------------------


def test_no_run_bounded_startup_optimize_in_daemon() -> None:
    """_run_bounded_startup_optimize must not exist in daemon module."""
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    assert not hasattr(
        _daemon, "_run_bounded_startup_optimize"
    ), "_run_bounded_startup_optimize was not deleted from daemon.py"


# ---------------------------------------------------------------------------
# 4. _post_bind_full_optimize is DELETED
# ---------------------------------------------------------------------------


def test_no_post_bind_full_optimize_in_daemon() -> None:
    """_post_bind_full_optimize must not exist in daemon module (inner function gone)."""
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    assert not hasattr(
        _daemon, "_post_bind_full_optimize"
    ), "_post_bind_full_optimize was not deleted from daemon.py"

    # Also verify it does not appear in the source text.
    src = inspect.getsource(_daemon)
    assert "_post_bind_full_optimize" not in src, (
        "_post_bind_full_optimize string found in daemon source"
    )


# ---------------------------------------------------------------------------
# 5. IAI_MCP_SKIP_STARTUP_OPTIMIZE env handling is DELETED
# ---------------------------------------------------------------------------


def test_no_skip_startup_optimize_env_handling() -> None:
    """IAI_MCP_SKIP_STARTUP_OPTIMIZE env-var handling must not appear in daemon source."""
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    src = inspect.getsource(_daemon)
    assert "IAI_MCP_SKIP_STARTUP_OPTIMIZE" not in src, (
        "IAI_MCP_SKIP_STARTUP_OPTIMIZE string found in daemon source — env handling not removed"
    )


# ---------------------------------------------------------------------------
# 6.: action verdict uses _label_map, not hnswlib.get_current_count()
# ---------------------------------------------------------------------------


def test_health_check_uses_label_map_for_action_verdict() -> None:
    """_hippo_health_check_on_boot source must compare sqlite_count == active_label_count."""
    src = inspect.getsource(_hippo_health_check_on_boot)
    assert "_label_map" in src, "Missing _label_map in health check source (M-05)"
    assert "hnsw_active_count" in src, "Missing hnsw_active_count in health check"
    assert "sqlite_count == active_label_count" in src, (
        "Parity check 'sqlite_count == active_label_count' not found (M-05 guard)"
    )
