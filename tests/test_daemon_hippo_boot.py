from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from iai_mcp.daemon import _hippo_health_check_on_boot
from iai_mcp.store import MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


def test_hippo_health_check_on_clean_store(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    assert result["action"] == "ok", f"Expected ok, got: {result}"
    assert result["sqlite_count"] == 0
    assert result["hnsw_active_count"] == 0
    assert "hnsw_raw_count" in result


def test_hippo_health_check_on_corrupted_hnsw(tmp_path: Path) -> None:
    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    hnsw_path.parent.mkdir(parents=True, exist_ok=True)
    hnsw_path.write_bytes(b"not a valid hnswlib index -- corrupted by test")

    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    assert isinstance(result, dict)
    assert "action" in result
    assert result["action"] in ("ok", "divergence_at_boot", "sqlite_count_failed")


def test_no_run_bounded_startup_optimize_in_daemon() -> None:
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    assert not hasattr(
        _daemon, "_run_bounded_startup_optimize"
    ), "_run_bounded_startup_optimize was not deleted from daemon.py"


def test_no_post_bind_full_optimize_in_daemon() -> None:
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    assert not hasattr(
        _daemon, "_post_bind_full_optimize"
    ), "_post_bind_full_optimize was not deleted from daemon.py"

    src = inspect.getsource(_daemon)
    assert "_post_bind_full_optimize" not in src, (
        "_post_bind_full_optimize string found in daemon source"
    )


def test_no_skip_startup_optimize_env_handling() -> None:
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    src = inspect.getsource(_daemon)
    assert "IAI_MCP_SKIP_STARTUP_OPTIMIZE" not in src, (
        "IAI_MCP_SKIP_STARTUP_OPTIMIZE string found in daemon source — env handling not removed"
    )


def test_health_check_uses_label_map_for_action_verdict() -> None:
    src = inspect.getsource(_hippo_health_check_on_boot)
    assert "_label_map" in src, "Missing _label_map in health check source (M-05)"
    assert "hnsw_active_count" in src, "Missing hnsw_active_count in health check"
    assert "sqlite_count == active_label_count" in src, (
        "Parity check 'sqlite_count == active_label_count' not found (M-05 guard)"
    )
