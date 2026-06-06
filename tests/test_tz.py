"""Tests for IANA timezone handling.

Uses IAI_MCP_STORE env var + tmp_path to isolate config.json file writes so
the user's real ~/.iai-mcp/config.json is never touched by the test suite.

Covers:
- detect_tz() returns a valid IANA key or falls back to UTC
- load_user_tz() reads config.json (if present), auto-seeds when absent
- Invalid IANA strings raise ZoneInfoNotFoundError
- to_local() converts tz-aware and naive datetimes
- Fresh ~/.iai-mcp dir triggers config.json auto-write on first load
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect IAI_MCP_STORE to a fresh tmpdir so config.json writes land there."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- detect_tz


def test_detect_tz_returns_iana_string():
    """detect_tz returns a non-empty string that ZoneInfo can resolve."""
    from iai_mcp.tz import detect_tz

    key = detect_tz()
    assert isinstance(key, str)
    assert len(key) > 0
    # ZoneInfo must be able to instantiate it without raising.
    ZoneInfo(key)  # noqa: B018 -- constructing is the check


def test_detect_tz_matches_system_or_utc_fallback():
    """detect_tz uses `datetime.astimezone().tzinfo.key` or falls back to UTC."""
    from iai_mcp.tz import detect_tz

    key = detect_tz()
    # On macOS/Linux the system tz usually has a.key; on minimal containers
    # the fallback is "UTC". Either is acceptable.
    assert key == "UTC" or "/" in key


# --------------------------------------------------------------- load_user_tz


def test_load_user_tz_reads_config(isolated_store):
    """Pre-populated config.json user.timezone is honoured."""
    from iai_mcp.tz import load_user_tz

    cfg = isolated_store / "config.json"
    cfg.write_text(json.dumps({"user": {"timezone": "Asia/Tokyo"}}))
    tz = load_user_tz()
    assert tz.key == "Asia/Tokyo"


def test_load_user_tz_defaults_on_missing_config(isolated_store):
    """No config.json -> load_user_tz returns a valid ZoneInfo (detect_tz result)."""
    from iai_mcp.tz import detect_tz, load_user_tz

    # Ensure fresh dir (no config.json yet)
    assert not (isolated_store / "config.json").exists()
    tz = load_user_tz()
    assert isinstance(tz, ZoneInfo)
    # The detected key should match detect_tz()'s result or at least round-trip.
    assert tz.key == detect_tz() or tz.key == "UTC"


def test_load_user_tz_rejects_invalid_iana(isolated_store):
    """Config with garbage IANA string raises ZoneInfoNotFoundError."""
    from iai_mcp.tz import load_user_tz

    cfg = isolated_store / "config.json"
    cfg.write_text(json.dumps({"user": {"timezone": "Garbage/Not-Real"}}))
    with pytest.raises(ZoneInfoNotFoundError):
        load_user_tz()


def test_load_user_tz_handles_malformed_json(isolated_store):
    """Malformed config.json -> fall back to detect_tz + auto-seed."""
    from iai_mcp.tz import load_user_tz

    cfg = isolated_store / "config.json"
    cfg.write_text("not-valid-json{")
    tz = load_user_tz()
    assert isinstance(tz, ZoneInfo)


# ---------------------------------------------------- config auto-seed


def test_config_auto_seeds_timezone_on_first_run(isolated_store):
    """Fresh dir -> load_user_tz writes detected key into config.json."""
    from iai_mcp.tz import load_user_tz

    assert not (isolated_store / "config.json").exists()
    load_user_tz()

    cfg_path = isolated_store / "config.json"
    assert cfg_path.exists()

    cfg = json.loads(cfg_path.read_text())
    assert "user" in cfg
    assert "timezone" in cfg["user"]
    # The seeded value is a valid IANA string.
    ZoneInfo(cfg["user"]["timezone"])  # noqa: B018


def test_config_autoseeded_value_stable_across_loads(isolated_store):
    """Calling load_user_tz twice returns the same TZ (no churn)."""
    from iai_mcp.tz import load_user_tz

    tz1 = load_user_tz()
    tz2 = load_user_tz()
    assert tz1.key == tz2.key


def test_config_load_respects_user_override(isolated_store):
    """User edits config.json after auto-seed -> next load honours the edit."""
    from iai_mcp.tz import load_user_tz

    load_user_tz()  # auto-seed

    cfg_path = isolated_store / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["user"]["timezone"] = "Europe/Moscow"
    cfg_path.write_text(json.dumps(cfg))

    tz = load_user_tz()
    assert tz.key == "Europe/Moscow"


# ------------------------------------------------------------------ to_local


def test_to_local_converts_utc():
    """Noon UTC in PDT (America/Los_Angeles, UTC-7) -> 5 AM local."""
    from iai_mcp.tz import to_local

    utc_dt = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    local = to_local(utc_dt, ZoneInfo("America/Los_Angeles"))
    # April 17 is PDT (UTC-7)
    assert local.hour == 5
    assert local.tzinfo.key == "America/Los_Angeles"


def test_to_local_handles_naive_datetime():
    """Naive input is treated as UTC."""
    from iai_mcp.tz import to_local

    naive = datetime(2026, 4, 17, 12, 0)  # no tzinfo
    local = to_local(naive, ZoneInfo("UTC"))
    assert local.tzinfo is not None
    assert local.hour == 12
