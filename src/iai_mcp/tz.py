"""D-34 IANA timezone handling (Plan 02-01, global-product mandate).

Every global-ready product must respect user timezone. We store all runtime
timestamps (events table, BudgetLedger, record created_at, etc.) in UTC and
render CLI output in the user's LOCAL timezone.

The user's timezone lives in ~/.iai-mcp/config.json under `user.timezone`
as an IANA string (e.g. "America/Los_Angeles", "Europe/Moscow", "Asia/Tokyo",
"UTC"). On first run we auto-detect from the system and seed the config file;
thereafter the user can edit config.json to override.

The sleep-cycle scheduler interprets `quiet_window` (22:00-06:00) in the
user's LOCAL time, not UTC. Multi-tenant architecture-ready: Phase 3+ deployments
can carry per-user_id tz maps.

Public surface:
- detect_tz() -> str         -- best-effort IANA key from system
- load_user_tz() -> ZoneInfo -- read config.json + auto-seed
- to_local(dt, tz=None)      -- convert UTC (or naive) to local TZ
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

CONFIG_FILENAME = "config.json"


def _config_path() -> Path:
    """Return the path to the user's config.json.

    Honours IAI_MCP_STORE env var so test isolation + multi-tenant layouts
    can redirect away from ~/.iai-mcp/.
    """
    env = os.environ.get("IAI_MCP_STORE")
    root = Path(env) if env else Path.home() / ".iai-mcp"
    return root / CONFIG_FILENAME


def detect_tz() -> str:
    """Auto-detect IANA timezone from the system. Falls back to "UTC"."""
    try:
        tz = datetime.now().astimezone().tzinfo
        if tz is None:
            return "UTC"
        # ZoneInfo has .key; plain datetime.timezone does not.
        key = getattr(tz, "key", None)
        if key:
            return str(key)
        return "UTC"
    except Exception:
        return "UTC"


def _seed_config(cfg_path: Path, tz_key: str) -> None:
    """Atomically write user.timezone into config.json.

    Preserves any existing keys in the file; only mutates user.timezone.
    Writes to a .tmp file first and os.replace()s over the target so a
    crashed process can never leave a half-written config.
    """
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.setdefault("user", {})
    if not isinstance(existing["user"], dict):
        existing["user"] = {}
    existing["user"]["timezone"] = tz_key
    tmp = cfg_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, cfg_path)


def load_user_tz() -> ZoneInfo:
    """Read user.timezone from config.json, auto-seed on first run.

    Behaviour:
    - config.json missing or malformed -> detect_tz() + write seed; return ZoneInfo.
    - config.json present + user.timezone is a valid IANA string -> return ZoneInfo.
    - config.json present + user.timezone is an INVALID IANA string -> raise
      zoneinfo.ZoneInfoNotFoundError. We refuse to silently override the user's
      edit; a hard error surfaces the typo.
    """
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = None
        if cfg is not None and isinstance(cfg, dict):
            user = cfg.get("user")
            if isinstance(user, dict):
                tz_key = user.get("timezone")
                if isinstance(tz_key, str) and tz_key.strip():
                    # Raises ZoneInfoNotFoundError on invalid IANA -- by design.
                    return ZoneInfo(tz_key)

    # No config (or config present but no user.timezone) -> detect + seed.
    detected = detect_tz()
    try:
        zi = ZoneInfo(detected)
    except Exception:
        detected = "UTC"
        zi = ZoneInfo("UTC")
    _seed_config(cfg_path, detected)
    return zi


def to_local(
    utc_dt: datetime,
    tz: ZoneInfo | None = None,
) -> datetime:
    """Convert a UTC (or naive-UTC-assumed) datetime into the target ZoneInfo.

    When tz is None, falls through to load_user_tz() -- but callers in hot paths
    should cache the ZoneInfo instance and pass it explicitly to avoid the
    per-call config.json read.
    """
    if tz is None:
        tz = load_user_tz()
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(tz)
