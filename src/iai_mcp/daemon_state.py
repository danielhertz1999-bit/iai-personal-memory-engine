from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE_PATH: Path = Path.home() / ".iai-mcp" / ".daemon-state.json"

DIGEST_SHOW_THRESHOLD_HOURS: int = 18

FIRST_TURN_TTL_HOURS: int = 24
MAX_FIRST_TURN_ENTRIES: int = 100


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".daemon-state.",
        suffix=".tmp",
        dir=str(STATE_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, STATE_PATH)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prune_stale_first_turn(
    state: dict,
    now: datetime | None = None,
    ttl_hours: int = FIRST_TURN_TTL_HOURS,
    max_entries: int = MAX_FIRST_TURN_ENTRIES,
) -> int:
    pending = state.get("first_turn_pending")
    if not isinstance(pending, dict) or not pending:
        return 0

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(hours=ttl_hours)

    def _as_dt(value: object) -> datetime:
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return datetime.fromtimestamp(0, tz=timezone.utc)
        return datetime.fromtimestamp(0, tz=timezone.utc)

    removed = 0
    for sid, value in list(pending.items()):
        dt = _as_dt(value)
        if dt < cutoff:
            pending.pop(sid, None)
            removed += 1
        elif not isinstance(value, str):
            pending[sid] = dt.isoformat()

    if len(pending) > max_entries:
        ordered = sorted(
            pending.items(),
            key=lambda kv: _as_dt(kv[1]),
            reverse=True,
        )
        keep = dict(ordered[:max_entries])
        removed += len(pending) - len(keep)
        state["first_turn_pending"] = keep

    return removed


def mark_session_opened(state: dict, session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        return
    pending = state.setdefault("first_turn_pending", {})
    pending[session_id] = datetime.now(timezone.utc).isoformat()
    prune_stale_first_turn(state)


def consume_first_turn(state: dict, session_id: str) -> bool:
    try:
        pending = state.get("first_turn_pending")
        if not isinstance(pending, dict):
            return False
        if pending.pop(session_id, False):
            try:
                save_state(state)
            except (OSError, TypeError, ValueError):
                pass
            return True
        return False
    except (KeyError, TypeError, AttributeError):
        return False


FIRST_TURN_PENDING_TTL_SEC_DEFAULT: float = 3600.0


def prune_first_turn_pending(
    state: dict,
    now: datetime | None = None,
    ttl_sec: float = FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
) -> tuple[dict, list[str]]:
    pending = state.get("first_turn_pending")
    if not isinstance(pending, dict) or not pending:
        return state, []

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(seconds=ttl_sec)

    dropped: list[str] = []
    fresh: dict = {}
    for sid, value in pending.items():
        if isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                dropped.append(sid)
                continue
            if ts < cutoff:
                dropped.append(sid)
                continue
            fresh[sid] = value
        else:
            dropped.append(sid)

    state["first_turn_pending"] = fresh
    return state, dropped


def get_pending_digest(state: dict, now: datetime) -> dict | None:
    last_shown = state.get("last_digest_shown_at")
    if last_shown:
        try:
            last_dt = datetime.fromisoformat(last_shown)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now_cmp = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
            if now_cmp - last_dt < timedelta(hours=DIGEST_SHOW_THRESHOLD_HOURS):
                return None
        except (TypeError, ValueError):
            pass

    digest = state.get("pending_digest")
    if not digest:
        return None

    now_cmp = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    state["last_digest_shown_at"] = now_cmp.isoformat()
    state.pop("pending_digest", None)
    save_state(state)
    return digest
