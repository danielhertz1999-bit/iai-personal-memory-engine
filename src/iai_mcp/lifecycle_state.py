"""-- typed schema + atomic load/save for lifecycle_state.json.

The 4-state lifecycle (WAKE / DROWSY / SLEEP / HIBERNATION) needs a single
source of truth on disk. Per LOCKED contract L2, the daemon is the ONLY
writer of `~/.iai-mcp/lifecycle_state.json`; wrappers
signal events via Unix socket OR atomic-write `~/.iai-mcp/wake.signal`
filesystem marker.

Persistence pattern mirrors `daemon_state.py` (-01) and
`maintenance.py` (-03):
- Writes via `tempfile.mkstemp` + `os.replace` (POSIX atomic rename).
- Crash mid-write leaves the prior file intact; readers either see
  the old complete blob or the new complete blob, never partial bytes.
- File mode 0o600 (user-only, matches T-04-07 mitigation).

Schema mirrors lifecycle_state.json spec.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypedDict

# Default location. Overridable for tests via the `path` arg of load/save.
LIFECYCLE_STATE_PATH: Path = Path.home() / ".iai-mcp" / "lifecycle_state.json"


class LifecycleState(str, Enum):
    """Four lifecycle states."""

    WAKE = "WAKE"
    DROWSY = "DROWSY"
    SLEEP = "SLEEP"
    HIBERNATION = "HIBERNATION"


class SleepCycleProgress(TypedDict, total=False):
    """Per-attempt progress of the multi-step sleep pipeline.

    All fields optional so the dict can be partially populated mid-cycle;
    `last_completed_step=0` and `attempt=1` represent a freshly-started cycle.
    """

    last_completed_step: int
    attempt: int
    last_error: str | None
    started_at: str  # ISO-8601 UTC


class Quarantine(TypedDict):
    """A failing sleep step can quarantine the cycle until `until_ts`."""

    until_ts: str   # ISO-8601 UTC
    reason: str
    since_ts: str   # ISO-8601 UTC


class LifecycleStateRecord(TypedDict):
    """On-disk schema for `lifecycle_state.json`.

    `sleep_cycle_progress` and `quarantine` are nullable; the rest are
    always present in a well-formed record. `shadow_run` toggles whether
    the state machine actually executes process termination on
    HIBERNATION (False post-) or merely logs the would-action.
    """

    current_state: str   # one of LifecycleState values
    since_ts: str        # ISO-8601 UTC
    last_activity_ts: str  # ISO-8601 UTC
    wrapper_event_seq: int
    sleep_cycle_progress: SleepCycleProgress | None
    quarantine: Quarantine | None
    shadow_run: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp with explicit `+00:00` suffix.

    `isoformat()` on a UTC-aware datetime emits `+00:00` rather than `Z`.
    Both forms are valid ISO-8601; downstream readers (CLI status, event
    log, Hypothesis tests) parse via `datetime.fromisoformat` which
    accepts the offset form.
    """
    return datetime.now(timezone.utc).isoformat()


def default_state() -> LifecycleStateRecord:
    """Return a fresh WAKE record with shadow_run=False (default).

    Used by `load_state` when the file is absent or malformed (self-heal),
    and by tests / callers that need a known starting point.

    Task 1.6 flipped the default from True to False:
    HIBERNATION transitions now actually exit the daemon process via the
    global shutdown event in `daemon.main()`. The legacy RSS-watchdog has
    been removed in Task 1.4; the lifecycle state machine owns shutdown
    authority.
    """
    now = _utc_now_iso()
    return {
        "current_state": LifecycleState.WAKE.value,
        "since_ts": now,
        "last_activity_ts": now,
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
    }


def _validate_record(raw: object) -> LifecycleStateRecord:
    """Reject malformed JSON; return a typed copy on success.

    A minimal schema check — enough to catch hand-edited corruption and
    out-of-band writes from a stale schema version, without pulling in
    pydantic for runtime validation. Reads stay zero-allocation past the
    JSON parse step.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"lifecycle_state record must be a JSON object, got {type(raw).__name__}"
        )

    required_str_keys = ("current_state", "since_ts", "last_activity_ts")
    for k in required_str_keys:
        v = raw.get(k)
        if not isinstance(v, str) or not v:
            raise ValueError(f"lifecycle_state.{k} must be a non-empty string, got {v!r}")

    state_value = raw["current_state"]
    if state_value not in {s.value for s in LifecycleState}:
        raise ValueError(
            f"lifecycle_state.current_state {state_value!r} is not a valid LifecycleState"
        )

    seq = raw.get("wrapper_event_seq")
    if not isinstance(seq, int) or seq < 0:
        raise ValueError(
            f"lifecycle_state.wrapper_event_seq must be a non-negative int, got {seq!r}"
        )

    shadow = raw.get("shadow_run")
    if not isinstance(shadow, bool):
        raise ValueError(
            f"lifecycle_state.shadow_run must be a bool, got {shadow!r}"
        )

    progress = raw.get("sleep_cycle_progress")
    if progress is not None and not isinstance(progress, dict):
        raise ValueError(
            f"lifecycle_state.sleep_cycle_progress must be dict or null, got {progress!r}"
        )

    quarantine = raw.get("quarantine")
    if quarantine is not None:
        if not isinstance(quarantine, dict):
            raise ValueError(
                f"lifecycle_state.quarantine must be dict or null, got {quarantine!r}"
            )
        for k in ("until_ts", "reason", "since_ts"):
            if not isinstance(quarantine.get(k), str):
                raise ValueError(
                    f"lifecycle_state.quarantine.{k} must be string"
                )

    # Cast is safe after the checks above; mypy/pylance accept the dict.
    return raw  # type: ignore[return-value]


def load_state(path: Path | None = None) -> LifecycleStateRecord:
    """Read `lifecycle_state.json`; return `default_state()` if absent.

    On JSON-decode error or schema-validation error: also returns a
    fresh default state. The legacy file is left in place (no auto-delete)
    so an operator can inspect it; `save_state` will overwrite it on the
    next persist.
    """
    target = path if path is not None else LIFECYCLE_STATE_PATH
    if not target.exists():
        return default_state()
    try:
        raw = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return default_state()
    try:
        return _validate_record(raw)
    except ValueError:
        return default_state()


def save_state(record: LifecycleStateRecord, path: Path | None = None) -> None:
    """Atomically persist `record` via tempfile + os.replace.

    Mirrors `daemon_state.save_state` (-01) bullet-for-bullet:
    creates parent dir if missing; writes to a sibling temp file in the
    same directory (required so os.replace is an atomic same-filesystem
    rename); fsyncs the file contents before rename so the data is on
    disk; chmods 0o600 before the swap so the visible file is never
    world-readable; on exception unlinks the temp file so /tmp does not
    accumulate.
    """
    target = path if path is not None else LIFECYCLE_STATE_PATH
    # Validate before writing so callers get an early ValueError on
    # malformed records rather than persisting garbage to disk.
    _validate_record(record)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".lifecycle_state.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
