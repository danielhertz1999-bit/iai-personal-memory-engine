"""PeriEventBuffer — synaptic tagging-and-capture buffer.

Holds a fixed-size deque of recent capture turns with their original tier.
When a STRONG_EVENT fires nearby in time, write_event triggers a flush over
the configured peri_event_window_sec and trigger_stc upgrades every eligible
semantic entry to episodic.

The buffer lives in process memory only; cross-respawn loss is
acceptable because the peri-event window is ~30 min and respawns are
rare in steady state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID


_VALID_TIERS = frozenset({"semantic", "episodic", "procedural"})


# Frozen entry tuple stored in the deque. Shape: (record_id, captured_at,
# original_tier). Frozen so deque entries can be safely shared with
# downstream consumers.
@dataclass(frozen=True)
class PeriEventEntry:
    record_id: UUID
    captured_at: datetime  # tz-aware UTC
    original_tier: str  # "semantic" | "episodic" | "procedural"


class PeriEventBuffer:
    """Fixed-size ring buffer of recent capture turns.

    - ring buffer respects maxlen (oldest entries evicted transparently)
    - flush_within_window filters by `(now - captured_at) <= window_sec`

    The buffer is intentionally decoupled from MemoryStore / events / capture.
    `trigger_stc` wires into store.upgrade_tier + events.write_event; that
    dependency lives inside the method so this module stays trivially
    unit-testable without any daemon or store imports.
    """

    def __init__(self, maxlen: int) -> None:
        if not isinstance(maxlen, int) or isinstance(maxlen, bool) or not (1 <= maxlen <= 1000):
            raise ValueError(
                f"PeriEventBuffer: invalid maxlen {maxlen!r}, expected int in [1, 1000]"
            )
        self._maxlen = maxlen
        self._entries: deque[PeriEventEntry] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, record_id: UUID, captured_at: datetime, tier: str) -> None:
        """Append a new entry. Deque maxlen evicts the oldest entry transparently."""
        if tier not in _VALID_TIERS:
            raise ValueError(
                f"PeriEventBuffer.add: invalid tier {tier!r}, "
                f"expected one of {{'semantic','episodic','procedural'}}"
            )
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        self._entries.append(
            PeriEventEntry(record_id=record_id, captured_at=captured_at, original_tier=tier)
        )

    def flush_within_window(self, now: datetime, window_sec: int) -> list[PeriEventEntry]:
        """Return entries with `0 <= (now - captured_at).total_seconds() <= window_sec`.

        Does NOT mutate the deque — trigger_stc decides which entries were
        upgraded and calls `clear_processed` for those, keeping semantics
        composable and letting dry-run leave the deque intact.

        Future-dated entries (negative delta) are filtered as clock-skew artifacts.
        """
        if not isinstance(window_sec, int) or isinstance(window_sec, bool) or window_sec <= 0:
            raise ValueError(
                f"PeriEventBuffer.flush_within_window: window_sec must be > 0, got {window_sec!r}"
            )
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        out: list[PeriEventEntry] = []
        for entry in self._entries:
            delta = (now - entry.captured_at).total_seconds()
            if 0 <= delta <= window_sec:
                out.append(entry)
        return out

    def clear_processed(self, record_ids: list[UUID]) -> None:
        """Remove every entry whose record_id appears in `record_ids`.

        Empty `record_ids` is a no-op. Preserves deque order and maxlen.
        """
        if not record_ids:
            return
        drop = set(record_ids)
        survivors = [e for e in self._entries if e.record_id not in drop]
        self._entries = deque(survivors, maxlen=self._maxlen)

    def trigger_stc(self, store, trigger_event_type: str) -> dict:
        """Flush peri-event window and upgrade eligible semantic entries to episodic.

        Invoked from events.write_event when the just-written event kind is
        in stc_config.strong_event_types (post-emit guard). For each
        entry whose original_tier == "semantic" and falls inside the
        configured peri-event window, calls store.upgrade_tier(...,
        new_tier="episodic", dry_run=cfg.dry_run).

        Per-entry try/except: a single upgrade failure does NOT abort the
        rest of the pass (logged + continue).

        Dry-run mode: upgrade_tier still emits stc_upgrade_pass events with
        dry_run_mode=True, but no row mutation happens AND the buffer is
        NOT cleared -- tests can inspect the surviving entries.

        Returns a small dict for observability: ``{"upgraded": [<uuid-str>],
        "dry_run": bool, "candidates": int}``. write_event currently ignores
        the return; future planners can wire it into a debug surface.
        """
        import logging
        from datetime import datetime, timezone

        # Per-call _load_stc_config re-read so mid-session env edits to
        # window_sec / strong_event_types / dry_run take effect without
        # daemon restart. Lazy import avoids circular load.
        from iai_mcp.daemon_config import _load_stc_config

        log = logging.getLogger(__name__)
        cfg = _load_stc_config()
        now = datetime.now(timezone.utc)
        candidates = self.flush_within_window(now, cfg.peri_event_window_sec)
        upgraded: list[str] = []
        for entry in candidates:
            if entry.original_tier != "semantic":
                continue
            try:
                ok = store.upgrade_tier(
                    entry.record_id,
                    "episodic",
                    trigger_event_type=trigger_event_type,
                    dry_run=cfg.dry_run,
                )
                if ok:
                    upgraded.append(str(entry.record_id))
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                log.warning(
                    "stc_upgrade_tier_failed",
                    extra={
                        "record_id": str(entry.record_id),
                        "trigger_event_type": trigger_event_type,
                        "err_type": type(exc).__name__,
                    },
                )
                continue

        if not cfg.dry_run and upgraded:
            # Remove processed entries so the next STRONG_EVENT does not
            # double-upgrade them. Dry-run leaves the buffer intact for
            # tests to inspect the surviving entries.
            upgraded_set = set(upgraded)
            self.clear_processed(
                [
                    entry.record_id
                    for entry in candidates
                    if str(entry.record_id) in upgraded_set
                ]
            )

        return {
            "upgraded": upgraded,
            "dry_run": cfg.dry_run,
            "candidates": len(candidates),
        }


# Singleton-on-daemon access mechanism. daemon.main registers the buffer
# at boot via set_buffer(buf); events.write_event and capture.capture_turn
# read it via get_buffer() with a None-guard so CLI paths and unit tests
# that never register stay no-ops. Tests pair set_buffer(buf) /
# set_buffer(None) around each case.

_BUFFER: Optional["PeriEventBuffer"] = None


def get_buffer() -> Optional["PeriEventBuffer"]:
    """Return the daemon-registered singleton, or None if no daemon is wired."""
    return _BUFFER


def set_buffer(buf: Optional["PeriEventBuffer"]) -> None:
    """Register (or clear) the singleton. None clears; non-PeriEventBuffer raises TypeError."""
    global _BUFFER
    if buf is not None and not isinstance(buf, PeriEventBuffer):
        raise TypeError(
            f"set_buffer: expected PeriEventBuffer or None, got {type(buf).__name__}"
        )
    _BUFFER = buf


def trigger_stc(store, trigger_event_type: str) -> dict:
    """module-level shim: delegate to the singleton buffer.

    Reconciles a signature drift between the function shape
    (``peri_event_buffer.trigger_stc(...)``) and the method shape
    (``PeriEventBuffer.trigger_stc(...)``). Exposed so callers can write
    ``from iai_mcp.peri_event_buffer import trigger_stc`` without first
    going through ``get_buffer()``. The buffer method on PeriEventBuffer
    remains the single source of behavior; this is a pure delegator.

    Returns the empty-pass shape when no buffer is registered (CLI /
    one-shot paths that never call ``set_buffer``).
    """
    buf = get_buffer()
    if buf is None:
        return {"upgraded": [], "dry_run": False, "candidates": 0}
    return buf.trigger_stc(store, trigger_event_type)
