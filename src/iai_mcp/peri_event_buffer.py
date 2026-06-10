
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID


_VALID_TIERS = frozenset({"semantic", "episodic", "procedural"})


@dataclass(frozen=True)
class PeriEventEntry:
    record_id: UUID
    captured_at: datetime
    original_tier: str


class PeriEventBuffer:

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
        if not record_ids:
            return
        drop = set(record_ids)
        survivors = [e for e in self._entries if e.record_id not in drop]
        self._entries = deque(survivors, maxlen=self._maxlen)

    def trigger_stc(self, store, trigger_event_type: str) -> dict:
        import logging
        from datetime import datetime, timezone

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


_BUFFER: Optional["PeriEventBuffer"] = None


def get_buffer() -> Optional["PeriEventBuffer"]:
    return _BUFFER


def set_buffer(buf: Optional["PeriEventBuffer"]) -> None:
    global _BUFFER
    if buf is not None and not isinstance(buf, PeriEventBuffer):
        raise TypeError(
            f"set_buffer: expected PeriEventBuffer or None, got {type(buf).__name__}"
        )
    _BUFFER = buf


def trigger_stc(store, trigger_event_type: str) -> dict:
    buf = get_buffer()
    if buf is None:
        return {"upgraded": [], "dry_run": False, "candidates": 0}
    return buf.trigger_stc(store, trigger_event_type)
