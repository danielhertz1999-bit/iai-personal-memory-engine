from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE

logger = logging.getLogger(__name__)


_record_buffer: dict[int, list[dict]] = {}
_record_last_flush_at: dict[int, datetime] = {}


def flush_record_buffer(store: "MemoryStore") -> int:
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        pending = _record_buffer.pop(store_id, [])
        if not pending:
            return 0
        try:
            store.db.open_table(RECORDS_TABLE).add(pending)
            _record_last_flush_at[store_id] = datetime.now(timezone.utc)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "flush_record_buffer_failed",
                extra={"n": len(pending), "err": str(exc)[:120]},
            )
        if pending:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    "lance_buffer_flush",
                    {"table": "records", "count": len(pending)},
                    severity="info",
                    buffered=False,
                )
            except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
                logger.debug("lance_buffer_flush telemetry failed: %s", str(exc)[:120])
        return len(pending)


def should_flush_record_buffer(store_id: int, max_size: int | None = None) -> bool:
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_RECORD_BUFFER_MAX", "500"))
        except ValueError:
            max_size = 500
    return len(_record_buffer.get(store_id, [])) >= max_size


def should_flush_record_buffer_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    if not _record_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec


_edge_buffer: dict[int, list[dict]] = {}
_edge_last_flush_at: dict[int, datetime] = {}


def flush_edge_buffer(store: "MemoryStore") -> int:
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        pending = _edge_buffer.pop(store_id, [])
        if not pending:
            return 0
        try:
            store.db.open_table(EDGES_TABLE).merge_insert(["src", "dst", "edge_type"]).execute(pending)
            _edge_last_flush_at[store_id] = datetime.now(timezone.utc)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "flush_edge_buffer_failed",
                extra={"n": len(pending), "err": str(exc)[:120]},
            )
        if pending:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    "lance_buffer_flush",
                    {"table": "edges", "count": len(pending)},
                    severity="info",
                    buffered=False,
                )
            except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
                logger.debug("lance_buffer_flush telemetry failed: %s", str(exc)[:120])
        return len(pending)


def should_flush_edge_buffer(store_id: int, max_size: int | None = None) -> bool:
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_EDGE_BUFFER_MAX", "500"))
        except ValueError:
            max_size = 500
    return len(_edge_buffer.get(store_id, [])) >= max_size


def should_flush_edge_buffer_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    if not _edge_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec
