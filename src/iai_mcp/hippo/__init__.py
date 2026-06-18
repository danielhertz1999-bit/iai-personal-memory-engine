from __future__ import annotations

import contextlib
import enum
import errno
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import pandas as pd
import pyarrow as pa

from iai_mcp.crypto import (
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import EMBED_DIM

_log = logging.getLogger(__name__)


class AccessMode(enum.Enum):

    EXCLUSIVE = "exclusive"
    SHARED = "shared"



HNSW_M = int(os.environ.get("IAI_MCP_HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.environ.get("IAI_MCP_HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF = int(os.environ.get("IAI_MCP_HNSW_EF", "50"))
HNSW_SAVE_INTERVAL = int(os.environ.get("IAI_MCP_HNSW_SAVE_INTERVAL", "200"))
RECALL_INDEX_EF = 200
HNSW_RESIZE_HEADROOM: float = 0.85
HNSW_INITIAL_CAPACITY: int = 10_000


_DEFAULT_IAI_ROOT = Path.home() / ".iai-mcp"


def _operator_home() -> Path:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, ImportError, AttributeError):
        return Path.home()


_REAL_IAI_ROOT = _operator_home() / ".iai-mcp"


def _resolve_root(path: str | Path | None = None) -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    if env_path:
        return Path(env_path)
    if path is not None:
        return Path(path)
    resolved = _DEFAULT_IAI_ROOT
    if os.environ.get("PYTEST_CURRENT_TEST") and resolved == _REAL_IAI_ROOT:
        raise RuntimeError(
            "hermeticity guard: store-root resolved to the real home store "
            "during a test run; tests must use a tmp path (autouse redirect "
            "fixture). This guard never fires in normal operation."
        )
    return resolved


class HippoLockHeldError(RuntimeError):

    def __init__(self, lock_path: Path | str, holding_pid: str = "unknown") -> None:
        self.lock_path = lock_path
        self.holding_pid = holding_pid
        msg = (
            f"Hippo storage at {Path(lock_path).parent} is already opened by another "
            f"process (pid={holding_pid}); cannot acquire exclusive lock on "
            f"{lock_path}. Stop the daemon or other hippo client first."
        )
        super().__init__(msg)


class ConsolidationPendingError(RuntimeError):

    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = lock_path
        msg = (
            f"Hippo consolidation in progress at {Path(lock_path).parent}; "
            "SHARED lock not acquired within the <1.5 s SLO. "
            "Retry after the consolidation window."
        )
        super().__init__(msg)


class HippoDecryptError(RuntimeError):
    pass


class HippoIntegrityError(RuntimeError):
    pass


_PROCESS_LOCKS: dict[str, tuple[int, int]] = {}
_PROCESS_LOCKS_SHARED: dict[str, tuple[int, int]] = {}
_PROCESS_LOCKS_GUARD: threading.Lock = threading.Lock()

_SHARED_RETRY_SLEEP_S: float = 0.040
_SHARED_MAX_RETRIES: int = 30
_SHARED_LOCK_TIMEOUT_S: float = 1.45


_ENCRYPTED_RECORD_COLUMNS: tuple[str, ...] = (
    "literal_surface",
    "provenance_json",
    "profile_modulation_gain_json",
)

_ENCRYPTED_EVENTS_COLUMNS: tuple[str, ...] = (
    "data_json",
)


from iai_mcp.hippo._table import (
    HippoTableList,
    HippoTable,
    HippoQuery,
    HippoMergeInsert,
    _normalize_to_row_list,
    _encode_embedding,
    _decode_embedding,
    _encode_row_for_insert,
    _decode_df_embedding,
    _decrypt_df_columns,
    _PA_TO_SQLITE,
    _pa_type_to_sqlite,
    _BOOL_COLUMNS,
    _STRICT_BOOL_COLUMNS,
    _sqlite_type_to_pa,
    _DDL_RECORDS,
    _DDL_RECORDS_INDEXES,
    _DDL_EDGES,
    _DDL_EDGES_INDEXES,
    _DDL_EVENTS,
    _DDL_EVENTS_INDEXES,
    _DDL_BUDGET_LEDGER,
    _DDL_BUDGET_LEDGER_INDEXES,
    _DDL_RATELIMIT_LEDGER,
    _DDL_HIPPO_META,
    _TABLE_SQL,
)



from iai_mcp.hippo._db import (
    _txn,
    _txn_owners,
    _txn_owners_lock,
    _validate_table_name,
    HippoDB,
)

from iai_mcp.hippo._recall import (
    _DIRECT_RECENCY_SQL,
    _DIRECT_RECENCY_SQL_LIMITED,
    _no_flock_recency_rows_from_store,
    reconcile_index_mid_run,
    direct_recency_rows_from_store,
    load_hnsw_readonly,
    _ann_lookup_client,
    degraded_semantic_recall,
)

__all__ = [
    "AccessMode",
    "ConsolidationPendingError",
    "HippoDB",
    "HippoTable",
    "HippoQuery",
    "HippoMergeInsert",
    "HippoTableList",
    "HippoLockHeldError",
    "HippoDecryptError",
    "HippoIntegrityError",
    "_ENCRYPTED_RECORD_COLUMNS",
    "_ENCRYPTED_EVENTS_COLUMNS",
    "direct_recency_rows_from_store",
    "load_hnsw_readonly",
    "_ann_lookup_client",
    "degraded_semantic_recall",
    "EMBED_DIM",
]
