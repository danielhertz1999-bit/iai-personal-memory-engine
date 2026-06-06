"""Storage backend: SQLite (metadata) + hnswlib (ANN) + AES-GCM per-field encryption
via crypto.py (wired by a later module).

Single-writer model enforced by fcntl exclusive lock on `<store_root>/hippo/.lock`.

This module provides the HippoDB / HippoTable / HippoQuery / HippoMergeInsert /
HippoTableList shim classes that expose the same call surface as the legacy storage
connection/table objects so that callers continue to work unchanged when the
storage backend is swapped.

Vector ANN search is backed by hnswlib (cosine space, M=16, ef_construction=200).
The hnswlib index is an in-memory derived structure rebuilt from SQLite whenever the
active-record count diverges. SQLite is the authoritative source of truth.

SQL injection surface notes
---------------------------
Table names embedded in SQL statements (``f"SELECT ... FROM {self._name}"``) are
validated at HippoTable construction time by ``_validate_table_name``, which rejects
any name that is not purely ``[A-Za-z0-9_]``. The five canonical table names are
all alphanumeric-plus-underscore; the only additional accepted names are those
created through ``HippoDB.create_table``, which applies the same validator.

WHERE predicates accepted by ``count_rows``, ``update``, ``delete``, and
``HippoQuery.where`` are passed through verbatim (legacy API compatibility).
The callers that produce these predicates are the same ones that previously sent
them to the legacy backend; no new injection surface is introduced beyond what already existed.
All VALUES placeholders use ``?`` bound parameters.
"""
from __future__ import annotations

import contextlib
import enum
import errno
import fcntl
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
    """HippoDB flock mode.

    EXCLUSIVE (default): LOCK_EX|LOCK_NB at open; backward-compatible.
        Used by daemon during boot/SLEEP/DREAMING and by writers.
    SHARED: LOCK_SH|LOCK_NB with short non-blocking retry bounded <1.5 s.
        Used by read clients and daemon during WAKE.
        Multiple processes may hold SHARED concurrently (normal flock SH).
    """

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


# ---------------------------------------------------------------------------
# Transaction helper
# ---------------------------------------------------------------------------

# External owner-map for cross-thread transaction detection.
#
# Keyed by id(conn) (connection identity, not the object itself — we never
# hold a strong reference here), value = threading.get_ident() of the thread
# that issued the most recent BEGIN on that connection via _txn.
#
# Guarded by a dedicated plain Lock (NOT _conn_lock): the tripwire purpose is
# to catch a future caller that enters _txn WITHOUT holding _conn_lock.  If
# the map read/write used _conn_lock, the foreign thread would block on that
# lock instead of hitting the branch that raises — the tripwire would be
# silently defeated.  A dedicated module-level lock avoids any ordering cycle
# with _hnsw_lock/_conn_lock because it is NEVER held across yield, across
# BEGIN/COMMIT/ROLLBACK, or while any other lock is held.
_txn_owners: dict[int, int] = {}
_txn_owners_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager  # type: ignore[misc]
def _txn(conn: "sqlite3.Connection"):
    """BEGIN/COMMIT/ROLLBACK bracket with cross-thread foreign-owner detection.

    Three cases on entry:

    1. ``conn.in_transaction`` is True AND the recorded owner equals the
       current thread → legitimate same-thread RLock re-entry (nested helper).
       Yields without issuing SQL; the outer transaction is not disturbed.

    2. ``conn.in_transaction`` is True AND NO owner is recorded → the caller
       issued its own outer BEGIN before entering ``_txn`` (caller-managed
       outer transaction).  Yields without issuing SQL; the outer transaction
       is left open for the caller.

    3. ``conn.in_transaction`` is True AND the recorded owner is a DIFFERENT
       thread → a still-unserialized site is racing a live transaction on the
       shared connection.  Raises :class:`HippoIntegrityError` immediately so
       the violation is loud and test-catchable rather than silently corrupt.

    When no outer transaction is active, the current thread is recorded as
    owner BEFORE ``BEGIN`` is issued, then the owner entry is unconditionally
    cleared in ``finally`` after COMMIT/ROLLBACK.  This ensures:

    - The owner is present for the full BEGIN→yield→COMMIT/ROLLBACK window so
      a concurrent thread entering ``_txn`` at any point in that window sees
      case 1 or 3 (never a false no-owner case 2).
    - A crash, ROLLBACK, or raised exception ALWAYS clears the entry; no stale
      owner survives to false-positive the next legitimate caller.
    """
    if conn.in_transaction:
        with _txn_owners_lock:
            owner = _txn_owners.get(id(conn))
        if owner is None:
            # Case 2: caller-managed outer transaction — yield without nesting.
            yield
            return
        if owner == threading.get_ident():
            # Case 1: same-thread re-entry — yield without nesting.
            yield
            return
        # Case 3: foreign thread owns the open transaction.
        raise HippoIntegrityError(
            f"Shared connection transaction owned by thread {owner} "
            f"observed by thread {threading.get_ident()} — a transactional "
            f"mutator site is missing _conn_lock serialization."
        )
    # No active transaction: record ownership, issue BEGIN, clear in finally.
    conn_id = id(conn)
    with _txn_owners_lock:
        _txn_owners[conn_id] = threading.get_ident()
    try:
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        conn.execute("COMMIT")
    finally:
        with _txn_owners_lock:
            _txn_owners.pop(conn_id, None)

# ---------------------------------------------------------------------------
# hnswlib index constants (env-overridable)
# ---------------------------------------------------------------------------

HNSW_M = int(os.environ.get("IAI_MCP_HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.environ.get("IAI_MCP_HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF = int(os.environ.get("IAI_MCP_HNSW_EF", "50"))
HNSW_SAVE_INTERVAL = int(os.environ.get("IAI_MCP_HNSW_SAVE_INTERVAL", "200"))
# Minimum ef for the daemon in-RAM recall index so k=K_CANDIDATES queries are
# approximately exact.  Raised globally at construction/load — race-free because
# it is a one-time write at index init, not a per-query set/restore.
RECALL_INDEX_EF = 200
HNSW_RESIZE_HEADROOM: float = 0.85   # resize 2x when usage > 85% of capacity
HNSW_INITIAL_CAPACITY: int = 10_000  # minimum capacity for a fresh index

# ---------------------------------------------------------------------------
# Storage root resolution — mirrors MemoryStore.__init__ logic.
# ---------------------------------------------------------------------------

_DEFAULT_IAI_ROOT = Path.home() / ".iai-mcp"


def _operator_home() -> Path:
    """Return the operator's real home directory, independent of ``$HOME``.

    Reads the login database (``/etc/passwd`` via ``pwd``) so the result is
    stable even when ``$HOME`` has been redirected to a temporary directory
    (e.g. inside a test harness, or in a child process that inherited a
    redirected environment). On non-POSIX platforms where ``pwd`` is
    unavailable it falls back to ``Path.home()``.
    """
    try:
        import pwd  # POSIX only; imported lazily so non-POSIX stays importable.

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, ImportError, AttributeError):
        return Path.home()


# Operator's real storage root, captured once at import time from the login
# database rather than ``$HOME``. Unlike _DEFAULT_IAI_ROOT (which the test
# suite redirects to a tmp directory), this sentinel keeps pointing at the
# operator's home store for the lifetime of the process even when $HOME has
# been redirected — including in a child process that inherited the redirected
# environment. It exists solely so the test-only backstop below can recognise
# an accidental resolution to the real store; a legitimate tmp store never
# equals it, so the guard does not misfire on isolated test stores. Inert in
# normal operation (the backstop is gated on a test-runner marker).
_REAL_IAI_ROOT = _operator_home() / ".iai-mcp"


def _resolve_root(path: str | Path | None = None) -> Path:
    """Resolve the iai-mcp storage root from env/arg, same priority as MemoryStore."""
    env_path = os.environ.get("IAI_MCP_STORE")
    if env_path:
        return Path(env_path)
    if path is not None:
        return Path(path)
    resolved = _DEFAULT_IAI_ROOT
    # Test-only backstop: under a test run, refuse a resolution to the real
    # operator store. This makes an accidental non-isolated open impossible by
    # construction even if a test fixture regresses. It never triggers in normal
    # operation because the sentinel is unset outside of test runs.
    if os.environ.get("PYTEST_CURRENT_TEST") and resolved == _REAL_IAI_ROOT:
        raise RuntimeError(
            "hermeticity guard: store-root resolved to the real home store "
            "during a test run; tests must use a tmp path (autouse redirect "
            "fixture). This guard never fires in normal operation."
        )
    return resolved


# ---------------------------------------------------------------------------
# Table name validation — prevents SQL identifier injection
# ---------------------------------------------------------------------------

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> str:
    """Validate that a table name is a safe SQL identifier.

    Accepts only names matching ``[A-Za-z_][A-Za-z0-9_]*`` so that
    f-string interpolation of the name into SQL statements cannot carry
    injection content. Raises ValueError on rejection.
    """
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return name


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class HippoLockHeldError(RuntimeError):
    """Raised when a second HippoDB() is opened on a path already held by this process.

    fcntl.flock is per-process exclusive: a second open() attempt on the same
    .lock file from any fd in the same process will also block (EAGAIN/EWOULDBLOCK).
    """

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
    """Raised when a SHARED client cannot acquire LOCK_SH within the 1.5 s SLO.

    The consolidation-intent flag (hippo/.consolidation-pending) persisted
    past the client budget, meaning the consolidator is actively draining
    SHARED holders before acquiring LOCK_EX.  The caller should retry after
    the consolidation window or open the store via the no-flock fallback path.
    """

    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = lock_path
        msg = (
            f"Hippo consolidation in progress at {Path(lock_path).parent}; "
            "SHARED lock not acquired within the <1.5 s SLO. "
            "Retry after the consolidation window."
        )
        super().__init__(msg)


class HippoDecryptError(RuntimeError):
    """Raised when a records-table ciphertext fails AES-GCM decryption.

    Records-table content (literal_surface, provenance_json,
    profile_modulation_gain_json) MUST decrypt successfully — silent
    empty-string fallback would violate the lossless-recall invariant.
    Events-table data_json keeps the lenient fallback (audit-only,
    empty payload is preferable to a crashed query).
    """


class HippoIntegrityError(RuntimeError):
    """Raised when an internal SQLite read returns an unexpected None result.

    count_rows() expects SELECT COUNT(*) to always return exactly one row.
    If fetchone() returns None, the connection is in an error state (e.g.
    cursor state invalidated by concurrent thread access). Raises this
    instead of the opaque TypeError so the caller can log the table name
    and connection state at the source of the failure.
    """


# ---------------------------------------------------------------------------
# Process-local reentrant lock registry
#
# fcntl.flock is per-process — a second open() on the same .lock file from
# any fd within the same process raises EAGAIN. This registry allows multiple
# HippoDB instances on the same path within one process to share the lock via
# refcounting: the first caller acquires via flock; subsequent callers dup the
# base fd and increment the refcount. close() decrements the refcount and only
# releases the flock when it reaches zero.
# ---------------------------------------------------------------------------

_PROCESS_LOCKS: dict[str, tuple[int, int]] = {}  # resolved_path -> (base_fd, refcount) [EXCLUSIVE]
_PROCESS_LOCKS_SHARED: dict[str, tuple[int, int]] = {}  # resolved_path -> (base_fd, refcount) [SHARED]
_PROCESS_LOCKS_GUARD: threading.Lock = threading.Lock()

# Client SHARED open: non-blocking retry parameters.
# Total client-observable wait is bounded STRICTLY < 1.5 s (the SLO).
# We probe for LOCK_SH|LOCK_NB up to _SHARED_MAX_RETRIES times with
# _SHARED_RETRY_SLEEP_S between attempts. At 40 ms × 30 = 1.2 s we stay
# safely under the 1.5 s SLO even at maximum contention.
_SHARED_RETRY_SLEEP_S: float = 0.040   # 40 ms between retries
_SHARED_MAX_RETRIES: int = 30          # 30 × 40 ms = 1.2 s total < 1.5 s SLO
_SHARED_LOCK_TIMEOUT_S: float = 1.45  # absolute wall-clock guard (< 1.5 s)


# ---------------------------------------------------------------------------
# Encrypted column whitelists — mirrors the existing MemoryStore boundary
# ---------------------------------------------------------------------------

_ENCRYPTED_RECORD_COLUMNS: tuple[str, ...] = (
    "literal_surface",
    "provenance_json",
    "profile_modulation_gain_json",
)

_ENCRYPTED_EVENTS_COLUMNS: tuple[str, ...] = (
    "data_json",
)


# ---------------------------------------------------------------------------
# HippoTableList — thin wrapper for list_tables() return value
# ---------------------------------------------------------------------------


class HippoTableList:
    """Return type for HippoDB.list_tables().

    Provides a ``.tables`` attribute (list[str]) so callers that do
    ``result.tables`` work, and also implements ``__iter__`` so that
    ``list(result)`` works as the fallback branch in migrate.py.
    """

    def __init__(self, tables: list[str]) -> None:
        self.tables: list[str] = tables

    def __iter__(self) -> Iterator[str]:
        return iter(self.tables)

    def __repr__(self) -> str:  # pragma: no cover
        return f"HippoTableList(tables={self.tables!r})"


# ---------------------------------------------------------------------------
# SQLite type mapping helpers
# ---------------------------------------------------------------------------

_PA_TO_SQLITE: dict[str, str] = {
    "int8": "INTEGER",
    "int16": "INTEGER",
    "int32": "INTEGER",
    "int64": "INTEGER",
    "uint8": "INTEGER",
    "uint16": "INTEGER",
    "uint32": "INTEGER",
    "uint64": "INTEGER",
    "float16": "REAL",
    "float32": "REAL",
    "float64": "REAL",
    "bool": "INTEGER",
    "string": "TEXT",
    "large_string": "TEXT",
    "binary": "BLOB",
    "large_binary": "BLOB",
}


def _pa_type_to_sqlite(t: pa.DataType) -> str:
    """Map a pyarrow DataType to a SQLite affinity string."""
    type_str = str(t)
    if type_str in _PA_TO_SQLITE:
        return _PA_TO_SQLITE[type_str]
    if pa.types.is_integer(t):
        return "INTEGER"
    if pa.types.is_floating(t):
        return "REAL"
    if pa.types.is_boolean(t):
        return "INTEGER"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return "BLOB"
    if pa.types.is_timestamp(t):
        return "TEXT"
    return "TEXT"


_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
    "detail_level",  # stored 0/1 in some contexts
})

# Columns that are stored as INTEGER but represent boolean values.
# detail_level is excluded because it's 0-5 range, not boolean.
_STRICT_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
})


def _sqlite_type_to_pa(col_name: str, type_str: str, embed_dim: int) -> pa.DataType:
    """Map a SQLite type string back to a pyarrow DataType.

    The embedding column gets the fixed-size list type so that callers
    using ``tbl.schema.field("embedding").type.list_size`` see the embed dim.
    Boolean columns (stored as INTEGER 0/1) are reported as pa.bool_() so
    callers using ``tbl.schema.field("pinned").type.equals(pa.bool_())``
    see the correct type.
    """
    t_upper = type_str.upper()
    if col_name == "embedding":
        return pa.list_(pa.float32(), embed_dim)
    if col_name in _STRICT_BOOL_COLUMNS:
        return pa.bool_()
    if t_upper in ("TEXT",):
        return pa.string()
    if t_upper in ("REAL",):
        return pa.float32()
    if t_upper in ("INTEGER",):
        return pa.int64()
    if t_upper in ("BLOB",):
        return pa.binary()
    # Fallback
    return pa.string()


# ---------------------------------------------------------------------------
# DDL: all five tables + _hippo_meta
# ---------------------------------------------------------------------------

_DDL_RECORDS = """\
CREATE TABLE IF NOT EXISTS records (
    vec_label       INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    tier            TEXT NOT NULL,
    literal_surface TEXT,
    aaak_index      TEXT,
    embedding       BLOB NOT NULL,
    structure_hv    BLOB,
    community_id    TEXT,
    centrality      REAL,
    detail_level    INTEGER,
    pinned          INTEGER,
    stability       REAL,
    difficulty      REAL,
    last_reviewed   TEXT,
    never_decay     INTEGER,
    never_merge     INTEGER,
    tombstoned_at   TEXT,
    schema_bypass   INTEGER,
    labile_until    TEXT,
    provenance_json TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT,
    tags_json       TEXT,
    language        TEXT,
    s5_trust_score  REAL,
    profile_modulation_gain_json TEXT,
    schema_version  INTEGER DEFAULT 1,
    wing            TEXT,
    room            TEXT,
    drawer          TEXT,
    valence         REAL DEFAULT 0.0,
    hv_tier              TEXT NOT NULL DEFAULT 'bsc',
    structure_hv_payload BLOB NOT NULL DEFAULT x'',
    embedding_pending    INTEGER NOT NULL DEFAULT 0
)"""

_DDL_RECORDS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_records_id        ON records(id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tier      ON records(tier)",
    "CREATE INDEX IF NOT EXISTS idx_records_community ON records(community_id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tomb      ON records(tombstoned_at) WHERE tombstoned_at IS NOT NULL",
    # Partial index for the embedding_pending recency reader (READ A).
    # Covers only pending rows so the planner selects this index for
    # WHERE embedding_pending = 1 scans without touching non-pending rows.
    "CREATE INDEX IF NOT EXISTS idx_records_pending   ON records(embedding_pending) WHERE embedding_pending=1",
]

_DDL_EDGES = """\
CREATE TABLE IF NOT EXISTS edges (
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.0,
    updated_at  TEXT,
    PRIMARY KEY (src, dst, edge_type)
)"""

_DDL_EDGES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
]

_DDL_EVENTS = """\
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    severity        TEXT,
    domain          TEXT,
    ts              TEXT NOT NULL,
    data_json       TEXT,
    session_id      TEXT,
    source_ids_json TEXT
)"""

_DDL_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind)",
    "CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
]

_DDL_BUDGET_LEDGER = """\
CREATE TABLE IF NOT EXISTS budget_ledger (
    date        TEXT,
    usd_spent   REAL,
    kind        TEXT,
    ts          TEXT
)"""

_DDL_BUDGET_LEDGER_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_budget_date_kind ON budget_ledger(date, kind)",
]

_DDL_RATELIMIT_LEDGER = """\
CREATE TABLE IF NOT EXISTS ratelimit_ledger (
    ts          TEXT,
    status_code INTEGER,
    endpoint    TEXT
)"""

_DDL_HIPPO_META = """\
CREATE TABLE IF NOT EXISTS _hippo_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)"""


# ---------------------------------------------------------------------------
# Static per-table SQL — string literals only, zero runtime taint
# ---------------------------------------------------------------------------
# Each entry is a dict of SQL statement templates for one canonical table.
# Using module-level string literals means semgrep's taint analysis sees
# constant values at every execute() call site, not runtime-concatenated strings.

_TABLE_SQL: dict[str, dict[str, str]] = {
    "records": {
        "count":          "SELECT COUNT(*) FROM records",
        "select_all":     "SELECT * FROM records",
        "delete_prefix":  "DELETE FROM records WHERE ",
        "pragma":         "PRAGMA table_info(records)",
        "update_prefix":  "UPDATE records SET ",
        "insert_prefix":  "INSERT INTO records ",
        "alter_prefix":   "ALTER TABLE records ADD COLUMN ",
    },
    "edges": {
        "count":          "SELECT COUNT(*) FROM edges",
        "select_all":     "SELECT * FROM edges",
        "delete_prefix":  "DELETE FROM edges WHERE ",
        "pragma":         "PRAGMA table_info(edges)",
        "update_prefix":  "UPDATE edges SET ",
        "insert_prefix":  "INSERT INTO edges ",
        "alter_prefix":   "ALTER TABLE edges ADD COLUMN ",
    },
    "events": {
        "count":          "SELECT COUNT(*) FROM events",
        "select_all":     "SELECT * FROM events",
        "delete_prefix":  "DELETE FROM events WHERE ",
        "pragma":         "PRAGMA table_info(events)",
        "update_prefix":  "UPDATE events SET ",
        "insert_prefix":  "INSERT INTO events ",
        "alter_prefix":   "ALTER TABLE events ADD COLUMN ",
    },
    "budget_ledger": {
        "count":          "SELECT COUNT(*) FROM budget_ledger",
        "select_all":     "SELECT * FROM budget_ledger",
        "delete_prefix":  "DELETE FROM budget_ledger WHERE ",
        "pragma":         "PRAGMA table_info(budget_ledger)",
        "update_prefix":  "UPDATE budget_ledger SET ",
        "insert_prefix":  "INSERT INTO budget_ledger ",
        "alter_prefix":   "ALTER TABLE budget_ledger ADD COLUMN ",
    },
    "ratelimit_ledger": {
        "count":          "SELECT COUNT(*) FROM ratelimit_ledger",
        "select_all":     "SELECT * FROM ratelimit_ledger",
        "delete_prefix":  "DELETE FROM ratelimit_ledger WHERE ",
        "pragma":         "PRAGMA table_info(ratelimit_ledger)",
        "update_prefix":  "UPDATE ratelimit_ledger SET ",
        "insert_prefix":  "INSERT INTO ratelimit_ledger ",
        "alter_prefix":   "ALTER TABLE ratelimit_ledger ADD COLUMN ",
    },
    "_hippo_meta": {
        "count":          "SELECT COUNT(*) FROM _hippo_meta",
        "select_all":     "SELECT * FROM _hippo_meta",
        "delete_prefix":  "DELETE FROM _hippo_meta WHERE ",
        "pragma":         "PRAGMA table_info(_hippo_meta)",
        "update_prefix":  "UPDATE _hippo_meta SET ",
        "insert_prefix":  "INSERT INTO _hippo_meta ",
        "alter_prefix":   "ALTER TABLE _hippo_meta ADD COLUMN ",
    },
}


# ---------------------------------------------------------------------------
# HippoDB
# ---------------------------------------------------------------------------


class HippoDB:
    """SQLite-backed storage connection.

    Opens (or creates) a brain.sqlite3 database under ``<root>/hippo/``.
    Acquires an fcntl LOCK_EX exclusive lock on ``<root>/hippo/.lock`` before
    opening the SQLite connection so that dual-process open attempts are
    detected and rejected with HippoLockHeldError rather than corrupting
    the database.

    ``isolation_level=None`` disables sqlite3's implicit transaction management,
    giving explicit control over BEGIN/COMMIT/ROLLBACK. All DDL and DML
    methods in this module issue explicit transactions where needed.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        crypto_key_provider: Callable[[], bytes] | None = None,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
        _lock_timeout_override: float | None = None,
    ) -> None:
        # If None, all encryption/decryption helpers are no-ops (test-friendly mode).
        # Production wires a CryptoKey-backed callable from the MemoryStore layer.
        self._crypto_key_provider: Callable[[], bytes] | None = crypto_key_provider
        self._access_mode: AccessMode = access_mode
        self._read_only: bool = read_only

        root = _resolve_root(path)
        self._store_root: Path = root
        self._hippo_dir: Path = root / "hippo"
        self._hippo_dir.mkdir(parents=True, exist_ok=True)

        self._lock_path: Path = self._hippo_dir / ".lock"
        self._lock_key: str = str(self._lock_path.resolve())

        if access_mode is AccessMode.EXCLUSIVE:
            self._acquire_exclusive_lock()
        else:
            self._acquire_shared_lock(
                lock_timeout_override=_lock_timeout_override,
            )

        # Lock acquired — now open the SQLite connection.
        db_path = self._hippo_dir / "brain.sqlite3"
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # manual BEGIN/COMMIT control (no implicit transactions)
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # SQLite ceiling for transient writer contention inside the WAL.
        # Distinct from the client-observable LOCK_SH wait bound (<1.5 s).
        self._conn.execute("PRAGMA busy_timeout=2000")
        if read_only:
            self._conn.execute("PRAGMA query_only=ON")

        _env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
        self._embed_dim: int = (
            int(_env_dim) if _env_dim and _env_dim.isdigit() else EMBED_DIM
        )
        self._closed: bool = False
        # hnswlib ANN index state ------------------------------------------------
        self._hnsw_path: Path = self._hippo_dir / "records.hnsw"
        self._hnsw_tmp_path: Path = self._hippo_dir / "records.hnsw.tmp"
        # threading.RLock: same thread may re-enter from nested helper calls;
        # the asyncio.Lock wrapper is added at the MemoryStore layer.
        # Initialized before _ensure_tables() so add_columns (called from
        # _reconcile_columns) can safely read _conn_lock during schema reconcile.
        self._hnsw_lock: threading.RLock = threading.RLock()
        # Serializes all conn.execute()+fetchone()/fetchall() pairs across worker
        # threads. Required because asyncio.to_thread() dispatches multiple tasks
        # to a shared thread pool, all sharing self._conn. CPython sqlite3 does not
        # protect cursor result sets between execute() and fetchone()/fetchall() when
        # another thread issues conn.execute("BEGIN") on the same connection
        # concurrently — causing fetchone() to return None.
        # RLock (not Lock): _rebuild_index_from_sqlite holds this lock and calls
        # _repopulate_label_map_from_sqlite, which uses cursor iteration on the same
        # connection. Re-entrant acquisition avoids deadlock in that call chain.
        # Lock ordering: _hnsw_lock must always be acquired BEFORE _conn_lock.
        self._conn_lock: threading.RLock = threading.RLock()
        if not read_only:
            self._ensure_tables()

        # Resolve embed_dim from _hippo_meta (allows non-default dims).
        # Skip for read_only opens against a potentially uninitialized store.
        if not read_only:
            meta_dim = self._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = 'embed_dim'"
            ).fetchone()
            if meta_dim is not None:
                self._embed_dim = int(meta_dim[0])
        # uuid → int64 vec_label mapping for active (non-tombstoned) records only.
        # Deleted entries are removed from _label_map even while the hnswlib slot
        # remains soft-deleted via mark_deleted.
        self._label_map: dict[str, int] = {}
        self._write_counter: int = 0

        # Populate _label_map first so _initialize_hnsw_index can compare counts.
        # SHARED read_only clients skip the hnswlib load and boot integrity rebuild
        # (both are write operations; read_only clients have no ANN access).
        if read_only:
            self._hnsw: hnswlib.Index | None = None  # type: ignore[assignment]
            # read_only stores may be opened before the table is created;
            # guard against OperationalError gracefully.
            try:
                self._repopulate_label_map_from_sqlite()
            except Exception:  # noqa: BLE001
                pass
        else:
            self._repopulate_label_map_from_sqlite()
            self._initialize_hnsw_index()

    # ------------------------------------------------------------------
    # Lock acquisition helpers (called only from __init__)
    # ------------------------------------------------------------------

    def _acquire_exclusive_lock(self) -> None:
        """Acquire LOCK_EX|LOCK_NB on hippo/.lock.

        Uses _PROCESS_LOCKS refcount registry so multiple HippoDB EXCLUSIVE
        instances on the same path within one process share the lock fd.
        Raises HippoLockHeldError on EAGAIN/EWOULDBLOCK.
        Raises if a SHARED lock on the same path is already held by this process
        (holding both SH and EX on the same path is a programming error).
        """
        with _PROCESS_LOCKS_GUARD:
            if self._lock_key in _PROCESS_LOCKS_SHARED:
                raise HippoLockHeldError(
                    self._lock_path,
                    "same-process-holds-SHARED",
                )
            held = _PROCESS_LOCKS.get(self._lock_key)
            if held is not None:
                base_fd, refcount = held
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS[self._lock_key] = (base_fd, refcount + 1)
            else:
                base_fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                os.chmod(str(self._lock_path), 0o600)
                try:
                    fcntl.flock(base_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    os.close(base_fd)
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        raise HippoLockHeldError(self._lock_path, "unknown") from exc
                    raise
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS[self._lock_key] = (base_fd, 1)

    def _acquire_shared_lock(
        self,
        lock_timeout_override: float | None = None,
    ) -> None:
        """Acquire LOCK_SH|LOCK_NB on hippo/.lock via short non-blocking retry loop.

        Contract:
        - Checks the consolidation-intent flag (hippo/.consolidation-pending)
          BEFORE each acquire attempt; if set, backs off rather than acquiring
          a new LOCK_SH (so the consolidator drains existing holders quickly).
        - After acquiring LOCK_SH, RE-CHECKS the intent flag; if it became set
          in the check-then-lock window (TOCTOU), releases LOCK_SH immediately
          and backs off (post-acquire recheck-release — H1 fix).
        - Total client-observable wait is bounded STRICTLY < 1.5 s.
        - Multiple SHARED holders on the same path within one process are fine
          (refcounted via _PROCESS_LOCKS_SHARED).
        - Raises HippoLockHeldError if a EXCLUSIVE lock on the same path is
          already held by this process.
        - Raises ConsolidationPendingError if the intent flag persists past budget.

        SIGKILL-fallback note: if the lock file is absent (store partially set up)
        we create it here (O_CREAT) so SHARED open is safe at any lifecycle stage.
        """
        _intent_path = self._hippo_dir / ".consolidation-pending"

        with _PROCESS_LOCKS_GUARD:
            if self._lock_key in _PROCESS_LOCKS:
                raise HippoLockHeldError(
                    self._lock_path,
                    "same-process-holds-EXCLUSIVE",
                )
            held_sh = _PROCESS_LOCKS_SHARED.get(self._lock_key)
            if held_sh is not None:
                # Same process already holds SHARED — increment refcount.
                base_fd, refcount = held_sh
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, refcount + 1)
                return

            # First SHARED open in this process — need to acquire flock.
            base_fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            os.chmod(str(self._lock_path), 0o600)

        # Release the guard while we spin-wait (may sleep); re-enter to register.
        _timeout = (
            lock_timeout_override
            if lock_timeout_override is not None
            else _SHARED_LOCK_TIMEOUT_S
        )
        deadline = time.monotonic() + _timeout
        acquired = False
        for _ in range(_SHARED_MAX_RETRIES + 1):
            # Pre-acquire intent check: if consolidation is pending, back off.
            if _intent_path.exists():
                if time.monotonic() >= deadline:
                    break
                time.sleep(_SHARED_RETRY_SLEEP_S)
                continue

            try:
                fcntl.flock(base_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_SHARED_RETRY_SLEEP_S)
                    continue
                os.close(base_fd)
                raise

            # Post-acquire recheck (H1 TOCTOU fix): if intent was set between
            # the precheck and acquiring LOCK_SH, release immediately and retry.
            if _intent_path.exists():
                fcntl.flock(base_fd, fcntl.LOCK_UN)
                if time.monotonic() >= deadline:
                    break
                time.sleep(_SHARED_RETRY_SLEEP_S)
                continue

            acquired = True
            break

        if not acquired:
            os.close(base_fd)
            raise ConsolidationPendingError(self._lock_path)

        # Register in the SHARED registry.
        with _PROCESS_LOCKS_GUARD:
            held_sh = _PROCESS_LOCKS_SHARED.get(self._lock_key)
            if held_sh is not None:
                # Another thread acquired concurrently — join its refcount.
                fcntl.flock(base_fd, fcntl.LOCK_UN)
                os.close(base_fd)
                base_fd2, refcount2 = held_sh
                self._lock_fd = os.dup(base_fd2)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd2, refcount2 + 1)
            else:
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, 1)

    # ------------------------------------------------------------------
    # Lock transition API (daemon FSM: downgrade EX→SH, escalate SH→EX)
    # ------------------------------------------------------------------

    def downgrade_to_shared(self) -> None:
        """Atomically convert the held LOCK_EX to LOCK_SH on the same fd.

        flock() conversion on the same fd is atomic on macOS/Linux.
        Updates _PROCESS_LOCKS → _PROCESS_LOCKS_SHARED and clears the
        consolidation-intent flag so new SHARED clients may proceed.
        Called by the daemon after the sleep pipeline completes (SLEEP→WAKE).
        """
        _intent_path = self._hippo_dir / ".consolidation-pending"

        with _PROCESS_LOCKS_GUARD:
            if self._access_mode is not AccessMode.EXCLUSIVE:
                return  # already SHARED or not applicable
            held = _PROCESS_LOCKS.get(self._lock_key)
            if held is None:
                return
            base_fd, refcount = held
            # Re-flock the same base_fd to LOCK_SH (atomic conversion).
            try:
                fcntl.flock(base_fd, fcntl.LOCK_SH)
            except OSError:
                return  # best-effort; daemon still has the fd
            # Move from EX registry to SH registry.
            del _PROCESS_LOCKS[self._lock_key]
            _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, refcount)
        self._access_mode = AccessMode.SHARED

        # Clear the intent flag so clients may proceed.
        try:
            _intent_path.unlink()
        except FileNotFoundError:
            pass

    def escalate_to_exclusive(self, intent_budget_ms: int = 4000) -> None:
        """Set the consolidation-intent flag then convert LOCK_SH → LOCK_EX.

        Protocol (yield protocol):
        1. Set hippo/.consolidation-pending so new SHARED clients back off.
        2. Poll LOCK_EX|LOCK_NB up to intent_budget_ms ms (40 ms steps) until
           all outstanding LOCK_SH holders release (they do so on the
           post-acquire recheck after seeing the intent flag).
        3. Convert the base_fd to LOCK_EX (same-fd flock conversion, atomic).
        4. Update registries.
        Called by the daemon before entering the sleep pipeline (WAKE→SLEEP).
        Raises HippoLockHeldError if the budget is exhausted.
        """
        _intent_path = self._hippo_dir / ".consolidation-pending"

        # Step 1: set the intent flag. This is a cross-process signal that
        # consolidation is in progress and is INDEPENDENT of this process's
        # flock mode — it must be set even when this daemon already holds
        # EXCLUSIVE. The daemon can reach the consolidation window still
        # EXCLUSIVE (e.g. it booted EXCLUSIVE and went boot→idle→SLEEP without
        # an intervening WAKE downgrade); without setting the flag here the
        # compaction VACUUM would run with no consolidation-intent signal, so
        # the maintenance guard warns ("hippo_compact_intent_missing") and a
        # racing client could open a fresh SHARED connection mid-VACUUM. Setting
        # the flag is idempotent (O_EXCL → FileExistsError swallowed), so it is
        # safe to do before the early-return.
        try:
            fd = os.open(str(_intent_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass  # idempotent

        if self._access_mode is AccessMode.EXCLUSIVE:
            return  # already EX — flock conversion is a no-op, but the intent
            # flag (set above) and the post-window downgrade clear still apply.

        # Step 2 + 3: poll for LOCK_EX on the base fd.
        with _PROCESS_LOCKS_GUARD:
            held = _PROCESS_LOCKS_SHARED.get(self._lock_key)
        if held is None:
            # No SH entry — open a new fd for EX.
            base_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        else:
            base_fd, _ = held

        deadline = time.monotonic() + intent_budget_ms / 1000.0
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(base_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    time.sleep(0.040)
                    continue
                raise

        if not acquired:
            if held is None:
                os.close(base_fd)
            raise HippoLockHeldError(self._lock_path, "escalate_timeout")

        # Step 4: update registries.
        with _PROCESS_LOCKS_GUARD:
            if held is not None:
                _, refcount = held
                del _PROCESS_LOCKS_SHARED[self._lock_key]
            else:
                refcount = 1
                self._lock_fd = os.dup(base_fd)
            _PROCESS_LOCKS[self._lock_key] = (base_fd, refcount)
        self._access_mode = AccessMode.EXCLUSIVE

    # ------------------------------------------------------------------
    # hnswlib index management
    # ------------------------------------------------------------------

    def _initialize_hnsw_index(self) -> None:
        """Load the hnswlib index from disk, or build a fresh one.

        Recovery order:
        1. If records.hnsw.tmp exists (interrupted atomic save), try loading it.
        2. If records.hnsw exists, try loading it.
        3. If both fail (or neither exists), init a fresh index.
        After loading, re-set ef (not persisted by save_index).
        Finally run a boot integrity check: if active label count diverges from
        SQLite active count, rebuild from SQLite BLOBs.
        """
        _sqlite_count_row = self._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
        if _sqlite_count_row is None:
            raise HippoIntegrityError(
                "_initialize_hnsw_index: SELECT COUNT(*) returned no row — "
                "connection may be in an error state"
            )
        sqlite_count = _sqlite_count_row[0]
        cap = max(HNSW_INITIAL_CAPACITY, sqlite_count * 2)

        loaded = False

        # Prefer .tmp if present — it is the most recently written serialization.
        for candidate in (self._hnsw_tmp_path, self._hnsw_path):
            if candidate.exists():
                try:
                    idx = hnswlib.Index(space="cosine", dim=self._embed_dim)
                    idx.load_index(str(candidate), max_elements=cap)
                    idx.set_ef(max(HNSW_EF, RECALL_INDEX_EF))  # MUST re-set after load
                    idx.set_num_threads(1)
                    self._hnsw: hnswlib.Index = idx
                    loaded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    _log.warning("Failed to load hnswlib index from %s: %s", candidate, exc)

        if not loaded:
            self._hnsw = hnswlib.Index(space="cosine", dim=self._embed_dim)
            self._hnsw.init_index(
                max_elements=cap,
                ef_construction=HNSW_EF_CONSTRUCTION,
                M=HNSW_M,
                allow_replace_deleted=True,
            )
            self._hnsw.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
            self._hnsw.set_num_threads(1)
            # If the disk file was corrupt (or missing) but SQLite has records,
            # rebuild immediately.  The integrity-check below uses _label_map
            # (which was pre-populated from SQLite), so label_count == sqlite_count
            # and the check would NOT fire — we must trigger the rebuild here.
            if sqlite_count > 0:
                _log.info(
                    "No valid hnswlib file found; rebuilding from %d SQLite records",
                    sqlite_count,
                )
                self._rebuild_index_from_sqlite()
                return

        # Boot integrity check: compare active-label count (excludes soft-deleted
        # hnswlib slots) against SQLite's active count.
        # Using len(self._label_map) instead of get_current_count() avoids spurious
        # rebuild triggers after normal tombstoning (M-05 fix).
        active_label_count = len(self._label_map)
        if active_label_count != sqlite_count:
            _log.info(
                "Boot integrity check: active labels=%d != sqlite count=%d — rebuilding",
                active_label_count,
                sqlite_count,
            )
            self._rebuild_index_from_sqlite()

    def _repopulate_label_map_from_sqlite(self) -> None:
        """Populate _label_map from the SQLite active records set.

        Must be called before _initialize_hnsw_index so the integrity check
        compares meaningful counts. Uses int() cast to guarantee Python int
        (not np.integer) in the map.

        Holds the connection lock around the iterated cursor: this method runs
        on a worker thread post-construction (e.g. the maintenance rebuild calls
        _rebuild_index_from_sqlite -> here), and the iterated cursor reads from
        the shared sqlite3.Connection, so a concurrent execute on another worker
        could reset its result set mid-iteration. The lock is _conn_lock only
        (never _hnsw_lock), so a caller already holding _hnsw_lock (the rebuild
        path) preserves the _hnsw_lock-before-_conn_lock order. RLock re-entrancy
        keeps any nested same-thread acquisition safe. The getattr/None fallback
        covers the early-construction call before the lock attribute exists.
        """
        _lock = getattr(self, "_conn_lock", None)
        if _lock is not None:
            with _lock:
                rows = self._conn.execute(
                    "SELECT id, vec_label FROM records"
                    " WHERE tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0"
                ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, vec_label FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchall()
        self._label_map.clear()
        for row in rows:
            self._label_map[row["id"]] = int(row["vec_label"])

    def _rebuild_index_from_sqlite(self) -> dict:
        """Rebuild the hnswlib index from scratch using SQLite embedding BLOBs.

        Fetches all active embeddings in a single scan, recreates the index,
        does a batched add_items, then atomically saves to disk and refreshes
        _label_map.  Returns a diagnostic dict.
        """
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT vec_label, embedding FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
                " ORDER BY vec_label"
            ).fetchall()

        n = len(rows)
        cap = max(HNSW_INITIAL_CAPACITY, n * 2)

        self._hnsw = hnswlib.Index(space="cosine", dim=self._embed_dim)
        self._hnsw.init_index(
            max_elements=cap,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
            allow_replace_deleted=True,
        )
        self._hnsw.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
        self._hnsw.set_num_threads(1)

        if n > 0:
            vecs = np.stack([
                np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
            ])
            labels = np.array([int(row["vec_label"]) for row in rows], dtype=np.int64)
            self._hnsw.add_items(vecs, labels)

        # Persist immediately so next boot loads the rebuilt index.
        self._save_index_atomic()

        # Refresh label map to match exactly the active set.
        self._repopulate_label_map_from_sqlite()

        return {"action": "rebuild", "rebuilt_count": n}

    def _save_index_atomic(self) -> None:
        """Atomically persist the hnswlib index: write to .tmp then os.replace.

        On POSIX (macOS APFS / Linux ext4/xfs), os.replace is atomic for a
        single-file rename on the same filesystem.  Durability is best-effort —
        SQLite is the source of truth; a failed save just means the next boot
        will rebuild from SQLite BLOBs.
        """
        try:
            self._hnsw.save_index(str(self._hnsw_tmp_path))
            os.replace(self._hnsw_tmp_path, self._hnsw_path)
        except OSError as exc:
            _log.warning("hnswlib index save failed: %s", exc)

    def _maybe_resize(self) -> None:
        """Grow the index capacity 2x when usage exceeds HNSW_RESIZE_HEADROOM.

        Uses get_current_count() (which includes soft-deleted slots) because
        resize is triggered by underlying slot exhaustion, not active-record
        semantics.  Must be called inside self._hnsw_lock.
        """
        current = self._hnsw.get_current_count()
        max_el = self._hnsw.get_max_elements()
        if max_el > 0 and current > HNSW_RESIZE_HEADROOM * max_el:
            self._hnsw.resize_index(max_el * 2)

    # ------------------------------------------------------------------
    # Pending-embedding management (H3 deferred-embed path)
    # ------------------------------------------------------------------

    def insert_pending_row(
        self,
        *,
        record_id: str,
        tier: str,
        literal_surface: str,
        tags_json: str,
        provenance_json: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        """Insert a pending-embedding row with a zero-vector BLOB.

        The row is immediately recency-recallable (embedding-independent).
        The daemon fills the real BLOB + clears embedding_pending on next wake.

        The embedding column is BLOB NOT NULL with a fixed-size arrow type —
        a NULL or empty BLOB would violate the constraint.  We store an
        embed_dim ZERO-VECTOR BLOB and set embedding_pending=1 to identify
        this row as deferred.  The normal store.insert() len-check is not
        reached (this path bypasses it).
        """
        import struct as _struct
        zero_blob = _struct.pack(f"<{self._embed_dim}f", *([0.0] * self._embed_dim))
        with self._conn_lock:
            self._conn.execute(
                "INSERT INTO records"
                " (id, tier, literal_surface, aaak_index, embedding, embedding_pending,"
                "  provenance_json, created_at, updated_at, tags_json,"
                "  community_id, detail_level, centrality, stability, difficulty,"
                "  pinned, never_decay, never_merge, s5_trust_score,"
                "  schema_version, language,"
                "  hv_tier, structure_hv_payload)"
                " VALUES (?, ?, ?, '', ?, 1, ?, ?, ?, ?, '', 1, 0.0, 0.0, 0.0,"
                "  0, 0, 0, 0.5, 1, 'en', 'bsc', x'')",
                (
                    record_id,
                    tier,
                    literal_surface,
                    zero_blob,
                    provenance_json,
                    created_at,
                    updated_at,
                    tags_json,
                ),
            )
            self._conn.commit()

    def has_pending_rows(self) -> bool:
        """Return True if any record has embedding_pending=1.

        Cheap dirty-check used to gate the wake re-embed/ingest/rebuild
        sequence so an idle wake is near-free.
        """
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1 LIMIT 1"
            ).fetchone()
        return row is not None

    def reembed_pending_rows(self, embedder: Any) -> int:
        """Re-embed all rows where embedding_pending=1.

        For each pending row: embed its literal_surface -> write the valid
        BLOB back (merge_insert style: UPDATE) -> clear embedding_pending.
        Returns the number of rows processed.

        The embedder must expose `.embed(text) -> list[float]`.
        """
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT id, literal_surface FROM records"
                " WHERE COALESCE(embedding_pending, 0) = 1"
                " AND tombstoned_at IS NULL"
            ).fetchall()
        count = 0
        for row in rows:
            rid = row["id"]
            surface = row["literal_surface"] or ""
            try:
                vec = list(embedder.embed(surface))
            except Exception as exc:  # noqa: BLE001
                _log.warning("reembed_pending_rows: embed failed for id=%s: %s", rid, exc)
                continue
            import struct as _struct
            blob = _struct.pack(f"<{len(vec)}f", *vec)
            with self._conn_lock:
                self._conn.execute(
                    "UPDATE records SET embedding = ?, embedding_pending = 0 WHERE id = ?",
                    (blob, rid),
                )
            count += 1
        if count > 0:
            with self._conn_lock:
                self._conn.commit()
        return count

    def ingest_pending_embeddings(self) -> int:
        """Ingest .pending-embeddings/{uuid}.npy sidecars into the hnswlib index.

        For each {uuid}.npy + {uuid}.json pair: load the vector, add_items,
        atomic-save, then remove the sidecar pair.  A partial pair (missing
        .json or a leftover .tmp) is skipped and retried on the next wake.
        Returns the number of sidecars ingested.
        """
        import json as _json
        import struct as _struct

        sidecar_dir = self._store_root / ".pending-embeddings"
        if not sidecar_dir.exists():
            return 0

        ingested = 0
        for npy_path in sorted(sidecar_dir.glob("*.npy")):
            uuid_str = npy_path.stem
            json_path = sidecar_dir / f"{uuid_str}.json"
            if not json_path.exists():
                _log.debug("ingest_pending_embeddings: skipping partial sidecar %s (no .json)", npy_path)
                continue
            try:
                vec_bytes = npy_path.read_bytes()
                n_floats = len(vec_bytes) // 4
                if n_floats == 0 or len(vec_bytes) % 4 != 0:
                    _log.warning("ingest_pending_embeddings: malformed .npy %s, skipping", npy_path)
                    continue
                vec = list(_struct.unpack(f"<{n_floats}f", vec_bytes))
                meta = _json.loads(json_path.read_text())
                vec_label = int(meta["vec_label"])
            except Exception as exc:  # noqa: BLE001
                _log.warning("ingest_pending_embeddings: failed to load %s: %s", npy_path, exc)
                continue

            import numpy as _np
            with self._hnsw_lock:
                self._maybe_resize()
                self._hnsw.add_items(
                    _np.array([vec], dtype=_np.float32),
                    _np.array([vec_label], dtype=_np.int64),
                )
                self._label_map[uuid_str] = vec_label
                self._save_index_atomic()

            # Remove sidecar pair after successful ingest.
            try:
                npy_path.unlink()
                json_path.unlink()
            except OSError as exc:
                _log.warning("ingest_pending_embeddings: cleanup failed for %s: %s", npy_path, exc)

            ingested += 1
        return ingested

    def pending_embeddings_wake_sequence(self, embedder: Any | None = None) -> dict:
        """Run the full ordered wake sequence: re-embed → ingest → rebuild + label-map refresh.

        GATED behind a dirty-check: if there are no pending rows, no sidecars, and
        no index-vs-sqlite count mismatch, the whole sequence is skipped (near-free).

        Order is mandatory:
        (1) reembed_pending_rows — fills BLOBs + clears flag
        (2) ingest_pending_embeddings — adds sidecar vectors to hnswlib
        (3) _rebuild_index_from_sqlite — reconciles index + refreshes label-map

        Returns a diagnostic dict.
        """
        # Dirty-check gate.
        has_pending = self.has_pending_rows()
        sidecar_dir = self._store_root / ".pending-embeddings"
        has_sidecars = sidecar_dir.exists() and any(sidecar_dir.glob("*.npy"))
        with self._conn_lock:
            non_pending_row = self._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        non_pending_count = non_pending_row[0] if non_pending_row else 0
        index_count = len(self._label_map)
        has_mismatch = (index_count != non_pending_count)

        if not has_pending and not has_sidecars and not has_mismatch:
            return {"action": "skip", "reason": "clean"}

        reembed_count = 0
        if has_pending and embedder is not None:
            reembed_count = self.reembed_pending_rows(embedder)

        ingest_count = 0
        if has_sidecars:
            ingest_count = self.ingest_pending_embeddings()

        rebuild_result = self._rebuild_index_from_sqlite()

        return {
            "action": "wake_sequence",
            "reembed_count": reembed_count,
            "ingest_count": ingest_count,
            "rebuild": rebuild_result,
        }

    # ------------------------------------------------------------------
    # Per-field encryption / decryption helpers
    # ------------------------------------------------------------------

    def _encrypt_for_uuid(self, uuid_str: str, value: str) -> str:
        """Encrypt a field with AD = lowercase uuid bytes; idempotent on already-encrypted input.

        Returns value unchanged when crypto_key_provider is None (no-op test path),
        when value is None, or when value is already prefixed with the ciphertext sentinel.
        """
        if self._crypto_key_provider is None:
            return value
        if value is None:
            return value
        if is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        return encrypt_field(value, key, associated_data=ad)

    def _decrypt_record_field(self, uuid_str: str, column: str, value: str) -> str:
        """Decrypt a records-table field; RAISES HippoDecryptError on AES-GCM failure.

        Records contain user memory content. Returning empty string on failure would
        silently drop memory — a violation of the lossless-recall invariant. Instead,
        a best-effort audit event is emitted and HippoDecryptError is raised so the
        caller surfaces the corruption rather than hiding it.

        Passes through values that are not encrypted (pre-encryption rows, None values,
        or when no key provider is configured).
        """
        if self._crypto_key_provider is None:
            return value
        if value is None or not is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        try:
            return decrypt_field(value, key, associated_data=ad)
        except Exception as exc:
            # Best-effort audit emit. Never let an audit-write failure mask the
            # original decrypt failure — swallow all exceptions here.
            try:
                self._emit_record_decrypt_failed(
                    uuid_str=uuid_str,
                    column=column,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
            except Exception:
                pass
            raise HippoDecryptError(
                f"records.{column} decrypt failed for id={uuid_str}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _decrypt_event_field(self, uuid_str: str, column: str, value: str) -> str:
        """Decrypt an events-table field; returns lenient empty fallback on failure.

        Events are audit/diagnostic rows; a missing decryption is preferable to
        a crashed query. Matches the existing events.py lenient-fallback contract.
        Returns '{}' for JSON-shaped columns (identified by '_json' suffix) or ''
        for plain text columns.

        Passes through values that are not encrypted (pre-encryption rows, None values,
        or when no key provider is configured).
        """
        if self._crypto_key_provider is None:
            return value
        if value is None or not is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        try:
            return decrypt_field(value, key, associated_data=ad)
        except Exception:
            return "{}" if column.endswith("_json") else ""

    def _emit_record_decrypt_failed(
        self,
        *,
        uuid_str: str,
        column: str,
        error: str,
    ) -> None:
        """Write an audit event noting a records-decrypt failure.

        Direct SQL INSERT into the events table — NOT routing through HippoTable.add
        to avoid encryption-recursion. The event itself carries no encrypted payload;
        it contains only the record id, column name, and sanitised error message.

        This method is best-effort: its caller wraps it in try/except so a failure
        here never masks the original HippoDecryptError.
        """
        import json as _json
        from uuid import uuid4

        event_id = str(uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        payload = _json.dumps({
            "record_id": uuid_str,
            "column": column,
            "error": error,
        })
        try:
            self._conn.execute(
                "INSERT INTO events (id, kind, severity, domain, ts, "
                "data_json, session_id, source_ids_json) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    "record_decrypt_failed",
                    "error",
                    "storage",
                    ts,
                    payload,
                    None,
                    None,
                ),
            )
        except Exception:
            # Truly best-effort. The HippoDecryptError raise is what matters.
            pass

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create all canonical tables inside a single transaction.

        Also reconciles the schema on every open: any column listed in the
        canonical DDL but missing from a pre-existing table is added via
        ``ALTER TABLE ADD COLUMN``. This handles the case where a store
        created by an older code version is opened by newer code that has
        added columns to the DDL — ``CREATE TABLE IF NOT EXISTS`` is a
        no-op when the table already exists, so without reconciliation
        the new columns would never appear.
        """
        conn = self._conn
        conn.execute("BEGIN")
        try:
            conn.execute(_DDL_RECORDS)
            # Reconcile columns BEFORE building indexes: a partial index that
            # references a column (e.g. embedding_pending) would fail on a
            # pre-migration store where the column does not yet exist.
            # _reconcile_columns guarantees all columns are present, then the
            # index loop runs safely on the fully-reconciled schema.
            self._reconcile_columns(
                "records",
                [
                    ("wing", "TEXT"),
                    ("room", "TEXT"),
                    ("drawer", "TEXT"),
                    ("valence", "REAL DEFAULT 0.0"),
                    ("hv_tier", "TEXT NOT NULL DEFAULT 'bsc'"),
                    ("structure_hv_payload", "BLOB NOT NULL DEFAULT x''"),
                    ("embedding_pending", "INTEGER NOT NULL DEFAULT 0"),
                ],
            )
            for idx in _DDL_RECORDS_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_EDGES)
            for idx in _DDL_EDGES_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_EVENTS)
            for idx in _DDL_EVENTS_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_BUDGET_LEDGER)
            for idx in _DDL_BUDGET_LEDGER_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_RATELIMIT_LEDGER)

            conn.execute(_DDL_HIPPO_META)
            conn.execute(
                "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
                ("schema_version", "1"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
                ("embed_dim", str(self._embed_dim)),
            )
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _reconcile_columns(
        self, table_name: str, expected: list[tuple[str, str]]
    ) -> None:
        """Idempotently add columns that exist in DDL but not in table.

        Each ``(column_name, sqlite_type_decl)`` pair is checked against the
        live ``PRAGMA table_info`` and added when absent. Plain-type columns
        (``TEXT``/``INTEGER``/``REAL``) delegate to :meth:`HippoTable.add_columns`
        so tests and operational code share a single ALTER-TABLE entry point.
        Columns whose declaration carries a ``DEFAULT`` clause go through a
        whitelisted direct ``ALTER TABLE`` because :meth:`HippoTable.add_columns`
        derives the type from :class:`pa.Field` and cannot express defaults.

        Any failure is re-raised as :class:`RuntimeError` naming the failing
        columns so an operator can identify which migration bailed
        (FAIL-LOUD: do not swallow migration errors).

        Identifiers are guarded: ``table_name`` and each ``column_name`` pass
        through :func:`_validate_table_name`, and the SQLite type fragment is
        whitelisted against a small fixed set of canonical declarations.
        """
        safe_table = _validate_table_name(table_name)
        # Whitelist of SQLite column-type fragments that may appear in
        # reconcile calls. Anything outside this set raises -- we do not
        # want arbitrary text concatenated into ALTER TABLE.
        plain_to_pa = {
            "TEXT": pa.string(),
            "INTEGER": pa.int64(),
            "REAL": pa.float64(),
            "BLOB": pa.binary(),
        }
        allowed_with_default = {
            "REAL DEFAULT 0.0",
            "INTEGER DEFAULT 0",
            "INTEGER DEFAULT 1",
            "TEXT NOT NULL DEFAULT 'bsc'",
            "BLOB NOT NULL DEFAULT x''",
            "INTEGER NOT NULL DEFAULT 0",
        }
        pragma_stmt = "PRAGMA table_info(" + safe_table + ")"  # nosemgrep
        _lock = getattr(self, "_conn_lock", None)
        if _lock is not None:
            with _lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()  # nosemgrep
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()  # nosemgrep
        existing = {row["name"] for row in _pragma_rows}

        tbl = self.open_table(safe_table)
        missing_plain: list[pa.Field] = []
        missing_with_default: list[tuple[str, str]] = []
        for col_name, sqlite_type in expected:
            if col_name in existing:
                continue
            if sqlite_type in plain_to_pa:
                missing_plain.append(pa.field(col_name, plain_to_pa[sqlite_type]))
            elif sqlite_type in allowed_with_default:
                missing_with_default.append((col_name, sqlite_type))
            else:
                raise RuntimeError(
                    f"_reconcile_columns rejected non-canonical type "
                    f"declaration {sqlite_type!r} for column {col_name!r}"
                )

        failing: list[str] = []
        if missing_plain:
            try:
                tbl.add_columns(missing_plain)
            except Exception:  # noqa: BLE001 -- aggregate names, raise once below
                failing.extend(f.name for f in missing_plain)

        for col_name, sqlite_type in missing_with_default:
            if col_name in failing:
                continue
            safe_col = _validate_table_name(col_name)
            alter_stmt = (
                "ALTER TABLE " + safe_table + " ADD COLUMN "
                + safe_col + " " + sqlite_type
            )  # nosemgrep
            try:
                self._conn.execute(alter_stmt)  # nosemgrep
            except Exception:  # noqa: BLE001 -- aggregate names, raise once below
                failing.append(col_name)

        if failing:
            raise RuntimeError(
                f"schema reconciliation failed for table {safe_table!r}: "
                f"could not add columns {failing!r}"
            )

    # ------------------------------------------------------------------
    # Table enumeration
    # ------------------------------------------------------------------

    def table_names(self) -> list[str]:
        """Return sorted list of user-defined table names."""
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [row["name"] for row in rows]

    def list_tables(self) -> HippoTableList:
        """Return a HippoTableList whose .tables attribute is list[str].

        This matches the legacy 0.30+ paginated-response shape used by
        migrate.py (result.tables accessor) while also supporting the
        ``list(result)`` fallback via HippoTableList.__iter__.
        """
        return HippoTableList(self.table_names())

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def open_table(self, name: str) -> "HippoTable":
        """Return a HippoTable bound to the named SQLite table."""
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def create_table(
        self,
        name: str,
        schema: pa.Schema | None = None,
        data: Any = None,
    ) -> "HippoTable":
        """Ensure a table exists; no-op when already created by _ensure_tables.

        The five canonical tables are created by _ensure_tables on open.
        This method exists for API compatibility with callers that call
        db.create_table() conditionally; it simply returns a HippoTable
        bound to the named table.
        """
        _validate_table_name(name)  # reject non-identifier names early
        if name not in self.table_names():
            if schema is not None:
                cols = []
                for f in schema:
                    sqlite_type = _pa_type_to_sqlite(f.type)
                    col_name = _validate_table_name(f.name)  # validate column names too
                    cols.append(f"{col_name} {sqlite_type}")  # nosemgrep: sql-injection
                # Table name already validated above. # nosemgrep: sql-injection
                ddl = f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(cols)})"  # nosemgrep: sql-injection
                self._conn.execute("BEGIN")
                try:
                    self._conn.execute(ddl)  # nosemgrep
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                self._conn.execute("COMMIT")
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def drop_table(self, name: str) -> None:
        """Drop a table by name (issues DROP TABLE IF EXISTS)."""
        _validate_table_name(name)
        self._conn.execute(f"DROP TABLE IF EXISTS {name}")  # nosemgrep

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Save the hnswlib index, commit, close the SQLite connection, and release the fcntl lock."""
        if self._closed:
            return
        self._closed = True
        # Save hnswlib index before closing SQLite (clean shutdown durability).
        if hasattr(self, "_hnsw"):
            try:
                with self._hnsw_lock:
                    self._save_index_atomic()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._conn.commit()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        if self._lock_fd is not None:
            lock_key = getattr(self, "_lock_key", None)
            access_mode = getattr(self, "_access_mode", AccessMode.EXCLUSIVE)
            registry = (
                _PROCESS_LOCKS_SHARED
                if access_mode is AccessMode.SHARED
                else _PROCESS_LOCKS
            )
            with _PROCESS_LOCKS_GUARD:
                held = registry.get(lock_key) if lock_key else None
                if held is not None:
                    base_fd, refcount = held
                    if refcount <= 1:
                        # Last holder: release the flock and close the base fd.
                        try:
                            fcntl.flock(base_fd, fcntl.LOCK_UN)
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            os.close(base_fd)
                        except Exception:  # noqa: BLE001
                            pass
                        del registry[lock_key]
                    else:
                        registry[lock_key] = (base_fd, refcount - 1)
                # Always close the dup fd for this instance.
                try:
                    os.close(self._lock_fd)
                except Exception:  # noqa: BLE001
                    pass
                self._lock_fd = None

    def __del__(self) -> None:
        """Best-effort release on GC. Swallows all exceptions to avoid
        resurrection warnings in dtor context."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "HippoDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Embedding encode / decode helpers
# ---------------------------------------------------------------------------


def _encode_embedding(vec: list[float] | np.ndarray | None) -> bytes | None:
    """Encode a float vector to the bytes BLOB stored in SQLite."""
    if vec is None:
        return None
    return np.array(vec, dtype=np.float32).tobytes()


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    """Decode a bytes BLOB from SQLite back to list[float]."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _encode_row_for_insert(row: dict) -> dict:
    """Return a copy of row with the embedding field BLOB-encoded."""
    out = dict(row)
    if "embedding" in out and out["embedding"] is not None:
        out["embedding"] = _encode_embedding(out["embedding"])
    return out


def _decode_df_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Decode the 'embedding' column of a DataFrame from bytes to list[float]."""
    if "embedding" in df.columns:
        df = df.copy()
        df["embedding"] = df["embedding"].apply(
            lambda b: _decode_embedding(b) if isinstance(b, (bytes, bytearray)) else b
        )
    return df


def _decrypt_df_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    decrypt_fn: "Callable[[str, str, str], str]",
) -> pd.DataFrame:
    """Apply per-row decryption to named columns in a DataFrame.

    Each row is decrypted with AD = row['id'] (the record or event UUID).
    Only columns that are present in the DataFrame are processed.

    ``decrypt_fn`` signature: (uuid_str: str, column: str, value: str) -> str.
    For records this raises HippoDecryptError on failure; for events it returns
    a lenient empty fallback.
    """
    active_cols = [c for c in columns if c in df.columns]
    if not active_cols or df.empty or "id" not in df.columns:
        return df
    df = df.copy()
    for col in active_cols:
        decrypted: list = []
        for _, row in df.iterrows():
            val = row[col]
            uid = str(row["id"])
            if val is None or not isinstance(val, str):
                decrypted.append(val)
            else:
                decrypted.append(decrypt_fn(uid, col, val))
        df[col] = decrypted
    return df


# ---------------------------------------------------------------------------
# HippoTable
# ---------------------------------------------------------------------------


class HippoTable:
    """Wraps a single SQLite table with a legacy-compatible API surface.

    SQL statements are looked up from the module-level ``_TABLE_SQL`` dict for
    the six canonical tables, giving semgrep string-literal constants at every
    execute() call site. For dynamic tables (created via HippoDB.create_table),
    ``self._sql`` falls back to None and methods build SQL from the validated
    ``self._name``; those paths carry explicit ``# nosemgrep`` suppressions.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        name: str,
        *,
        embed_dim: int,
        db: "HippoDB | None" = None,
        ann_index: Any = None,
    ) -> None:
        self._name = _validate_table_name(name)
        self._conn = conn
        self._embed_dim = embed_dim
        self._db: "HippoDB | None" = db   # HippoDB reference for hnswlib access
        self._ann_index = ann_index
        # Use static SQL dict for the six canonical tables; None for dynamic.
        self._sql: dict[str, str] | None = _TABLE_SQL.get(self._name)

    def _stmt(self, key: str) -> str:
        """Return a pre-built SQL statement for canonical tables, or raise."""
        if self._sql is not None:
            return self._sql[key]
        raise KeyError(f"No pre-built SQL for key {key!r} on dynamic table {self._name!r}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def count_rows(self, filter: str | None = None) -> int:  # noqa: A002
        """Return the number of rows, optionally filtered by a SQL WHERE predicate.

        The ``filter`` argument is a raw SQL predicate (legacy API compat).
        WHERE predicates from callers are already-validated SQL strings (tier
        names, UUID literals from _uuid_literal, IS NULL checks) — the same
        injection surface that existed under the legacy backend.

        Thread safety: acquires self._db._conn_lock around the execute()+fetchone()
        pair. Without the lock, concurrent threads calling conn.execute("BEGIN") on
        the same shared connection can reset cursor state between execute() and
        fetchone(), causing fetchone() to return None for SELECT COUNT(*) — which
        then raises TypeError 'NoneType' object is not subscriptable. The lock
        serialises all execute()+fetchone() pairs that read a single-row result.
        """
        if self._sql is not None:
            base = self._sql["count"]
        else:
            base = "SELECT COUNT(*) FROM " + self._name  # nosemgrep
        stmt = (base + " WHERE " + filter) if filter else base  # nosemgrep
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                row = self._conn.execute(stmt).fetchone()
        else:
            row = self._conn.execute(stmt).fetchone()
        if row is None:
            raise HippoIntegrityError(
                f"count_rows({self._name!r}): SELECT COUNT(*) returned no row — "
                f"connection may be in an error state. "
                f"(filter={filter!r}, in_transaction={getattr(self._conn, 'in_transaction', '?')})",
            )
        return int(row[0])

    def to_pandas(self) -> pd.DataFrame:
        """Return all rows as a DataFrame.

        Embedding BLOBs are decoded to list[float]. Encrypted text columns are
        returned as raw ciphertext (iai:enc:v1: prefix). Callers at the
        MemoryStore boundary decrypt via _from_row / _decrypt_for_record.

        Thread safety: for the records table, acquires _conn_lock around the
        pd.read_sql_query call so concurrent HippoTable.add() writes (which
        hold _hnsw_lock -> _conn_lock) cannot reset the shared
        sqlite3.Connection cursor state between execute() and fetchall(),
        which would cause pd.read_sql_query to return an empty/truncated
        DataFrame.  The reader holds _conn_lock only (never _hnsw_lock), so
        there is no lock cycle with the writer path (hnsw->conn order).
        """
        if self._sql is not None:
            stmt = self._sql["select_all"]
        else:
            stmt = "SELECT * FROM " + self._name  # nosemgrep
        # Hold _conn_lock for EVERY table (not just records): a read opens a
        # SQLite read transaction on the shared connection, and a concurrent
        # VACUUM (consolidation) on the same connection fails with "database
        # table is locked" / "SQL statements in progress" unless reads and
        # VACUUM are mutually exclusive on _conn_lock. The reader takes
        # _conn_lock only (never _hnsw_lock) so the _hnsw_lock-before-_conn_lock
        # order is preserved.
        if self._db is not None:
            with self._db._conn_lock:
                df = pd.read_sql_query(stmt, self._conn)
        else:
            df = pd.read_sql_query(stmt, self._conn)
        df = _decode_df_embedding(df)
        return df

    def _decrypt_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-field decryption to a DataFrame based on this table's policy.

        Records table: uses the strict decrypt path (raises HippoDecryptError).
        Events table: uses the lenient path (empty fallback on failure).
        All other tables: returned unchanged.
        """
        if self._db is None or self._db._crypto_key_provider is None:
            return df
        if self._name == "records":
            return _decrypt_df_columns(
                df, _ENCRYPTED_RECORD_COLUMNS, self._db._decrypt_record_field
            )
        if self._name == "events":
            return _decrypt_df_columns(
                df, _ENCRYPTED_EVENTS_COLUMNS, self._db._decrypt_event_field
            )
        return df

    def _encrypt_rows(self, rows: list[dict]) -> list[dict]:
        """Encrypt the appropriate columns in each row before INSERT.

        Returns a new list of dicts (original dicts are not mutated).
        No-op when no crypto_key_provider is configured.
        """
        if self._db is None or self._db._crypto_key_provider is None:
            return rows
        if self._name == "records":
            enc_cols = _ENCRYPTED_RECORD_COLUMNS
        elif self._name == "events":
            enc_cols = _ENCRYPTED_EVENTS_COLUMNS
        else:
            return rows
        result = []
        for row in rows:
            uid = str(row.get("id", ""))
            if not uid:
                result.append(row)
                continue
            new_row = dict(row)
            for col in enc_cols:
                val = new_row.get(col)
                if val is not None:
                    new_row[col] = self._db._encrypt_for_uuid(uid, val)
            result.append(new_row)
        return result

    def search(self, vector: Any = None, **kwargs: Any) -> "HippoQuery":
        """Return a chainable HippoQuery for this table.

        When vector is None, falls through to SQL SELECT with WHERE/LIMIT.
        When vector is provided, performs hnswlib cosine ANN search and returns
        a HippoQuery pre-seeded with the top-k vec_labels so that .to_pandas()
        returns the matching records with a ``_distance`` column.

        The k (limit) can be specified via .limit(k) on the returned query;
        default k=10 applies if .limit() is not called before .to_pandas().
        """
        if vector is None:
            return HippoQuery(
                self._conn,
                self._name,
                embed_dim=self._embed_dim,
                db=self._db,
            )

        if self._db is None:
            raise NotImplementedError("ANN search requires a HippoDB reference")

        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        return HippoQuery(
            self._conn,
            self._name,
            embed_dim=self._embed_dim,
            ann_vector=vec,
            ann_db=self._db,
            db=self._db,
        )

    def list_versions(self) -> list[dict]:
        """Return a single-element version list for legacy API compatibility.

        SQLite has no MVCC. This stub satisfies callers that count versions
        for diagnostics (all will observe version=1).
        """
        return [{"version": 1, "ts": datetime.now(timezone.utc).isoformat()}]

    def optimize(
        self,
        cleanup_older_than: Any = None,
        delete_unverified: bool = False,
        **kwargs: Any,
    ) -> dict:
        """No-op optimize stub. Real compaction lives in maintenance.py."""
        return {"compaction": "noop_hippo"}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, rows: Any) -> None:
        """Insert rows into the table. Accepts list[dict], pa.Table, or pd.DataFrame.

        For the records table, each row is inserted individually (not via executemany)
        so that cursor.lastrowid gives the vec_label for each row, which is then added
        to the hnswlib index. For all other tables, executemany is used.

        Encrypted columns are encrypted before INSERT — the SQLite layer sees ciphertext,
        not plaintext. Encryption requires the row to have an 'id' field (UUID string)
        used as associated data (AAD).

        VALUES use ``?`` bound parameters. Column names come from the row dict
        keys (the canonical DDL column names, all alphanumeric-plus-underscore).
        """
        row_list = _normalize_to_row_list(rows)
        if not row_list:
            return
        # Apply per-field encryption before embedding encode (disjoint columns).
        row_list = self._encrypt_rows(row_list)
        encoded = [_encode_row_for_insert(r) for r in row_list]
        cols = list(encoded[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        if self._sql is not None:
            stmt = self._sql["insert_prefix"] + "(" + col_names + ") VALUES (" + placeholders + ")"
        else:
            stmt = "INSERT INTO " + self._name + " (" + col_names + ") VALUES (" + placeholders + ")"  # nosemgrep

        if self._name == "records" and self._db is not None:
            # Records table: insert one row at a time to capture per-row vec_label,
            # then add to hnswlib index inside the lock.
            # Lock order: _hnsw_lock (outer) -> _conn_lock (inner).
            # _conn_lock inside _hnsw_lock ensures the cursor execute() +
            # lastrowid read are atomic with respect to concurrent to_pandas()
            # readers that hold _conn_lock alone (no deadlock: reader never
            # takes _hnsw_lock, so there is no lock cycle).
            db = self._db
            with db._hnsw_lock:
                with db._conn_lock:
                    with _txn(self._conn):
                        for r, enc in zip(row_list, encoded):
                            cursor = self._conn.execute(stmt, tuple(enc.get(c) for c in cols))  # nosemgrep
                            vec_label = int(cursor.lastrowid)
                            # Add to hnswlib immediately (still inside transaction so
                            # a crash rolls back both SQLite and drops the in-memory state).
                            emb_raw = r.get("embedding")
                            if emb_raw is not None:
                                emb_vec = np.array(emb_raw, dtype=np.float32).reshape(1, -1)
                                db._hnsw.add_items(emb_vec, np.array([vec_label], dtype=np.int64))
                                db._label_map[str(r["id"])] = vec_label
                                db._write_counter += 1
                            db._maybe_resize()
                # Periodic save (outside the row loop but still inside _hnsw_lock).
                if db._write_counter > 0 and db._write_counter % HNSW_SAVE_INTERVAL == 0:
                    db._save_index_atomic()
        else:
            # Non-records tables: no hnswlib work — use efficient executemany.
            # Serialize on _conn_lock (NOT _hnsw_lock): this write issues
            # BEGIN/COMMIT on the shared sqlite3.Connection, and a concurrent
            # consolidation VACUUM holds _conn_lock for its checkpoint+VACUUM
            # window. Guarding this write with _hnsw_lock would NOT mutually
            # exclude it from VACUUM (different lock), leaving the connection's
            # transaction open during VACUUM -> "database table is locked".
            # No hnswlib work happens here, so _conn_lock alone is correct and
            # respects the _hnsw_lock-before-_conn_lock order (no _hnsw_lock taken).
            if self._db is not None:
                lock_ctx = self._db._conn_lock
            else:
                lock_ctx = contextlib.nullcontext()
            with lock_ctx:
                with _txn(self._conn):
                    self._conn.executemany(stmt, [tuple(r.get(c) for c in cols) for r in encoded])  # nosemgrep

    def update(self, where: str, values: dict[str, Any]) -> None:
        """Update rows matching the WHERE predicate with the given column values.

        Column names and VALUES use bound ``?`` parameters; WHERE is a raw SQL
        predicate from callers (legacy API compat).

        When any of the updated columns are in the encrypted whitelist, the WHERE
        clause MUST be id-keyed (``id = '<uuid>'``) so we can extract the AAD for
        correct AES-GCM encryption. A non-id-keyed WHERE on an encrypted column
        raises ValueError to prevent silent AAD-binding regressions.
        """
        if not values:
            return

        # Determine whether any encrypted column is being updated.
        enc_cols: tuple[str, ...] = ()
        if self._db is not None and self._db._crypto_key_provider is not None:
            if self._name == "records":
                enc_cols = _ENCRYPTED_RECORD_COLUMNS
            elif self._name == "events":
                enc_cols = _ENCRYPTED_EVENTS_COLUMNS

        encrypted_being_updated = [c for c in values if c in enc_cols]
        if encrypted_being_updated:
            # Extract uuid from WHERE clause: must match ``id = '<uuid>'`` or
            # ``id = "<uuid>"`` pattern (the canonical form from store.py callers).
            match = re.search(r"""id\s*=\s*['"]([^'"]+)['"]""", where)
            if match is None:
                raise ValueError(
                    f"Encrypted column(s) {encrypted_being_updated!r} can only be "
                    f"updated with an id-keyed WHERE clause (e.g. \"id = '<uuid>'\") "
                    f"for AAD binding. Received WHERE: {where!r}"
                )
            uuid_str = match.group(1)
            encrypted_values = dict(values)
            for col in encrypted_being_updated:
                encrypted_values[col] = self._db._encrypt_for_uuid(
                    uuid_str, encrypted_values[col]
                )
            values = encrypted_values

        # Encode any list-valued columns (embedding) to BLOB before binding.
        encoded_values: dict = {}
        for col, val in values.items():
            if col == "embedding" and isinstance(val, (list, np.ndarray)):
                encoded_values[col] = _encode_embedding(val)
            else:
                encoded_values[col] = val

        set_clause = ", ".join(col + "=?" for col in encoded_values)
        if self._sql is not None:
            stmt = self._sql["update_prefix"] + set_clause + " WHERE " + where
        else:
            stmt = "UPDATE " + self._name + " SET " + set_clause + " WHERE " + where  # nosemgrep
        # Serialize the full BEGIN..COMMIT window under _conn_lock so a
        # concurrent VACUUM or merge_insert on the shared connection cannot
        # interleave with this transaction.  Matches the lock-guarded pattern
        # used by add() and HippoMergeInsert.execute() for the same reason.
        _lock_m1 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m1:
            with _txn(self._conn):
                self._conn.execute(stmt, list(encoded_values.values()))  # nosemgrep

    def delete(self, where: str) -> None:
        """Delete rows matching the WHERE predicate (legacy API compat).

        For the records table, first collects (id, vec_label) of matching rows,
        then issues DELETE, then calls mark_deleted on the hnswlib labels and
        removes them from _label_map — all inside the hnswlib lock.
        """
        if self._name == "records" and self._db is not None:
            db = self._db
            with db._hnsw_lock:
                # Lock order: _hnsw_lock (outer) -> _conn_lock (inner).
                # _conn_lock now covers BOTH the SELECT and the DELETE so the
                # full BEGIN..COMMIT window is serialized against concurrent
                # VACUUM, merge_insert, and add writers on the shared connection.
                sel_sql = "SELECT id, vec_label FROM records WHERE " + where  # nosemgrep
                del_sql = "DELETE FROM records WHERE " + where  # nosemgrep
                with db._conn_lock:
                    affected = self._conn.execute(sel_sql).fetchall()  # nosemgrep
                    with _txn(self._conn):
                        self._conn.execute(del_sql)  # nosemgrep
                for row in affected:
                    label = int(row["vec_label"])
                    try:
                        db._hnsw.mark_deleted(label)
                    except RuntimeError:
                        pass  # already deleted — harmless
                    db._label_map.pop(str(row["id"]), None)
            return

        if self._sql is not None:
            stmt = self._sql["delete_prefix"] + where
        else:
            stmt = "DELETE FROM " + self._name + " WHERE " + where  # nosemgrep
        # Serialize the full BEGIN..COMMIT window under _conn_lock so a
        # concurrent VACUUM or merge_insert on the shared connection cannot
        # interleave with this transaction.  Matches the pattern used by add()
        # and HippoMergeInsert.execute().
        _lock_m2 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m2:
            with _txn(self._conn):
                self._conn.execute(stmt)  # nosemgrep

    def merge_insert(self, key_cols: str | list[str]) -> "HippoMergeInsert":
        """Return a HippoMergeInsert builder for the given key column(s)."""
        if isinstance(key_cols, str):
            key_cols = [key_cols]
        return HippoMergeInsert(self, list(key_cols))

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    @property
    def schema(self) -> pa.Schema:
        """Return a pyarrow.Schema derived from PRAGMA table_info.

        The embedding column gets ``pa.list_(pa.float32(), embed_dim)`` so that
        callers using ``tbl.schema.field("embedding").type.list_size`` see the
        embedding dimension.
        """
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"  # nosemgrep
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        fields: list[pa.Field] = []
        for row in pragma_rows:
            col_name = row["name"]
            type_str = row["type"] if row["type"] else "TEXT"
            pa_type = _sqlite_type_to_pa(col_name, type_str, self._embed_dim)
            nullable = not bool(row["notnull"])
            fields.append(pa.field(col_name, pa_type, nullable=nullable))
        return pa.schema(fields)

    def add_columns(self, fields: list[pa.Field]) -> None:
        """Idempotently add columns via ALTER TABLE ADD COLUMN.

        Columns that already exist (detected via PRAGMA table_info) are silently
        skipped. Actual SQLite errors are propagated to the caller.
        """
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
            alter_prefix = self._sql["alter_prefix"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"  # nosemgrep
            alter_prefix = "ALTER TABLE " + self._name + " ADD COLUMN "  # nosemgrep
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        existing = {row["name"] for row in _pragma_rows}
        for f in fields:
            if f.name in existing:
                continue
            sqlite_type = _pa_type_to_sqlite(f.type)
            col_name = _validate_table_name(f.name)
            self._conn.execute(alter_prefix + col_name + " " + sqlite_type)  # nosemgrep
            existing.add(f.name)

    def drop_columns(self, column_names: list[str]) -> None:
        """Drop columns via ALTER TABLE DROP COLUMN (SQLite 3.35+).

        Columns that do not exist are silently skipped. Requires SQLite 3.35 or
        later; raises RuntimeError on older versions.

        Column names are validated via _validate_table_name before use.
        Table name is validated in HippoTable.__init__.
        """
        import sqlite3 as _sqlite3
        # SQLite 3.35 added DROP COLUMN support.
        major, minor, _ = (int(x) for x in _sqlite3.sqlite_version.split("."))
        if (major, minor) < (3, 35):
            raise RuntimeError(
                f"ALTER TABLE DROP COLUMN requires SQLite >= 3.35; "
                f"installed: {_sqlite3.sqlite_version}"
            )
        # Mirror add_columns: build SQL on separate lines with nosemgrep,
        # then execute the pre-built string (same pattern, same safety guarantee).
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"  # nosemgrep
        drop_prefix = "ALTER TABLE " + self._name + " DROP COLUMN "  # nosemgrep
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        existing = {row["name"] for row in _pragma_rows}
        for col in column_names:
            if col not in existing:
                continue
            col_name = _validate_table_name(col)
            self._conn.execute(drop_prefix + col_name)  # nosemgrep
            existing.discard(col)


# ---------------------------------------------------------------------------
# HippoQuery — chainable SQL SELECT builder
# ---------------------------------------------------------------------------


class HippoQuery:
    """Chainable query builder returned by HippoTable.search().

    The table name is always validated (it passes through from HippoTable).
    WHERE predicates are raw SQL passed from callers (legacy compat).

    When ``ann_vector`` and ``ann_db`` are provided, ``to_pandas`` performs
    an hnswlib knn_query and returns the matching records with a ``_distance``
    column (1 - cosine similarity, so 0 = identical).  The query executes
    under ``ann_db._hnsw_lock`` to serialise concurrent access.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        *,
        embed_dim: int,
        ann_vector: "np.ndarray | None" = None,
        ann_db: "HippoDB | None" = None,
        db: "HippoDB | None" = None,
    ) -> None:
        self._conn = conn
        self._table_name = _validate_table_name(table_name)  # validated
        self._embed_dim = embed_dim
        self._where_clauses: list[str] = []
        self._select_cols: list[str] | None = None
        self._limit_val: int | None = None
        self._ann_vector: "np.ndarray | None" = ann_vector
        self._ann_db: "HippoDB | None" = ann_db
        # HippoDB reference for decryption on non-ANN paths.
        # ann_db is always set on the ANN path; db covers the non-ANN path.
        self._db: "HippoDB | None" = db if db is not None else ann_db

    # ------------------------------------------------------------------
    # Builder chain
    # ------------------------------------------------------------------

    def where(self, predicate: str) -> "HippoQuery":
        """Append a SQL WHERE predicate. Chainable."""
        self._where_clauses.append(predicate)
        return self

    def select(self, columns: list[str]) -> "HippoQuery":
        """Restrict the returned columns. Chainable."""
        self._select_cols = list(columns)
        return self

    def limit(self, n: int) -> "HippoQuery":
        """Set a LIMIT on the result. Chainable."""
        self._limit_val = n
        return self

    def distance_type(self, metric: str) -> "HippoQuery":
        """Accept a distance-type hint for API compatibility. No-op: the hnswlib
        index is always initialised in cosine space; the hint is ignored.
        Chainable.
        """
        # Accepted but intentionally unused — cosine is the only supported metric.
        return self

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def _build_sql(self) -> str:
        col_clause = (
            ", ".join(self._select_cols) if self._select_cols else "*"
        )
        # Table name validated at __init__. # nosemgrep: sql-injection
        sql = f"SELECT {col_clause} FROM {self._table_name}"  # nosemgrep: sql-injection
        if self._where_clauses:
            sql += " WHERE " + " AND ".join(
                f"({clause})" for clause in self._where_clauses
            )
        if self._limit_val is not None:
            sql += f" LIMIT {self._limit_val}"
        return sql

    def to_pandas(self) -> pd.DataFrame:
        """Execute the query and return a DataFrame. Empty result returns empty DataFrame.

        When this query was created with an ANN vector, performs hnswlib knn_query
        under the write lock, fetches the matching rows by vec_label, and attaches
        a ``_distance`` column (cosine distance = 1 - similarity, ∈ [0, 2]).
        Results are returned in rank order (closest first).

        Encrypted columns are returned as raw ciphertext (iai:enc:v1: prefix).
        Callers at the MemoryStore boundary decrypt via _from_row / _decrypt_for_record.
        """
        if self._ann_vector is not None and self._ann_db is not None:
            return self._ann_to_pandas()
        sql = self._build_sql()
        # Hold _conn_lock around the SQL fetch so a concurrent consolidation
        # VACUUM (which holds _conn_lock) does not race this read's SQLite read
        # transaction -> "database table is locked". Reader takes _conn_lock
        # only; no _hnsw_lock here, so lock order is preserved.
        _lock = self._db._conn_lock if self._db is not None else None
        if _lock is not None:
            with _lock:
                df = pd.read_sql_query(sql, self._conn)
        else:
            df = pd.read_sql_query(sql, self._conn)
        df = _decode_df_embedding(df)
        return df

    def _ann_to_pandas(self) -> pd.DataFrame:
        """Execute hnswlib knn_query and return matching rows with _distance column."""
        db = self._ann_db
        k = self._limit_val if self._limit_val is not None else 10

        with db._hnsw_lock:
            active_count = len(db._label_map)
            if active_count == 0:
                return pd.DataFrame()
            # Clamp k to the number of available items to avoid hnswlib RuntimeError.
            k_clamped = min(k, active_count)
            labels, distances = db._hnsw.knn_query(self._ann_vector, k=k_clamped)

        # knn_query returns shape (1, k) since we passed a single query vector.
        flat_labels: list[int] = labels[0].tolist()
        flat_distances: list[float] = distances[0].tolist()

        if not flat_labels:
            return pd.DataFrame()

        # Fetch matching rows in one query using an IN clause with bound params.
        placeholders = ", ".join("?" for _ in flat_labels)
        # Table name validated at __init__. # nosemgrep: sql-injection
        sql = (  # nosemgrep: sql-injection
            f"SELECT * FROM {self._table_name} WHERE vec_label IN ({placeholders})"
        )
        if self._where_clauses:
            sql += " AND " + " AND ".join(f"({c})" for c in self._where_clauses)

        # _hnsw_lock block above is already closed; take _conn_lock only for the
        # SQL fetch so a concurrent VACUUM does not race it. Order preserved
        # (_hnsw_lock released before _conn_lock acquired).
        _lock = db._conn_lock if db is not None else None
        if _lock is not None:
            with _lock:
                df = pd.read_sql_query(sql, self._conn, params=flat_labels)
        else:
            df = pd.read_sql_query(sql, self._conn, params=flat_labels)
        df = _decode_df_embedding(df)

        if df.empty:
            return df

        # Build a distance lookup keyed on vec_label (Python int) and attach _distance.
        dist_map: dict[int, float] = {
            int(lbl): float(d) for lbl, d in zip(flat_labels, flat_distances)
        }
        df["_distance"] = df["vec_label"].apply(lambda lbl: dist_map.get(int(lbl), float("nan")))

        # Sort by distance ascending (closest first) to preserve rank order.
        df = df.sort_values("_distance").reset_index(drop=True)
        return df

    def _decrypt_query_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decrypt encrypted columns in the query result DataFrame.

        Records table: strict path (raises HippoDecryptError on failure).
        Events table: lenient path (empty fallback on failure).
        All other tables: returned unchanged.
        """
        if self._db is None or self._db._crypto_key_provider is None:
            return df
        if self._table_name == "records":
            return _decrypt_df_columns(
                df, _ENCRYPTED_RECORD_COLUMNS, self._db._decrypt_record_field
            )
        if self._table_name == "events":
            return _decrypt_df_columns(
                df, _ENCRYPTED_EVENTS_COLUMNS, self._db._decrypt_event_field
            )
        return df

    def to_batches(self, batch_size: int = 1000) -> Iterator[pa.RecordBatch]:
        """Yield pyarrow.RecordBatch slices of the query result.

        The cursor execute() + the entire fetchmany() drain run under the shared
        connection lock (mirroring to_pandas). The shared sqlite3.Connection runs
        with check_same_thread=False, so a concurrent add()/execute on the same
        connection from another worker thread can reset cursor state between
        execute() and a later fetchmany() -- corrupting the result set
        (truncated rows / IndexError / InterfaceError). Holding the connection
        lock across the cursor's full life closes that race.

        Design: the batches are materialized into memory under the lock, the
        cursor is closed deterministically, the lock is released, and only then
        are the batches yielded. This bounds the lock hold to the drain (never
        spanning arbitrary consumer time) and guarantees the cursor never
        outlives the lock. Materialization is acceptable here: this path
        projects a small set of columns over the member set, so the buffered
        result is bounded; it is not a full-record fan-out.
        """
        sql = self._build_sql()
        _lock = self._db._conn_lock if self._db is not None else None
        if _lock is not None:
            with _lock:
                batches = self._drain_to_batches(sql, batch_size)
        else:
            batches = self._drain_to_batches(sql, batch_size)
        yield from batches

    def _drain_to_batches(
        self, sql: str, batch_size: int
    ) -> list[pa.RecordBatch]:
        """Drain the cursor fully into a list of RecordBatch, closing the cursor.

        The caller holds the connection lock (when a HippoDB is attached) so the
        execute() + fetchmany() loop runs without a concurrent writer resetting
        the shared cursor. The cursor is closed in a finally so it never leaks,
        even if a fetch raises partway through the drain.
        """
        batches: list[pa.RecordBatch] = []
        cursor = self._conn.execute(sql)
        try:
            column_names = [desc[0] for desc in cursor.description]
            while True:
                raw_rows = cursor.fetchmany(batch_size)
                if not raw_rows:
                    break
                data: dict[str, list] = {c: [] for c in column_names}
                for row in raw_rows:
                    for col in column_names:
                        data[col].append(row[col])
                # Decode embedding column within the batch
                if "embedding" in data:
                    data["embedding"] = [
                        _decode_embedding(b)
                        if isinstance(b, (bytes, bytearray))
                        else b
                        for b in data["embedding"]
                    ]
                batches.append(pa.record_batch(data))
        finally:
            cursor.close()
        return batches


# ---------------------------------------------------------------------------
# HippoMergeInsert — INSERT ... ON CONFLICT ... DO UPDATE SET ...
# ---------------------------------------------------------------------------


class HippoMergeInsert:
    """Builder for upsert operations on a HippoTable.

    Pattern::

        tbl.merge_insert(["src", "dst", "edge_type"])
           .when_matched_update_all()
           .execute(rows)
    """

    def __init__(self, table: HippoTable, key_cols: list[str]) -> None:
        self._table = table
        self._key_cols = key_cols
        self._update_all: bool = False

    def when_matched_update_all(self) -> "HippoMergeInsert":
        """Set the matched-update mode (the only supported mode). Chainable."""
        self._update_all = True
        return self

    def execute(self, data: Any) -> None:
        """Upsert data into the table.

        Accepts ``pa.Table``, ``pd.DataFrame``, or ``list[dict]``. An empty
        data argument is a no-op.

        When only a subset of the table's columns are provided (the common
        case for provenance / partial-update callers), we emit a plain
        UPDATE WHERE statement — not INSERT+ON CONFLICT — so that rows without
        all NOT-NULL columns do not trigger constraint errors.  Unmatched rows
        are silently ignored (matching the legacy when_matched_update_all()
        semantics).

        When the data rows contain ALL table columns (full-row upsert, e.g.
        edges re-insert), we fall back to INSERT ... ON CONFLICT ... DO UPDATE.

        Encrypted columns are encrypted before the SQL is emitted.
        """
        rows = _normalize_to_row_list(data)
        if not rows:
            return

        # Apply per-field encryption before embedding encode (disjoint columns).
        rows = self._table._encrypt_rows(rows)
        encoded = [_encode_row_for_insert(r) for r in rows]
        all_cols = list(encoded[0].keys())
        non_key = [c for c in all_cols if c not in self._key_cols]
        key_conflict = ", ".join(self._key_cols)
        conn = self._table._conn

        # Serialize every db._conn BEGIN/COMMIT + executemany under the shared
        # connection RLock. The daemon fans consolidation work across threads
        # via asyncio.to_thread (a co-occurrence writer here racing a VACUUM/
        # checkpoint elsewhere); without this lock two threads interleave their
        # transaction brackets on the one check_same_thread=False connection and
        # corrupt its transaction state ("cannot start a transaction within a
        # transaction" / "cannot commit - no transaction is active" / "bad
        # parameter or other API misuse"). _conn_lock is a re-entrant RLock so
        # any inner helper that re-acquires it is safe; lock order is
        # _hnsw_lock-before-_conn_lock and no _hnsw_lock is taken here.
        _db = self._table._db
        _conn_lock = (
            _db._conn_lock if _db is not None else contextlib.nullcontext()
        )

        with _conn_lock:
            # Detect partial-column update: provided columns < table column count.
            # Use pure UPDATE to avoid NOT NULL violations on absent columns.
            try:
                actual_cols_count = len(
                    conn.execute(  # nosemgrep
                        f"SELECT * FROM {self._table._name} LIMIT 0"  # nosemgrep
                    ).description or []
                )
            except Exception:  # noqa: BLE001
                actual_cols_count = len(all_cols)

            is_partial = len(all_cols) < actual_cols_count

            if is_partial and non_key and self._update_all:
                # Partial update: UPDATE only matched rows; ignore unmatched.
                update_clause = ", ".join(f"{c}=?" for c in non_key)
                where_clause = " AND ".join(f"{k}=?" for k in self._key_cols)
                # Column names from canonical DDL. # nosemgrep: sql-injection
                sql = (  # nosemgrep: sql-injection
                    f"UPDATE {self._table._name} SET {update_clause} WHERE {where_clause}"
                )
                # Params: non-key values first, then key values (for WHERE).
                params = [
                    tuple(r.get(c) for c in non_key) + tuple(r.get(k) for k in self._key_cols)
                    for r in encoded
                ]
                with _txn(conn):
                    conn.executemany(sql, params)
                return

            # Full-row upsert: INSERT ... ON CONFLICT ... DO UPDATE.
            placeholders = ", ".join("?" for _ in all_cols)
            col_names = ", ".join(all_cols)

            if non_key:
                update_clause = ", ".join(f"{c}=excluded.{c}" for c in non_key)
                # Table name validated in HippoTable.__init__. Column names from DDL.
                sql = (  # nosemgrep: sql-injection
                    f"INSERT INTO {self._table._name} ({col_names}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT ({key_conflict}) DO UPDATE SET {update_clause}"
                )
            else:
                sql = (  # nosemgrep: sql-injection
                    f"INSERT INTO {self._table._name} ({col_names}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT ({key_conflict}) DO NOTHING"
                )

            with _txn(conn):
                conn.executemany(sql, [tuple(r.get(c) for c in all_cols) for r in encoded])


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def _normalize_to_row_list(data: Any) -> list[dict]:
    """Convert ``pa.Table``, ``pd.DataFrame``, or ``list[dict]`` to list[dict]."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if isinstance(data, pa.Table):
        return data.to_pylist()
    # Fallback: try iterable of dicts
    return list(data)


# ---------------------------------------------------------------------------
# No-flock direct recency reader (REQ-1)
# ---------------------------------------------------------------------------

# Fixed SQL statement (no user-controlled interpolation; semgrep gate satisfied).
_DIRECT_RECENCY_SQL = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " hv_tier, structure_hv_payload,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE tombstoned_at IS NULL ORDER BY created_at DESC"
)

# Bounded variant for the CC-A daemon-down degrade path.  Standalone literal
# (implicit string adjacency — no + operator) so the SQL-concat semgrep rule
# does not fire.  The LIMIT integer value is bound via the params tuple at
# call time; it is never interpolated into the SQL text.
_DIRECT_RECENCY_SQL_LIMITED = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " hv_tier, structure_hv_payload,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE tombstoned_at IS NULL ORDER BY created_at DESC"
    " LIMIT ?"
)


def _no_flock_recency_rows_from_store(
    db_path: Path,
    limit: "int | None" = None,
) -> list[dict]:
    """Bare no-flock query_only sqlite3 reader — SIGKILL-missing-shm fallback.

    Opens ``brain.sqlite3`` with a normal read-write sqlite3 connection +
    PRAGMA query_only=ON.  NO ``hippo/.lock`` flock acquired and NO hnswlib
    index loaded.  A normal-rw connection with query_only=ON can reconstruct
    the WAL shm file in-place (unlike mode=ro, which raises READONLY_CANTINIT
    when the shm is absent after a non-clean exit).

    This fallback is used when:
    - The primary SHARED+read_only HippoDB path fails (e.g. ConsolidationPendingError,
      or the lock file is absent in a partially set-up store), OR
    - The store is in a SIGKILL-residue state where even a non-blocking LOCK_SH
      open could be unreliable (e.g. WAL shm absent and lock file missing).

    Parameters
    ----------
    limit:
        When provided, uses _DIRECT_RECENCY_SQL_LIMITED (a pre-built standalone
        constant with a parameterized LIMIT clause) so at most ``limit`` rows are
        fetched from SQLite.  The integer value is bound via the params tuple.
        When None, _DIRECT_RECENCY_SQL (no LIMIT) returns all rows — preserving
        the unbounded behaviour of the direct_recency.py consumer.

    Returns [] on any error (graceful degrade).
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        if limit is not None:
            cursor = conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (limit,))
        else:
            cursor = conn.execute(_DIRECT_RECENCY_SQL)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def reconcile_index_mid_run(hippo: "HippoDB") -> dict:
    """Trigger a mid-run integrity rebuild on an already-open HippoDB.

    Equivalent to the boot integrity rebuild but callable at any time
    (not only from __init__).  Used by the daemon wake sequence and
    available for test driving.  Delegates to _rebuild_index_from_sqlite
    which excludes pending rows and atomically saves.
    """
    return hippo._rebuild_index_from_sqlite()


def direct_recency_rows_from_store(
    store_root: "Path | str",
    limit: "int | None" = None,
) -> list[dict]:
    """Return raw record rows from brain.sqlite3 via the SHARED+read_only path.

    Primary path (RESEARCH Pattern 2): opens Hippo with
    ``HippoDB(store_root, access_mode=SHARED, read_only=True)`` which:
      - Acquires LOCK_SH|LOCK_NB (non-blocking, <1.5 s budget, honors the
        consolidation-intent flag and performs the post-acquire recheck).
      - Sets PRAGMA query_only=ON + PRAGMA busy_timeout=2000.
      - Skips hnswlib index load (ANN not needed for recency surface).

    This path is now safe against a daemon in WAKE (which holds LOCK_SH,
    compatible with a client LOCK_SH) AND delivers the F6 BUSY-latency
    guarantee (intent flag causes the client to back off rather than hang
    through a VACUUM window).

    SIGKILL-missing-shm fallback: when the primary SHARED path fails (e.g.
    ConsolidationPendingError or any other open error), the helper falls
    back to the bare no-flock query_only sqlite3 reader.  The fallback
    handles the SIGKILL-residue edge case where the
    WAL shm file is absent and even a non-blocking LOCK_SH open should not
    be attempted (lock file may be absent too).  This maintains the REQ-1
    test contract for ``test_recency_read_daemon_down_sigkill``.

    Parameters
    ----------
    limit:
        When provided, fetches at most ``limit`` rows using the pre-built
        _DIRECT_RECENCY_SQL_LIMITED constant (parameterized LIMIT — integer
        value bound via params tuple, not interpolated).  When None, all rows
        are returned — preserving the full-result behaviour of the unbounded
        consumer in direct_recency.

    Returns a list of row dicts (the same columns ``MemoryStore._from_row``
    consumes).  Returns [] on any error so callers degrade gracefully.
    """
    root = Path(store_root)
    db_path = root / "hippo" / "brain.sqlite3"
    if not db_path.exists():
        return []

    # Primary: SHARED+read_only HippoDB open via a short-probe flock attempt.
    # We attempt LOCK_SH with a ~200 ms timeout: if the consolidation-intent
    # flag is set or LOCK_EX is held (e.g. VACUUM), we fall back immediately
    # to the no-flock reader, keeping the ≤1.5 s recency SLO.
    # The full ≤1.45 s _SHARED_LOCK_TIMEOUT_S budget is used for normal client
    # opens (store.py MemoryStore); the recency path uses a shorter probe.
    db: "HippoDB | None" = None
    try:
        db = HippoDB(
            root,
            access_mode=AccessMode.SHARED,
            read_only=True,
            _lock_timeout_override=0.20,
        )
        with db._conn_lock:
            if limit is not None:
                cursor = db._conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (limit,))
            else:
                cursor = db._conn.execute(_DIRECT_RECENCY_SQL)
            rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001 — fall through to no-flock fallback
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    # Fallback: bare no-flock query_only reader (SIGKILL-residue path).
    return _no_flock_recency_rows_from_store(db_path, limit=limit)


# ---------------------------------------------------------------------------
# Client ANN helpers (read-only; CLIENT-facing, not daemon-internal)
# ---------------------------------------------------------------------------


def load_hnsw_readonly(store_root: "str | Path", embed_dim: int) -> "hnswlib.Index | None":
    """Load records.hnsw from disk as a transient read-only hnswlib index.

    CLIENT primitive for the daemon-degraded semantic path.  Loads ONLY
    ``records.hnsw`` (the atomically-written stable file) — never
    ``records.hnsw.tmp`` (a half-written .tmp would
    crash knn_query).  Does NOT write, does NOT trigger a rebuild.
    Returns None when the file is absent or fails to load.
    """
    hnsw_path = Path(store_root) / "hippo" / "records.hnsw"
    if not hnsw_path.exists():
        return None
    try:
        idx = hnswlib.Index(space="cosine", dim=embed_dim)
        idx.load_index(str(hnsw_path), max_elements=0)  # 0 = use persisted capacity
        idx.set_ef(200)
        idx.set_num_threads(1)
        return idx
    except Exception:  # noqa: BLE001 — corrupt or incompatible index
        return None


def _ann_lookup_client(
    store_root: "str | Path",
    cue_vec: "list[float]",
    *,
    k: int = 10,
    embed_dim: int = EMBED_DIM,
) -> "list[int]":
    """Return hnswlib vec_labels matching cue_vec using a client-loaded index.

    Loads records.hnsw in the client process (no daemon).  Returns a list
    of integer vec_labels (may be empty if the index is absent, corrupt, or
    has no active elements).  Caller maps labels back to record ids via the
    SQLite records table.
    """
    idx = load_hnsw_readonly(store_root, embed_dim)
    if idx is None or idx.get_current_count() == 0:
        return []
    try:
        k_actual = min(k, idx.get_current_count())
        cue_np = np.array(cue_vec, dtype=np.float32).reshape(1, -1)
        labels_arr, _distances = idx.knn_query(cue_np, k=k_actual)
        return [int(lbl) for lbl in labels_arr[0]]
    except Exception:  # noqa: BLE001 — index incompatible or corrupted
        return []


def degraded_semantic_recall(
    store_root: "str | Path",
    cue: str,
    limit: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    """STANDALONE client-invocable degraded recall (REQ-3b, H2).

    Opens HippoDB(SHARED, read_only=True) in the CLIENT process and returns
    a functional recency/temporal result tagged with ``_degraded=True``.

    Design:
    - No daemon involvement — caller uses this ONLY when the full
      memory_recall RPC is unreachable (daemon down / LOCK_EX window).
    - Returns recency/temporal records (recent_user_turns or direct
      recency rows), never empty as a hard-fail, never bank.
    - Results are dicts with keys: literal_surface, score, _degraded.
    - Pending rows (embedding_pending=1) are included per CL4-H1.

    Falls back to bare direct_recency_rows_from_store if HippoDB SHARED
    open fails (e.g. SIGKILL lock residue).
    """
    root = Path(store_root)

    # Primary: SHARED HippoDB, direct SQL recency read.
    # CC-A: use _DIRECT_RECENCY_SQL_LIMITED (pre-built standalone constant with
    # parameterized LIMIT) to fetch at most `limit` rows from SQLite — no
    # fetch-all-then-truncate.  The integer is bound via the params tuple.
    db: "HippoDB | None" = None
    try:
        db = HippoDB(
            root,
            access_mode=AccessMode.SHARED,
            read_only=True,
            _lock_timeout_override=0.25,
        )
        with db._conn_lock:
            rows = db._conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (limit,)).fetchall()
        row_dicts = [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        row_dicts = []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    if not row_dicts:
        # Last-ditch: no-flock path (SIGKILL residue tolerance).
        row_dicts = direct_recency_rows_from_store(root, limit=limit)

    # Decrypt surfaces using the crypto key (CryptoKey resolves the
    # IAI_MCP_CRYPTO_PASSPHRASE env var or the key file — same path as MemoryStore).
    _crypto_key: "bytes | None" = None
    try:
        from iai_mcp.crypto import CryptoKey as _CryptoKey
        _crypto_key = _CryptoKey(store_root=root).get_or_create()
    except Exception:  # noqa: BLE001 — no key available: leave ciphertext as-is
        pass

    try:
        from iai_mcp.crypto import decrypt_field as _decrypt_field, is_encrypted as _is_enc
    except Exception:  # noqa: BLE001
        _decrypt_field = None  # type: ignore[assignment]
        _is_enc = None  # type: ignore[assignment]

    # Map rows to the return dict shape the callers expect.
    seen_ids: set[str] = set()
    results: list[dict] = []
    for row in row_dicts:
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        surface = row.get("literal_surface") or ""
        # Decrypt surface if still ciphertext and key is available.
        if surface and _crypto_key is not None and _is_enc is not None and _decrypt_field is not None:
            try:
                if _is_enc(surface):
                    aad = row_id.encode("utf-8")
                    surface = _decrypt_field(surface, _crypto_key, aad)
            except Exception:  # noqa: BLE001 — leave ciphertext if decrypt fails
                pass
        results.append({
            "literal_surface": surface,
            "score": 0.0,
            "_degraded": True,
            "_source": "direct-store",
        })
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
