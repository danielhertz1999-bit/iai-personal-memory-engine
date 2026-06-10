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

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


_txn_owners: dict[int, int] = {}
_txn_owners_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager  # type: ignore[misc]
def _txn(conn: "sqlite3.Connection"):
    if conn.in_transaction:
        with _txn_owners_lock:
            owner = _txn_owners.get(id(conn))
        if owner is None:
            yield
            return
        if owner == threading.get_ident():
            yield
            return
        raise HippoIntegrityError(
            f"Shared connection transaction owned by thread {owner} "
            f"observed by thread {threading.get_ident()} — a transactional "
            f"mutator site is missing _conn_lock serialization."
        )
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


_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> str:
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return name


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


class HippoTableList:

    def __init__(self, tables: list[str]) -> None:
        self.tables: list[str] = tables

    def __iter__(self) -> Iterator[str]:
        return iter(self.tables)

    def __repr__(self) -> str:  # pragma: no cover
        return f"HippoTableList(tables={self.tables!r})"


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
    "detail_level",
})

_STRICT_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
})


def _sqlite_type_to_pa(col_name: str, type_str: str, embed_dim: int) -> pa.DataType:
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
    return pa.string()


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


class HippoDB:

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        crypto_key_provider: Callable[[], bytes] | None = None,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
        _lock_timeout_override: float | None = None,
    ) -> None:
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

        db_path = self._hippo_dir / "brain.sqlite3"
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=2000")
        if read_only:
            self._conn.execute("PRAGMA query_only=ON")

        _env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
        self._embed_dim: int = (
            int(_env_dim) if _env_dim and _env_dim.isdigit() else EMBED_DIM
        )
        self._closed: bool = False
        self._hnsw_path: Path = self._hippo_dir / "records.hnsw"
        self._hnsw_tmp_path: Path = self._hippo_dir / "records.hnsw.tmp"
        self._hnsw_lock: threading.RLock = threading.RLock()
        self._conn_lock: threading.RLock = threading.RLock()
        if not read_only:
            self._ensure_tables()

        if not read_only:
            meta_dim = self._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = 'embed_dim'"
            ).fetchone()
            if meta_dim is not None:
                self._embed_dim = int(meta_dim[0])
        self._label_map: dict[str, int] = {}
        self._write_counter: int = 0

        if read_only:
            self._hnsw: hnswlib.Index | None = None  # type: ignore[assignment]
            try:
                self._repopulate_label_map_from_sqlite()
            except Exception:  # noqa: BLE001
                pass
        else:
            self._repopulate_label_map_from_sqlite()
            self._initialize_hnsw_index()


    def _acquire_exclusive_lock(self) -> None:
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
        _intent_path = self._hippo_dir / ".consolidation-pending"

        with _PROCESS_LOCKS_GUARD:
            if self._lock_key in _PROCESS_LOCKS:
                raise HippoLockHeldError(
                    self._lock_path,
                    "same-process-holds-EXCLUSIVE",
                )
            held_sh = _PROCESS_LOCKS_SHARED.get(self._lock_key)
            if held_sh is not None:
                base_fd, refcount = held_sh
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, refcount + 1)
                return

            base_fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            os.chmod(str(self._lock_path), 0o600)

        _timeout = (
            lock_timeout_override
            if lock_timeout_override is not None
            else _SHARED_LOCK_TIMEOUT_S
        )
        deadline = time.monotonic() + _timeout
        acquired = False
        for _ in range(_SHARED_MAX_RETRIES + 1):
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

        with _PROCESS_LOCKS_GUARD:
            held_sh = _PROCESS_LOCKS_SHARED.get(self._lock_key)
            if held_sh is not None:
                fcntl.flock(base_fd, fcntl.LOCK_UN)
                os.close(base_fd)
                base_fd2, refcount2 = held_sh
                self._lock_fd = os.dup(base_fd2)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd2, refcount2 + 1)
            else:
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, 1)


    def downgrade_to_shared(self) -> None:
        _intent_path = self._hippo_dir / ".consolidation-pending"

        with _PROCESS_LOCKS_GUARD:
            if self._access_mode is not AccessMode.EXCLUSIVE:
                return
            held = _PROCESS_LOCKS.get(self._lock_key)
            if held is None:
                return
            base_fd, refcount = held
            try:
                fcntl.flock(base_fd, fcntl.LOCK_SH)
            except OSError:
                return
            del _PROCESS_LOCKS[self._lock_key]
            _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, refcount)
        self._access_mode = AccessMode.SHARED

        try:
            _intent_path.unlink()
        except FileNotFoundError:
            pass

    def escalate_to_exclusive(self, intent_budget_ms: int = 4000) -> None:
        _intent_path = self._hippo_dir / ".consolidation-pending"

        try:
            fd = os.open(str(_intent_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass

        if self._access_mode is AccessMode.EXCLUSIVE:
            return

        with _PROCESS_LOCKS_GUARD:
            held = _PROCESS_LOCKS_SHARED.get(self._lock_key)
        if held is None:
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

        with _PROCESS_LOCKS_GUARD:
            if held is not None:
                _, refcount = held
                del _PROCESS_LOCKS_SHARED[self._lock_key]
            else:
                refcount = 1
                self._lock_fd = os.dup(base_fd)
            _PROCESS_LOCKS[self._lock_key] = (base_fd, refcount)
        self._access_mode = AccessMode.EXCLUSIVE


    def _initialize_hnsw_index(self) -> None:
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

        for candidate in (self._hnsw_tmp_path, self._hnsw_path):
            if candidate.exists():
                try:
                    idx = hnswlib.Index(space="cosine", dim=self._embed_dim)
                    idx.load_index(str(candidate), max_elements=cap)
                    idx.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
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
            if sqlite_count > 0:
                _log.info(
                    "No valid hnswlib file found; rebuilding from %d SQLite records",
                    sqlite_count,
                )
                self._rebuild_index_from_sqlite()
                return

        active_label_count = len(self._label_map)
        if active_label_count != sqlite_count:
            _log.info(
                "Boot integrity check: active labels=%d != sqlite count=%d — rebuilding",
                active_label_count,
                sqlite_count,
            )
            self._rebuild_index_from_sqlite()

    def _repopulate_label_map_from_sqlite(self) -> None:
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

        self._save_index_atomic()

        self._repopulate_label_map_from_sqlite()

        return {"action": "rebuild", "rebuilt_count": n}

    def _save_index_atomic(self) -> None:
        try:
            self._hnsw.save_index(str(self._hnsw_tmp_path))
            os.replace(self._hnsw_tmp_path, self._hnsw_path)
        except OSError as exc:
            _log.warning("hnswlib index save failed: %s", exc)

    def _maybe_resize(self) -> None:
        current = self._hnsw.get_current_count()
        max_el = self._hnsw.get_max_elements()
        if max_el > 0 and current > HNSW_RESIZE_HEADROOM * max_el:
            self._hnsw.resize_index(max_el * 2)


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
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1 LIMIT 1"
            ).fetchone()
        return row is not None

    def reembed_pending_rows(self, embedder: Any) -> int:
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

            try:
                npy_path.unlink()
                json_path.unlink()
            except OSError as exc:
                _log.warning("ingest_pending_embeddings: cleanup failed for %s: %s", npy_path, exc)

            ingested += 1
        return ingested

    def pending_embeddings_wake_sequence(self, embedder: Any | None = None) -> dict:
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


    def _encrypt_for_uuid(self, uuid_str: str, value: str) -> str:
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
        if self._crypto_key_provider is None:
            return value
        if value is None or not is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        try:
            return decrypt_field(value, key, associated_data=ad)
        except Exception as exc:
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
            pass


    def _ensure_tables(self) -> None:
        conn = self._conn
        conn.execute("BEGIN")
        try:
            conn.execute(_DDL_RECORDS)
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
        safe_table = _validate_table_name(table_name)
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
        pragma_stmt = "PRAGMA table_info(" + safe_table + ")"
        _lock = getattr(self, "_conn_lock", None)
        if _lock is not None:
            with _lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
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
            )
            try:
                self._conn.execute(alter_stmt)
            except Exception:  # noqa: BLE001 -- aggregate names, raise once below
                failing.append(col_name)

        if failing:
            raise RuntimeError(
                f"schema reconciliation failed for table {safe_table!r}: "
                f"could not add columns {failing!r}"
            )


    def table_names(self) -> list[str]:
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [row["name"] for row in rows]

    def list_tables(self) -> HippoTableList:
        return HippoTableList(self.table_names())


    def open_table(self, name: str) -> "HippoTable":
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def create_table(
        self,
        name: str,
        schema: pa.Schema | None = None,
        data: Any = None,
    ) -> "HippoTable":
        _validate_table_name(name)
        if name not in self.table_names():
            if schema is not None:
                cols = []
                for f in schema:
                    sqlite_type = _pa_type_to_sqlite(f.type)
                    col_name = _validate_table_name(f.name)
                    cols.append(f"{col_name} {sqlite_type}")
                ddl = f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(cols)})"
                self._conn.execute("BEGIN")
                try:
                    self._conn.execute(ddl)
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                self._conn.execute("COMMIT")
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def drop_table(self, name: str) -> None:
        _validate_table_name(name)
        self._conn.execute(f"DROP TABLE IF EXISTS {name}")


    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
                try:
                    os.close(self._lock_fd)
                except Exception:  # noqa: BLE001
                    pass
                self._lock_fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "HippoDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _encode_embedding(vec: list[float] | np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.array(vec, dtype=np.float32).tobytes()


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _encode_row_for_insert(row: dict) -> dict:
    out = dict(row)
    if "embedding" in out and out["embedding"] is not None:
        out["embedding"] = _encode_embedding(out["embedding"])
    return out


def _decode_df_embedding(df: pd.DataFrame) -> pd.DataFrame:
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


class HippoTable:

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
        self._db: "HippoDB | None" = db
        self._ann_index = ann_index
        self._sql: dict[str, str] | None = _TABLE_SQL.get(self._name)

    def _stmt(self, key: str) -> str:
        if self._sql is not None:
            return self._sql[key]
        raise KeyError(f"No pre-built SQL for key {key!r} on dynamic table {self._name!r}")


    def count_rows(self, filter: str | None = None) -> int:  # noqa: A002
        if self._sql is not None:
            base = self._sql["count"]
        else:
            base = "SELECT COUNT(*) FROM " + self._name
        stmt = (base + " WHERE " + filter) if filter else base
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
        if self._sql is not None:
            stmt = self._sql["select_all"]
        else:
            stmt = "SELECT * FROM " + self._name
        if self._db is not None:
            with self._db._conn_lock:
                df = pd.read_sql_query(stmt, self._conn)
        else:
            df = pd.read_sql_query(stmt, self._conn)
        df = _decode_df_embedding(df)
        return df

    def _decrypt_df(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return [{"version": 1, "ts": datetime.now(timezone.utc).isoformat()}]

    def optimize(
        self,
        cleanup_older_than: Any = None,
        delete_unverified: bool = False,
        **kwargs: Any,
    ) -> dict:
        return {"compaction": "noop_hippo"}


    def add(self, rows: Any) -> None:
        row_list = _normalize_to_row_list(rows)
        if not row_list:
            return
        row_list = self._encrypt_rows(row_list)
        encoded = [_encode_row_for_insert(r) for r in row_list]
        cols = list(encoded[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        if self._sql is not None:
            stmt = self._sql["insert_prefix"] + "(" + col_names + ") VALUES (" + placeholders + ")"
        else:
            stmt = "INSERT INTO " + self._name + " (" + col_names + ") VALUES (" + placeholders + ")"

        if self._name == "records" and self._db is not None:
            db = self._db
            with db._hnsw_lock:
                with db._conn_lock:
                    with _txn(self._conn):
                        for r, enc in zip(row_list, encoded):
                            cursor = self._conn.execute(stmt, tuple(enc.get(c) for c in cols))
                            vec_label = int(cursor.lastrowid)
                            emb_raw = r.get("embedding")
                            if emb_raw is not None:
                                emb_vec = np.array(emb_raw, dtype=np.float32).reshape(1, -1)
                                db._hnsw.add_items(emb_vec, np.array([vec_label], dtype=np.int64))
                                db._label_map[str(r["id"])] = vec_label
                                db._write_counter += 1
                            db._maybe_resize()
                if db._write_counter > 0 and db._write_counter % HNSW_SAVE_INTERVAL == 0:
                    db._save_index_atomic()
        else:
            if self._db is not None:
                lock_ctx = self._db._conn_lock
            else:
                lock_ctx = contextlib.nullcontext()
            with lock_ctx:
                with _txn(self._conn):
                    self._conn.executemany(stmt, [tuple(r.get(c) for c in cols) for r in encoded])

    def update(self, where: str, values: dict[str, Any]) -> None:
        if not values:
            return

        enc_cols: tuple[str, ...] = ()
        if self._db is not None and self._db._crypto_key_provider is not None:
            if self._name == "records":
                enc_cols = _ENCRYPTED_RECORD_COLUMNS
            elif self._name == "events":
                enc_cols = _ENCRYPTED_EVENTS_COLUMNS

        encrypted_being_updated = [c for c in values if c in enc_cols]
        if encrypted_being_updated:
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
            stmt = "UPDATE " + self._name + " SET " + set_clause + " WHERE " + where
        _lock_m1 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m1:
            with _txn(self._conn):
                self._conn.execute(stmt, list(encoded_values.values()))

    def delete(self, where: str) -> None:
        if self._name == "records" and self._db is not None:
            db = self._db
            with db._hnsw_lock:
                sel_sql = "SELECT id, vec_label FROM records WHERE " + where
                del_sql = "DELETE FROM records WHERE " + where
                with db._conn_lock:
                    affected = self._conn.execute(sel_sql).fetchall()
                    with _txn(self._conn):
                        self._conn.execute(del_sql)
                for row in affected:
                    label = int(row["vec_label"])
                    try:
                        db._hnsw.mark_deleted(label)
                    except RuntimeError:
                        pass
                    db._label_map.pop(str(row["id"]), None)
            return

        if self._sql is not None:
            stmt = self._sql["delete_prefix"] + where
        else:
            stmt = "DELETE FROM " + self._name + " WHERE " + where
        _lock_m2 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m2:
            with _txn(self._conn):
                self._conn.execute(stmt)

    def merge_insert(self, key_cols: str | list[str]) -> "HippoMergeInsert":
        if isinstance(key_cols, str):
            key_cols = [key_cols]
        return HippoMergeInsert(self, list(key_cols))


    @property
    def schema(self) -> pa.Schema:
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"
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
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
            alter_prefix = self._sql["alter_prefix"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"
            alter_prefix = "ALTER TABLE " + self._name + " ADD COLUMN "
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
            self._conn.execute(alter_prefix + col_name + " " + sqlite_type)
            existing.add(f.name)

    def drop_columns(self, column_names: list[str]) -> None:
        import sqlite3 as _sqlite3
        major, minor, _ = (int(x) for x in _sqlite3.sqlite_version.split("."))
        if (major, minor) < (3, 35):
            raise RuntimeError(
                f"ALTER TABLE DROP COLUMN requires SQLite >= 3.35; "
                f"installed: {_sqlite3.sqlite_version}"
            )
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"
        drop_prefix = "ALTER TABLE " + self._name + " DROP COLUMN "
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
            self._conn.execute(drop_prefix + col_name)
            existing.discard(col)


class HippoQuery:

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
        self._table_name = _validate_table_name(table_name)
        self._embed_dim = embed_dim
        self._where_clauses: list[str] = []
        self._select_cols: list[str] | None = None
        self._limit_val: int | None = None
        self._ann_vector: "np.ndarray | None" = ann_vector
        self._ann_db: "HippoDB | None" = ann_db
        self._db: "HippoDB | None" = db if db is not None else ann_db


    def where(self, predicate: str) -> "HippoQuery":
        self._where_clauses.append(predicate)
        return self

    def select(self, columns: list[str]) -> "HippoQuery":
        self._select_cols = list(columns)
        return self

    def limit(self, n: int) -> "HippoQuery":
        self._limit_val = n
        return self

    def distance_type(self, metric: str) -> "HippoQuery":
        return self


    def _build_sql(self) -> str:
        col_clause = (
            ", ".join(self._select_cols) if self._select_cols else "*"
        )
        sql = f"SELECT {col_clause} FROM {self._table_name}"
        if self._where_clauses:
            sql += " WHERE " + " AND ".join(
                f"({clause})" for clause in self._where_clauses
            )
        if self._limit_val is not None:
            sql += f" LIMIT {self._limit_val}"
        return sql

    def to_pandas(self) -> pd.DataFrame:
        if self._ann_vector is not None and self._ann_db is not None:
            return self._ann_to_pandas()
        sql = self._build_sql()
        _lock = self._db._conn_lock if self._db is not None else None
        if _lock is not None:
            with _lock:
                df = pd.read_sql_query(sql, self._conn)
        else:
            df = pd.read_sql_query(sql, self._conn)
        df = _decode_df_embedding(df)
        return df

    def _ann_to_pandas(self) -> pd.DataFrame:
        db = self._ann_db
        k = self._limit_val if self._limit_val is not None else 10

        with db._hnsw_lock:
            active_count = len(db._label_map)
            if active_count == 0:
                return pd.DataFrame()
            k_clamped = min(k, active_count)
            labels, distances = db._hnsw.knn_query(self._ann_vector, k=k_clamped)

        flat_labels: list[int] = labels[0].tolist()
        flat_distances: list[float] = distances[0].tolist()

        if not flat_labels:
            return pd.DataFrame()

        placeholders = ", ".join("?" for _ in flat_labels)
        sql = (  # nosemgrep: sql-injection
            f"SELECT * FROM {self._table_name} WHERE vec_label IN ({placeholders})"
        )
        if self._where_clauses:
            sql += " AND " + " AND ".join(f"({c})" for c in self._where_clauses)

        _lock = db._conn_lock if db is not None else None
        if _lock is not None:
            with _lock:
                df = pd.read_sql_query(sql, self._conn, params=flat_labels)
        else:
            df = pd.read_sql_query(sql, self._conn, params=flat_labels)
        df = _decode_df_embedding(df)

        if df.empty:
            return df

        dist_map: dict[int, float] = {
            int(lbl): float(d) for lbl, d in zip(flat_labels, flat_distances)
        }
        df["_distance"] = df["vec_label"].apply(lambda lbl: dist_map.get(int(lbl), float("nan")))

        df = df.sort_values("_distance").reset_index(drop=True)
        return df

    def _decrypt_query_df(self, df: pd.DataFrame) -> pd.DataFrame:
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


class HippoMergeInsert:

    def __init__(self, table: HippoTable, key_cols: list[str]) -> None:
        self._table = table
        self._key_cols = key_cols
        self._update_all: bool = False

    def when_matched_update_all(self) -> "HippoMergeInsert":
        self._update_all = True
        return self

    def execute(self, data: Any) -> None:
        rows = _normalize_to_row_list(data)
        if not rows:
            return

        rows = self._table._encrypt_rows(rows)
        encoded = [_encode_row_for_insert(r) for r in rows]
        all_cols = list(encoded[0].keys())
        non_key = [c for c in all_cols if c not in self._key_cols]
        key_conflict = ", ".join(self._key_cols)
        conn = self._table._conn

        _db = self._table._db
        _conn_lock = (
            _db._conn_lock if _db is not None else contextlib.nullcontext()
        )

        with _conn_lock:
            try:
                actual_cols_count = len(
                    conn.execute(  # nosemgrep
                        f"SELECT * FROM {self._table._name} LIMIT 0"
                    ).description or []
                )
            except Exception:  # noqa: BLE001
                actual_cols_count = len(all_cols)

            is_partial = len(all_cols) < actual_cols_count

            if is_partial and non_key and self._update_all:
                update_clause = ", ".join(f"{c}=?" for c in non_key)
                where_clause = " AND ".join(f"{k}=?" for k in self._key_cols)
                sql = (  # nosemgrep: sql-injection
                    f"UPDATE {self._table._name} SET {update_clause} WHERE {where_clause}"
                )
                params = [
                    tuple(r.get(c) for c in non_key) + tuple(r.get(k) for k in self._key_cols)
                    for r in encoded
                ]
                with _txn(conn):
                    conn.executemany(sql, params)
                return

            placeholders = ", ".join("?" for _ in all_cols)
            col_names = ", ".join(all_cols)

            if non_key:
                update_clause = ", ".join(f"{c}=excluded.{c}" for c in non_key)
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


def _normalize_to_row_list(data: Any) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if isinstance(data, pa.Table):
        return data.to_pylist()
    return list(data)


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
    return hippo._rebuild_index_from_sqlite()


def direct_recency_rows_from_store(
    store_root: "Path | str",
    limit: "int | None" = None,
) -> list[dict]:
    root = Path(store_root)
    db_path = root / "hippo" / "brain.sqlite3"
    if not db_path.exists():
        return []

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

    return _no_flock_recency_rows_from_store(db_path, limit=limit)


def load_hnsw_readonly(store_root: "str | Path", embed_dim: int) -> "hnswlib.Index | None":
    hnsw_path = Path(store_root) / "hippo" / "records.hnsw"
    if not hnsw_path.exists():
        return None
    try:
        idx = hnswlib.Index(space="cosine", dim=embed_dim)
        idx.load_index(str(hnsw_path), max_elements=0)
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
    root = Path(store_root)

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
        row_dicts = direct_recency_rows_from_store(root, limit=limit)

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

    seen_ids: set[str] = set()
    results: list[dict] = []
    for row in row_dicts:
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        surface = row.get("literal_surface") or ""
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
