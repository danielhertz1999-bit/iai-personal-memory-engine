"""Parent-side read-only export helpers used by the periodic graph refresh.

Open a dedicated read-only sqlite3 connection (`file:...?mode=ro` URI plus
`PRAGMA query_only=ON`), wrap the entire export in a single BEGIN DEFERRED
transaction for snapshot consistency, and walk records and edges via
narrow-projection keyset pagination.

The edges SELECT JOINs against the records table on both src and dst with
the active-records predicate `tombstoned_at IS NULL AND
COALESCE(embedding_pending, 0) = 0` (identical to the active_records_count
predicate used elsewhere in the store), so any edge with a filtered
endpoint is dropped at the streaming SQL layer and no dangling endpoint
reaches the worker. The shared write lock on the main HippoDB connection
is never acquired on this path.
"""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Iterator


DEFAULT_RECORDS_CHUNK_SIZE: int = 2000
DEFAULT_EDGES_CHUNK_SIZE: int = 5000


def open_ro_connection(db_path: Path) -> sqlite3.Connection:
    """Open a dedicated read-only sqlite3 connection.

    Uses the `file:...?mode=ro` URI form so SQLite refuses any DML at the
    file layer, plus `PRAGMA query_only=ON` for defense-in-depth and
    `PRAGMA busy_timeout=2000` so a brief writer-side contention does not
    abort the export. The caller is responsible for closing the connection
    in a `finally`.
    """
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


@contextlib.contextmanager
def read_transaction(conn: sqlite3.Connection):
    """Wrap the export pass in one BEGIN DEFERRED so every paginated SELECT
    sees a single consistent WAL snapshot."""
    conn.execute("BEGIN DEFERRED")
    try:
        yield
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        try:
            conn.execute("COMMIT")
        except sqlite3.Error:
            pass


def iter_records_chunks(
    conn: sqlite3.Connection,
    chunk_size: int = DEFAULT_RECORDS_CHUNK_SIZE,
) -> Iterator[list[tuple[str, bytes]]]:
    """Yield lists of `(id_str, embedding_blob_bytes)` of length <= chunk_size.

    Keyset cursor on the integer `vec_label` primary key. The predicate
    `tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0` matches
    the active-records semantics applied elsewhere in the store, so the
    streamed records are exactly the records the system considers
    recall-visible.
    """
    cursor = -1
    sql = (
        "SELECT vec_label, id, embedding FROM records"
        " WHERE tombstoned_at IS NULL"
        " AND COALESCE(embedding_pending, 0) = 0"
        " AND vec_label > ?"
        " ORDER BY vec_label"
        " LIMIT ?"
    )
    while True:
        rows = conn.execute(sql, (cursor, chunk_size)).fetchall()
        if not rows:
            return
        chunk: list[tuple[str, bytes]] = []
        for row in rows:
            chunk.append((str(row["id"]), bytes(row["embedding"])))
            cursor = int(row["vec_label"])
        yield chunk
        if len(rows) < chunk_size:
            return


def iter_edges_chunks(
    conn: sqlite3.Connection,
    chunk_size: int = DEFAULT_EDGES_CHUNK_SIZE,
) -> Iterator[list[tuple[str, str, float]]]:
    """Yield lists of `(src_str, dst_str, weight_float)` of length <= chunk_size.

    The JOIN against records on both endpoints, with the same active-records
    predicate as `iter_records_chunks`, drops any edge whose src or dst is
    filtered. The `edge_type` is consumed as part of the compound keyset
    cursor but not yielded — the worker hardcodes `"hebbian"` to match the
    in-process build's edge type.
    """
    src = ""
    dst = ""
    edge_type = ""
    sql = (
        "SELECT e.src, e.dst, e.edge_type, e.weight"
        " FROM edges e"
        " JOIN records rs ON rs.id = e.src"
        "                AND rs.tombstoned_at IS NULL"
        "                AND COALESCE(rs.embedding_pending, 0) = 0"
        " JOIN records rd ON rd.id = e.dst"
        "                AND rd.tombstoned_at IS NULL"
        "                AND COALESCE(rd.embedding_pending, 0) = 0"
        " WHERE (e.src, e.dst, e.edge_type) > (?, ?, ?)"
        " ORDER BY e.src, e.dst, e.edge_type"
        " LIMIT ?"
    )
    while True:
        rows = conn.execute(sql, (src, dst, edge_type, chunk_size)).fetchall()
        if not rows:
            return
        chunk: list[tuple[str, str, float]] = []
        for row in rows:
            chunk.append((
                str(row["src"]),
                str(row["dst"]),
                float(row["weight"]),
            ))
            src = str(row["src"])
            dst = str(row["dst"])
            edge_type = str(row["edge_type"])
        yield chunk
        if len(rows) < chunk_size:
            return
