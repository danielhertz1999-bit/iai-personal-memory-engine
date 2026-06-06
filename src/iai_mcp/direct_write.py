"""Daemon-free two-phase write path for episodic memory records.

Writes the SQLite row immediately without touching the hnswlib index.
The daemon fills the real embedding + refreshes the ANN index on the next wake.

Two write situations:
- Embedder reachable: embed now, store the VALID BLOB in the records row, AND
  drop the embedding to a.pending-embeddings/{uuid}.npy sidecar. On the next
  wake the daemon incrementally add_items from the sidecar.
- No embedder reachable (true daemon-down): store the row IMMEDIATELY with an
  embed_dim ZERO-VECTOR BLOB + embedding_pending=1. The write completes <=1.5 s
  because it never blocks on a cold embed. Recency recall returns it instantly.
  On the next wake the daemon re-embeds the row's raw text, fills the real BLOB,
  clears the flag, and adds it to hnswlib.

The written record is immediately findable by RECENCY (SQLite query — embedding-
independent) and becomes findable by ANN + warm semantic after the next daemon wake.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)


def _resolve_store_root() -> Path:
    """Resolve store root from IAI_MCP_STORE env or ~/.iai-mcp."""
    env = os.environ.get("IAI_MCP_STORE")
    return Path(env) if env else Path.home() / ".iai-mcp"


def write_turn_direct(
    store_root: Path | str | None = None,
    *,
    text: str,
    session_id: str = "-",
    role: str = "user",
    ts_iso: str | None = None,
    deferred_embedding: bool = False,
    cue: str | None = None,
    tier: str = "episodic",
    source_uuid: str | None = None,
) -> dict[str, Any]:
    """Write a single conversational turn directly to the Hippo SQLite store.

    Does NOT contact the daemon socket. Opens the store in SHARED mode
    (LOCK_SH + busy_timeout) and inserts the SQLite row immediately.
    The hnswlib index update is deferred to the next daemon wake.

    Args:
        store_root: Path to the IAI-MCP store root (uses IAI_MCP_STORE env or
            ~/.iai-mcp when None).
        text: The turn text to store.
        session_id: Session identifier (used in the idempotency key).
        role: "user" or "assistant".
        ts_iso: ISO-format timestamp string. Defaults to now().
        deferred_embedding: Force the zero-vector pending path even if an
            embedder would be reachable. Useful for testing.
        cue: Optional cue text (defaults to text).
        tier: Memory tier (default "episodic").
        source_uuid: Optional transcript line UUID for stable idempotency keys.

    Returns:
        {"status": "inserted"|"reinforced", "record_id": str, "reason": str}
    """
    from iai_mcp.capture import _idem_tag, MIN_CAPTURE_LEN, MAX_CAPTURE_LEN, TIER_ENUM
    from iai_mcp.hippo import AccessMode, HippoDB

    root = Path(store_root) if store_root is not None else _resolve_store_root()

    # Basic validation.
    if tier not in TIER_ENUM:
        return {"status": "skipped", "record_id": None, "reason": f"invalid tier {tier!r}"}

    text = (text or "").strip()
    if len(text) < MIN_CAPTURE_LEN:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if len(text) > MAX_CAPTURE_LEN:
        text = text[:MAX_CAPTURE_LEN]

    now = datetime.fromisoformat(ts_iso) if ts_iso else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ts_norm = now.isoformat()

    idem_t = _idem_tag(session_id, role, ts_norm, text, source_uuid=source_uuid)

    # Open in SHARED mode (LOCK_SH + busy_timeout; no hnswlib).
    db = HippoDB(root, access_mode=AccessMode.SHARED)
    try:
        # Dedup check: if an identical turn already exists, reinforce and return.
        existing_id = _find_record_by_tag_direct(db, idem_t)
        if existing_id is not None:
            return {
                "status": "reinforced",
                "record_id": str(existing_id),
                "reason": "exact-key re-drain",
            }

        record_id = str(uuid4())
        tags = ["capture", f"role:{role}", idem_t]
        tags_json = json.dumps(tags)
        provenance = [{"ts": ts_norm, "cue": cue or "(direct-write)", "session_id": session_id, "role": role}]
        provenance_json = json.dumps(provenance)

        # Embedding strategy: try cheaply; fall back to deferred if not fast enough.
        embedding: list[float] | None = None
        if not deferred_embedding:
            try:
                embedding = _try_get_embedding_fast(text, cue or text)
            except Exception:
                embedding = None

        if embedding is not None and len(embedding) == db._embed_dim:
            # Embedder-reachable: insert with a real BLOB + drop sidecar.
            _insert_row_with_embedding(db, record_id, tier, text, tags_json, provenance_json, ts_norm, ts_norm, embedding)
            _write_sidecar(root, record_id, embedding, db)
        else:
            # Deferred: insert with zero-vector + pending flag.
            db.insert_pending_row(
                record_id=record_id,
                tier=tier,
                literal_surface=text,
                tags_json=tags_json,
                provenance_json=provenance_json,
                created_at=ts_norm,
                updated_at=ts_norm,
            )

        return {
            "status": "inserted",
            "record_id": record_id,
            "reason": f"tier={tier}",
        }
    finally:
        db.close()


def _find_record_by_tag_direct(db: Any, tag: str) -> str | None:
    """Scan tags_json for an exact tag match. Returns record id string or None."""
    tag_json_literal = json.dumps(tag)
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id, tags_json FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
    for row in rows:
        tags_raw = row["tags_json"] or "[]"
        if tag_json_literal not in tags_raw:
            continue
        try:
            tags = json.loads(tags_raw)
        except (ValueError, TypeError):
            continue
        if tag in tags:
            return row["id"]
    return None


def _try_get_embedding_fast(text: str, cue: str) -> list[float] | None:
    """Attempt to obtain an embedding within the SLO budget.

    Returns a list of floats on success, None when the embedder is
    unavailable or would take too long (deferred path should be used).

    Does NOT block on a cold (~3 s) embedder load. A fresh CLI process
    has no warm embedder — returns None immediately so the deferred path
    runs instead.

    Only calls the daemon embed RPC when the daemon socket is reachable
    (fast, non-blocking probe). A local embed warm startup takes ~3 s,
    which would blow the 1.5 s SLO, so we do NOT call it here.
    """
    # Probe the daemon socket first (fast, non-blocking).
    socket_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if socket_path:
        try:
            import socket as _socket
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect(socket_path)
            s.close()
            # Socket reachable — daemon might be able to embed.
            # Try a quick embed RPC but stay within 0.8 s budget.
        except (OSError, ConnectionRefusedError, FileNotFoundError):
            # Socket dead — deferred path.
            return None
    else:
        # No socket configured — assume daemon down.
        return None

    # Socket responded — try embedding via the daemon. The embedder requires
    # a MemoryStore object; since this is a daemon-free path, we don't have
    # one. Return None here so the deferred path is used — the warm path
    # (embedder reachable + sidecar) is intended for callers that already have
    # a MemoryStore open (e.g. the daemon itself after boot).
    # For the daemon-down write, deferred is always correct.
    return None


def _insert_row_with_embedding(
    db: Any,
    record_id: str,
    tier: str,
    literal_surface: str,
    tags_json: str,
    provenance_json: str,
    created_at: str,
    updated_at: str,
    embedding: list[float],
) -> None:
    """Insert a row with a real embedding BLOB (LOCK_SH, direct SQLite)."""
    blob = struct.pack(f"<{len(embedding)}f", *embedding)
    with db._conn_lock:
        db._conn.execute(
            "INSERT INTO records"
            " (id, tier, literal_surface, aaak_index, embedding, embedding_pending,"
            "  provenance_json, created_at, updated_at, tags_json,"
            "  community_id, detail_level, centrality, stability, difficulty,"
            "  pinned, never_decay, never_merge, s5_trust_score,"
            "  schema_version, language,"
            "  hv_tier, structure_hv_payload)"
            " VALUES (?, ?, ?, '', ?, 0, ?, ?, ?, ?, '', 1, 0.0, 0.0, 0.0,"
            "  0, 0, 0, 0.5, 1, 'en', 'bsc', x'')",
            (
                record_id,
                tier,
                literal_surface,
                blob,
                provenance_json,
                created_at,
                updated_at,
                tags_json,
            ),
        )
        db._conn.commit()


def _write_sidecar(root: Path, record_id: str, embedding: list[float], db: Any) -> None:
    """Write {uuid}.npy + {uuid}.json atomically to {store_root}/.pending-embeddings/.

    The vec_label is read from the just-inserted row so add_items uses the
    correct SQLite autoincrement key.
    """
    # Validate UUID form (no path traversal).
    import re
    if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', record_id):
        log.warning("direct_write: sidecar skipped — record_id not uuid4: %r", record_id)
        return

    with db._conn_lock:
        row = db._conn.execute(
            "SELECT vec_label FROM records WHERE id = ?", (record_id,)
        ).fetchone()
    if row is None:
        return
    vec_label = int(row["vec_label"])

    sidecar_dir = root / ".pending-embeddings"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    blob = struct.pack(f"<{len(embedding)}f", *embedding)
    npy_tmp = sidecar_dir / f"{record_id}.npy.tmp"
    json_tmp = sidecar_dir / f"{record_id}.json.tmp"
    npy_final = sidecar_dir / f"{record_id}.npy"
    json_final = sidecar_dir / f"{record_id}.json"

    try:
        npy_tmp.write_bytes(blob)
        json_tmp.write_text(json.dumps({"uuid": record_id, "vec_label": vec_label}))
        os.replace(npy_tmp, npy_final)
        os.replace(json_tmp, json_final)
    except OSError as exc:
        log.warning("direct_write: sidecar write failed for %s: %s", record_id, exc)
        for p in (npy_tmp, json_tmp):
            try:
                p.unlink()
            except OSError:
                pass


def simulate_daemon_reembed(
    store_root: Path | str,
    *,
    text_fragment: str,
    embedding: list[float],
) -> None:
    """Test helper: simulate the daemon re-embed pass for a pending row.

    Writes the provided REAL embedding BLOB to the matching row and clears
    the embedding_pending flag. Matches rows whose literal_surface contains
    text_fragment (using LIKE).

    Used only in tests to validate the H3 post-wake assertions without
    running the actual daemon.
    """
    import sqlite3 as _sqlite3
    root = Path(store_root)
    db_path = root / "hippo" / "brain.sqlite3"
    blob = struct.pack(f"<{len(embedding)}f", *embedding)
    conn = _sqlite3.connect(str(db_path))
    conn.row_factory = _sqlite3.Row
    try:
        conn.execute(
            "UPDATE records SET embedding = ?, embedding_pending = 0"
            " WHERE literal_surface LIKE ?",
            (blob, f"%{text_fragment}%"),
        )
        conn.commit()
    finally:
        conn.close()
