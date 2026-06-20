from __future__ import annotations

import json
import logging
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)


def _resolve_store_root() -> Path:
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
    from iai_mcp.capture import _idem_tag, MIN_CAPTURE_LEN, MAX_CAPTURE_LEN, TIER_ENUM
    from iai_mcp.hippo import AccessMode, HippoDB

    root = Path(store_root) if store_root is not None else _resolve_store_root()

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

    db = HippoDB(root, access_mode=AccessMode.SHARED)
    try:
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

        embedding: list[float] | None = None
        if not deferred_embedding:
            try:
                embedding = _try_get_embedding_fast(text, cue or text)
            except Exception:
                embedding = None

        if embedding is not None and len(embedding) == db._embed_dim:
            _insert_row_with_embedding(db, record_id, tier, text, tags_json, provenance_json, ts_norm, ts_norm, embedding)
            _write_sidecar(root, record_id, embedding, db)
        else:
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
    from iai_mcp._ipc import IS_WINDOWS, make_sync_ipc_socket
    # On POSIX only proceed when IAI_DAEMON_SOCKET_PATH is explicitly set
    if not IS_WINDOWS and not os.environ.get("IAI_DAEMON_SOCKET_PATH"):
        return None
    try:
        s, addr = make_sync_ipc_socket()
        s.settimeout(0.1)
        s.connect(addr)
        s.close()
    except (OSError, ConnectionRefusedError, FileNotFoundError):
        return None
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
        json_tmp.write_text(json.dumps({"uuid": record_id, "vec_label": vec_label}), encoding="utf-8")
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
