"""Regression: reembed_pending_rows must embed the PLAINTEXT literal_surface,
not the iai:enc:v1: ciphertext, on an encrypted store.

Pre-existing bug (found 2026-06-21): HippoDB.reembed_pending_rows read
literal_surface straight from SQLite and fed it to embedder.embed() without
decrypting. On an encrypted deployment that meant every embedding_pending=1 row
re-embedded by this path got an embedding of the ciphertext = garbage vector.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM


def _vec_from_text(text: str) -> list[float]:
    """Deterministic, text-sensitive unit vector. embed(plaintext) and
    embed(ciphertext) therefore differ, which is what makes the test meaningful."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return _vec_from_text(text)


def _brain_path(tmp_path: Path) -> Path:
    return tmp_path / "hippo" / "brain.sqlite3"


def _insert_pending(db: HippoDB, rid: str, surface: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    db.insert_pending_row(
        record_id=rid,
        tier="episodic",
        literal_surface=surface,
        tags_json="[]",
        provenance_json="{}",
        created_at=now,
        updated_at=now,
    )


def _read_row(db_path: Path, rid: str) -> tuple[list[float], int]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT embedding, embedding_pending FROM records WHERE id = ?", (rid,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    blob, pending = row[0], row[1]
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob)), int(pending)


def _set_literal_surface(db_path: Path, rid: str, value: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE records SET literal_surface = ? WHERE id = ?", (value, rid))
        conn.commit()
    finally:
        conn.close()


def test_reembed_pending_decrypts_before_embedding(tmp_path: Path) -> None:
    key = os.urandom(32)
    db = HippoDB(tmp_path, crypto_key_provider=lambda: key)
    try:
        plaintext = "a secret pending memory worth embedding"
        rid = str(uuid4())
        _insert_pending(db, rid, plaintext)

        # Simulate encrypted-at-rest: replace the stored surface with ciphertext
        # carrying the canonical AAD (uuid.lower()).
        ciphertext = db._encrypt_for_uuid(rid, plaintext)
        assert ciphertext.startswith("iai:enc:v1:")
        _set_literal_surface(_brain_path(tmp_path), rid, ciphertext)

        emb = _RecordingEmbedder()
        n = db.reembed_pending_rows(emb)

        assert n == 1
        # The embedder was handed PLAINTEXT, never the ciphertext.
        assert emb.seen == [plaintext]
        assert not emb.seen[0].startswith("iai:enc:v1:")

        stored_vec, pending = _read_row(_brain_path(tmp_path), rid)
        assert pending == 0
        # Stored vector matches embed(plaintext)...
        assert stored_vec == pytest.approx(_vec_from_text(plaintext), abs=1e-6)
        # ...and is NOT embed(ciphertext) (the old, buggy result).
        assert stored_vec != pytest.approx(_vec_from_text(ciphertext), abs=1e-6)
    finally:
        db.close()


def test_reembed_pending_plaintext_store_still_works(tmp_path: Path) -> None:
    """No crypto provider: decrypt is a no-op and the path behaves as before."""
    db = HippoDB(tmp_path)
    try:
        plaintext = "a plain pending memory"
        rid = str(uuid4())
        _insert_pending(db, rid, plaintext)

        emb = _RecordingEmbedder()
        n = db.reembed_pending_rows(emb)

        assert n == 1
        assert emb.seen == [plaintext]
        stored_vec, pending = _read_row(_brain_path(tmp_path), rid)
        assert pending == 0
        assert stored_vec == pytest.approx(_vec_from_text(plaintext), abs=1e-6)
    finally:
        db.close()


def test_reembed_pending_undecryptable_row_stays_pending(tmp_path: Path) -> None:
    """A row whose ciphertext can't be decrypted (wrong AAD) must not be embedded
    with garbage and must remain embedding_pending=1 for a later retry."""
    key = os.urandom(32)
    db = HippoDB(tmp_path, crypto_key_provider=lambda: key)
    try:
        plaintext = "undecryptable pending memory"
        rid = str(uuid4())
        _insert_pending(db, rid, plaintext)

        # Encrypt under a DIFFERENT record id's AAD so decrypt fails for `rid`.
        wrong_ciphertext = db._encrypt_for_uuid(str(uuid4()), plaintext)
        _set_literal_surface(_brain_path(tmp_path), rid, wrong_ciphertext)

        emb = _RecordingEmbedder()
        n = db.reembed_pending_rows(emb)

        assert n == 0
        assert emb.seen == []  # never reached embed()
        _, pending = _read_row(_brain_path(tmp_path), rid)
        assert pending == 1  # left for retry, not poisoned
    finally:
        db.close()
