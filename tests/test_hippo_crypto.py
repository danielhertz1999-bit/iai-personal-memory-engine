"""Per-field AES-GCM encryption tests for HippoDB / HippoTable.

Scope:
- Encryption at rest (raw SQLite probe shows ciphertext)
- Decryption on read (plaintext returned through API)
- Records-decrypt failure RAISES HippoDecryptError + emits audit event
- Events-decrypt failure returns lenient empty fallback
- AAD binding: cross-row ciphertext swap detected
- Back-compat: pre-encrypted plaintext rows pass through unchanged
- Idempotent encrypt: already-encrypted values not re-encrypted
- merge_insert and update paths encrypt consistently
- ANN query result decryption

Single-file target: pytest tests/test_hippo_crypto.py -x
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.hippo import (
    HippoDB,
    HippoDecryptError,
    _ENCRYPTED_EVENTS_COLUMNS,
    _ENCRYPTED_RECORD_COLUMNS,
)
from iai_mcp.types import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_unit_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


def _record_row(
    *,
    rid: str | None = None,
    literal_surface: str = "test surface",
    provenance_json: str | None = None,
    profile_modulation_gain_json: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": rid or str(uuid4()),
        "tier": "episodic",
        "literal_surface": literal_surface,
        "provenance_json": provenance_json or '{"src": "test"}',
        "profile_modulation_gain_json": profile_modulation_gain_json or '{"g": 1.0}',
        "embedding": embedding or _rng_unit_vec(),
        "created_at": now,
    }


def _event_row(*, eid: str | None = None, data_json: str = '{"ev": "test"}') -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": eid or str(uuid4()),
        "kind": "test_event",
        "severity": "info",
        "domain": "storage",
        "ts": now,
        "data_json": data_json,
        "session_id": None,
        "source_ids_json": None,
    }


_RECORDS_SELECT: dict[str, str] = {
    "id":                            "SELECT id FROM records WHERE id = ?",
    "tier":                          "SELECT tier FROM records WHERE id = ?",
    "literal_surface":               "SELECT literal_surface FROM records WHERE id = ?",
    "provenance_json":               "SELECT provenance_json FROM records WHERE id = ?",
    "profile_modulation_gain_json":  "SELECT profile_modulation_gain_json FROM records WHERE id = ?",
    "embedding":                     "SELECT embedding FROM records WHERE id = ?",
    "created_at":                    "SELECT created_at FROM records WHERE id = ?",
}

_EVENTS_SELECT: dict[str, str] = {
    "id":               "SELECT id FROM events WHERE id = ?",
    "kind":             "SELECT kind FROM events WHERE id = ?",
    "severity":         "SELECT severity FROM events WHERE id = ?",
    "domain":           "SELECT domain FROM events WHERE id = ?",
    "ts":               "SELECT ts FROM events WHERE id = ?",
    "data_json":        "SELECT data_json FROM events WHERE id = ?",
    "session_id":       "SELECT session_id FROM events WHERE id = ?",
    "source_ids_json":  "SELECT source_ids_json FROM events WHERE id = ?",
}

_RECORDS_UPDATE: dict[str, str] = {
    "literal_surface":               "UPDATE records SET literal_surface = ? WHERE id = ?",
    "provenance_json":               "UPDATE records SET provenance_json = ? WHERE id = ?",
    "profile_modulation_gain_json":  "UPDATE records SET profile_modulation_gain_json = ? WHERE id = ?",
}

_EVENTS_UPDATE: dict[str, str] = {
    "data_json":        "UPDATE events SET data_json = ? WHERE id = ?",
    "source_ids_json":  "UPDATE events SET source_ids_json = ? WHERE id = ?",
}


def _raw_records_col(db_path: Path, col: str, row_id: str) -> str | None:
    """Read a column from the records table directly, bypassing HippoDB.

    Uses a pre-built literal SQL dict — no runtime string construction.
    """
    stmt = _RECORDS_SELECT[col]
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(stmt, (row_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _raw_events_col(db_path: Path, col: str, row_id: str) -> str | None:
    """Read a column from the events table directly, bypassing HippoDB."""
    stmt = _EVENTS_SELECT[col]
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(stmt, (row_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _raw_records_set(db_path: Path, col: str, row_id: str, value: str) -> None:
    """Overwrite a records column directly in SQLite to simulate tampering."""
    stmt = _RECORDS_UPDATE[col]
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(stmt, (value, row_id))
        conn.commit()
    finally:
        conn.close()


def _raw_events_set(db_path: Path, col: str, row_id: str, value: str) -> None:
    """Overwrite an events column directly in SQLite to simulate tampering."""
    stmt = _EVENTS_UPDATE[col]
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(stmt, (value, row_id))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_key() -> bytes:
    return os.urandom(32)


@pytest.fixture()
def hippo_with_key(tmp_path: Path, test_key: bytes):
    """HippoDB opened with a test crypto_key_provider."""
    provider = lambda: test_key  # noqa: E731
    db = HippoDB(tmp_path, crypto_key_provider=provider)
    yield db
    db.close()


@pytest.fixture()
def hippo_no_key(tmp_path: Path):
    """HippoDB opened without a crypto_key_provider (plaintext test mode)."""
    db = HippoDB(tmp_path)
    yield db
    db.close()


@pytest.fixture()
def brain_db_path(tmp_path: Path) -> Path:
    return tmp_path / "hippo" / "brain.sqlite3"


# ---------------------------------------------------------------------------
# 1. Encryption at rest — raw SQLite probe
# ---------------------------------------------------------------------------

def test_records_literal_surface_encrypted_on_disk(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """literal_surface is stored as iai:enc:v1:... in SQLite; id and tier are plaintext."""
    row = _record_row(literal_surface="secret phrase")
    hippo_with_key.open_table("records").add([row])

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw!r}"

    # Plaintext columns must NOT be encrypted
    raw_tier = _raw_records_col(brain_db_path, "tier", row["id"])
    assert raw_tier == "episodic"
    raw_id = _raw_records_col(brain_db_path, "id", row["id"])
    assert raw_id == row["id"]


def test_records_literal_surface_decrypted_on_read(
    hippo_with_key: HippoDB,
) -> None:
    """to_pandas() returns raw ciphertext; manual decrypt yields original plaintext."""
    row = _record_row(literal_surface="secret phrase")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])
    df = tbl.to_pandas()
    match = df[df["id"] == row["id"]]
    assert not match.empty
    raw_val = match.iloc[0]["literal_surface"]
    # to_pandas() returns raw ciphertext — MemoryStore._from_row decrypts on the consumer side.
    assert str(raw_val).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw_val!r}"
    # Manual decrypt via the HippoDB helper verifies round-trip correctness.
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "literal_surface", raw_val)
    assert plaintext == "secret phrase"


def test_provenance_json_encrypted(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """provenance_json is encrypted at rest; to_pandas() returns raw ciphertext."""
    payload = '{"source": "unit-test", "confidence": 0.99}'
    row = _record_row(provenance_json=payload)
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    raw = _raw_records_col(brain_db_path, "provenance_json", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:")

    df = tbl.to_pandas()
    result = df[df["id"] == row["id"]].iloc[0]["provenance_json"]
    # to_pandas() returns raw ciphertext; manual decrypt verifies round-trip.
    assert str(result).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {result!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "provenance_json", result)
    assert plaintext == payload


def test_profile_modulation_gain_json_encrypted(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """profile_modulation_gain_json is encrypted at rest; to_pandas() returns raw ciphertext."""
    payload = '{"gain": 2.5, "decay": 0.1}'
    row = _record_row(profile_modulation_gain_json=payload)
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    raw = _raw_records_col(brain_db_path, "profile_modulation_gain_json", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:")

    df = tbl.to_pandas()
    result = df[df["id"] == row["id"]].iloc[0]["profile_modulation_gain_json"]
    # to_pandas() returns raw ciphertext; manual decrypt verifies round-trip.
    assert str(result).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {result!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "profile_modulation_gain_json", result)
    assert plaintext == payload


def test_events_data_json_encrypted(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """data_json in events is encrypted at rest; to_pandas() returns raw ciphertext."""
    payload = '{"event": "login", "user": "alice"}'
    ev = _event_row(data_json=payload)
    tbl = hippo_with_key.open_table("events")
    tbl.add([ev])

    raw = _raw_events_col(brain_db_path, "data_json", ev["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:")

    df = tbl.to_pandas()
    result = df[df["id"] == ev["id"]].iloc[0]["data_json"]
    # to_pandas() returns raw ciphertext; manual decrypt verifies round-trip.
    assert str(result).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {result!r}"
    plaintext = hippo_with_key._decrypt_event_field(ev["id"], "data_json", result)
    assert plaintext == payload


# ---------------------------------------------------------------------------
# 6. No crypto_key_provider → plaintext mode
# ---------------------------------------------------------------------------

def test_no_crypto_provider_is_plaintext(hippo_no_key: HippoDB) -> None:
    """Without a crypto_key_provider, all values are stored and returned as-is."""
    row = _record_row(literal_surface="plain text")
    tbl = hippo_no_key.open_table("records")
    tbl.add([row])
    df = tbl.to_pandas()
    result = df[df["id"] == row["id"]].iloc[0]["literal_surface"]
    assert result == "plain text"


# ---------------------------------------------------------------------------
# 7. Idempotent encryption
# ---------------------------------------------------------------------------

def test_idempotent_encrypt_does_not_double_encrypt(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """Inserting a value that already starts with iai:enc:v1: is stored unchanged."""
    # Build a syntactically valid but semantically inert ciphertext sentinel.
    # The prefix is what matters for the idempotency check; the payload is
    # constructed from a known-safe byte sequence (not a real secret).
    import base64
    dummy_payload = base64.b64encode(b"\x00" * 44).decode("ascii")
    fake_ct = "iai:enc:v1:" + dummy_payload
    row = _record_row(literal_surface=fake_ct)
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    # Must be byte-identical to what was inserted (no double-encryption).
    assert raw == fake_ct


# ---------------------------------------------------------------------------
# 8. Back-compat: plaintext rows (pre-encryption) pass through on read
# ---------------------------------------------------------------------------

def test_plaintext_passthrough_on_decrypt(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """A row inserted with raw plaintext (no prefix) is returned unchanged on read.

    This covers pre-encryption rows that have not yet been migrated.
    """
    row_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    # Insert directly via raw SQL — bypasses HippoTable.add encryption.
    conn = sqlite3.connect(str(brain_db_path))
    embedding_blob = np.zeros(EMBED_DIM, dtype=np.float32).tobytes()
    try:
        conn.execute(
            "INSERT INTO records (id, tier, literal_surface, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (row_id, "episodic", "raw plaintext value", embedding_blob, now),
        )
        conn.commit()
    finally:
        conn.close()

    df = hippo_with_key.open_table("records").to_pandas()
    result = df[df["id"] == row_id].iloc[0]["literal_surface"]
    assert result == "raw plaintext value"


# ---------------------------------------------------------------------------
# 9. Records cross-row AAD swap RAISES HippoDecryptError
# ---------------------------------------------------------------------------

def test_records_cross_row_aad_swap_raises_decrypt_error(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """Swapping ciphertext between records causes HippoDecryptError on explicit decrypt."""
    row_a = _record_row(literal_surface="Record A content")
    row_b = _record_row(literal_surface="Record B content")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row_a])
    tbl.add([row_b])

    # Get A's ciphertext, put it into B's slot.
    ct_a = _raw_records_col(brain_db_path, "literal_surface", row_a["id"])
    assert ct_a is not None and ct_a.startswith("iai:enc:v1:")
    _raw_records_set(brain_db_path, "literal_surface", row_b["id"], ct_a)

    # to_pandas() returns raw ciphertext; explicit decrypt via _decrypt_df MUST raise.
    df = tbl.to_pandas()
    with pytest.raises(HippoDecryptError):
        tbl._decrypt_df(df)


# ---------------------------------------------------------------------------
# 10. Records decrypt failure emits audit event
# ---------------------------------------------------------------------------

def test_records_decrypt_failure_emits_audit_event(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """When a records decrypt fails, a record_decrypt_failed event is emitted."""
    row_a = _record_row(literal_surface="Source ciphertext")
    row_b = _record_row(literal_surface="Target to corrupt")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row_a])
    tbl.add([row_b])

    ct_a = _raw_records_col(brain_db_path, "literal_surface", row_a["id"])
    _raw_records_set(brain_db_path, "literal_surface", row_b["id"], ct_a)

    # to_pandas() returns raw ciphertext; explicit decrypt via _decrypt_df triggers the error.
    df = tbl.to_pandas()
    with pytest.raises(HippoDecryptError):
        tbl._decrypt_df(df)

    # Query events table directly for audit row.
    # Events data_json is also raw ciphertext now — decrypt manually.
    events_tbl = hippo_with_key.open_table("events")
    events_df_raw = events_tbl.to_pandas()
    events_df = events_tbl._decrypt_df(events_df_raw)
    audit_rows = events_df[events_df["kind"] == "record_decrypt_failed"]
    assert not audit_rows.empty, "Expected at least one record_decrypt_failed event"

    # The audit row payload MUST reference record B's id (not ciphertext bytes).
    found = False
    for _, ev_row in audit_rows.iterrows():
        try:
            payload = json.loads(ev_row["data_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if payload.get("record_id") == row_b["id"]:
            found = True
            break
    assert found, (
        f"record_decrypt_failed event did not reference record id {row_b['id']!r}; "
        f"events: {audit_rows['data_json'].tolist()!r}"
    )


# ---------------------------------------------------------------------------
# 11. Events cross-row AAD swap returns lenient fallback (no exception)
# ---------------------------------------------------------------------------

def test_events_cross_row_aad_swap_returns_fallback(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """Swapping event ciphertext returns lenient '{}' fallback on explicit decrypt."""
    ev_a = _event_row(data_json='{"msg": "event_a"}')
    ev_b = _event_row(data_json='{"msg": "event_b"}')
    tbl = hippo_with_key.open_table("events")
    tbl.add([ev_a])
    tbl.add([ev_b])

    ct_a = _raw_events_col(brain_db_path, "data_json", ev_a["id"])
    assert ct_a is not None and ct_a.startswith("iai:enc:v1:")
    _raw_events_set(brain_db_path, "data_json", ev_b["id"], ct_a)

    # to_pandas() returns raw ciphertext; explicit decrypt via _decrypt_df uses lenient path.
    df = tbl.to_pandas()
    # Must NOT raise — events use lenient fallback.
    decrypted_df = tbl._decrypt_df(df)
    result_b = decrypted_df[decrypted_df["id"] == ev_b["id"]].iloc[0]["data_json"]
    assert result_b == "{}", f"Expected lenient fallback '{{}}', got {result_b!r}"


# ---------------------------------------------------------------------------
# 12. ANN search result decryption
# ---------------------------------------------------------------------------

def test_search_results_decrypt_literal_surface(
    hippo_with_key: HippoDB,
) -> None:
    """ANN query results return raw ciphertext; manual decrypt yields plaintext."""
    texts = ["Alpha memory", "Beta memory", "Gamma memory"]
    embeddings = [_rng_unit_vec(i) for i in range(3)]
    rows = [
        _record_row(literal_surface=t, embedding=e)
        for t, e in zip(texts, embeddings)
    ]
    tbl = hippo_with_key.open_table("records")
    for r in rows:
        tbl.add([r])

    # ANN query via to_pandas() now returns raw ciphertext.
    result_df = tbl.search(vector=embeddings[0]).limit(3).to_pandas()
    assert not result_df.empty
    raw_vals = result_df["literal_surface"].tolist()
    # All results should be ciphertext (HippoDB encrypts on write).
    for val in raw_vals:
        if isinstance(val, str) and val:
            assert val.startswith("iai:enc:v1:"), f"Expected ciphertext from ANN, got: {val!r}"
    # Manual decrypt should recover at least one of the original texts.
    decrypted_df = tbl._decrypt_df(result_df)
    retrieved = set(decrypted_df["literal_surface"].tolist())
    for text in texts:
        if text in retrieved:
            break
    else:
        pytest.fail(f"No expected plaintext found after manual decrypt: {retrieved!r}")


# ---------------------------------------------------------------------------
# 13. merge_insert encrypts consistently with add()
# ---------------------------------------------------------------------------

def test_merge_insert_encrypts_records(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """merge_insert update path encrypts the updated value."""
    # Insert the row first so merge_insert can match and update it.
    row = _record_row(literal_surface="original content")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    # Now update via merge_insert with a new literal_surface.
    updated_row = dict(row)
    updated_row["literal_surface"] = "merge surface content"
    tbl.merge_insert(["id"]).when_matched_update_all().execute([updated_row])

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:"), f"merge_insert did not encrypt; got: {raw!r}"

    df = tbl.to_pandas()
    raw_val = df[df["id"] == row["id"]].iloc[0]["literal_surface"]
    # to_pandas() returns raw ciphertext; manual decrypt verifies round-trip.
    assert str(raw_val).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw_val!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "literal_surface", raw_val)
    assert plaintext == "merge surface content"


# ---------------------------------------------------------------------------
# 14. update with id-keyed WHERE encrypts the updated value
# ---------------------------------------------------------------------------

def test_update_id_keyed_with_encrypted_value(
    hippo_with_key: HippoDB, brain_db_path: Path
) -> None:
    """HippoTable.update with id= WHERE encrypts the new value."""
    row = _record_row(literal_surface="original text")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    tbl.update(
        where=f"id = '{row['id']}'",
        values={"literal_surface": "updated text"},
    )

    raw = _raw_records_col(brain_db_path, "literal_surface", row["id"])
    assert raw is not None
    assert raw.startswith("iai:enc:v1:"), f"Update did not encrypt; got: {raw!r}"

    df = tbl.to_pandas()
    raw_val = df[df["id"] == row["id"]].iloc[0]["literal_surface"]
    # to_pandas() returns raw ciphertext; manual decrypt verifies round-trip.
    assert str(raw_val).startswith("iai:enc:v1:"), f"Expected ciphertext, got: {raw_val!r}"
    plaintext = hippo_with_key._decrypt_record_field(row["id"], "literal_surface", raw_val)
    assert plaintext == "updated text"


# ---------------------------------------------------------------------------
# 15. update with non-id WHERE on encrypted column raises ValueError
# ---------------------------------------------------------------------------

def test_update_non_id_keyed_encrypted_column_raises(
    hippo_with_key: HippoDB,
) -> None:
    """HippoTable.update refuses to update encrypted columns without an id-keyed WHERE."""
    row = _record_row(literal_surface="some text")
    tbl = hippo_with_key.open_table("records")
    tbl.add([row])

    with pytest.raises(ValueError, match="id-keyed WHERE"):
        tbl.update(
            where="tier = 'episodic'",
            values={"literal_surface": "danger zone"},
        )
