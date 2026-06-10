from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    store_for_test: dict[tuple[str, str], str] = {}

    def fake_get(service: str, username: str):
        return store_for_test.get((service, username))

    def fake_set(service: str, username: str, password: str) -> None:
        store_for_test[(service, username)] = password

    def fake_delete(service: str, username: str) -> None:
        store_for_test.pop((service, username), None)

    monkeypatch.setattr(_keyring, "get_password", fake_get)
    monkeypatch.setattr(_keyring, "set_password", fake_set)
    monkeypatch.setattr(_keyring, "delete_password", fake_delete)
    yield store_for_test


def _make(text: str = "hello", language: str = "en", detail: int = 2):
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[{"ts": "2026-04-17T12:00:00Z", "cue": "original cue", "session_id": "s1"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["topic:test"],
        language=language,
        profile_modulation_gain={"learnedKnob": 0.42},
    )


def test_insert_writes_encrypted_literal_surface_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make(text="top-secret Russian phrase: Привет")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["literal_surface"].startswith("iai:enc:v1:")


def test_insert_writes_encrypted_provenance_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["provenance_json"].startswith("iai:enc:v1:")


def test_insert_writes_encrypted_profile_modulation_gain_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["profile_modulation_gain_json"].startswith("iai:enc:v1:")


def test_embedding_remains_plaintext_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    emb = list(row["embedding"])
    assert len(emb) == store.embed_dim
    assert emb[0] == pytest.approx(0.1)


def test_language_remains_plaintext_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make(language="ru", text="Привет")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["language"] == "ru"


def test_tags_remain_plaintext_on_disk(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    tags = json.loads(row["tags_json"])
    assert tags == ["topic:test"]


def test_get_decrypts_literal_surface(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    text = "Alice said: пусть каждое слово сохранится точно"
    rec = _make(text=text)
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == text


def test_get_decrypts_provenance(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.provenance == rec.provenance


def test_get_decrypts_profile_modulation_gain(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.profile_modulation_gain == rec.profile_modulation_gain


def test_all_records_decrypts_all_rows(tmp_path):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    r1 = _make(text="first")
    r2 = _make(text="второй")
    store.insert(r1)
    store.insert(r2)

    all_r = store.all_records()
    texts = {r.literal_surface for r in all_r}
    assert "first" in texts
    assert "второй" in texts


def test_query_similar_still_works_after_encryption(tmp_path):
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM
    store = MemoryStore(path=tmp_path)
    rec = _make(text="probe me")
    store.insert(rec)
    hits = store.query_similar([0.1] * EMBED_DIM, k=5)
    assert len(hits) >= 1
    assert hits[0][0].literal_surface == "probe me"


def test_encrypted_row_cannot_be_decrypted_with_wrong_key(tmp_path, monkeypatch):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path)
    rec = _make(text="sensitive")
    store.insert(rec)

    store._crypto_key = b"\xff" * 32  # type: ignore[attr-defined]
    with pytest.raises(Exception):
        store.get(rec.id)


def test_ad_binding_prevents_row_swap(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.store import _uuid_literal

    store = MemoryStore(path=tmp_path)
    r_a = _make(text="row A secret")
    r_b = _make(text="row B secret")
    store.insert(r_a)
    store.insert(r_b)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    ct_a = df[df["id"] == str(r_a.id)].iloc[0]["literal_surface"]

    tbl.update(
        where=f"id = '{_uuid_literal(r_b.id)}'",
        values={"literal_surface": ct_a},
    )

    with pytest.raises(Exception):
        store.get(r_b.id)


def test_get_passes_through_plaintext_rows(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.store import _uuid_literal

    store = MemoryStore(path=tmp_path)
    rec = _make(text="plaintext-legacy")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where=f"id = '{_uuid_literal(rec.id)}'",
        values={
            "literal_surface": "plaintext-legacy",
            "provenance_json": json.dumps(rec.provenance),
            "profile_modulation_gain_json": json.dumps(rec.profile_modulation_gain),
        },
    )

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == "plaintext-legacy"
    assert got.provenance == rec.provenance
    assert got.profile_modulation_gain == rec.profile_modulation_gain


def test_append_provenance_batch_still_writes_encrypted(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)

    new_entry = {"ts": "2026-04-17T13:00:00Z", "cue": "batch cue", "session_id": "s2"}
    store.append_provenance_batch([(rec.id, new_entry)])

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["provenance_json"].startswith("iai:enc:v1:")

    got = store.get(rec.id)
    assert got is not None
    cues = [p["cue"] for p in got.provenance]
    assert "batch cue" in cues


def test_append_provenance_single_still_writes_encrypted(tmp_path):
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    store = MemoryStore(path=tmp_path)
    rec = _make()
    store.insert(rec)
    store.append_provenance(rec.id, {"ts": "x", "cue": "y", "session_id": "z"})

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["provenance_json"].startswith("iai:enc:v1:")


def test_reopen_store_with_same_keyring_decrypts(tmp_path):
    from iai_mcp.store import MemoryStore
    s1 = MemoryStore(path=tmp_path)
    rec = _make(text="persistent secret")
    s1.insert(rec)
    del s1

    s2 = MemoryStore(path=tmp_path)
    got = s2.get(rec.id)
    assert got is not None
    assert got.literal_surface == "persistent secret"
