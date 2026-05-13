"""iai-mcp crypto + iai-mcp migrate --from=2 --to=3 CLI tests.

Originally ; updated in W1 to retire the keyring
backend in favor of a file-backed primary backend at
`{IAI_MCP_STORE}/.crypto.key` (32 raw bytes, mode 0o600). The
`_isolated_keyring` autouse fixture is gone — CLI tests now monkeypatch
IAI_MCP_STORE to a tmp_path and pre-create / inspect the file directly.

Commands under test:
- `iai-mcp crypto status`         -> JSON-ish status of file backend + user_id
- `iai-mcp crypto rotate`         -> rotate key + re-encrypt all records
- `iai-mcp migrate --from=2 --to=3 [--dry-run]`  -> encryption migration
"""
from __future__ import annotations

import json
import os
import secrets
import stat
from datetime import datetime, timezone
from uuid import uuid4

import pytest


def test_cli_crypto_status_shows_file_backend(tmp_path, monkeypatch, capsys):
    """W1 RED — `iai-mcp crypto status` reports the file backend.

    Pre-creates a 32-byte 0o600 `.crypto.key` in the store root, calls the
    status command, asserts:
      - exit code 0
      - output mentions backend=file
      - output includes the file path (or at least its filename)
      - output exposes mode 0o600
      - NO mention of "keyring" (the backend is gone in W2)

    RED until W2: cmd_crypto_status still emits keyring fields + has no
    `backend: file` shape.
    """
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_status

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_status(args)
    out = capsys.readouterr().out
    out_lower = out.lower()
    assert exit_code == 0
    assert "default" in out
    # New file-backend output contract:
    assert "file" in out_lower, f"status must report backend=file; got:\n{out}"
    assert ".crypto.key" in out, f"status must include the file path; got:\n{out}"
    assert "600" in out, f"status must expose mode 0o600; got:\n{out}"
    # The keyring shape is gone in W2:
    assert "keyring" not in out_lower, (
        f"status must NOT mention keyring (backend retired in 07.10); got:\n{out}"
    )


def test_cli_crypto_rotate_regenerates_key(tmp_path, monkeypatch, capsys):
    """W1 RED — `iai-mcp crypto rotate` writes a fresh key to the
    file backend AND re-encrypts records under the new key.

    Pre-creates a `.crypto.key` (key A) at 0o600, seeds a record encrypted
    under key A, calls rotate, asserts:
      - the file now contains different 32 bytes at mode 0o600
      - the seeded record's ciphertext was re-encrypted (different blob,
        still iai:enc:v1: prefixed, decrypts to the original plaintext
        through the rotated wrapper)

    RED until W2/W3 ship the file-backend + cache-invalidate fix.
    """
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    # Seed key A in the file backend.
    key_path = tmp_path / ".crypto.key"
    key_a = secrets.token_bytes(32)
    key_path.write_bytes(key_a)
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_rotate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    # Seed a record under the initial key.
    store = MemoryStore()
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="rotation test content",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    store.insert(rec)
    initial_ct = store.db.open_table(RECORDS_TABLE).to_pandas()[
        lambda df: df["id"] == str(rec.id)
    ].iloc[0]["literal_surface"]
    assert initial_ct.startswith("iai:enc:v1:")

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_rotate(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "rotat" in out.lower()

    # File backend invariant: the key file now holds different 32 bytes
    # at mode 0o600.
    new_key_bytes = key_path.read_bytes()
    assert len(new_key_bytes) == 32
    assert new_key_bytes != key_a, "rotate must write a fresh key to the file"
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600, f"rotated key file must be 0o600, got 0o{mode:03o}"

    # Data invariant: the seeded record was re-encrypted under the new key.
    # store2 picks up the rotated key from the file backend; the AESGCM
    # wrapper cache is freshly built from the new key.
    store2 = MemoryStore()
    post_ct = store2.db.open_table(RECORDS_TABLE).to_pandas()[
        lambda df: df["id"] == str(rec.id)
    ].iloc[0]["literal_surface"]
    assert post_ct.startswith("iai:enc:v1:")
    assert post_ct != initial_ct  # Re-encrypted under a new key.
    # Content round-trip still works through the rotated key.
    got = store2.get(rec.id)
    assert got is not None
    assert got.literal_surface == "rotation test content"


def test_cli_migrate_to_3_dry_run_counts_plaintext_rows(tmp_path, monkeypatch, capsys):
    """iai-mcp migrate --from=2 --to=3 --dry-run prints a plaintext-row count."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore()
    # Forcibly add a PLAINTEXT row directly to the table (bypass insert()'s encryption).
    rid = uuid4()
    row = {
        "id": str(rid),
        "tier": "episodic",
        "literal_surface": "plain legacy",
        "aaak_index": "",
        "embedding": [0.1] * EMBED_DIM,
        "structure_hv": b"",
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 2,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "provenance_json": json.dumps([{"ts": "x", "cue": "y", "session_id": "z"}]),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "tags_json": json.dumps([]),
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": json.dumps({}),
        "schema_version": 2,
    }
    store.db.open_table(RECORDS_TABLE).add([row])

    args = argparse.Namespace(from_=2, to=3, dry_run=True, verbose=False)
    exit_code = cmd_migrate(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    # Output mentions a record count + the word migrate/would.
    assert "would" in out.lower() or "dry" in out.lower() or "migrat" in out.lower()
    assert "1" in out  # We planted exactly one plaintext row.


def test_cli_migrate_to_3_encrypts_plaintext_rows(tmp_path, monkeypatch, capsys):
    """`iai-mcp migrate --from=2 --to=3` actually encrypts plaintext rows."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore()
    rid = uuid4()
    row = {
        "id": str(rid),
        "tier": "episodic",
        "literal_surface": "still-plaintext",
        "aaak_index": "",
        "embedding": [0.1] * EMBED_DIM,
        "structure_hv": b"",
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 2,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "provenance_json": json.dumps([]),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "tags_json": json.dumps([]),
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": json.dumps({}),
        "schema_version": 2,
    }
    store.db.open_table(RECORDS_TABLE).add([row])

    args = argparse.Namespace(from_=2, to=3, dry_run=False, verbose=False)
    exit_code = cmd_migrate(args)
    assert exit_code == 0

    df = store.db.open_table(RECORDS_TABLE).to_pandas()
    post = df[df["id"] == str(rid)].iloc[0]
    assert post["literal_surface"].startswith("iai:enc:v1:")


def test_cli_migrate_to_3_rejects_unsupported_version_pair(
    tmp_path, monkeypatch, capsys
):
    """--from=9 --to=42 is rejected with a clear error + non-zero exit."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate

    args = argparse.Namespace(from_=9, to=42, dry_run=False, verbose=False)
    exit_code = cmd_migrate(args)
    err = capsys.readouterr().err.lower()
    out = capsys.readouterr().out.lower()
    assert exit_code != 0
    # Some guidance in stderr or stdout.
    assert ("unsupported" in err or "invalid" in err or
            "unsupported" in out or "invalid" in out)


def test_neural_map_bench_passes_after_encryption(tmp_path):
    """bench/neural_map N=100 must still pass <100ms p95 post-encryption."""
    from bench.neural_map import run_neural_map_bench, D_SPEED_P95_MS

    out = run_neural_map_bench(n=100, iterations=10, store_path=tmp_path, seed=0)
    assert out["n"] == 100
    assert out["iterations"] == 10
    assert out["passed"] is True, (
        f"D-SPEED regression post-encryption: p95={out['latency_ms_p95']} ms "
        f">= {D_SPEED_P95_MS} ms"
    )


def test_cli_crypto_init_creates_fresh_file(tmp_path, monkeypatch, capsys):
    """`iai-mcp crypto init` creates a fresh 32-byte 0o600 file.

    No file pre-existing; no keyring needed; resulting file must be exactly
    32 bytes at mode 0o600, exit 0, output cites the path. The key bytes
    themselves MUST NOT appear in stdout.
    """
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    assert not key_path.exists()

    from iai_mcp.cli import cmd_crypto_init

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_init(args)
    out = capsys.readouterr().out
    assert exit_code == 0

    assert key_path.exists()
    assert key_path.stat().st_size == 32
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600, f"init key file must be 0o600, got 0o{mode:03o}"
    # Output cites the path so the user knows where the key lives.
    assert ".crypto.key" in out
    # The 32 raw key bytes MUST NOT appear in the output (D-09 — no key disclosure).
    raw = key_path.read_bytes()
    # Stdout is decoded; a binary blob would not round-trip cleanly. Sanity:
    # check that no run of >=4 raw bytes appears in stdout.
    for i in range(0, 32, 4):
        chunk = raw[i:i + 4]
        # Skip null-padded windows that could trivially collide with text.
        if chunk == b"\x00\x00\x00\x00":
            continue
        assert chunk.decode("latin-1") not in out, (
            "init must not print key bytes to stdout"
        )


def test_cli_crypto_init_refuses_when_file_exists(tmp_path, monkeypatch, capsys):
    """`iai-mcp crypto init` refuses if `.crypto.key` exists.

    Pre-create any-content file at the canonical path; `init` must exit 1
    with an error pointing at the path. File contents must be unchanged.
    """
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    pre = secrets.token_bytes(32)
    key_path.write_bytes(pre)
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_init

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_init(args)
    err = capsys.readouterr().err
    assert exit_code == 1
    assert ".crypto.key" in err
    # File contents unchanged.
    assert key_path.read_bytes() == pre


def test_cli_crypto_rotate_invalidates_aesgcm_cache(tmp_path, monkeypatch):
    """/ T-07.10-08 — `cmd_crypto_rotate` MUST invalidate the
    cached AESGCM after writing the fresh key.

    The rotate test above (`test_cli_crypto_rotate_regenerates_key`) reads
    post-rotate state via a fresh `MemoryStore()` which sidesteps the cache
    entirely; removing the hook would not break it. This test pins the hook
    directly via `unittest.mock.patch.object` so a future refactor that drops
    the `store._invalidate_aesgcm_cache()` line is caught immediately.
    """
    import argparse
    from unittest.mock import patch

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    # Seed a key file so the rotate path proceeds normally.
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_rotate
    from iai_mcp.store import MemoryStore

    args = argparse.Namespace(user_id="default")
    with patch.object(
        MemoryStore, "_invalidate_aesgcm_cache", autospec=True
    ) as m:
        exit_code = cmd_crypto_rotate(args)

    assert exit_code == 0
    assert m.called, (
        "cmd_crypto_rotate must call store._invalidate_aesgcm_cache() "
        "after assigning the new key (, T-07.10-08)"
    )
