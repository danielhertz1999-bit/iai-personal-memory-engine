"""Phase 07.10 W1 RED: file-backed crypto key {`_try_file_get`, `_try_file_set`,
get_or_create priority, migrate-to-file CLI}.

Locks the executable spec for the file-backed crypto key per CONTEXT.md
D-05 / / D-11. All 9 tests are RED until W2 (crypto.py file
backend) and W3 (cmd_crypto_migrate_to_file) land.

Failure shapes that count as a correct RED signal in this plan:

- TypeError: CryptoKey() got an unexpected keyword argument 'store_root'
  (W2 adds the kwarg)
- AttributeError: 'CryptoKey' object has no attribute '_try_file_get'
  / '_try_file_set' / '_key_file_path'
- ImportError: cannot import name 'cmd_crypto_migrate_to_file'
  (W3 lands the CLI command)

Imports of the new symbols stay INSIDE each test body so module-level
collection succeeds: pytest must be able to ENUMERATE the 9 tests and
then fail each one at assertion time, not crash at collection.
"""
from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

import pytest


# ---------------------------------------------------------------- _try_file_get

def test_try_file_get_returns_bytes_on_valid_0o600_file(tmp_path: Path) -> None:
    """D-11 case 1 — read 32 raw bytes back from a 0o600 key file."""
    from iai_mcp.crypto import CryptoKey

    key_bytes = secrets.token_bytes(32)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(key_bytes)
    os.chmod(key_path, 0o600)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    got = ck._try_file_get()
    assert got == key_bytes
    assert isinstance(got, bytes)
    assert len(got) == 32


def test_try_file_get_rejects_world_or_group_bits(tmp_path: Path) -> None:
    """D-06 / case 2 — mode 0o644 is refused with CryptoKeyError ('insecure mode')."""
    from iai_mcp.crypto import CryptoKey, CryptoKeyError

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x00" * 32)
    os.chmod(key_path, 0o644)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    with pytest.raises(CryptoKeyError) as exc_info:
        ck._try_file_get()
    assert "insecure mode" in str(exc_info.value).lower()


def test_try_file_get_rejects_wrong_length(tmp_path: Path) -> None:
    """D-05 / case 3 — a 31-byte file is rejected with 'wrong length'."""
    from iai_mcp.crypto import CryptoKey, CryptoKeyError

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x01" * 31)  # short by 1 byte
    os.chmod(key_path, 0o600)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    with pytest.raises(CryptoKeyError) as exc_info:
        ck._try_file_get()
    assert "wrong length" in str(exc_info.value).lower()


def test_try_file_get_rejects_foreign_uid(tmp_path: Path, monkeypatch) -> None:
    """D-06 / case 4 — st_uid != geteuid() is refused with 'uid' in message.

    The fake_stat is path-scoped: only the key file gets the foreign-uid
    treatment. Any other os.stat call (pytest internals, library imports)
    delegates to the real os.stat. Returns a full os.stat_result tuple so
    the call shape stays compatible with anything that subscripts it.
    """
    from iai_mcp.crypto import CryptoKey, CryptoKeyError

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x02" * 32)
    os.chmod(key_path, 0o600)

    real_stat = os.stat
    real_result = real_stat(key_path)
    foreign_uid = (os.geteuid() + 12345) & 0xFFFF  # almost certainly not us

    # os.stat_result is constructible from a 10-tuple of (mode, ino, dev,
    # nlink, uid, gid, size, atime, mtime, ctime).
    forged = os.stat_result((
        real_result.st_mode,
        real_result.st_ino,
        real_result.st_dev,
        real_result.st_nlink,
        foreign_uid,
        real_result.st_gid,
        real_result.st_size,
        real_result.st_atime,
        real_result.st_mtime,
        real_result.st_ctime,
    ))

    target_str = str(key_path)

    def fake_stat(path, *args, **kwargs):
        # Path-scoped: only the key file gets the foreign-uid treatment.
        try:
            path_str = str(path)
        except Exception:
            return real_stat(path, *args, **kwargs)
        if path_str == target_str:
            return forged
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fake_stat)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    with pytest.raises(CryptoKeyError) as exc_info:
        ck._try_file_get()
    assert "uid" in str(exc_info.value).lower()


# ---------------------------------------------------------------- _try_file_set

def test_try_file_set_writes_atomic_with_0o600(tmp_path: Path) -> None:
    """D-07 / case 5 — atomic write produces a 0o600 file with exact bytes.

    Also asserts NO `.crypto.key.tmp.<pid>` survives after the call:
    a leaked tmp would prove the rename was non-atomic or the cleanup
    branch was skipped.
    """
    from iai_mcp.crypto import CryptoKey

    payload = b"\x00" * 32
    ck = CryptoKey(user_id="t", store_root=tmp_path)
    ck._try_file_set(payload)

    key_path = tmp_path / ".crypto.key"
    assert key_path.exists()
    assert key_path.read_bytes() == payload
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600

    # Stale tmp scan: the dir must not contain any `.crypto.key.tmp.*` artifacts.
    leftover_tmps = list(tmp_path.glob(".crypto.key.tmp.*"))
    assert leftover_tmps == [], f"leaked tmp files: {leftover_tmps}"


def test_try_file_set_cleans_stale_tmp(tmp_path: Path) -> None:
    """D-07 / case 6 — stale `.crypto.key.tmp.<pid>` is removed before the new write."""
    from iai_mcp.crypto import CryptoKey

    stale_tmp = tmp_path / ".crypto.key.tmp.99999"
    stale_tmp.write_bytes(b"GARBAGE-FROM-CRASHED-PRIOR-RUN")

    payload = b"\x01" * 32
    ck = CryptoKey(user_id="t", store_root=tmp_path)
    ck._try_file_set(payload)

    # Stale tmp gone, final key file present with new payload.
    assert not stale_tmp.exists(), "stale tmp must be cleaned up before the new write"
    key_path = tmp_path / ".crypto.key"
    assert key_path.exists()
    assert key_path.read_bytes() == payload


# ---------------------------------------------------------------- get_or_create priority

def test_get_or_create_prefers_file_over_passphrase(
    tmp_path: Path, monkeypatch
) -> None:
    """D-11 case 7 — file backend wins over passphrase env var.

    Pre-write a valid key file (key A); also set IAI_MCP_CRYPTO_PASSPHRASE
    (which would derive a different key B). get_or_create() must return
    key A (file priority).
    """
    from iai_mcp.crypto import CryptoKey

    key_a = secrets.token_bytes(32)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(key_a)
    os.chmod(key_path, 0o600)

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    got = ck.get_or_create()
    assert got == key_a, "file-backed key must win over passphrase fallback"


# ---------------------------------------------------------------- migrate-to-file CLI

def test_cmd_crypto_migrate_to_file_happy_path(
    tmp_path: Path, monkeypatch
) -> None:
    """D-11 case 8 — migrate-to-file reads keyring, writes file, round-trip OK.

    Patches `keyring.get_password` BEFORE importing the command so the
    local `import keyring` inside cmd_crypto_migrate_to_file picks up
    the monkeypatched attribute (Python caches modules).
    """
    import argparse
    import base64
    import keyring as _keyring

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    keyring_key = secrets.token_bytes(32)
    keyring_blob = base64.urlsafe_b64encode(keyring_key).decode("ascii")

    def fake_get(service: str, username: str) -> str | None:
        return keyring_blob

    def fake_delete(service: str, username: str) -> None:
        pass

    monkeypatch.setattr(_keyring, "get_password", fake_get)
    monkeypatch.setattr(_keyring, "delete_password", fake_delete)

    from iai_mcp.cli import cmd_crypto_migrate_to_file  # ImportError until W3 — RED.

    args = argparse.Namespace(
        user_id="default", keep_keychain=True, delete_keychain=False
    )
    exit_code = cmd_crypto_migrate_to_file(args)
    assert exit_code == 0

    key_path = tmp_path / ".crypto.key"
    assert key_path.exists()
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600
    assert key_path.read_bytes() == keyring_key, (
        "file contents must equal the round-tripped keyring key bytes"
    )


def test_cmd_crypto_migrate_to_file_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    """D-11 case 9 — file already present → no-op success, NO keyring touch.

    keyring.get_password is patched to raise AssertionError; if the
    idempotent path ever calls it, the test fails with a specific message.
    """
    import argparse
    import keyring as _keyring

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    # Pre-create a valid file so the command takes the idempotent branch.
    pre_existing = secrets.token_bytes(32)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(pre_existing)
    os.chmod(key_path, 0o600)

    def assert_not_called(*args, **kwargs):
        raise AssertionError(
            "keyring touched on idempotent path — migrate-to-file must "
            "skip keyring entirely when the file is already present"
        )

    monkeypatch.setattr(_keyring, "get_password", assert_not_called)
    monkeypatch.setattr(_keyring, "delete_password", assert_not_called)

    from iai_mcp.cli import cmd_crypto_migrate_to_file  # ImportError until W3 — RED.

    args = argparse.Namespace(
        user_id="default", keep_keychain=True, delete_keychain=False
    )
    exit_code = cmd_crypto_migrate_to_file(args)
    assert exit_code == 0
    # File contents unchanged.
    assert key_path.read_bytes() == pre_existing
