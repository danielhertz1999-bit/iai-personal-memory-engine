"""crypto.py AES-256-GCM primitives + file-backed key storage.

Updated to retire the keyring
backend (which deadlocked the daemon under launchd via the macOS
Keychain ACL prompt) in favor of a file-backed primary backend at
`{IAI_MCP_STORE}/.crypto.key` (32 raw bytes, mode 0o600, uid-validated).

Covers:
- encrypt_field / decrypt_field round-trip (byte-for-byte)
- Cyrillic / CJK / Arabic round-trip (across languages)
- Associated data binding (swapped AD -> InvalidTag)
- Tamper detection (mutated ciphertext -> InvalidTag)
- is_encrypted prefix check
- Passphrase fallback when no `.crypto.key` file is present
  (via IAI_MCP_CRYPTO_PASSPHRASE), deterministic across instances

File-backend specific behavior (file priority, uid/mode validation,
atomic write) is exercised in tests/test_crypto_file_backend.py.
"""
from __future__ import annotations

import os
import pytest


def test_crypto_module_exports() -> None:
    """crypto.py exposes encrypt_field / decrypt_field / is_encrypted / CryptoKey."""
    from iai_mcp import crypto
    assert hasattr(crypto, "encrypt_field")
    assert hasattr(crypto, "decrypt_field")
    assert hasattr(crypto, "is_encrypted")
    assert hasattr(crypto, "CryptoKey")
    assert hasattr(crypto, "derive_key_from_passphrase")


def test_crypto_roundtrip_basic() -> None:
    """encrypt(plaintext) -> decrypt -> byte-for-byte equal."""
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x00" * 32
    plaintext = "hello world"
    ciphertext = encrypt_field(plaintext, key)
    assert isinstance(ciphertext, str)
    recovered = decrypt_field(ciphertext, key)
    assert recovered == plaintext


def test_crypto_roundtrip_cyrillic() -> None:
    """+: Russian text byte-for-byte preserved."""
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x01" * 32
    plaintext = "Привет, мир! Это тест шифрования."
    ciphertext = encrypt_field(plaintext, key)
    recovered = decrypt_field(ciphertext, key)
    assert recovered == plaintext
    # Byte-level equality after utf-8 encode+decode cycle.
    assert recovered.encode("utf-8") == plaintext.encode("utf-8")


def test_crypto_roundtrip_cjk() -> None:
    """+: Japanese / Chinese round-trip."""
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x02" * 32
    plaintext = "こんにちは世界。これは暗号化テストです。"
    ciphertext = encrypt_field(plaintext, key)
    assert decrypt_field(ciphertext, key) == plaintext


def test_crypto_roundtrip_arabic() -> None:
    """+: Arabic round-trip."""
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x03" * 32
    plaintext = "مرحبا بالعالم. هذا اختبار تشفير."
    ciphertext = encrypt_field(plaintext, key)
    assert decrypt_field(ciphertext, key) == plaintext


def test_crypto_empty_string_roundtrip() -> None:
    """Empty plaintext encrypts and decrypts cleanly."""
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x04" * 32
    assert decrypt_field(encrypt_field("", key), key) == ""


def test_crypto_associated_data_binding() -> None:
    """Ciphertext encrypted with AD=A cannot be decrypted with AD=B (InvalidTag)."""
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x05" * 32
    ciphertext = encrypt_field("secret", key, associated_data=b"record_id_A")
    with pytest.raises(InvalidTag):
        decrypt_field(ciphertext, key, associated_data=b"record_id_B")


def test_crypto_associated_data_roundtrip_when_matching() -> None:
    """With matching AD the round-trip succeeds."""
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x06" * 32
    ad = b"record_id_matching"
    ct = encrypt_field("secret", key, associated_data=ad)
    assert decrypt_field(ct, key, associated_data=ad) == "secret"


def test_crypto_tamper_detection() -> None:
    """A single-bit flip in ciphertext raises InvalidTag on decrypt."""
    import base64
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x07" * 32
    ct = encrypt_field("secret", key)
    # Strip the prefix, flip one byte in the base64 payload, re-wrap.
    prefix = "iai:enc:v1:"
    assert ct.startswith(prefix)
    payload_b64 = ct[len(prefix):]
    raw = bytearray(base64.b64decode(payload_b64))
    # Flip the byte after the nonce (12 bytes) -- tamper the ciphertext itself.
    raw[15] ^= 0x01
    tampered = prefix + base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(InvalidTag):
        decrypt_field(tampered, key)


def test_crypto_wrong_key_fails() -> None:
    """Decrypt with a different key raises InvalidTag."""
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key_a = b"\x08" * 32
    key_b = b"\x09" * 32
    ct = encrypt_field("secret", key_a)
    with pytest.raises(InvalidTag):
        decrypt_field(ct, key_b)


def test_is_encrypted_prefix_true() -> None:
    """is_encrypted returns True for strings that start with iai:enc:v1:"""
    from iai_mcp.crypto import encrypt_field, is_encrypted
    key = b"\x0a" * 32
    ct = encrypt_field("hello", key)
    assert is_encrypted(ct) is True


def test_is_encrypted_prefix_false() -> None:
    """is_encrypted returns False for plaintext / None / empty / wrong prefix."""
    from iai_mcp.crypto import is_encrypted
    assert is_encrypted("plaintext") is False
    assert is_encrypted("") is False
    assert is_encrypted("iai:enc:v0:abc") is False  # Different version
    assert is_encrypted("foo:bar") is False


def test_crypto_unique_nonce_per_encrypt() -> None:
    """Two encryptions of the same plaintext under the same key produce different ciphertexts."""
    from iai_mcp.crypto import encrypt_field
    key = b"\x0b" * 32
    ct1 = encrypt_field("repeat", key)
    ct2 = encrypt_field("repeat", key)
    assert ct1 != ct2  # Random nonce ensures ciphertext differs


def test_derive_key_from_passphrase_deterministic() -> None:
    """Same passphrase + same salt -> same derived key (PBKDF2)."""
    from iai_mcp.crypto import derive_key_from_passphrase
    salt = b"saltsaltsaltsalt"  # 16 bytes
    k1 = derive_key_from_passphrase("hunter2", salt)
    k2 = derive_key_from_passphrase("hunter2", salt)
    assert k1 == k2
    assert len(k1) == 32  # 256 bits


def test_derive_key_from_passphrase_different_salts() -> None:
    """Same passphrase, different salts -> different keys."""
    from iai_mcp.crypto import derive_key_from_passphrase
    salt_a = b"A" * 16
    salt_b = b"B" * 16
    assert derive_key_from_passphrase("same", salt_a) != derive_key_from_passphrase("same", salt_b)


def test_derive_key_uses_600k_iterations() -> None:
    """OWASP 2023: PBKDF2-HMAC-SHA256 recommends 600k iterations minimum."""
    from iai_mcp import crypto
    assert crypto.PBKDF2_ITERATIONS >= 600_000


def test_crypto_key_passphrase_fallback_when_file_missing(
    tmp_path, monkeypatch
) -> None:
    """W1 RED — file-backed CryptoKey falls back to passphrase
    when no `.crypto.key` file exists in store_root.

    Priority order under the new backend: file -> passphrase env var
    -> CryptoKeyError. This test exercises the second tier: file is absent,
    IAI_MCP_CRYPTO_PASSPHRASE is set, get_or_create() must return a 32-byte
    derived key that is deterministic across instances (same passphrase +
    same salt -> same key). NO keyring mocking — the keyring backend is
    gone in W2, so this test must not depend on it.

    RED until W2: CryptoKey does not yet accept store_root kwarg.
    """
    from iai_mcp import crypto

    # No `.crypto.key` written to tmp_path -> file backend miss.
    assert not (tmp_path / ".crypto.key").exists()

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2-fallback")

    ck = crypto.CryptoKey(user_id="t", store_root=tmp_path)
    key1 = ck.get_or_create()
    assert isinstance(key1, bytes)
    assert len(key1) == 32

    # Same passphrase + same user_id (salt) -> same derived key on a fresh
    # instance with the same store_root.
    ck2 = crypto.CryptoKey(user_id="t", store_root=tmp_path)
    key2 = ck2.get_or_create()
    assert key1 == key2
