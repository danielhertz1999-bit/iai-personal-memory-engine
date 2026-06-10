from __future__ import annotations

import os
import pytest


def test_crypto_module_exports() -> None:
    from iai_mcp import crypto
    assert hasattr(crypto, "encrypt_field")
    assert hasattr(crypto, "decrypt_field")
    assert hasattr(crypto, "is_encrypted")
    assert hasattr(crypto, "CryptoKey")
    assert hasattr(crypto, "derive_key_from_passphrase")


def test_crypto_roundtrip_basic() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x00" * 32
    plaintext = "hello world"
    ciphertext = encrypt_field(plaintext, key)
    assert isinstance(ciphertext, str)
    recovered = decrypt_field(ciphertext, key)
    assert recovered == plaintext


def test_crypto_roundtrip_cyrillic() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x01" * 32
    plaintext = "Привет, мир! Это тест шифрования."
    ciphertext = encrypt_field(plaintext, key)
    recovered = decrypt_field(ciphertext, key)
    assert recovered == plaintext
    assert recovered.encode("utf-8") == plaintext.encode("utf-8")


def test_crypto_roundtrip_cjk() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x02" * 32
    plaintext = "こんにちは世界。これは暗号化テストです。"
    ciphertext = encrypt_field(plaintext, key)
    assert decrypt_field(ciphertext, key) == plaintext


def test_crypto_roundtrip_arabic() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x03" * 32
    plaintext = "مرحبا بالعالم. هذا اختبار تشفير."
    ciphertext = encrypt_field(plaintext, key)
    assert decrypt_field(ciphertext, key) == plaintext


def test_crypto_empty_string_roundtrip() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x04" * 32
    assert decrypt_field(encrypt_field("", key), key) == ""


def test_crypto_associated_data_binding() -> None:
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x05" * 32
    ciphertext = encrypt_field("secret", key, associated_data=b"record_id_A")
    with pytest.raises(InvalidTag):
        decrypt_field(ciphertext, key, associated_data=b"record_id_B")


def test_crypto_associated_data_roundtrip_when_matching() -> None:
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x06" * 32
    ad = b"record_id_matching"
    ct = encrypt_field("secret", key, associated_data=ad)
    assert decrypt_field(ct, key, associated_data=ad) == "secret"


def test_crypto_tamper_detection() -> None:
    import base64
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key = b"\x07" * 32
    ct = encrypt_field("secret", key)
    prefix = "iai:enc:v1:"
    assert ct.startswith(prefix)
    payload_b64 = ct[len(prefix):]
    raw = bytearray(base64.b64decode(payload_b64))
    raw[15] ^= 0x01
    tampered = prefix + base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(InvalidTag):
        decrypt_field(tampered, key)


def test_crypto_wrong_key_fails() -> None:
    from cryptography.exceptions import InvalidTag
    from iai_mcp.crypto import encrypt_field, decrypt_field
    key_a = b"\x08" * 32
    key_b = b"\x09" * 32
    ct = encrypt_field("secret", key_a)
    with pytest.raises(InvalidTag):
        decrypt_field(ct, key_b)


def test_is_encrypted_prefix_true() -> None:
    from iai_mcp.crypto import encrypt_field, is_encrypted
    key = b"\x0a" * 32
    ct = encrypt_field("hello", key)
    assert is_encrypted(ct) is True


def test_is_encrypted_prefix_false() -> None:
    from iai_mcp.crypto import is_encrypted
    assert is_encrypted("plaintext") is False
    assert is_encrypted("") is False
    assert is_encrypted("iai:enc:v0:abc") is False
    assert is_encrypted("foo:bar") is False


def test_crypto_unique_nonce_per_encrypt() -> None:
    from iai_mcp.crypto import encrypt_field
    key = b"\x0b" * 32
    ct1 = encrypt_field("repeat", key)
    ct2 = encrypt_field("repeat", key)
    assert ct1 != ct2


def test_derive_key_from_passphrase_deterministic() -> None:
    from iai_mcp.crypto import derive_key_from_passphrase
    salt = b"saltsaltsaltsalt"
    k1 = derive_key_from_passphrase("hunter2", salt)
    k2 = derive_key_from_passphrase("hunter2", salt)
    assert k1 == k2
    assert len(k1) == 32


def test_derive_key_from_passphrase_different_salts() -> None:
    from iai_mcp.crypto import derive_key_from_passphrase
    salt_a = b"A" * 16
    salt_b = b"B" * 16
    assert derive_key_from_passphrase("same", salt_a) != derive_key_from_passphrase("same", salt_b)


def test_derive_key_uses_600k_iterations() -> None:
    from iai_mcp import crypto
    assert crypto.PBKDF2_ITERATIONS >= 600_000


def test_crypto_key_passphrase_fallback_when_file_missing(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp import crypto

    assert not (tmp_path / ".crypto.key").exists()

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2-fallback")

    ck = crypto.CryptoKey(user_id="t", store_root=tmp_path)
    key1 = ck.get_or_create()
    assert isinstance(key1, bytes)
    assert len(key1) == 32

    ck2 = crypto.CryptoKey(user_id="t", store_root=tmp_path)
    key2 = ck2.get_or_create()
    assert key1 == key2
