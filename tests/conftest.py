"""Project-wide pytest fixtures for the IAI-MCP test suite.

(file-based crypto key migration) removed the keyring backend
from `iai_mcp.crypto.CryptoKey.get_or_create()`. Pre-existing tests that
exercised the daemon, store, events, recall, and CLI paths relied on the
keyring auto-fallback to source the encryption key in test environments.
After , the runtime path is **file → passphrase env → error**
with no keyring fallback, so those tests now hit `CryptoKeyError` unless
either the file or the passphrase is set.

This module's autouse fixture sets `IAI_MCP_CRYPTO_PASSPHRASE` to a fixed
test passphrase for every test session, restoring the deterministic
`derive_key_from_passphrase(...)` path that the test suite expects.
Production behavior is unaffected — the production daemon never sets
this env var and instead reads the 32-byte file at `{IAI_MCP_STORE}/.crypto.key`
written by `iai-mcp crypto migrate-to-file` or `iai-mcp crypto init`.

The dedicated file-backend tests in `tests/test_crypto_file_backend.py`
override this fixture per-test by clearing the env var or by writing an
explicit `.crypto.key` file in their `tmp_path` fixtures.
"""
from __future__ import annotations

import os

import pytest


_TEST_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30-phase-07.10"


@pytest.fixture(autouse=True)
def _crypto_passphrase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set IAI_MCP_CRYPTO_PASSPHRASE for every test unless already set.

    Tests that need to assert the absent-passphrase / missing-key error
    path can still call `monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE",
    raising=False)` inside the test body to override this default.
    """
    if "IAI_MCP_CRYPTO_PASSPHRASE" not in os.environ:
        monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_PASSPHRASE)
