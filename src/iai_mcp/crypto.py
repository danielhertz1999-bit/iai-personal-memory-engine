"""AES-256-GCM encryption-at-rest primitives + file-backed key storage.

Ciphertext format (string-encoded for string-column storage):

    iai:enc:v1:<base64(nonce || ciphertext || tag)>

Components:
- prefix "iai:enc:v1:" (identifies encrypted payload; enables mixed
                  plaintext/ciphertext coexistence during v2->v3 migration)
- nonce 12 random bytes (AES-GCM standard IV length)
- ciphertext+tag AESGCM.encrypt(nonce, plaintext_utf8, associated_data) output;
                  the 16-byte GCM authentication tag is appended by AESGCM.

Associated data (AD) is the UUID bytes of the record id: this binds the
ciphertext to its row so an attacker with write access cannot swap ciphertext
values between rows (tampering mitigation).

Key storage (file-backed primary, no keyring at module scope):
- Primary: a 32-raw-byte file at ``{store_root}/.crypto.key`` (default
  ``~/.iai-mcp/.crypto.key``), mode ``0o600``, owner-uid validated. Resolved
  via the ``store_root`` constructor argument (single-source path, threaded
  from ``MemoryStore.root`` — see). When ``store_root`` is
  ``None`` the path is read lazily from ``IAI_MCP_STORE`` env or the
  ``DEFAULT_STORAGE_PATH`` (``~/.iai-mcp``).
- Fallback: passphrase via ``IAI_MCP_CRYPTO_PASSPHRASE`` env var (CI / fresh
  installs / non-interactive environments). Key derived via PBKDF2-HMAC-
  SHA256 with 600_000 iterations (OWASP 2023 recommendation) and a per-user
  salt (``sha256(user_id)[:16]``). Deterministic given passphrase + user_id,
  so the same machine survives reboots without persisting anything new.
- If neither path resolves, ``CryptoKey.get_or_create()`` raises
  ``CryptoKeyError`` with a dual-remediation message naming
  ``iai-mcp crypto migrate-to-file`` (existing macOS Keychain key from before), ``iai-mcp crypto init`` (fresh install), and the
  ``IAI_MCP_CRYPTO_PASSPHRASE`` env var (CI / non-interactive). No silent
  key generation — that would render existing data unreadable.

The migration CLI command ``iai-mcp crypto migrate-to-file`` keeps
a function-local ``import keyring`` to read an existing macOS Keychain key
once and write it to the file backend; this module never imports ``keyring``
at file scope, so daemon boot under launchd does not block on the Keychain
ACL prompt.

Module contract:
- encrypt_field(plaintext, key, associated_data) -> str (prefixed base64)
- decrypt_field(ciphertext_b64, key, associated_data) -> str
- is_encrypted(field) -> bool
- CryptoKey(user_id, store_root=None).get_or_create() / rotate() / delete()
- derive_key_from_passphrase(passphrase, salt) -> bytes (32)

Design invariants:
- No keys stored in the SQLite store; only ciphertext.
- File backend missing degrades to passphrase fallback; absent both,
  refusal is loud with an actionable error pointing at both remediation paths.
- Encryption is lossless: decrypt(encrypt(x)) == x byte-for-byte.
"""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# Crypto constants (module-scope for grep-discoverability).
CIPHERTEXT_PREFIX: str = "iai:enc:v1:"
NONCE_BYTES: int = 12          # AES-GCM standard IV length
KEY_BYTES: int = 32            # 256-bit key
PBKDF2_ITERATIONS: int = 600_000  # OWASP 2023 minimum for PBKDF2-HMAC-SHA256
SERVICE_NAME_DEFAULT: str = "iai-mcp"

# Default storage root mirrors store.DEFAULT_STORAGE_PATH so a CryptoKey that
# is constructed without a ``store_root`` argument resolves to the same
# location MemoryStore would have used. Kept as a module-private to avoid
# importing store.py here (would create a circular import).
_DEFAULT_STORE_ROOT: Path = Path.home() / ".iai-mcp"
_KEY_FILE_NAME: str = ".crypto.key"


class CryptoKeyError(RuntimeError):
    """Raised when a CryptoKey cannot be loaded or created.

    Typical triggers:
    - The key file exists at the resolved path but is unreadable, has an
      insecure mode, is owned by a different uid, or has the wrong length.
    - Neither a key file NOR ``IAI_MCP_CRYPTO_PASSPHRASE`` is present;
      ``MemoryStore`` surfaces the error so the daemon refuses to start with
      a clear actionable message instead of silently proceeding without
      encryption.
    """


def is_encrypted(field: Optional[str]) -> bool:
    """Cheap prefix check supporting mixed-plaintext/ciphertext coexistence.

    Returns True only when `field` is a non-empty string that starts with the
    exact version prefix `iai:enc:v1:`. Used by:
    - store._decrypt_fields to know whether to attempt decryption
    - migrate_encryption_v2_to_v3 to skip already-encrypted rows
    """
    if not field or not isinstance(field, str):
        return False
    return field.startswith(CIPHERTEXT_PREFIX)


def encrypt_field(
    plaintext: str,
    key: bytes,
    associated_data: bytes = b"",
) -> str:
    """AES-256-GCM encrypt a UTF-8 string; return prefixed base64 ciphertext.

    The nonce is generated randomly with secrets.token_bytes (not os.urandom
    for slight additional entropy guarantees). A fresh nonce is REQUIRED for
    every call with a given key -- reusing a nonce with AES-GCM breaks the
    security of both messages.

    Parameters
    ----------
    plaintext:
        Any UTF-8 string (including empty string). Cyrillic / CJK / Arabic
        preserved byte-for-byte.
    key:
        32-byte (256-bit) key. Typically sourced from CryptoKey.get_or_create().
    associated_data:
        Arbitrary bytes that are authenticated but not encrypted. In this
        codebase: the record id in UUID-string form (binds ciphertext to row).

    Returns
    -------
    str: "iai:enc:v1:" + base64(nonce || ciphertext || tag)
    """
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes (got {len(key)})")
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(NONCE_BYTES)
    ct_with_tag = aesgcm.encrypt(
        nonce, plaintext.encode("utf-8"), associated_data or None
    )
    payload = nonce + ct_with_tag
    return CIPHERTEXT_PREFIX + base64.b64encode(payload).decode("ascii")


def decrypt_field(
    ciphertext_b64: str,
    key: bytes,
    associated_data: bytes = b"",
) -> str:
    """Decrypt a prefixed base64 AES-256-GCM payload back to a UTF-8 string.

    Raises cryptography.exceptions.InvalidTag on:
    - Wrong key
    - Tampered ciphertext (single-bit flip in nonce / ct / tag)
    - Mismatched associated_data (even one byte off)

    Raises ValueError if the field doesn't carry the iai:enc:v1: prefix -- the
    caller should have guarded with is_encrypted() first.
    """
    if not is_encrypted(ciphertext_b64):
        raise ValueError("field is not iai:enc:v1:-prefixed ciphertext")
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes (got {len(key)})")
    payload_b64 = ciphertext_b64[len(CIPHERTEXT_PREFIX):]
    payload = base64.b64decode(payload_b64)
    if len(payload) < NONCE_BYTES + 16:  # nonce + min GCM tag
        raise ValueError("ciphertext payload too short")
    nonce = payload[:NONCE_BYTES]
    ct_with_tag = payload[NONCE_BYTES:]
    aesgcm = AESGCM(key)
    plaintext_bytes = aesgcm.decrypt(
        nonce, ct_with_tag, associated_data or None
    )
    return plaintext_bytes.decode("utf-8")


def derive_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256 key derivation for the passphrase-fallback path.

    Parameters
    ----------
    passphrase:
        User-supplied passphrase (via IAI_MCP_CRYPTO_PASSPHRASE env var in the
        current design -- first-run prompt is future work when we have a CLI
        interaction point).
    salt:
        16+ bytes of salt. In practice the CryptoKey fallback uses
        sha256(user_id)[:16] so the derived key is deterministic per
        (passphrase, user_id) pair on a given machine.

    Returns 32 bytes (256-bit) suitable for AESGCM.
    """
    if len(salt) < 16:
        raise ValueError(f"salt must be at least 16 bytes (got {len(salt)})")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


class CryptoKey:
    """File-backed 256-bit AES key with passphrase fallback.

     redesign:
        File backend at ``{store_root}/.crypto.key`` (32 raw bytes, mode
        ``0o600``, owner-uid validated) is the primary. Passphrase via
        ``IAI_MCP_CRYPTO_PASSPHRASE`` is the second-tier fallback. If neither
        resolves, ``get_or_create()`` raises ``CryptoKeyError`` with an
        actionable error message naming both remediation paths plus
        ``iai-mcp crypto migrate-to-file`` (one-time migration of an existing
        Keychain key) and ``iai-mcp crypto init`` (fresh install).

    Usage:
        ck = CryptoKey(user_id="default", store_root=Path("~/.iai-mcp"))
        key = ck.get_or_create() # 32 bytes; reads from file or falls back
                                   # to passphrase
        #...
        new_key = ck.rotate() # writes a fresh key file (atomic temp+rename);
                                   # caller is responsible for re-encrypting data
        ck.delete() # remove the key file (test teardown / uninstall)

    Multi-user ready: each ``user_id`` derives its own passphrase salt
    (``sha256(user_id)[:16]``). The current product ships a single
    ``user_id="default"`` but the architecture supports per-user isolation for
    future multi-tenant deployments. (The file backend itself is currently
    single-tenant — one ``.crypto.key`` per store root.)

    Thread-safety: instance-level ``_cached_key`` hides repeated
    ``get_or_create()`` calls from the file backend (one read per process
    lifetime, not per call).
    """

    SERVICE_NAME: str = SERVICE_NAME_DEFAULT

    def __init__(
        self,
        user_id: str = "default",
        store_root: Path | None = None,
    ) -> None:
        self.user_id = user_id
        self.store_root: Path | None = store_root
        self._cached_key: Optional[bytes] = None

    # ---------------------------------------------------------------- helpers

    def _passphrase_salt(self) -> bytes:
        """Per-user salt for the passphrase fallback; deterministic across runs."""
        return hashlib.sha256(self.user_id.encode("utf-8")).digest()[:16]

    def _key_file_path(self) -> Path:
        """Resolve ``{store_root}/.crypto.key``.

        Lazy resolution: if ``self.store_root`` was not supplied at
        construction, read ``IAI_MCP_STORE`` env or fall back to the project
        default ``~/.iai-mcp`` — the same precedence ``MemoryStore.__init__``
        uses. Resolving here (not in ``__init__``) lets a test set
        ``IAI_MCP_STORE`` after a CryptoKey instance was already created
        without the kwarg.
        """
        if self.store_root is not None:
            root = Path(self.store_root)
        else:
            env_path = os.environ.get("IAI_MCP_STORE")
            root = Path(env_path) if env_path else _DEFAULT_STORE_ROOT
        return root / _KEY_FILE_NAME

    def _try_file_get(self) -> Optional[bytes]:
        """Return 32 raw bytes from the key file; ``None`` if the file is absent.

          strict validation:
        - mode strictly ``0o600`` — refuse if any group/world bits are set
          (``mode & 0o077 != 0``) with ``CryptoKeyError("...insecure mode...")``
        - ``st_uid == os.geteuid()`` — refuse files owned by a different user
          with ``CryptoKeyError("...uid...")``
        - file length exactly ``KEY_BYTES`` — refuse with
          ``CryptoKeyError("...wrong length...")``

        Each rejection emits a distinct error message so misconfigurations are
        diagnosable at a glance.
        """
        path = self._key_file_path()
        if not path.exists():
            return None
        # Use ``os.stat`` rather than ``Path.stat`` so test harnesses can
        # monkeypatch ``os.stat`` to simulate foreign-uid scenarios at the
        # syscall boundary (W1 case 4 path-scoped fake stat).
        st = os.stat(path)
        # Mode check: owner-only bits permitted.
        if st.st_mode & 0o077 != 0:
            raise CryptoKeyError(
                f"crypto key file at {path} has insecure mode "
                f"0o{st.st_mode & 0o777:03o}; expected 0o600 "
                f"(run: chmod 0o600 {path})"
            )
        # UID check: refuse files owned by a different user.
        if st.st_uid != os.geteuid():
            raise CryptoKeyError(
                f"crypto key file at {path} is owned by uid={st.st_uid}; "
                f"current process runs as uid={os.geteuid()} (refusing to read)"
            )
        raw = path.read_bytes()
        if len(raw) != KEY_BYTES:
            raise CryptoKeyError(
                f"crypto key file at {path} has wrong length {len(raw)} "
                f"(expected {KEY_BYTES})"
            )
        return raw

    def _try_file_set(self, key: bytes) -> None:
        """Atomically write ``key`` to the key file.

        Pattern:
        1. ``mkdir -p`` the parent directory.
        2. Remove any stale ``{path}.tmp.*`` siblings from prior crashed runs.
        3. Open ``{path}.tmp.{pid}`` with ``O_CREAT|O_EXCL|O_WRONLY`` mode
           ``0o600`` — refuses if a tmp file at the same pid already exists.
        4. ``os.fchmod(fd, 0o600)`` BEFORE writing bytes — defends against
           umask quirks, makes the mode-restriction window zero.
        5. ``os.write`` + ``os.fsync`` + ``os.close``.
        6. ``os.rename`` the tmp file to the final path (atomic on POSIX).

        ``ValueError`` is raised if ``key`` is not exactly ``KEY_BYTES`` long.
        """
        if len(key) != KEY_BYTES:
            raise ValueError(f"key must be {KEY_BYTES} bytes (got {len(key)})")
        final = self._key_file_path()
        final.parent.mkdir(parents=True, exist_ok=True)
        # Clean stale tmp files from prior crashed runs so the new write is
        # never confused by leftover state.
        for stale in final.parent.glob(f"{final.name}.tmp.*"):
            try:
                stale.unlink()
            except OSError:
                # Best-effort cleanup; if unlink fails we still proceed and
                # the EXCL open below will refuse if our pid happens to
                # collide with a leftover.
                pass
        tmp = final.parent / f"{final.name}.tmp.{os.getpid()}"
        # ``O_CREAT | O_EXCL | O_WRONLY`` refuses if a tmp at this exact pid
        # already exists; combined with the cleanup above, this guarantees a
        # fresh write path. ``mode=0o600`` is enforced atomically by ``open``.
        fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            # Explicit ``fchmod`` BEFORE writing bytes: defends against any
            # umask quirk that might subtly relax the mode after open. The
            # window where the tmp file exists with permissive bits is zero.
            os.fchmod(fd, 0o600)
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(str(tmp), str(final))

    # -------------------------------------------------------- public API

    def get_or_create(self) -> bytes:
        """Return the 256-bit AES key for this user_id.

         priority:
        1. Instance cache (``self._cached_key``) — avoids repeated file reads.
        2. File backend (``_try_file_get``) — returns the 32 raw bytes from
           ``{store_root}/.crypto.key`` if present, else ``None``.
        3. Passphrase fallback — derives a key from
           ``IAI_MCP_CRYPTO_PASSPHRASE`` via PBKDF2; deterministic given
           ``(passphrase, user_id)``. The derived key is NOT written to disk
           — it lives only in the instance cache for the session.
        4. Otherwise raise ``CryptoKeyError`` naming all remediation paths
           (``iai-mcp crypto migrate-to-file``, ``iai-mcp crypto init``,
           ``IAI_MCP_CRYPTO_PASSPHRASE``).
        """
        if self._cached_key is not None:
            return self._cached_key

        # Priority 1: file backend.
        existing = self._try_file_get()
        if existing is not None:
            self._cached_key = existing
            return existing

        # Priority 2: passphrase fallback (CI / non-interactive / fresh-install opt-in).
        passphrase = os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        if passphrase:
            derived = derive_key_from_passphrase(passphrase, self._passphrase_salt())
            self._cached_key = derived
            return derived

        # Priority 3: refuse with a dual-remediation error message.
        path = self._key_file_path()
        raise CryptoKeyError(
            f"crypto key file not found at {path} and IAI_MCP_CRYPTO_PASSPHRASE "
            f"is not set.\n"
            f"\n"
            f"To fix:\n"
            f"  - Existing install (key was in macOS Keychain): "
            f"run `iai-mcp crypto migrate-to-file` from a Terminal where the "
            f"Keychain prompt can appear, then click \"Always Allow\".\n"
            f"  - Fresh install: run `iai-mcp crypto init` to generate a new key "
            f"file, OR set IAI_MCP_CRYPTO_PASSPHRASE to a strong passphrase "
            f"(suitable for CI or non-interactive environments)."
        )

    def rotate(self) -> bytes:
        """Generate a fresh 32-byte key, write it to the key file, return it.

         : rotation is now an atomic file-write operation,
        irrespective of how the previous key was sourced. Caller is responsible
        for re-encrypting any existing ciphertext under the old key (see
        ``iai-mcp crypto rotate`` CLI; re-encryption is an application-layer
        concern). The cached instance key is updated so subsequent calls in
        the same process see the new key.
        """
        fresh = secrets.token_bytes(KEY_BYTES)
        self._try_file_set(fresh)
        self._cached_key = fresh
        return fresh

    def delete(self) -> None:
        """Remove the key file (and drop the cache). Idempotent on absent files."""
        self._cached_key = None
        path = self._key_file_path()
        try:
            path.unlink()
        except FileNotFoundError:
            # Idempotent: nothing to delete.
            pass
