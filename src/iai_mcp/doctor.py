"""Daemon health doctor: PASS/WARN/FAIL checklist + up to 4-action repair sequence.

Reversibility-by-default. Default mode is diagnose-only (zero mutations).
--apply confirms each destructive action; --apply --yes skips confirmations.

Guards:
- User consent: doctor --apply respects [y/N] confirmations unless --yes is
  also passed; no destructive action without explicit consent.
- Clean uninstall: doctor --apply may unlink stale ~/.iai-mcp/.daemon.sock
  ONLY. Lock file + state file are managed by daemon_state.save_state /
  iai-mcp daemon uninstall.
- Fail-loud: doctor surfaces failures with explicit user-readable diagnosis,
  never silently masks daemon death.
- Wrong-PID-kill mitigation: every kill action verifies BOTH os.kill(pid, 0)
  liveness AND psutil.Process(pid).cmdline() contains 'iai_mcp.core' (orphan
  target) or 'iai_mcp.daemon' (live target) before SIGTERM. Mitigates PID
  reuse on macOS (PIDs cycle within minutes).

Exit codes:
  0 = all checks PASS (WARN does NOT flip to 1)
  1 = one or more FAIL (no --apply)
  2 = --apply ran but final re-check still has FAIL

This module has NO LLM code and NO paid-API env var references.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import sqlite3
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Recovery action timing constants. Tuned so a launchd-managed daemon has
# time to react (KeepAlive bounces in 1-2s on macOS) and a manual respawn
# can finish bge-small load (~3-10s) plus store open (~1s).
_LAUNCHD_REACT_DELAY_SEC = 2.0
_RESPAWN_BIND_TIMEOUT_SEC = 8.0
_RESPAWN_POLL_INTERVAL_SEC = 0.1


# -----------------------------------------------------------------------------
# Result + action dataclasses
# -----------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single doctor check.

    Attributes:
        name: Stable label printed verbatim (e.g. "(a) daemon process alive").
        passed: True iff the check is healthy. WARN rows count as ``passed=True``
            so they do NOT flip the doctor's exit code to 1 — they're advisory.
        detail: One-line explanation; printed verbatim after the
            [PASS]/[WARN]/[FAIL] tag.
        status: One of "PASS", "WARN", "FAIL". Lets check_h
            emit the WARN tri-state without breaking the 3-arg construction
            pattern used by ~14 sites in test_doctor_checklist.py. When
            unspecified, derives from ``passed`` (True → "PASS", False → "FAIL").
    """

    name: str
    passed: bool
    detail: str
    status: str = ""

    def __post_init__(self) -> None:
        # Default-derive `status` from `passed` so legacy 3-arg construction
        # continues to work unchanged. Explicit ``status="WARN"`` is the only
        # way to produce a WARN row.
        if not self.status:
            self.status = "PASS" if self.passed else "FAIL"


@dataclass
class RepairAction:
    """A single --apply repair step.

    Attributes:
        label: Short slug used in audit events + log lines (e.g. "respawn_daemon").
        description: Human-readable phrasing shown in [y/N] prompt.
        destructive: True iff the action mutates state or kills processes; gated
            by [y/N] confirmation when --yes is not passed.
        execute: Callable returning (success, message, duration_ms).
    """

    label: str
    description: str
    destructive: bool
    execute: Callable[[], tuple[bool, str, int]]


# -----------------------------------------------------------------------------
# Helpers — socket path resolution honoring IAI_DAEMON_SOCKET_PATH
# -----------------------------------------------------------------------------


def _resolve_socket_path() -> Path:
    """Return the socket path honoring IAI_DAEMON_SOCKET_PATH env override.

    The env override is the test isolation mechanism; production users have
    no env var set and fall back to ~/.iai-mcp/.daemon.sock.
    """
    env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if env_path:
        return Path(env_path)
    from iai_mcp.cli import SOCKET_PATH

    return Path(SOCKET_PATH)


async def _socket_status_probe(socket_path: Path, timeout: float) -> dict | None:
    """One-shot NDJSON `{type: status}` round-trip against socket_path.

    Returns the daemon's reply dict, or None if the daemon is unreachable
    (socket missing / connect refused / no reply within timeout).

    Distinct from cli._send_socket_request — that helper hard-codes the home
    socket path; the doctor needs to honor IAI_DAEMON_SOCKET_PATH so test
    isolation works.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path=str(socket_path)),
            timeout=timeout,
        )
    except (FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError, OSError):
        return None
    try:
        writer.write((json.dumps({"type": "status"}) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except Exception as exc:
        logger.debug("socket status probe failed: %s", exc)
        return None
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass  # cleanup best-effort


# -----------------------------------------------------------------------------
# 6 individual checks
# -----------------------------------------------------------------------------


def check_a_daemon_alive() -> CheckResult:
    """(a) daemon process alive.

    PID source-of-truth is `~/.iai-mcp/.daemon-state.json`
    (`daemon_pid` is stamped on boot; the .lock file is fcntl-only and
    contains zero PID bytes).

    Wrong-PID kill mitigation: verifies BOTH os.kill(pid, 0) liveness AND
    psutil.cmdline contains 'iai_mcp.daemon'. Without the cmdline check,
    a recycled PID belonging to an unrelated process would falsely appear
    healthy.
    """
    from iai_mcp.daemon_state import load_state

    try:
        state = load_state() or {}
    except Exception as e:
        logger.debug("check_a: daemon-state.json unreadable: %s", e)
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"daemon-state.json unreadable: {type(e).__name__}: {e}",
        )

    pid = state.get("daemon_pid")
    if pid is None:
        return CheckResult(
            "(a) daemon process alive",
            False,
            "ABSENT (no daemon_pid in state — daemon never booted or already shut down)",
        )

    # Reject obviously-garbage PID values (negative / non-int / > INT_MAX)
    # from a corrupted state file before they reach os.kill, which raises
    # OverflowError for out-of-range ints. ProcessLookupError is the right
    # semantic here — the "process" is unreachable / bogus.
    if not isinstance(pid, int) or pid < 1 or pid > 2**31 - 1:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"daemon_pid={pid!r} is not a valid PID (corrupt state?)",
        )

    # Liveness probe via signal 0 (no actual signal sent).
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"PID {pid} in state but no process found",
        )
    except PermissionError:
        # Process exists but is owned by another UID (extremely unlikely on a
        # single-user machine; would mean PID reuse to a system process).
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"PID {pid} exists but is not owned by this user",
        )
    except OSError as e:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"liveness probe failed: {type(e).__name__}: {e}",
        )

    # Wrong-PID-kill mitigation: confirm the live PID is actually our daemon.
    try:
        import psutil

        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline() or [])
        if "iai_mcp.daemon" not in cmdline:
            return CheckResult(
                "(a) daemon process alive",
                False,
                f"PID {pid} is NOT iai_mcp.daemon (got: {proc.name()!r})",
            )
    except Exception as e:  # noqa: BLE001 — psutil edge cases all roll up here
        logger.debug("check_a: psutil verify PID %d failed: %s", pid, e)
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"could not verify PID {pid}: {type(e).__name__}: {e}",
        )

    return CheckResult(
        "(a) daemon process alive",
        True,
        f"PID {pid} (iai_mcp.daemon)",
    )


async def _socket_connect_probe(socket_path: Path, timeout: float) -> str | None:
    """Connect-only probe: open a unix socket and close immediately.

    Returns None on success, or a string describing the failure (errno or
    exception class). The (b) row uses this in preference to a status
    round-trip because the daemon's status handler can take seconds while
    the socket itself is fully reachable; (b) asserts "socket file fresh",
    which is a kernel-level question (does connect() succeed?). Daemon
    responsiveness belongs to a separate diagnostic (out of scope here).
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path=str(socket_path)),
            timeout=timeout,
        )
    except FileNotFoundError:
        return "FileNotFoundError"
    except ConnectionRefusedError:
        return "ConnectionRefusedError"
    except asyncio.TimeoutError:
        return f"TimeoutError after {int(timeout * 1000)} ms"
    except OSError as e:
        return f"OSError errno={e.errno}: {e.strerror or e}"
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass  # cleanup best-effort
    return None


def check_b_socket_fresh() -> CheckResult:
    """(b) socket file fresh.

    `~/.iai-mcp/.daemon.sock` (or IAI_DAEMON_SOCKET_PATH override) exists
    AND a kernel-level `connect()` succeeds within a generous 1-second
    window.

    Connect-only (no status round-trip). The previous implementation issued
    a `{type: status}` round-trip with a 250 ms wall, which empirically
    false-FAILed on a healthy daemon whose status handler can take 1-8 s
    to reply under normal WAKE load (20/20 FAIL on a daemon with PID alive,
    fsm_state=WAKE, accepting connections in 0.2 ms).

    Since the row name is literally "socket file fresh" and (a) already
    verifies the daemon PID + cmdline, a successful `connect()` is the
    correct signal here. Daemon-responsiveness diagnostics belong to a
    separate row (out of scope here; tracked as follow-up).
    """
    socket_path = _resolve_socket_path()
    if not socket_path.exists():
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"{socket_path} does not exist",
        )

    t0 = time.monotonic()
    try:
        err = asyncio.run(_socket_connect_probe(socket_path, timeout=1.0))
    except Exception as e:  # noqa: BLE001 — surface any unexpected probe failure
        logger.debug("check_b: socket probe failed: %s", e)
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"connect failed: {type(e).__name__}: {e}",
        )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if err is not None:
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"{socket_path} present but unreachable: {err}",
        )
    return CheckResult(
        "(b) socket file fresh",
        True,
        f"{socket_path} connected in {elapsed_ms} ms",
    )


def check_c_lock_healthy() -> CheckResult:
    """(c) lock file healthy.

    Probes the storage contention lock (``<root>/hippo/.lock``) — the real
    awake-path read/write lock. A held lock (the consolidation process holding
    exclusive, or an active recall holding shared) means a non-blocking shared
    acquire cannot be taken: that is HEALTHY, not broken. An acquirable lock
    means the store is idle: also HEALTHY. Only an OS-level error (permission,
    corrupt path) is a FAIL.

    The probe is read-only: it takes a shared lock non-blocking and releases it
    at once, never disturbing a live consolidation process and never touching
    the consolidation intent flag. It opens the file read-only (no create) so it
    cannot fabricate a spurious lock file.
    """
    import errno as _errno
    import fcntl as _fcntl

    lock_path = _resolve_hippo_db_path().parent / ".lock"
    if not lock_path.exists():
        # Fresh install / store never opened — not a failure.
        return CheckResult(
            "(c) lock file healthy",
            True,
            f"{lock_path} absent (store not yet initialized)",
        )
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
            _fcntl.flock(fd, _fcntl.LOCK_UN)  # release immediately
            return CheckResult(
                "(c) lock file healthy",
                True,
                f"{lock_path} acquirable (store idle)",
            )
        except OSError as exc:
            if exc.errno in (_errno.EAGAIN, _errno.EWOULDBLOCK):
                return CheckResult(
                    "(c) lock file healthy",
                    True,
                    f"{lock_path} held (consolidating or recall active — normal)",
                )
            raise
    except Exception as e:  # noqa: BLE001 — fcntl/OSError/permission all FAIL
        logger.debug("check_c: store-lock probe failed: %s", e)
        return CheckResult(
            "(c) lock file healthy",
            False,
            f"store-lock probe failed: {type(e).__name__}: {e}",
        )
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass  # cleanup best-effort


def check_d_no_orphan_core() -> CheckResult:
    """(d) zero orphan iai_mcp.core processes.

    NO `iai_mcp.core` processes should exist anywhere — wrappers spawn the
    singleton daemon, never a per-wrapper core. Any hit here is a stale
    process that wastes ~1.2 GB RSS and confuses cross-client memory.
    """
    try:
        import psutil

        orphans: list[int] = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
                if "iai_mcp.core" in cl:
                    orphans.append(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not orphans:
            return CheckResult(
                "(d) no orphan iai_mcp.core procs",
                True,
                "0 found",
            )
        return CheckResult(
            "(d) no orphan iai_mcp.core procs",
            False,
            f"{len(orphans)} found: PIDs {orphans}",
        )
    except Exception as e:  # noqa: BLE001 — psutil edge cases
        logger.debug("check_d: psutil probe failed: %s", e)
        return CheckResult(
            "(d) no orphan iai_mcp.core procs",
            False,
            f"psutil probe failed: {type(e).__name__}: {e}",
        )


def check_e_state_file_valid() -> CheckResult:
    """(e) daemon state file valid.

    `~/.iai-mcp/.daemon-state.json` either:
      - does not exist (daemon never booted — acceptable, NOT a bug); OR
      - parses as JSON AND `fsm_state` ∈ {WAKE, DROWSY, SLEEP, SLEEPING,
        DREAMING, HIBERNATION}.

    The whitelist mirrors the canonical lifecycle FSM in
    `src/iai_mcp/lifecycle.py` plus the legacy daemon.py per-REM-cycle
    `DREAMING` sub-state and the historical alias `SLEEPING`. The
    pair (`SLEEP`, `SLEEPING`) is preserved during the gradual rename
    between the two FSMs and the bridge module `fsm_reconcile.py`.
    """
    from iai_mcp.daemon_state import load_state

    try:
        state = load_state() or {}
    except Exception as e:  # noqa: BLE001 — corrupt JSON / IO error
        logger.debug("check_e: daemon state unreadable: %s", e)
        return CheckResult(
            "(e) daemon state file valid",
            False,
            f"unreadable: {type(e).__name__}: {e}",
        )

    fsm_state = state.get("fsm_state")
    if fsm_state is None:
        # No state file (or no fsm_state key) is acceptable when daemon has
        # never booted. A separate check (a) catches the "never booted but
        # should have" case.
        return CheckResult(
            "(e) daemon state file valid",
            True,
            "no state file (daemon never booted — not a bug)",
        )

    valid = {"WAKE", "DROWSY", "SLEEP", "SLEEPING", "DREAMING", "HIBERNATION"}
    if fsm_state in valid:
        return CheckResult(
            "(e) daemon state file valid",
            True,
            f"fsm_state={fsm_state}",
        )
    return CheckResult(
        "(e) daemon state file valid",
        False,
        f"fsm_state={fsm_state!r} not in {sorted(valid)}",
    )


def check_f_hippo_readable() -> CheckResult:
    """(f) hippo storage readable.

    Open a MemoryStore handle. The constructor opens the Hippo storage
    backend; if the file is corrupt / permission-denied / disk-full, the
    constructor raises and we report FAIL.

    Daemon-running carve-out: when the live daemon already holds the store,
    the exclusive open here fast-fails with a lock-held signal. That is the
    HEALTHY running state, not a defect — the store IS readable, the daemon
    simply has it. Mirroring the (c) lock probe, we report PASS in that case.
    A different open failure (corruption, permission, missing path) is still a
    genuine FAIL, and when the daemon is OFF the exclusive open still really
    verifies the store opens.
    """
    import sqlite3

    from iai_mcp.hippo import HippoLockHeldError

    try:
        from iai_mcp.store import MemoryStore

        MemoryStore()
        return CheckResult(
            "(f) hippo storage readable",
            True,
            "Hippo storage opens without error",
        )
    except HippoLockHeldError as e:
        # A live daemon holds the store — healthy, not broken.
        logger.debug("check_f: store held by running daemon: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            True,
            "store held by the live daemon — normal",
        )
    except sqlite3.OperationalError as e:
        # The shared SQLite connection can surface the daemon's lock as a
        # "database is locked" OperationalError on some platforms. Treat that
        # specific message as the healthy held state; any other operational
        # error is a real open failure.
        if "database is locked" in str(e).lower():
            logger.debug("check_f: store held by running daemon (sqlite): %s", e)
            return CheckResult(
                "(f) hippo storage readable",
                True,
                "store held by the live daemon — normal",
            )
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            False,
            f"open failed: {type(e).__name__}: {e}",
        )
    except Exception as e:  # noqa: BLE001 — surface any open failure
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            False,
            f"open failed: {type(e).__name__}: {e}",
        )


# -----------------------------------------------------------------------------
# multi-binder detection
# -----------------------------------------------------------------------------


def _extract_binder_pids(lsof_output: str, target_socket: Path) -> set[int]:
    """Parse lsof -F pn output. Format alternates lines:

       p<pid>
       n<filename>

    Each PID is followed by 0+ name entries until next p<pid>. Return the
    set of PIDs whose name == str(target_socket).

    Defense-in-depth helper for check_g_no_dup_binders. Pure parser, no I/O —
    accepts the captured stdout and returns the matching PID set.
    """
    pids: set[int] = set()
    current_pid: int | None = None
    target = str(target_socket)
    for line in lsof_output.splitlines():
        if line.startswith("p"):
            try:
                current_pid = int(line[1:])
            except ValueError:
                current_pid = None
        elif line.startswith("n") and current_pid is not None:
            name = line[1:]
            if name == target:
                pids.add(current_pid)
    return pids


def check_g_no_dup_binders() -> CheckResult:
    """(g) no duplicate processes bound to socket — TOCTOU race aftermath detector.

    Even with launchd as the only spawn vector in production,
    a user can manually `python -m iai_mcp.daemon` while one is already
    running. lsof -U reports all processes holding the AF_UNIX socket fd;
    if >1, we have a singleton-invariant violation that no other check
    catches (check_a inspects state.json:daemon_pid; a second daemon that
    never wrote state is invisible to check_a).

    lsof unavailable (rare on macOS, possible on minimal Linux) returns
    PASS-with-skip per the existing check_d_no_orphan_core pattern.
    """
    socket_path = _resolve_socket_path()
    if not socket_path.exists():
        return CheckResult(
            "(g) no dup binders",
            True,
            "no socket file (skip)",
        )
    try:
        # -U: AF_UNIX only; -F pn: machine-parseable, p-prefix=PID, n-prefix=name
        result = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return CheckResult(
            "(g) no dup binders",
            True,
            f"lsof unavailable: {e} (skip)",
        )
    binder_pids = _extract_binder_pids(result.stdout, socket_path)
    if len(binder_pids) <= 1:
        return CheckResult(
            "(g) no dup binders",
            True,
            f"{len(binder_pids)} binder(s)",
        )
    return CheckResult(
        "(g) no dup binders",
        False,
        f"{len(binder_pids)} processes bound to socket: {sorted(binder_pids)}",
    )


# -----------------------------------------------------------------------------
# file-backed crypto-key state check
# -----------------------------------------------------------------------------


def check_h_crypto_file_state() -> CheckResult:
    """Detect 'key file missing + Keychain entry exists' state.

    Detection matrix:
        | file present + valid | keyring entry | output |
        | yes                  | any           | PASS   |
        | no                   | yes           | WARN — `migrate-to-file` hint |
        | no                   | no/error      | PASS   (clean fresh-install state) |
        | yes (malformed)      | any           | FAIL   (CryptoKeyError detail)     |

    Imports of ``iai_mcp.crypto`` and ``keyring`` are LOCAL (function-scope)
    so the doctor module stays keyring-clean unless this check actually runs.
    Production daemon boot does NOT import ``keyring``; only the doctor's
    diagnostic-time probe does.

    WARN rows return ``passed=True`` (advisory only) — see ``CheckResult``
    docstring. The exit code stays 0 when only WARNs are present; ``cmd_doctor``
    prints a top-of-output remediation hint via ``_format_top_of_output_hint``.
    """
    # LOCAL imports keep the doctor module's footprint clean.
    from iai_mcp.crypto import CryptoKey, CryptoKeyError, SERVICE_NAME_DEFAULT

    ck = CryptoKey(user_id="default")
    path = ck._key_file_path()

    # Branch 1: file exists — validate via _try_file_get (mode + uid + length).
    if path.exists():
        try:
            ck._try_file_get()
            return CheckResult(
                "(h) crypto key file state",
                True,
                f"crypto key file present at {path} (mode 0o600, valid)",
                status="PASS",
            )
        except CryptoKeyError as exc:
            return CheckResult(
                "(h) crypto key file state",
                False,
                f"crypto key file is malformed: {exc}",
                status="FAIL",
            )

    # Branch 2: file missing — probe keyring for a pre-migration entry.
    # LOCAL imports here too: keyring is not imported at module top of
    # doctor.py.
    keyring_has_key = False
    keyring_probe_failed = False
    try:
        import keyring as _keyring
        import keyring.errors as _keyring_errors
    except ImportError:
        _keyring = None
        _keyring_errors = None  # type: ignore[assignment]

    if _keyring is not None:
        try:
            existing = _keyring.get_password(SERVICE_NAME_DEFAULT, "default")
            keyring_has_key = existing is not None
        except _keyring_errors.NoKeyringError:
            # No backend (Linux without Secret Service, etc.) — clean state.
            pass
        except _keyring_errors.KeyringError:
            # Backend exists but the read failed — could be ACL hang in a
            # non-interactive context. Mark as probe-failed; still emit a
            # WARN so the user is informed.
            keyring_probe_failed = True
        except Exception as e:  # noqa: BLE001 — defensive against keyring backend quirks
            logger.debug("check_h: keyring probe failed: %s", e)
            keyring_probe_failed = True

    if keyring_has_key:
        return CheckResult(
            "(h) crypto key file state",
            True,  # WARN does NOT flip exit code
            (
                f"crypto key file missing at {path}, but a Keychain entry was found.\n"
                f"  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key."
            ),
            status="WARN",
        )
    if keyring_probe_failed:
        return CheckResult(
            "(h) crypto key file state",
            True,  # WARN does NOT flip exit code
            (
                f"crypto key file missing at {path}; Keychain probe could not complete "
                f"(may indicate non-interactive context). If you have an existing Keychain key, "
                f"run `iai-mcp crypto migrate-to-file` from a Terminal."
            ),
            status="WARN",
        )

    # Branch 3: clean fresh-install state.
    return CheckResult(
        "(h) crypto key file state",
        True,
        (
            f"crypto key file absent at {path} and no Keychain entry detected. "
            f"Fresh install — run `iai-mcp crypto init` or set IAI_MCP_CRYPTO_PASSPHRASE."
        ),
        status="PASS",
    )


# -----------------------------------------------------------------------------
# Hippo storage diagnostic rows
# -----------------------------------------------------------------------------

# Expected schema_version for the current Hippo database layout.
_HIPPO_EXPECTED_SCHEMA_VERSION = "1"


def _resolve_hippo_db_path() -> Path:
    """Return the canonical path of brain.sqlite3 for the active store.

    Honors ``IAI_MCP_STORE`` env (test isolation + multi-tenant layout)
    before falling back to the default home-derived layout. Mirrors the
    resolution pattern in MemoryStore.__init__ so the doctor row inspects
    the SAME file the daemon would actually open.
    """
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "hippo" / "brain.sqlite3"


def check_i_hippo_db_size() -> CheckResult:
    """(i) hippo db size: report brain.sqlite3 file size in MB.

    Status thresholds:
      - PASS: size < 500 MB -- healthy steady state.
      - WARN: 500 <= size < 2048 MB -- recommend compaction.
      - FAIL: size >= 2048 MB -- run compaction immediately.

    Edge cases:
      - brain.sqlite3 absent (fresh install or no writes yet) -> PASS.
      - OSError while stat-ing -> WARN with the error class+message.

    INV-7 preserved: this check runs only when the user invokes
    ``iai-mcp doctor`` -- no background polling, no daemon-side work.
    """
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail="brain.sqlite3 not present yet (fresh install or no writes yet)",
            status="PASS",
        )
    try:
        size_bytes = db_path.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,  # WARN, not FAIL: probe failure is advisory.
            detail=f"stat failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 500:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=f"{size_mb:.1f} MB — healthy",
            status="PASS",
        )
    if size_mb < 2048:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,  # WARN -- advisory only.
            detail=(
                f"{size_mb:.1f} MB — consider "
                f"`iai-mcp maintenance compact-hippo --apply --yes`"
            ),
            status="WARN",
        )
    return CheckResult(
        name="(i) hippo db size",
        passed=False,
        detail=f"{size_mb:.1f} MB — run compaction immediately",
        status="FAIL",
    )


# -----------------------------------------------------------------------------
# daemon wake/sleep cycle diagnostic rows
# -----------------------------------------------------------------------------


def _resolve_wrappers_dir() -> Path:
    """Return the canonical path of the wrapper heartbeat directory.

    Honors ``IAI_MCP_STORE`` env (test isolation + multi-tenant layout)
    before falling back to ``~/.iai-mcp``.
    The heartbeat scanner watches ``<root>/wrappers/`` for the per-wrapper
    ``heartbeat-<pid>-<uuid>.json`` files written by the MCP wrapper.
    """
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "wrappers"


def check_m_heartbeat_scanner() -> CheckResult:
    """(m) heartbeat scanner health: PASS unless the wrappers dir is unreadable.

    The daemon's heartbeat scanner aggregates
    per-wrapper heartbeat files in ``~/.iai-mcp/wrappers/`` to decide WAKE
    vs. BEDTIME. This row surfaces the current per-status breakdown so the
    user can see at a glance whether stale / orphan files are accumulating.

    Status rules:
      - PASS: wrappers dir absent (fresh install) OR scan succeeds.
      - FAIL: wrappers dir exists but cannot be enumerated (permission /
        FUSE error). The probe failure is reported with the error class so
        the user can correct the underlying filesystem issue.

    Display: ``"n=3 fresh, 1 stale, 0 orphan"``. STALE / ORPHAN counts are
    reported even though they are advisory — they indicate to the user that
    a wrapper crashed without cleaning up, which is a benign but
    diagnostically interesting state.
    """
    from iai_mcp.heartbeat_scanner import HeartbeatScanner, HeartbeatStatus

    wrappers_dir = _resolve_wrappers_dir()
    if not wrappers_dir.exists():
        return CheckResult(
            name="(m) heartbeat scanner",
            passed=True,
            detail=(
                f"{wrappers_dir} not present yet (fresh install or no "
                "wrapper has refreshed yet)"
            ),
            status="PASS",
        )

    scanner = HeartbeatScanner(wrappers_dir)
    try:
        entries = scanner.scan()
    except OSError as exc:
        return CheckResult(
            name="(m) heartbeat scanner",
            passed=False,
            detail=(
                f"could not scan {wrappers_dir}: "
                f"{type(exc).__name__}: {exc}"
            ),
            status="FAIL",
        )

    fresh = sum(1 for e in entries if e.status is HeartbeatStatus.FRESH)
    stale = sum(1 for e in entries if e.status is HeartbeatStatus.STALE)
    orphan = sum(1 for e in entries if e.status is HeartbeatStatus.ORPHAN)
    return CheckResult(
        name="(m) heartbeat scanner",
        passed=True,
        detail=f"n={fresh} fresh, {stale} stale, {orphan} orphan",
        status="PASS",
    )


def _resolve_lifecycle_state_path() -> Path:
    """Return the path of ``lifecycle_state.json`` honoring IAI_MCP_STORE.

    Mirrors the pattern in ``_resolve_wrappers_dir`` so the doctor rows
    behave consistently with the heartbeat-scanner row when
    the user runs under a non-default store path.
    """
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "lifecycle_state.json"


def _resolve_lifecycle_log_dir() -> Path:
    """Return the directory of lifecycle event-log JSONL files."""
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "logs"


def _format_relative_short(ts_iso: str, *, now: Any = None) -> str:
    """Return a short elapsed-string ("12 min", "3 h", "2 d") for a UTC ts.

    Doctor uses a tighter format than `cli._format_relative` because each
    row prints on a single 80-col line; the trailing units stay singular
    ("min" not "minutes") to keep the alignment tight.
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    try:
        ts = _dt.fromisoformat(ts_iso)
    except (TypeError, ValueError):
        return "?"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_tz.utc)
    moment = now if now is not None else _dt.now(_tz.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_tz.utc)
    seconds = int((moment - ts).total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} h"
    days = hours // 24
    return f"{days} d"


def check_j_lifecycle_current_state() -> CheckResult:
    """(j) lifecycle current state.

    Reads ``lifecycle_state.json`` and reports the current state plus how
    long the daemon has been in it. Always PASS — informational row. The
    state file self-heals on missing/corrupt content (returns default WAKE),
    so this row never fails on a fresh install.
    """
    from iai_mcp.lifecycle_state import load_state

    state_path = _resolve_lifecycle_state_path()
    record = load_state(state_path)
    current = record.get("current_state", "WAKE")
    since_ts = record.get("since_ts", "?")
    elapsed = _format_relative_short(since_ts)
    shadow_run = record.get("shadow_run", True)

    detail = f"{current} since {elapsed} (shadow_run={'true' if shadow_run else 'false'})"
    return CheckResult(
        name="(j) lifecycle current state",
        passed=True,
        detail=detail,
        status="PASS",
    )


def check_k_lifecycle_history_24h() -> CheckResult:
    """(k) lifecycle history 24h.

    Counts state-transition events in today's + yesterday's lifecycle
    event-log JSONL files, broken down by Wake/Sleep cycles. INFO row —
    always PASS.

    Implementation: parse ``lifecycle-events-YYYY-MM-DD.jsonl`` for
    today + yesterday (UTC), filter ``event=='state_transition'``,
    aggregate counts. Files absent / unparseable -> "0 transitions"
    rather than failure. The 24h window is approximate (UTC-day-bucket
    so a transition at 23:59 yesterday + 00:01 today is a 2-event
    window); precise sliding 24h is not needed for the operator
    summary.
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    log_dir = _resolve_lifecycle_log_dir()
    if not log_dir.exists():
        return CheckResult(
            name="(k) lifecycle history 24h",
            passed=True,
            detail="no event log yet (fresh install or daemon never run)",
            status="PASS",
        )

    log = LifecycleEventLog(log_dir=log_dir)
    now = _dt.now(_tz.utc)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - _td(days=1)).strftime("%Y-%m-%d")

    transitions: list[dict[str, Any]] = []
    for date_str in (yesterday, today):
        try:
            events = log.read_all(date_str=date_str)
        except OSError:
            continue
        for ev in events:
            if ev.get("event") == "state_transition":
                transitions.append(ev)

    # Bucket transitions by destination state for a quick summary.
    counts: dict[str, int] = {}
    for ev in transitions:
        to = ev.get("to") or "?"
        counts[to] = counts.get(to, 0) + 1

    if not transitions:
        return CheckResult(
            name="(k) lifecycle history 24h",
            passed=True,
            detail="0 transitions in last 24h",
            status="PASS",
        )

    summary = ", ".join(f"{state}={n}" for state, n in sorted(counts.items()))
    return CheckResult(
        name="(k) lifecycle history 24h",
        passed=True,
        detail=f"{len(transitions)} transitions ({summary})",
        status="PASS",
    )


def check_l_sleep_cycle_status() -> CheckResult:
    """(l) sleep cycle quarantine status.

    Reads ``lifecycle_state.json.quarantine`` sub-record. Status rules:

      - PASS: ``quarantine`` is None / absent (sleep pipeline healthy).
      - PASS: ``quarantine`` present but ``until_ts`` already in the
        past (auto-recovery will clear it on next ``run()``).
      - WARN: ``quarantine`` active for less than 12 hours.
      - FAIL: ``quarantine`` active 12 hours or more (operator should
        run ``iai-mcp maintenance sleep-cycle --reset-quarantine``).
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import load_state

    state_path = _resolve_lifecycle_state_path()
    record = load_state(state_path)
    quarantine = record.get("quarantine")
    if quarantine is None:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=True,
            detail="no quarantine active",
            status="PASS",
        )

    reason = quarantine.get("reason", "?")
    until_ts = quarantine.get("until_ts", "?")
    since_ts = quarantine.get("since_ts", "?")

    # Compute age since quarantine entered.
    now = _dt.now(_tz.utc)
    try:
        since = _dt.fromisoformat(since_ts)
        if since.tzinfo is None:
            since = since.replace(tzinfo=_tz.utc)
        age_hours = (now - since).total_seconds() / 3600.0
    except (TypeError, ValueError):
        age_hours = 0.0

    # Auto-recovery branch: until_ts already in the past.
    try:
        until = _dt.fromisoformat(until_ts)
        if until.tzinfo is None:
            until = until.replace(tzinfo=_tz.utc)
        expired = until <= now
    except (TypeError, ValueError):
        expired = False

    if expired:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=True,
            detail=(
                f"quarantine expired (until={until_ts}); will clear on next "
                f"sleep-cycle run; reason={reason}"
            ),
            status="PASS",
        )

    detail = (
        f"quarantined for {age_hours:.1f}h; until={until_ts}; reason={reason}"
    )

    if age_hours >= 12.0:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=False,
            detail=(
                f"{detail}; run `iai-mcp maintenance sleep-cycle "
                "--reset-quarantine` to clear"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(l) sleep cycle quarantine",
        passed=True,  # WARN is advisory; does not flip exit code.
        detail=detail,
        status="WARN",
    )


def check_n_hid_idle_source() -> CheckResult:
    """(n) HID idle source health: PASS if HIDIdleTime present, WARN if not.

    Reports which hardware-grounded idle signals are reachable on the
    current host. ``HIDIdleTime`` (via
    ``ioreg -c IOHIDSystem``) is the primary signal; ``pmset -g log`` is
    the secondary System/Display Sleep event source.

    Status rules:
      - PASS: ``available_signals`` includes ``"HIDIdleTime"``.
      - WARN: signal list empty (will fall back to heartbeat-only L6 — the
        daemon stays correct but loses the hardware backstop). Advisory
        only — does NOT flip the doctor exit code (mirrors check_i WARN).

    Display includes the current ``HIDIdleTime`` value and pmset state so
    the user can see what the L6 sleep predicate is evaluating right now.
    """
    from iai_mcp.idle_detector import IdleDetector

    detector = IdleDetector()
    status = detector.status()

    hid_str = (
        f"{status.hid_idle_sec}s"
        if status.hid_idle_sec is not None
        else "unavailable"
    )
    pmset_str = "recent-sleep" if status.pmset_recent_sleep else "clean"
    signals_str = (
        ",".join(status.available_signals) if status.available_signals else "none"
    )
    detail = (
        f"HIDIdleTime: {hid_str}, pmset: {pmset_str}, available: {signals_str}"
    )

    if "HIDIdleTime" in status.available_signals:
        return CheckResult(
            name="(n) HID idle source",
            passed=True,
            detail=detail,
            status="PASS",
        )
    return CheckResult(
        name="(n) HID idle source",
        passed=True,  # WARN — advisory only, does not flip exit code.
        detail=(
            f"{detail}; L6 will fall back to heartbeat-idle only"
        ),
        status="WARN",
    )


def check_w_no_permanent_failed() -> CheckResult:
    """(w) no permanent-failed captures: WARN when terminal capture files exist.

    Scans the deferred-captures directory for .permanent-failed-*.jsonl files.
    These are terminal evidence files created when earlier drain passes failed
    repeatedly. Their presence means captured user turns were not ingested and
    may be recoverable via ``iai-mcp drain-permanent-failed``.

    Status rules:
      - PASS: directory absent or zero .permanent-failed-*.jsonl files.
      - WARN: one or more files found — advisory; data is not lost but
        operator action is recommended to complete recovery.

    This check is pure filesystem — no MemoryStore open, no socket.
    Honors ``IAI_MCP_STORE`` env override for test isolation.
    """
    import fnmatch

    env_store = os.environ.get("IAI_MCP_STORE")
    if env_store:
        deferred_dir = Path(env_store).parent / ".deferred-captures"
    else:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"

    if not deferred_dir.exists():
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="deferred-captures dir absent — nothing to recover",
        )

    count = 0
    try:
        for entry in os.scandir(deferred_dir):
            if entry.is_file() and fnmatch.fnmatch(entry.name, "*.permanent-failed-*.jsonl"):
                count += 1
    except OSError as exc:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail=f"could not scan deferred-captures dir: {exc}",
            status="WARN",
        )

    if count == 0:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="No permanent-failed capture files",
        )
    return CheckResult(
        name="(w) no permanent-failed captures",
        passed=True,
        detail=(
            f"{count} permanent-failed capture file(s) — "
            "run 'iai-mcp drain-permanent-failed' to recover"
        ),
        status="WARN",
    )


# Direct CPU-feature probe so the row is correct even when `import lancedb`
# would SIGILL on an illegal opcode. has_avx2() is imported inside the
# function so tests can monkeypatch iai_mcp.cpu_features.has_avx2 against
# this binding.
def check_z_avx2_support() -> CheckResult:
    """(z) AVX2 CPU support: PASS if AVX2 is present (or N/A on ARM/unknown).

    Consults cpu_features.has_avx2() directly so the row is correct even on
    a host where ``import lancedb`` would SIGILL. Falls back to
    ``iai_mcp.store.CPU_HAS_AVX2`` only if the cpu_features probe itself
    raises.

    Status rules:
      - PASS: has_avx2() True (or ARM Mac / unknown-platform fallback).
      - FAIL: has_avx2() False -- "this host lacks AVX2 -- LanceDB cannot
        load; iai-mcp memory store is unavailable. Deploy on an AVX2-
        equipped host (any Intel CPU 2013+; AMD Excavator 2015+;
        Mac M-series ARM is unaffected)."
    """
    from iai_mcp.cpu_features import has_avx2

    try:
        avx2_ok = has_avx2()
    except Exception as exc:  # noqa: BLE001 -- defensive against probe quirks
        # Secondary fallback: store.CPU_HAS_AVX2 reflects whether `import
        # lancedb` itself succeeded as a Python-level operation. A False
        # there is a positive signal that lancedb cannot load; default to
        # True (benefit of the doubt) if even that import is unreachable.
        try:
            from iai_mcp.store import CPU_HAS_AVX2
            avx2_ok = CPU_HAS_AVX2
        except Exception:  # noqa: BLE001 -- store may itself be unimportable
            avx2_ok = True
        logger.debug(
            "check_z: has_avx2() probe failed: %s; fallback=%s",
            exc,
            avx2_ok,
        )

    if avx2_ok:
        return CheckResult(
            name="(z) AVX2 CPU support",
            passed=True,
            detail="AVX2 available (or N/A on this architecture)",
            status="PASS",
        )
    return CheckResult(
        name="(z) AVX2 CPU support",
        passed=False,
        detail=(
            "this host lacks AVX2 -- LanceDB cannot load; iai-mcp memory "
            "store is unavailable. Deploy on an AVX2-equipped host (any "
            "Intel CPU 2013+; AMD Excavator 2015+; Mac M-series ARM is "
            "unaffected)."
        ),
        status="FAIL",
    )


def _format_top_of_output_hint(results: list[CheckResult]) -> str | None:
    """Return a `> hint:` line for any WARN row from check_h, else None.

    The migration remediation surfaces at the TOP of doctor's output (above
    the row-by-row print) so a user running ``iai-mcp doctor`` after
    upgrading from a Keychain-backed install sees the fix BEFORE they hit
    the eight-row checklist.

    The detail of the WARN row is multi-line (first line = state description,
    second line = actionable command). The hint flattens both lines into a
    single output line so the actionable command is visible at the top —
    a one-liner that omits the command would be useless.
    """
    for r in results:
        if r.name == "(h) crypto key file state" and r.status == "WARN":
            # Flatten the multi-line detail into a single hint line — strip
            # leading whitespace so the actionable command does not lose
            # readability when concatenated.
            flat = " ".join(line.strip() for line in r.detail.splitlines() if line.strip())
            return f"> hint: {flat}"
    return None


# Headless host helpers. Auto-detect is gated to Linux because macOS
# (Quartz) does not set DISPLAY / WAYLAND_DISPLAY -- applying the
# auto-detect verbatim would mask the IdleDetector signal on every Mac
# desktop, including hosts where HIDIdleTime reads fine. The explicit
# `--headless` flag forces headless mode on any host.

_HEADLESS_DOWNGRADE_ROWS: frozenset[str] = frozenset({
    "(b) socket file fresh",
    "(n) HID idle source",
})


def is_headless(*, force: bool = False) -> bool:
    """Return True iff the host has no display server reachable.

    Linux auto-detect: both ``DISPLAY`` and ``WAYLAND_DISPLAY`` unset.
    macOS / other: only True if ``force`` is True (the explicit
    ``--headless`` flag was passed).
    """
    if force:
        return True
    if platform.system() != "Linux":
        return False
    return (
        os.environ.get("DISPLAY") is None
        and os.environ.get("WAYLAND_DISPLAY") is None
    )


def _apply_headless_downgrade(
    results: list[CheckResult], headless: bool
) -> list[CheckResult]:
    """Downgrade FAIL -> WARN for (b) socket fresh + (n) HID idle source.

    Pure-function helper exposed for unit testing. Only mutates rows
    listed in ``_HEADLESS_DOWNGRADE_ROWS`` and only those currently
    FAILing -- PASS rows stay PASS; rows already at WARN stay WARN. The
    downgraded WARN inherits the original FAIL ``detail`` so the user
    still sees the underlying probe message (e.g., "<path> present but
    unreachable").

    Returns the same list (in-place mutation) so callers can chain.
    """
    if not headless:
        return results
    for r in results:
        if r.name in _HEADLESS_DOWNGRADE_ROWS and r.status == "FAIL":
            r.passed = True
            r.status = "WARN"
    return results


def check_o_subscription_credentials() -> CheckResult:
    """(o) Claude subscription credentials are present + non-expired.

    The daemon's nightly REM consolidation invokes `claude -p`
    through the user's existing `~/.claude/.credentials.json` OAuth blob.
    This row surfaces three failure modes BEFORE the next REM cycle:

      - credentials_file_missing: the user never ran `claude /login`, or
        Claude Code was uninstalled. Daemon falls back to Tier-0 statistical
        consolidation; LLM critic + nightly insight are silently skipped.
      - not_subscription: the auth blob carries a non-subscription
        `subscriptionType` (e.g. a community / unknown tier). Same
        fallback; recoverable by upgrading the Claude.ai plan.
      - credentials_expired / missing_inference_scope: refresh window
        lapsed or the token was minted without the `user:inference` scope.
        The next `claude -p` would hang or 401; fail-fast here.

    PASS: credentials are valid + non-expired + scoped for inference.
    WARN: missing / expired / wrong tier (daemon still runs in Tier-0).
    """
    try:
        from iai_mcp.claude_cli import verify_credentials_subscription
    except Exception as exc:  # noqa: BLE001 -- defensive
        return CheckResult(
            name="(o) Claude subscription credentials",
            passed=True,
            detail=f"unable to import claude_cli ({exc}); skipping",
            status="WARN",
        )

    result = verify_credentials_subscription()
    if result.get("ok"):
        sub_type = result.get("subscription_type") or result.get("billing_type") or "unknown"
        return CheckResult(
            name="(o) Claude subscription credentials",
            passed=True,
            detail=f"valid {sub_type} subscription with inference scope",
            status="PASS",
        )

    reason = result.get("reason", "unknown")
    return CheckResult(
        name="(o) Claude subscription credentials",
        passed=True,  # WARN is advisory — Tier-0 fallback still works
        detail=(
            f"reason={reason}; daemon will fall back to local Tier-0 "
            "consolidation (no LLM critic, no nightly insight). Run "
            "`claude /login` to restore subscription path."
        ),
        status="WARN",
    )


def check_q_iai_cli_reachable() -> CheckResult:
    """(q) `iai` user-facing CLI installed + reachable.

    Checks `iai --version` exits 0. PASS confirms the entry point landed
    in PATH after the most recent `pip install -e .`. WARN when the
    binary is absent (likely a stale install missing the entry
    point); informational, since the daemon and `iai-mcp` operator CLI
    are unaffected.
    """
    import shutil

    iai_path = shutil.which("iai")
    if iai_path is None:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,  # advisory only -- daemon is unaffected
            detail=(
                "iai not in PATH. Re-run `pip install -e .` from the repo "
                "root to register the entry point."
            ),
            status="WARN",
        )

    try:
        completed = subprocess.run(  # noqa: S603 -- argv list, no shell
            [iai_path, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=f"iai found at {iai_path} but invocation failed: {exc}",
            status="WARN",
        )

    if completed.returncode != 0:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=(
                f"iai --version exited {completed.returncode}: "
                f"{completed.stderr.strip()[:120]}"
            ),
            status="WARN",
        )

    version_line = (completed.stdout or completed.stderr).strip().splitlines()[0:1]
    version = version_line[0] if version_line else "?"
    return CheckResult(
        name="(q) iai CLI reachable",
        passed=True,
        detail=f"{iai_path} -> {version}",
        status="PASS",
    )


def check_r_hippo_hnsw_loadable() -> CheckResult:
    """(r) hippo hnsw index loadable.

    Probes the records.hnsw file: presence check, zero-byte guard, and
    hnswlib.Index load. Status rules:
      - PASS: file loads without error.
      - WARN: file absent (HippoDB rebuilds the index from SQLite on next boot).
      - FAIL: file is zero bytes (corrupt; rebuild needed) OR hnswlib.load_index
        raises.

    INV-7 preserved: this check runs only on user request.
    """
    hnsw_path = _resolve_hippo_db_path().parent / "records.hnsw"
    if not hnsw_path.exists():
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=True,  # advisory only — rebuild is automatic on next boot.
            detail="records.hnsw absent (HippoDB rebuilds from SQLite on next boot)",
            status="WARN",
        )
    try:
        size = hnsw_path.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=f"stat failed: {type(exc).__name__}: {exc}",
            status="FAIL",
        )
    if size == 0:
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=(
                "records.hnsw is zero bytes (corrupt; rebuild needed — "
                "restart the daemon to trigger automatic rebuild)"
            ),
            status="FAIL",
        )
    try:
        import hnswlib as _hnswlib
        from iai_mcp.types import EMBED_DIM

        idx = _hnswlib.Index(space="cosine", dim=EMBED_DIM)
        idx.load_index(str(hnsw_path), max_elements=0)
    except Exception as exc:  # noqa: BLE001 — surface any load failure
        logger.debug("check_r: hnswlib.load_index failed: %s", exc)
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=(
                f"hnswlib.load_index failed: {type(exc).__name__}: {exc} "
                "(restart the daemon to trigger automatic rebuild)"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(r) hippo hnsw index",
        passed=True,
        detail=f"{size / (1024 * 1024):.1f} MB",
        status="PASS",
    )


def check_s_hippo_schema_version() -> CheckResult:
    """(s) hippo schema version matches expected.

    Opens brain.sqlite3 directly via sqlite3.connect (lightweight, no
    embedder load) to read _hippo_meta.schema_version. Compares to
    _HIPPO_EXPECTED_SCHEMA_VERSION. WARN on mismatch; FAIL on read error.

    The sqlite3 probe uses WAL reader semantics (read-only SELECT) so it
    cannot interfere with a live daemon writer. timeout=2.0 prevents hangs
    against a VACUUM-locked database.

    Status rules:
      - PASS: db absent (fresh install) OR schema_version matches expected.
      - WARN: schema_version present but mismatched.
      - FAIL: sqlite3.Error on connect/query OR _hippo_meta missing the row.

    INV-7 preserved: this check runs only on user request.
    """
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,
            detail="db absent (fresh install)",
            status="PASS",
        )
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        row = conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error as exc:
        return CheckResult(
            name="(s) hippo schema version",
            passed=False,
            detail=f"sqlite3 query failed: {type(exc).__name__}: {exc}",
            status="FAIL",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if row is None:
        return CheckResult(
            name="(s) hippo schema version",
            passed=False,
            detail="_hippo_meta missing schema_version row",
            status="FAIL",
        )
    value = str(row[0])
    expected = _HIPPO_EXPECTED_SCHEMA_VERSION
    if value != expected:
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,  # advisory WARN — daemon can still run.
            detail=f"schema_version={value} (expected {expected})",
            status="WARN",
        )
    return CheckResult(
        name="(s) hippo schema version",
        passed=True,
        detail=f"schema_version={value}",
        status="PASS",
    )


def check_t_hippo_compacted_freshness() -> CheckResult:
    """(t) hippo_compacted event freshness: WARN if no compaction in last 24h.

    Queries the events store for the most recent ``hippo_compacted`` event.
    Status rules:
      - PASS: event found and within 24 hours.
      - WARN: no event found (store fresh or daemon never ran compaction), OR
        most recent event is older than 24 hours.

    This is an advisory-only check — compaction is not required for
    correctness; it is a storage hygiene signal.

    INV-7 preserved: this check runs only on user request.
    """
    import sqlite3
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.hippo import HippoLockHeldError

    try:
        from iai_mcp.events import query_events
        from iai_mcp.store import MemoryStore

        store = MemoryStore()
        events = query_events(store, kind="hippo_compacted", limit=1)
    except HippoLockHeldError as exc:
        # The live daemon holds the store; this advisory probe is simply
        # deferred — a benign normal state, not a problem to flag.
        logger.debug("check_t: store held by running daemon: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="deferred — daemon holds the store (normal)",
            status="PASS",
        )
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.debug("check_t: store held by running daemon (sqlite): %s", exc)
            return CheckResult(
                name="(t) hippo_compacted freshness",
                passed=True,
                detail="deferred — daemon holds the store (normal)",
                status="PASS",
            )
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,  # WARN only — cannot query events.
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,  # WARN only — cannot query events.
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )

    if not events:
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,  # WARN only — no compaction recorded yet.
            detail="no hippo_compacted event found (fresh install or compaction not yet run)",
            status="WARN",
        )

    # Most recent event is events[0] (query returns newest-first).
    last_event = events[0]
    ts_str = last_event.get("timestamp") or last_event.get("ts") or ""
    try:
        ts = _dt.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_tz.utc)
        now = _dt.now(_tz.utc)
        age_hours = (now - ts).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="last hippo_compacted event timestamp unparseable",
            status="WARN",
        )

    if age_hours <= 24.0:
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail=f"last hippo_compacted event {age_hours:.1f}h ago",
            status="PASS",
        )
    return CheckResult(
        name="(t) hippo_compacted freshness",
        passed=True,  # WARN — advisory; compaction not a hard requirement.
        detail=(
            f"last hippo_compacted event {age_hours:.1f}h ago "
            f"(consider `iai-mcp maintenance compact-hippo --apply --yes`)"
        ),
        status="WARN",
    )


def check_u_recall_centrality_regression() -> CheckResult:
    """(u) recall centrality regression: WARN if 24h median centrality_ms > 30ms.

    Reads ``recall_timing`` events emitted by ``_recall_core`` at a tunable
    sample rate (``IAI_MCP_RECALL_SAMPLE_RATE``, default 0.1 = 1-in-10).
    Threshold 30ms is the acceptance gate for the centrality hot-path
    measured against the post-Hippo baseline.

    Status rules:
      - PASS: events present AND median centrality_ms <= 30ms.
      - WARN: no events in last 24h (daemon idle or sampling missed),
              OR median centrality_ms > 30ms (also emits a
              ``health_concern`` event for downstream operator audit).

    Advisory-only — centrality is a ranking signal, not a correctness
    invariant. INV-7 preserved: this check runs only on user request.
    """
    import sqlite3
    import statistics
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.hippo import HippoLockHeldError

    try:
        from iai_mcp.events import query_events, write_event
        from iai_mcp.store import MemoryStore

        store = MemoryStore()
        since = _dt.now(_tz.utc) - _td(hours=24)
        events = query_events(
            store, kind="recall_timing", since=since, limit=1000
        )
    except HippoLockHeldError as exc:
        # The live daemon holds the store; this advisory probe is simply
        # deferred — a benign normal state, not a problem to flag.
        logger.debug("check_u: store held by running daemon: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="deferred — daemon holds the store (normal)",
            status="PASS",
        )
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.debug("check_u: store held by running daemon (sqlite): %s", exc)
            return CheckResult(
                name="(u) recall centrality regression",
                passed=True,
                detail="deferred — daemon holds the store (normal)",
                status="PASS",
            )
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,  # WARN only — cannot query events.
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,  # WARN only — cannot query events.
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )

    if not events:
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="no recall_timing events in last 24h (daemon idle or sampling missed)",
            status="WARN",
        )

    centrality_values: list[float] = []
    for ev in events:
        payload = ev.get("data") or {}
        cv = payload.get("centrality_ms")
        if cv is None:
            continue
        try:
            centrality_values.append(float(cv))
        except (TypeError, ValueError):
            continue
    if not centrality_values:
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="recall_timing events present but centrality_ms missing/invalid",
            status="WARN",
        )

    median_ms = statistics.median(centrality_values)
    if median_ms > 30.0:
        # Emit a health_concern event so downstream operators can audit;
        # best-effort, must never break the doctor row itself.
        try:
            write_event(
                store,
                kind="health_concern",
                data={"centrality_median_ms": float(median_ms)},
                severity="warning",
            )
        except Exception as exc:  # noqa: BLE001 — telemetry best-effort
            logger.debug("check_u: health_concern emit failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,  # WARN only — advisory; recall ranking still works.
            detail=(
                f"centrality_ms median {median_ms:.1f}ms > 30ms threshold "
                f"(n_events={len(centrality_values)})"
            ),
            status="WARN",
        )
    return CheckResult(
        name="(u) recall centrality regression",
        passed=True,
        detail=(
            f"centrality_ms median {median_ms:.1f}ms <= 30ms "
            f"(n_events={len(centrality_values)})"
        ),
        status="PASS",
    )


def check_v_native_embedder() -> CheckResult:
    """(v) native Rust embedder: import, smoke-encode, backend==rust assertion.

    Probes the Rust native extension at runtime: imports iai_mcp_native,
    instantiates the production Embedder, runs one tiny encode, and asserts
    384-d finite output and active backend == rust.

    PASS: encode returns a 384-d finite vector and _backend == "rust".
    FAIL: import fails (wheel absent or broken), backend assertion fails, or
          encode produces unexpected output — detail includes a rebuild hint.
    """
    import math

    try:
        import iai_mcp_native  # noqa: F401
        from iai_mcp.embed import Embedder

        emb = Embedder()
        assert emb._backend == "rust", f"backend={emb._backend!r}"
        vec = emb.embed("smoke")
        assert len(vec) == 384, f"expected 384 dims, got {len(vec)}"
        assert all(math.isfinite(float(x)) for x in vec[:3]), (
            "non-finite values in output"
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="(v) native Rust embedder",
            passed=False,
            detail=(
                f"{type(exc).__name__}: {exc} — rebuild with: "
                "cd rust/iai_mcp_native && maturin develop --release"
            ),
        )
    return CheckResult(
        name="(v) native Rust embedder",
        passed=True,
        detail="encode ok, backend=rust, 384-dim",
    )


def check_p_anthropic_sdk_absent() -> CheckResult:
    """(p) anthropic SDK is NOT importable.

    The `anthropic>=0.40.0` package was dropped from `pyproject.toml`
    runtime dependencies. A fresh `pip install -e .` will not install the
    SDK. This row catches the stale-install case where an older venv still
    carries `anthropic` site-packages from a prior install — the
    daemon does not USE the SDK, but its presence is a sanity-check
    failure that operators can resolve with `pip uninstall anthropic`.

    PASS: ImportError on `import anthropic` (clean install).
    WARN: SDK importable (stale site-packages; daemon is fine but the
    cleanup recommendation surfaces here).
    """
    try:
        import anthropic  # noqa: F401 -- presence-probe only
        return CheckResult(
            name="(p) anthropic SDK absent",
            passed=True,  # advisory only — daemon does not use the SDK
            detail=(
                "anthropic SDK is importable in this venv. It was dropped "
                "as a runtime dependency; this is likely leftover site-packages "
                "from an older install. Run `pip uninstall anthropic` "
                "to clean up."
            ),
            status="WARN",
        )
    except ImportError:
        return CheckResult(
            name="(p) anthropic SDK absent",
            passed=True,
            detail="ImportError as expected (subscription-only path)",
            status="PASS",
        )


def run_diagnosis() -> list[CheckResult]:
    """Execute all checks in order, returning the result list.

    Final order: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, z
    Total: 23 rows.
    """
    return [
        check_a_daemon_alive(),
        check_b_socket_fresh(),
        check_c_lock_healthy(),
        check_d_no_orphan_core(),
        check_e_state_file_valid(),
        check_f_hippo_readable(),
        check_g_no_dup_binders(),
        check_h_crypto_file_state(),
        check_i_hippo_db_size(),
        # Lifecycle visibility rows.
        check_j_lifecycle_current_state(),
        check_k_lifecycle_history_24h(),
        check_l_sleep_cycle_status(),
        # Daemon wake/sleep cycle rows.
        check_m_heartbeat_scanner(),
        check_n_hid_idle_source(),
        # Subscription-path rows.
        check_o_subscription_credentials(),
        check_p_anthropic_sdk_absent(),
        # User-facing iai CLI reachability.
        check_q_iai_cli_reachable(),
        # Hippo storage health rows.
        check_r_hippo_hnsw_loadable(),
        check_s_hippo_schema_version(),
        check_t_hippo_compacted_freshness(),
        # Recall-latency regression telemetry.
        check_u_recall_centrality_regression(),
        # Native Rust embedder runtime verification.
        check_v_native_embedder(),
        # Permanent-failed capture file accumulation.
        check_w_no_permanent_failed(),
        # AVX2 graceful diagnostic stays last.
        check_z_avx2_support(),
    ]


def print_checklist(results: list[CheckResult]) -> None:
    """Print the PASS/WARN/FAIL checklist row by row."""
    print("iai doctor — daemon health check\n")
    for r in results:
        # WARN tag is distinct from PASS/FAIL so the user sees the advisory
        # state at a glance.
        if r.status == "WARN":
            tag = "[WARN]"
        elif r.passed:
            tag = "[PASS]"
        else:
            tag = "[FAIL]"
        print(f"  {tag} {r.name:<40} {r.detail}")


# -----------------------------------------------------------------------------
# Repair actions
# -----------------------------------------------------------------------------


def _kill_orphan_cores() -> tuple[bool, str, int]:
    """Action 1: SIGTERM every iai_mcp.core process (verified by cmdline match).

    Wrong-PID-kill mitigation: only kills processes whose psutil cmdline
    contains the literal substring 'iai_mcp.core'. A recycled PID belonging
    to an unrelated process is skipped (its cmdline differs).
    """
    import psutil

    t0 = time.monotonic()
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.core" not in cl:
                continue
            pid = p.info["pid"]
            # Wrong-PID-kill mitigation: cmdline is verified above; signal
            # the live PID. SIGTERM (not SIGKILL) gives the core a chance to
            # finalize any in-flight store writes.
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except OSError as e:
            failed.append((p.info.get("pid", -1), str(e)))
    duration_ms = int((time.monotonic() - t0) * 1000)
    if failed:
        return (
            False,
            f"killed {len(killed)} ({killed}); FAILED on {failed}",
            duration_ms,
        )
    return True, f"killed {len(killed)} orphan(s): {killed}", duration_ms


def _unlink_stale_socket() -> tuple[bool, str, int]:
    """Action 2: unlink ~/.iai-mcp/.daemon.sock (or env-resolved path) if present.

    C4 CLEAN UNINSTALL: doctor only unlinks the socket file. Lock file +
    state file are owned by `iai-mcp daemon uninstall`.
    """
    socket_path = _resolve_socket_path()
    t0 = time.monotonic()
    if not socket_path.exists():
        return True, "no stale socket to unlink", int((time.monotonic() - t0) * 1000)
    try:
        socket_path.unlink()
        return True, f"unlinked {socket_path}", int((time.monotonic() - t0) * 1000)
    except OSError as e:
        return False, f"unlink failed: {e}", int((time.monotonic() - t0) * 1000)


def _respawn_daemon() -> tuple[bool, str, int]:
    """Action 3: spawn `python -m iai_mcp.daemon` detached.

    No-op-with-sleep when launchd plist is present AND we are using the
    default (home-derived) socket path: launchd's KeepAlive will respawn
    the daemon within 1-2s on macOS, so we yield rather than double-spawn.
    If IAI_DAEMON_SOCKET_PATH is set to a non-default value (test isolation
    or developer custom session), launchd's plist (which does not export
    the env override) cannot resurrect THIS daemon — manual respawn is
    required.

    Manual respawn passes os.environ.copy() so IAI_DAEMON_SOCKET_PATH +
    IAI_MCP_STORE propagate to the child process. Without env propagation,
    test recovery would always spawn against the user's real ~/.iai-mcp/
    paths instead of the test-isolated path.
    """
    from iai_mcp.cli import LAUNCHD_TARGET

    t0 = time.monotonic()
    socket_path = _resolve_socket_path()

    # launchd-managed: yield to KeepAlive ONLY if the user is targeting the
    # default socket path. A custom IAI_DAEMON_SOCKET_PATH means launchd's
    # plist (which has no env overrides) cannot revive this daemon — fall
    # through to manual respawn.
    using_default_socket = os.environ.get("IAI_DAEMON_SOCKET_PATH") is None
    if (
        using_default_socket
        and LAUNCHD_TARGET
        and Path(LAUNCHD_TARGET).expanduser().exists()
    ):
        time.sleep(_LAUNCHD_REACT_DELAY_SEC)
        return (
            True,
            "launchd-managed (KeepAlive will respawn)",
            int((time.monotonic() - t0) * 1000),
        )

    try:
        subprocess.Popen(
            [sys.executable, "-m", "iai_mcp.daemon"],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 — spawn failure is a recovery error
        logger.debug("respawn daemon failed: %s", e)
        return (
            False,
            f"respawn failed: {type(e).__name__}: {e}",
            int((time.monotonic() - t0) * 1000),
        )

    # Wait for the bind. bge-small first-load is 3-10s on cold cache plus
    # store open ~1s; an 8s budget covers most warm-cache machines and
    # is supplemented by a final re-check in cmd_doctor.
    deadline = time.monotonic() + _RESPAWN_BIND_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if socket_path.exists():
            duration_ms = int((time.monotonic() - t0) * 1000)
            return (
                True,
                f"daemon respawned (socket bound in {duration_ms} ms)",
                duration_ms,
            )
        time.sleep(_RESPAWN_POLL_INTERVAL_SEC)
    duration_ms = int((time.monotonic() - t0) * 1000)
    return (
        False,
        f"daemon respawn timed out (socket not bound after {_RESPAWN_BIND_TIMEOUT_SEC}s)",
        duration_ms,
    )


def _kill_dup_binders() -> tuple[bool, str, int]:
    """Repair action: keep oldest-etime binder, SIGKILL the rest.

    Identifies binders via lsof -F pn, sorts by psutil create_time ascending
    (oldest process = max etime = most accumulated client traffic), keeps
    that one, SIGKILLs the rest.

    Wrong-PID-kill mitigation: only kills processes whose psutil cmdline
    contains the literal substring 'iai_mcp.daemon' — anyone running 2 daemons
    against the SAME socket file is by definition violating singleton, but the
    cmdline filter still protects against PID reuse (a recycled PID belonging
    to an unrelated process is skipped).

    Race tolerance: processes that disappear between lsof enumeration and
    psutil.Process(pid) construction are silently skipped (NoSuchProcess /
    AccessDenied caught) — the natural concurrency between detection and
    repair MUST NOT crash the doctor.
    """
    import psutil

    t0 = time.monotonic()
    socket_path = _resolve_socket_path()
    try:
        result = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return (
            False,
            f"lsof unavailable: {e}",
            int((time.monotonic() - t0) * 1000),
        )
    binder_pids = _extract_binder_pids(result.stdout, socket_path)
    if len(binder_pids) <= 1:
        return (
            True,
            f"{len(binder_pids)} dup binders to kill",
            int((time.monotonic() - t0) * 1000),
        )

    # Compute etime for each PID; "oldest" = max(time.time() - create_time).
    # Skip PIDs that disappear between lsof and psutil (race).
    pid_etimes: list[tuple[int, float]] = []
    for pid in binder_pids:
        try:
            p = psutil.Process(pid)
            create_time = p.create_time()  # epoch seconds
            pid_etimes.append((pid, time.time() - create_time))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not pid_etimes:
        return (
            False,
            "all binders disappeared between lsof and psutil",
            int((time.monotonic() - t0) * 1000),
        )

    # Sort longest-etime first; keep the oldest, kill the rest.
    pid_etimes.sort(key=lambda x: x[1], reverse=True)
    keep_pid = pid_etimes[0][0]
    kill_candidates = [pid for pid, _ in pid_etimes[1:]]

    killed: list[int] = []
    for pid in kill_candidates:
        try:
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline() or [])
            if "iai_mcp.daemon" not in cmdline:
                # Wrong-PID-kill mitigation: never SIGKILL a non-daemon process,
                # even if lsof reported it bound to our socket path (PID reuse).
                continue
            p.kill()  # SIGKILL — these are stuck duplicate binders
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Let the kills settle so a follow-up check_g sees the post-kill state.
    time.sleep(_LAUNCHD_REACT_DELAY_SEC)
    return (
        True,
        f"kept PID {keep_pid} (oldest); killed {killed}",
        int((time.monotonic() - t0) * 1000),
    )


def _plan_repair_actions(results: list[CheckResult]) -> list[RepairAction]:
    """Map FAIL checks to repair actions in dependency order.

    Ordering:
      1. unlink stale socket  (lets next bind succeed cleanly)
      2. kill dup binders     (multi-binder cleanup)
      3. kill orphan cores    (frees store write-locks held by stale cores)
      4. respawn daemon       (binds fresh)
    """
    actions: list[RepairAction] = []
    fail_names = {r.name for r in results if not r.passed}

    if "(b) socket file fresh" in fail_names:
        actions.append(
            RepairAction(
                label="unlink_stale_socket",
                description="unlink stale ~/.iai-mcp/.daemon.sock",
                destructive=True,
                execute=_unlink_stale_socket,
            )
        )

    if "(g) no dup binders" in fail_names:
        actions.append(
            RepairAction(
                label="kill_dup_binders",
                description="keep oldest-etime daemon binder, SIGKILL the rest",
                destructive=True,
                execute=_kill_dup_binders,
            )
        )

    if "(d) no orphan iai_mcp.core procs" in fail_names:
        actions.append(
            RepairAction(
                label="kill_orphan_cores",
                description="SIGTERM every orphan iai_mcp.core process",
                destructive=True,
                execute=_kill_orphan_cores,
            )
        )

    if "(a) daemon process alive" in fail_names:
        actions.append(
            RepairAction(
                label="respawn_daemon",
                description="spawn `python -m iai_mcp.daemon` detached",
                # Spawning a long-lived background process IS state-changing
                # (uses ~1.2GB RAM, holds the socket, runs REM cycles). Treat
                # as destructive so --apply (without --yes) prompts the user.
                # Without this, an unprompted respawn could surprise users who
                # ran `--apply` to see what it WOULD do.
                destructive=True,
                execute=_respawn_daemon,
            )
        )

    return actions


def _prompt_action(action: RepairAction) -> bool:
    """Strict 'y' confirmation prompt; EOFError-safe.

    EOFError on closed stdin returns empty string → False. Empty / 'n' /
    anything-else → False. Only literal lowercase 'y' (after strip) → True.
    """
    try:
        response = input(f"  [y/N] {action.description}: ")
    except EOFError:
        response = ""
    return response.strip().lower() == "y"


# -----------------------------------------------------------------------------
# CLI dispatch entry point
# -----------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run full diagnosis and optional repair sequence."""
    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    if yes and not apply:
        print(
            "[warn] --yes without --apply is meaningless; ignoring --yes.",
            file=sys.stderr,
        )

    # Step 1: diagnosis (read-only, always runs).
    results = run_diagnosis()
    # Downgrade (b) socket fresh + (n) HID idle source FAIL -> WARN on
    # headless hosts so the operator sees a clean, advisory output.
    headless = is_headless(force=bool(getattr(args, "headless", False)))
    results = _apply_headless_downgrade(results, headless)
    total = len(results)
    # Surface the migration remediation at the TOP, before the row-by-row
    # print, so users upgrading from a Keychain-backed install see the fix
    # before they parse the checklist.
    hint = _format_top_of_output_hint(results)
    if hint is not None:
        print(hint)
        print()
    print_checklist(results)
    fail_count = sum(1 for r in results if not r.passed)

    if fail_count == 0:
        print("\nAll checks passed. Exit 0.")
        return 0

    if not apply:
        print(
            f"\n{fail_count}/{total} FAIL. Run with --apply to attempt recovery. Exit 1."
        )
        return 1

    # Step 2: --apply repair sequence.
    print(
        f"\n{fail_count}/{total} FAIL. Attempting recovery (--apply{' --yes' if yes else ''}):\n"
    )
    actions = _plan_repair_actions(results)
    if not actions:
        print(
            "(no automated repair actions for the FAILs above; manual intervention required)"
        )
    for action in actions:
        if action.destructive and not yes:
            if not _prompt_action(action):
                print(f"  [skipped] {action.description}")
                continue
        ok, msg, ms = action.execute()
        tag = "[done]" if ok else "[FAIL]"
        print(f"  {tag} {action.label}: {msg} ({ms} ms)")
        # Audit-trail event. Audit must NEVER block recovery — wrap
        # in a broad try/except and silently swallow any failure (the store
        # may be unreadable per check (f) FAIL).
        try:
            from iai_mcp.events import write_event
            from iai_mcp.store import MemoryStore

            write_event(
                MemoryStore(),
                kind="doctor_action",
                data={
                    "action": action.label,
                    "target": action.description,
                    "success": ok,
                    "duration_ms": ms,
                    "detail": msg,
                },
            )
        except Exception as e:
            logger.debug("doctor audit event write failed: %s", e)

    # Step 3: re-run all checks.
    print("\nRe-running checks ...")
    final_results = run_diagnosis()
    print_checklist(final_results)
    final_fails = [r.name for r in final_results if not r.passed]
    if not final_fails:
        print(f"\nFIXED. All {len(final_results)} checks pass. Exit 0.")
        return 0
    print(f"\nSTILL BROKEN: {final_fails}. Exit 2.")
    return 2
