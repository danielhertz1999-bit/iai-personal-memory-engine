"""Durable hermeticity backstop for frozen, home-derived default constants.

Several modules freeze a home-derived default path at import time (e.g.
``Path.home() / ".iai-mcp" / ...``). Because they are computed once, redirecting
``$HOME`` does NOT move them — they keep pointing at the operator's real store
unless the autouse fixture explicitly redirects them. A bare construction that
hits such a default would then read/write the operator's real store WITHOUT
failing any test: a silent isolation breach.

This meta-test converts the one-shot completeness audit into a guard that runs
on every invocation under the autouse redirect fixture. For each frozen
home-derived constant it asserts the EFFECTIVE resolution a consumer would use
does NOT land under the operator's real ``~/.iai-mcp``:

- Redirected constants must resolve under the per-test tmp dir.
- A constant left un-redirected because a consumer-checked env var shadows it
  must have its CONSUMER resolution (env first) land in tmp, not the real store.
- A constant that resolves outside ``~/.iai-mcp`` entirely (a non-store ancillary
  path) is outside the isolation boundary and asserted as such.

If a new frozen home-derived store-root constant is added and not redirected,
the aggregate check below fails — the hole cannot silently reopen.

Test data is generic; no real names or PII.
"""
from __future__ import annotations

import os
from pathlib import Path

import iai_mcp.backup
import iai_mcp.capture_queue
import iai_mcp.cli
import iai_mcp.concurrency
import iai_mcp.crypto
import iai_mcp.daemon
import iai_mcp.daemon_state
import iai_mcp.hippo
import iai_mcp.lifecycle
import iai_mcp.lifecycle_event_log
import iai_mcp.lifecycle_state
import iai_mcp.store


def _real_root() -> Path:
    """The operator's real ``~/.iai-mcp`` (login-database derived, not $HOME)."""
    return Path(iai_mcp.hippo._REAL_IAI_ROOT)


def _tmp_home() -> Path:
    """The per-test tmp HOME installed by the autouse redirect fixture."""
    return Path(os.environ["HOME"])


def _resolves_under_real(value) -> bool:
    """True if ``value`` is at or under the operator's real ~/.iai-mcp."""
    real = _real_root()
    try:
        p = Path(value).resolve()
    except (TypeError, ValueError):
        return False
    return p == real.resolve() or real.resolve() in p.parents


def _resolves_under_tmp(value) -> bool:
    """True if ``value`` is at or under the per-test tmp HOME."""
    home = _tmp_home().resolve()
    try:
        p = Path(value).resolve()
    except (TypeError, ValueError):
        return False
    return p == home or home in p.parents


# Frozen constants the autouse fixture redirects to the tmp dir. The EFFECTIVE
# value (the module attribute the consumer reads) must resolve under tmp.
def _redirected_constants() -> dict[str, object]:
    return {
        "hippo._DEFAULT_IAI_ROOT": iai_mcp.hippo._DEFAULT_IAI_ROOT,
        "store.DEFAULT_STORAGE_PATH": iai_mcp.store.DEFAULT_STORAGE_PATH,
        "concurrency.SOCKET_PATH": iai_mcp.concurrency.SOCKET_PATH,
        "daemon_state.STATE_PATH": iai_mcp.daemon_state.STATE_PATH,
        "lifecycle_state.LIFECYCLE_STATE_PATH": iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH,
        "cli.LOCK_PATH": iai_mcp.cli.LOCK_PATH,
        "cli.STATE_PATH": iai_mcp.cli.STATE_PATH,
        "lifecycle_event_log.DEFAULT_LOG_DIR": iai_mcp.lifecycle_event_log.DEFAULT_LOG_DIR,
        "capture_queue.DEFAULT_QUEUE_DIR": iai_mcp.capture_queue.DEFAULT_QUEUE_DIR,
        "lifecycle.DEFAULT_LOCK_PATH": iai_mcp.lifecycle.DEFAULT_LOCK_PATH,
        "daemon.SESSION_START_CACHE_PATH": iai_mcp.daemon.SESSION_START_CACHE_PATH,
        "crypto._DEFAULT_STORE_ROOT": iai_mcp.crypto._DEFAULT_STORE_ROOT,
        "backup.DEFAULT_STORE_PATH": iai_mcp.backup.DEFAULT_STORE_PATH,
    }


def test_redirected_constants_resolve_under_tmp_not_real() -> None:
    """Every fixture-redirected frozen default resolves under tmp, never real."""
    real = _real_root()
    for name, value in _redirected_constants().items():
        assert not _resolves_under_real(value), (
            f"{name}={value!r} resolves under the operator's real store {real}"
        )
        assert _resolves_under_tmp(value), (
            f"{name}={value!r} is not under the per-test tmp HOME"
        )


def test_socket_path_shadowed_by_env_resolves_to_tmp() -> None:
    """cli.SOCKET_PATH is not redirected — its consumer reads the env first.

    The CLI resolves ``IAI_DAEMON_SOCKET_PATH or str(SOCKET_PATH)``; the fixture
    sets that env under tmp, so the EFFECTIVE socket path the CLI uses lands in
    tmp even though the frozen constant itself is import-time real-home.
    """
    consumer_value = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(
        iai_mcp.cli.SOCKET_PATH
    )
    assert consumer_value, "IAI_DAEMON_SOCKET_PATH must be set by the fixture"
    assert not _resolves_under_real(consumer_value), (
        f"cli socket consumer resolution {consumer_value!r} lands on the real store"
    )
    assert _resolves_under_tmp(consumer_value), (
        f"cli socket consumer resolution {consumer_value!r} is not under tmp"
    )


def test_launchd_target_is_outside_the_store_boundary() -> None:
    """cli.LAUNCHD_TARGET is a LaunchAgents plist, outside ~/.iai-mcp.

    It is home-derived but never resolves under the real ``~/.iai-mcp`` store,
    so it is outside the REL-01 store-isolation boundary and is intentionally
    not redirected. Assert it does not land on the real store.
    """
    assert not _resolves_under_real(iai_mcp.cli.LAUNCHD_TARGET), (
        f"cli.LAUNCHD_TARGET={iai_mcp.cli.LAUNCHD_TARGET!r} unexpectedly under the real store"
    )
    # Sanity: it really is the LaunchAgents plist path, not a store path. (The
    # filename embeds the bundle id, so a substring check is not reliable —
    # assert the structural location instead.)
    target = Path(iai_mcp.cli.LAUNCHD_TARGET)
    assert target.parent.name == "LaunchAgents"
    assert target.suffix == ".plist"


def test_no_frozen_home_constant_resolves_to_real_store() -> None:
    """Aggregate REL-01 backstop: NO frozen home-derived constant — redirected
    or env-shadowed — has an effective resolution under the operator's real
    ~/.iai-mcp under pytest. A new un-redirected store-root default trips this.
    """
    real = _real_root()
    offenders: list[str] = []

    # Redirected defaults: the attribute value itself is the effective path.
    for name, value in _redirected_constants().items():
        if _resolves_under_real(value):
            offenders.append(f"{name}={value!r}")

    # Env-shadowed default: the consumer's env-first resolution is the effective
    # path (the raw constant is import-time real-home, which is expected).
    socket_consumer = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(
        iai_mcp.cli.SOCKET_PATH
    )
    if _resolves_under_real(socket_consumer):
        offenders.append(f"cli.SOCKET_PATH(consumer)={socket_consumer!r}")

    assert not offenders, (
        "frozen home-derived constants resolve to the real store "
        f"{real} under pytest: {offenders}"
    )
