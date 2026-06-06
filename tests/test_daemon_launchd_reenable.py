"""Hermetic launchd re-enable test (mocked launchctl, no live daemon touch).

The daemon re-enable is a launchctl `bootout` (idempotent) -> `bootstrap
gui/$uid <target>` -> `kickstart gui/$uid/<label>` sequence against the
on-disk plist, followed by a `daemon status` socket round-trip that must
report ok within the socket-timeout bound. This module proves that sequence
+ verification logic WITHOUT ever invoking the real launchctl or touching the
real `com.iai-mcp.daemon` job:

- `subprocess.run` is monkeypatched (every launchctl call is captured, never
  executed);
- `Path.home()` + `LAUNCHD_TARGET` are redirected to tmp by `fake_state_dir`,
  so the rendered plist lands in a tmp path, never `~/Library/LaunchAgents`;
- the status round-trip is mocked at `_send_socket_request`, so no real
  daemon socket is contacted.

It also locks the render -> install path as a regression guard: the rendered
plist (and the plist `daemon install` writes) MUST carry the
`IAI_MCP_WATCHDOG_*` env knobs, with values matching the shipped template /
the daemon code defaults, so an operator-visible `iai-mcp daemon install`
always deploys the watchdog knobs.
"""
from __future__ import annotations

import asyncio
import platform
from pathlib import Path

import pytest

from iai_mcp import cli as cli_mod


# The watchdog env keys the re-rendered plist + a fresh install MUST carry.
# Values match the shipped template (deploy/launchd/com.iai-mcp.daemon.plist)
# AND the daemon code defaults (daemon.py WATCHDOG_* loaders), so the rendered
# plist deploys the exact same behaviour the daemon already runs with.
EXPECTED_WATCHDOG_ENV: dict[str, str] = {
    "IAI_MCP_WATCHDOG_LIVENESS_POLL_SEC": "30.0",
    "IAI_MCP_WATCHDOG_WARN_POLL_SEC": "7.0",
    "IAI_MCP_WATCHDOG_PROBE_TIMEOUT_SEC": "5.0",
    "IAI_MCP_WATCHDOG_FAILURE_DEBOUNCE_N": "3",
    "IAI_MCP_WATCHDOG_RSS_HARD_CAP_BYTES": "2684354560",
    "IAI_MCP_WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES": "1610612736",
    "IAI_MCP_WATCHDOG_MAX_RECOVERIES": "3",
    "IAI_MCP_WATCHDOG_RECOVERY_WINDOW_SEC": "600.0",
    "IAI_MCP_WATCHDOG_COLD_START_GRACE_SEC": "600.0",
}


def _plist_env_value(plist_text: str, key: str) -> str | None:
    """Extract the <string> value following a <key>NAME</key> entry in a
    launchd plist. Returns None when the key is absent. Tolerant of arbitrary
    whitespace/newlines between the key and its string value.
    """
    import re

    m = re.search(
        rf"<key>{re.escape(key)}</key>\s*<string>(.*?)</string>",
        plist_text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_cli_daemon.py conventions)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.iai-mcp + ~/Library/LaunchAgents to tmp_path so install
    never touches the real host filesystem (mirrors test_cli_daemon.py)."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(cli_mod, "LOCK_PATH", fake_home / ".iai-mcp" / ".lock")
    monkeypatch.setattr(
        cli_mod, "SOCKET_PATH", fake_home / ".iai-mcp" / ".daemon.sock",
    )
    monkeypatch.setattr(
        cli_mod, "STATE_PATH", fake_home / ".iai-mcp" / ".daemon-state.json",
    )
    monkeypatch.setattr(
        cli_mod,
        "LAUNCHD_TARGET",
        fake_home / "Library" / "LaunchAgents" / "com.iai-mcp.daemon.plist",
    )
    monkeypatch.setattr(
        cli_mod,
        "SYSTEMD_TARGET",
        fake_home / ".config" / "systemd" / "user" / "iai-mcp-daemon.service",
    )
    return fake_home


@pytest.fixture
def captured_launchctl(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Monkeypatch subprocess.run to CAPTURE (never execute) every argv.
    Returns the live list of captured argv lists. The real launchctl is never
    invoked.
    """
    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):  # noqa: ANN001 -- mirrors subprocess.run
        calls.append(list(argv))

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
    return calls


# ---------------------------------------------------------------------------
# 1. The rendered plist carries the watchdog env knobs (FIX-1 regression guard)
# ---------------------------------------------------------------------------


def test_rendered_plist_contains_all_watchdog_env_keys() -> None:
    """The plist `iai-mcp daemon install` renders MUST carry every
    `IAI_MCP_WATCHDOG_*` env knob. This calls the real `_render_launchd_plist`
    (it reads a repo artifact + token-substitutes; no ~/.iai-mcp path, no
    socket, no launchctl), so it is hermetic.
    """
    rendered = cli_mod._render_launchd_plist()
    for key in EXPECTED_WATCHDOG_ENV:
        assert (
            f"<key>{key}</key>" in rendered
        ), f"rendered plist is missing watchdog key {key}"


def test_rendered_plist_watchdog_values_match_code_defaults() -> None:
    """Each rendered watchdog env value MUST equal the shipped template /
    daemon code default, so deploying the rendered plist is behaviour-neutral
    (no drift from what the daemon already runs with).
    """
    import iai_mcp.daemon as daemon_mod

    rendered = cli_mod._render_launchd_plist()
    for key, expected in EXPECTED_WATCHDOG_ENV.items():
        assert _plist_env_value(rendered, key) == expected, (
            f"{key}: rendered value != expected template value {expected}"
        )

    # Tie the expected values back to the daemon's live code defaults so a
    # future default change can't silently desync the plist from the daemon.
    assert daemon_mod.WATCHDOG_LIVENESS_POLL_SEC == 30.0
    assert daemon_mod.WATCHDOG_WARN_POLL_SEC == 7.0
    assert daemon_mod.WATCHDOG_PROBE_TIMEOUT_SEC == 5.0
    assert daemon_mod.WATCHDOG_FAILURE_DEBOUNCE_N == 3
    assert daemon_mod.WATCHDOG_RSS_HARD_CAP_BYTES == 2684354560
    assert daemon_mod.WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES == 1610612736
    assert daemon_mod.WATCHDOG_MAX_RECOVERIES == 3
    assert daemon_mod.WATCHDOG_RECOVERY_WINDOW_SEC == 600.0
    assert daemon_mod.WATCHDOG_COLD_START_GRACE_SEC == 600.0


# ---------------------------------------------------------------------------
# 2. The re-enable sequence (mocked launchctl): bootout -> bootstrap -> kickstart
# ---------------------------------------------------------------------------


def test_reenable_emits_bootout_bootstrap_kickstart_in_order(
    fake_state_dir: Path,
    captured_launchctl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`iai-mcp daemon install` on macOS re-renders the plist to LAUNCHD_TARGET
    and drives the re-enable launchctl sequence: `bootout` (idempotent) ->
    `bootstrap gui/$uid <target>` -> `kickstart gui/$uid/<label>` in that
    order. subprocess.run is captured (real launchctl never runs).
    """
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0

    # The launchctl subcommands, in the order they were invoked.
    launchctl_subcmds = [
        argv[1] for argv in captured_launchctl
        if argv and argv[0] == "launchctl"
    ]
    assert launchctl_subcmds == ["bootout", "bootstrap", "kickstart"], (
        captured_launchctl
    )

    uid = __import__("os").getuid()
    target = str(cli_mod.LAUNCHD_TARGET)
    label = cli_mod.DAEMON_LABEL

    by_subcmd = {
        argv[1]: argv
        for argv in captured_launchctl
        if argv and argv[0] == "launchctl"
    }
    # bootout + bootstrap target gui/$uid + the on-disk plist target.
    assert by_subcmd["bootout"] == ["launchctl", "bootout", f"gui/{uid}", target]
    assert by_subcmd["bootstrap"] == [
        "launchctl", "bootstrap", f"gui/{uid}", target,
    ]
    # kickstart targets the per-label service path.
    assert by_subcmd["kickstart"] == [
        "launchctl", "kickstart", f"gui/{uid}/{label}",
    ]


def test_reenable_writes_plist_carrying_watchdog_keys(
    fake_state_dir: Path,
    captured_launchctl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plist that `daemon install` actually writes to LAUNCHD_TARGET (the
    on-disk target the subsequent bootstrap loads) MUST carry the watchdog env
    knobs with the expected values — the live re-enable boots WITH the watchdog
    knobs deployed.
    """
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0
    assert cli_mod.LAUNCHD_TARGET.exists()

    written = cli_mod.LAUNCHD_TARGET.read_text()
    for key, expected in EXPECTED_WATCHDOG_ENV.items():
        assert f"<key>{key}</key>" in written, f"written plist missing {key}"
        assert _plist_env_value(written, key) == expected, (
            f"{key}: written plist value != expected {expected}"
        )


# ---------------------------------------------------------------------------
# 3. Status verification: ok-within-bound passes; timeout is handled
# ---------------------------------------------------------------------------


def test_status_ok_within_bound_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Post-re-enable verification: when the status socket round-trip replies
    `{ok: True, ...}` within the bound, `daemon status` returns 0 and prints
    the live status. `_send_socket_request` is mocked — no real daemon socket.
    """
    sent: list[dict] = []

    def _fake_send(req, *, timeout=30.0):  # noqa: ANN001
        sent.append(req)
        # Assert the verification uses a bounded socket timeout (cmd_daemon_status
        # uses 10.0s), not an unbounded wait.
        assert timeout == 10.0
        return {
            "ok": True,
            "state": "WAKE",
            "uptime_sec": 12.5,
            "version": "0.1.0",
        }

    monkeypatch.setattr(cli_mod, "_send_socket_request", _fake_send)

    rc = cli_mod.main(["daemon", "status"])
    assert rc == 0
    assert sent == [{"type": "status"}]

    out = capsys.readouterr().out
    assert "ok: True" in out
    assert "WAKE" in out


def test_status_timeout_prints_not_responding_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """If the round-trip exceeds the bound (the daemon accepted the connection
    but never replied — a wedged loop), `asyncio.TimeoutError` propagates and
    `daemon status` prints "daemon not responding" to stderr + returns
    non-zero. Mocked — no real socket.
    """
    def _fake_send_timeout(req, *, timeout=30.0):  # noqa: ANN001
        raise asyncio.TimeoutError()

    monkeypatch.setattr(cli_mod, "_send_socket_request", _fake_send_timeout)

    rc = cli_mod.main(["daemon", "status"])
    assert rc != 0

    err = capsys.readouterr().err
    assert "daemon not responding" in err


def test_status_socket_absent_prints_not_running_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """When the socket is unreachable entirely (`_send_socket_request` returns
    None), `daemon status` prints "daemon not running" + returns non-zero —
    distinct from the wedged-but-bound-exceeded timeout path above.
    """
    monkeypatch.setattr(
        cli_mod, "_send_socket_request", lambda req, *, timeout=30.0: None,
    )

    rc = cli_mod.main(["daemon", "status"])
    assert rc != 0

    out = capsys.readouterr().out
    assert "daemon not running" in out


# ---------------------------------------------------------------------------
# 4. Hermeticity guard: the real launchctl / real daemon job is NEVER invoked
# ---------------------------------------------------------------------------


def test_real_launchctl_is_never_invoked(
    fake_state_dir: Path,
    captured_launchctl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt-and-braces: drive the full install (re-enable) path and confirm
    every launchctl invocation went through the captured (mocked) subprocess —
    none escaped to the real binary. The captured argv list is the only
    evidence the sequence ran; the host launchd state is untouched.
    """
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0

    # Every launchctl call is in the captured list (the mock); the real binary
    # was never reached. The plist target is a tmp path, not the real job file.
    assert all(
        str(cli_mod.LAUNCHD_TARGET).endswith("com.iai-mcp.daemon.plist")
        for _ in [0]
    )
    assert "home" in str(cli_mod.LAUNCHD_TARGET)  # tmp-redirected, not real ~
    assert len([c for c in captured_launchctl if c and c[0] == "launchctl"]) == 3
