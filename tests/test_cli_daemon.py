"""iai-mcp daemon subcommand group tests.

Verifies dispatcher wiring, install/uninstall flow with consent banner,
launchd / systemd template rendering with sys.executable substitution,
version skew detection in `daemon status`, and clean uninstall
(removes plist/unit + all 3 state files).

All subprocess calls (launchctl, systemctl, loginctl, tail, journalctl) are
monkeypatched so the suite never touches the host's actual launchd/systemd.

Socket-talking subcommands (status / force-rem / pause / logs) are exercised
against the `_ThreadedFakeDaemon` helper (lifted from
tests/test_core_bedtime_inject.py pattern -- a fake daemon that survives
multiple asyncio.run() teardowns by running on a dedicated background loop).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import platform
import sys
import tempfile
import threading
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp import cli as cli_mod


# ---------------------------------------------------------------------------
# Threaded fake daemon (survives multiple asyncio.run teardowns)
# ---------------------------------------------------------------------------


class _ThreadedFakeDaemon:
    """Fake daemon NDJSON server on a background loop.

    Each request line is captured. Each request gets `reply` written back
    (or a per-request reply via `reply_fn(req)` if provided).
    """

    def __init__(
        self,
        path: Path,
        captured: list,
        reply: dict | None = None,
        reply_fn=None,
    ) -> None:
        self.path = path
        self.captured = captured
        self.reply = reply
        self.reply_fn = reply_fn
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _handle(reader, writer):
                try:
                    line = await reader.readline()
                    if line:
                        req = json.loads(line.decode("utf-8"))
                        self.captured.append(req)
                        if self.reply_fn is not None:
                            resp = self.reply_fn(req)
                        else:
                            resp = self.reply or {}
                        writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                        await writer.drain()
                finally:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass

            async def _serve():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._server = await asyncio.start_unix_server(
                    _handle, path=str(self.path),
                )
                self._ready.set()
                async with self._server:
                    await self._server.serve_forever()

            try:
                self._loop.run_until_complete(_serve())
            except asyncio.CancelledError:
                pass
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0), "fake daemon failed to start"

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return

        async def _shutdown():
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=5.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_socket(tmp_path: Path) -> Path:
    """Short unix-socket path (macOS ~104-byte limit)."""
    candidate = tmp_path / "d.sock"
    if len(str(candidate)) > 100:
        candidate = Path(tempfile.mkdtemp(prefix="iai-clitest-")) / "d.sock"
    return candidate


@pytest.fixture
def fake_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.iai-mcp + ~/Library/LaunchAgents + ~/.config/systemd/user
    to tmp_path-rooted equivalents, so install/uninstall never touches the
    real host filesystem."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    # Re-resolve the constants after Path.home() is patched.
    monkeypatch.setattr(
        cli_mod, "LOCK_PATH", fake_home / ".iai-mcp" / ".lock",
    )
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


# ---------------------------------------------------------------------------
# Test 1: dry-run does NOT write any file
# ---------------------------------------------------------------------------


def test_install_dry_run_writes_no_file(
    fake_state_dir: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    rc = cli_mod.main(["daemon", "install", "--dry-run", "--yes"])
    assert rc == 0
    assert not cli_mod.LAUNCHD_TARGET.exists()
    out = capsys.readouterr().out
    assert "Would install to" in out
    # sys.executable is substituted in dry-run output
    assert sys.executable in out


# ---------------------------------------------------------------------------
# Test 2: install on macOS writes plist with sys.executable + invokes launchctl
# ---------------------------------------------------------------------------


def test_install_macos_writes_plist_with_sys_executable(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0
    assert cli_mod.LAUNCHD_TARGET.exists()
    contents = cli_mod.LAUNCHD_TARGET.read_text()
    # Absolute sys.executable must be substituted into plist.
    assert sys.executable in contents
    # USERNAME placeholder substituted (not present literally)
    assert "{USERNAME}" not in contents
    # launchctl bootstrap + kickstart called
    assert any("bootstrap" in " ".join(c) for c in calls), calls
    assert any("kickstart" in " ".join(c) for c in calls), calls


# ---------------------------------------------------------------------------
# Test 3: install on Linux writes systemd unit + invokes systemctl + loginctl
# ---------------------------------------------------------------------------


def test_install_linux_writes_unit_and_invokes_systemctl(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setenv("USER", "testuser")
    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        class _R:
            returncode = 0
            # Simulate Linger=no on the first show-user, then Linger=yes after enable
            _show_count = [0]
            stdout = (
                "Linger=no" if argv[:2] == ["loginctl", "show-user"]
                else ""
            )
            stderr = ""
        return _R()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0
    assert cli_mod.SYSTEMD_TARGET.exists()
    contents = cli_mod.SYSTEMD_TARGET.read_text()
    assert sys.executable in contents
    # loginctl invoked at least twice (show + enable + re-verify)
    loginctl_calls = [c for c in calls if c and c[0] == "loginctl"]
    assert len(loginctl_calls) >= 2, loginctl_calls
    # systemctl --user daemon-reload AND enable --now invoked
    cmd_strs = [" ".join(c) for c in calls]
    assert any("systemctl --user daemon-reload" in s for s in cmd_strs), cmd_strs
    assert any("systemctl --user enable --now iai-mcp-daemon.service" in s for s in cmd_strs), cmd_strs


# ---------------------------------------------------------------------------
# Test 4: consent banner blocks on stdin; non-`y` responses abort
# ---------------------------------------------------------------------------


def test_install_without_yes_prompts_consent_banner_aborts(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    # Don't actually call subprocess
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    # Strict gate: ONLY exact lowercase "y" (after .strip()) proceeds.
    # Everything else -- empty, "n", "N", "yes", "no", "true", numeric -- aborts.
    for response in ["", "n", "N", "yes", "no", "true", "1", "0", "yeah", "nope"]:
        monkeypatch.setattr(
            "builtins.input", lambda _prompt="", r=response: r,
        )
        rc = cli_mod.main(["daemon", "install"])
        assert rc == 1, f"non-strict-y response {response!r} should abort"
        # State file should not exist (install did not proceed)
        assert not cli_mod.LAUNCHD_TARGET.exists()

    err = capsys.readouterr().err
    # Banner must mention key phrases.
    # RAM budget is ~400 MB (bge-small-en-v1.5 kept warm).
    assert "~400 MB" in err or "400 MB" in err
    assert "1%" in err
    assert "iai-mcp daemon uninstall" in err


def test_install_with_lowercase_y_proceeds(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    rc = cli_mod.main(["daemon", "install"])
    assert rc == 0
    assert cli_mod.LAUNCHD_TARGET.exists()


def test_install_consent_records_audit_trail(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit trail: explicit consent writes a timestamped JSON receipt
    under ~/.iai-mcp/.consent-*.json so a later forensic review can confirm
    the user actually consented (not bypassed via --yes)."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    rc = cli_mod.main(["daemon", "install"])
    assert rc == 0
    consent_files = list((fake_state_dir / ".iai-mcp").glob(".consent-*.json"))
    assert consent_files, "expected at least one .consent-<ts>.json audit receipt"
    payload = json.loads(consent_files[0].read_text())
    assert payload.get("consent") is True
    assert "ts" in payload


# ---------------------------------------------------------------------------
# Test 5: macOS uninstall removes plist + all 3 state files (C4)
# ---------------------------------------------------------------------------


def test_uninstall_macos_removes_plist_and_all_state_files(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())

    # Pre-seed the plist + 3 state files
    cli_mod.LAUNCHD_TARGET.parent.mkdir(parents=True, exist_ok=True)
    cli_mod.LAUNCHD_TARGET.write_text("<plist></plist>")
    state_dir = fake_state_dir / ".iai-mcp"
    state_dir.mkdir(parents=True, exist_ok=True)
    cli_mod.LOCK_PATH.write_text("")
    cli_mod.SOCKET_PATH.write_text("")
    cli_mod.STATE_PATH.write_text("{}")

    rc = cli_mod.main(["daemon", "uninstall", "--yes"])
    assert rc == 0
    # all 4 artefacts gone
    assert not cli_mod.LAUNCHD_TARGET.exists()
    assert not cli_mod.LOCK_PATH.exists()
    assert not cli_mod.SOCKET_PATH.exists()
    assert not cli_mod.STATE_PATH.exists()


# ---------------------------------------------------------------------------
# Test 6: Linux uninstall removes unit + all 3 state files (C4)
# ---------------------------------------------------------------------------


def test_uninstall_linux_removes_unit_and_all_state_files(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda argv, **k: (calls.append(list(argv)) or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()),
    )

    cli_mod.SYSTEMD_TARGET.parent.mkdir(parents=True, exist_ok=True)
    cli_mod.SYSTEMD_TARGET.write_text("[Service]")
    state_dir = fake_state_dir / ".iai-mcp"
    state_dir.mkdir(parents=True, exist_ok=True)
    cli_mod.LOCK_PATH.write_text("")
    cli_mod.SOCKET_PATH.write_text("")
    cli_mod.STATE_PATH.write_text("{}")

    rc = cli_mod.main(["daemon", "uninstall", "--yes"])
    assert rc == 0
    assert not cli_mod.SYSTEMD_TARGET.exists()
    assert not cli_mod.LOCK_PATH.exists()
    assert not cli_mod.SOCKET_PATH.exists()
    assert not cli_mod.STATE_PATH.exists()
    cmd_strs = [" ".join(c) for c in calls]
    assert any("systemctl --user disable --now iai-mcp-daemon.service" in s for s in cmd_strs), cmd_strs


# ---------------------------------------------------------------------------
# Test 7: status round-trip + daemon-down message
# ---------------------------------------------------------------------------


def test_status_socket_round_trip(
    short_socket: Path,
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(cli_mod, "SOCKET_PATH", short_socket)
    # CLI resolves IAI_DAEMON_SOCKET_PATH before the SOCKET_PATH constant
    # (env wins, by contract), and the autouse fixture points that env at an
    # empty tmp socket; redirect it to this test's live socket (last-wins).
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(short_socket))
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(
        short_socket,
        captured,
        reply={
            "ok": True,
            "state": "WAKE",
            "uptime_sec": 42.5,
            "version": "0.1.0",
        },
    )
    daemon.start()
    try:
        rc = cli_mod.main(["daemon", "status"])
        assert rc == 0
    finally:
        daemon.stop()

    out = capsys.readouterr().out
    assert "WAKE" in out
    assert "42" in out
    # request was sent
    assert captured == [{"type": "status"}]


def test_status_daemon_down(
    short_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(cli_mod, "SOCKET_PATH", short_socket)
    assert not short_socket.exists()
    rc = cli_mod.main(["daemon", "status"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "daemon not running" in out


# ---------------------------------------------------------------------------
# Test 8: status version skew warns when daemon != installed
# ---------------------------------------------------------------------------


def test_status_warns_on_version_skew(
    short_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(cli_mod, "SOCKET_PATH", short_socket)
    # CLI resolves IAI_DAEMON_SOCKET_PATH before the SOCKET_PATH constant
    # (env wins, by contract), and the autouse fixture points that env at an
    # empty tmp socket; redirect it to this test's live socket (last-wins).
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(short_socket))
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(
        short_socket,
        captured,
        reply={
            "ok": True,
            "state": "WAKE",
            "version": "0.0.1-OLD",
        },
    )
    daemon.start()
    try:
        rc = cli_mod.main(["daemon", "status"])
        assert rc == 0
    finally:
        daemon.stop()

    err = capsys.readouterr().err
    assert "version" in err.lower()
    assert "0.0.1-OLD" in err
    assert "restart" in err.lower()


# ---------------------------------------------------------------------------
# Test 9: configure subcommands persist to state file
# ---------------------------------------------------------------------------


def test_configure_set_budget_persists(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # daemon_state.STATE_PATH must mirror our fake home for save_state to land
    # in the right place. We patch BOTH cli_mod.STATE_PATH AND the daemon_state
    # module's constant in one shot.
    from iai_mcp import daemon_state
    monkeypatch.setattr(daemon_state, "STATE_PATH", cli_mod.STATE_PATH)
    cli_mod.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    rc = cli_mod.main(["daemon", "configure", "set-budget", "0.02"])
    assert rc == 0
    state = json.loads(cli_mod.STATE_PATH.read_text())
    assert state["daily_quota_pct_override"] == pytest.approx(0.02)


def test_configure_set_cycle_count_persists(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp import daemon_state
    monkeypatch.setattr(daemon_state, "STATE_PATH", cli_mod.STATE_PATH)
    cli_mod.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rc = cli_mod.main(["daemon", "configure", "set-cycle-count", "5"])
    assert rc == 0
    state = json.loads(cli_mod.STATE_PATH.read_text())
    assert state["cycle_count_override"] == 5


def test_configure_disable_claude_persists(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp import daemon_state
    monkeypatch.setattr(daemon_state, "STATE_PATH", cli_mod.STATE_PATH)
    cli_mod.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rc = cli_mod.main(["daemon", "configure", "disable-claude"])
    assert rc == 0
    state = json.loads(cli_mod.STATE_PATH.read_text())
    assert state["claude_enabled"] is False


# ---------------------------------------------------------------------------
# Test 10: force-rem socket message
# ---------------------------------------------------------------------------


def test_force_rem_sends_correct_message(
    short_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_mod, "SOCKET_PATH", short_socket)
    # CLI resolves IAI_DAEMON_SOCKET_PATH before the SOCKET_PATH constant
    # (env wins, by contract), and the autouse fixture points that env at an
    # empty tmp socket; redirect it to this test's live socket (last-wins).
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(short_socket))
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(
        short_socket, captured, reply={"ok": True, "cycles_completed": 1},
    )
    daemon.start()
    try:
        rc = cli_mod.main(["daemon", "force-rem"])
        assert rc == 0
    finally:
        daemon.stop()
    assert len(captured) == 1
    assert captured[0]["type"] == "force_rem"
    assert isinstance(captured[0].get("ts"), str)


# ---------------------------------------------------------------------------
# Test 11: pause N
# ---------------------------------------------------------------------------


def test_pause_sends_seconds_arg(
    short_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_mod, "SOCKET_PATH", short_socket)
    # CLI resolves IAI_DAEMON_SOCKET_PATH before the SOCKET_PATH constant
    # (env wins, by contract), and the autouse fixture points that env at an
    # empty tmp socket; redirect it to this test's live socket (last-wins).
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(short_socket))
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(short_socket, captured, reply={"ok": True})
    daemon.start()
    try:
        rc = cli_mod.main(["daemon", "pause", "300"])
        assert rc == 0
    finally:
        daemon.stop()
    assert captured == [{"type": "pause", "seconds": 300}]


def test_resume_sends_resume_message(
    short_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_mod, "SOCKET_PATH", short_socket)
    # CLI resolves IAI_DAEMON_SOCKET_PATH before the SOCKET_PATH constant
    # (env wins, by contract), and the autouse fixture points that env at an
    # empty tmp socket; redirect it to this test's live socket (last-wins).
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(short_socket))
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(short_socket, captured, reply={"ok": True})
    daemon.start()
    try:
        rc = cli_mod.main(["daemon", "resume"])
        assert rc == 0
    finally:
        daemon.stop()
    assert captured == [{"type": "resume"}]


# ---------------------------------------------------------------------------
# Test 12: start / stop dispatch correct argv on each platform
# ---------------------------------------------------------------------------


def test_start_macos_uses_launchctl_kickstart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda argv, **k: (calls.append(list(argv)) or type("R", (), {"returncode": 0})()),
    )
    rc = cli_mod.main(["daemon", "start"])
    assert rc == 0
    cmd_strs = [" ".join(c) for c in calls]
    assert any("launchctl kickstart" in s for s in cmd_strs), cmd_strs


def test_stop_macos_disables_keepalive_then_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """daemon stop (macOS) now disables KeepAlive (launchctl bootout) and
    self-issues a signal to the lockfile PID -- it no longer relies on the
    in-loop-handler `launchctl kill SIGTERM` path."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls, sig = _patch_stop_collaborators(
        monkeypatch, pid=9090, alive_sequence=[True, False],
    )
    rc = cli_mod.main(["daemon", "stop"])
    assert rc == 0
    run_strs = [" ".join(c[1]) for c in calls if c[0] == "run"]
    assert any("launchctl bootout" in s for s in run_strs), calls
    # The legacy in-loop `launchctl kill SIGTERM` path is gone.
    assert not any("launchctl kill" in s for s in run_strs), calls
    assert ("kill", 9090, sig.SIGTERM) in calls


def test_start_linux_uses_systemctl_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda argv, **k: (calls.append(list(argv)) or type("R", (), {"returncode": 0})()),
    )
    rc = cli_mod.main(["daemon", "start"])
    assert rc == 0
    assert any(c[:4] == ["systemctl", "--user", "start", "iai-mcp-daemon.service"] for c in calls), calls


def test_stop_linux_uses_systemctl_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda argv, **k: (calls.append(list(argv)) or type("R", (), {"returncode": 0})()),
    )
    rc = cli_mod.main(["daemon", "stop"])
    assert rc == 0
    assert any(c[:4] == ["systemctl", "--user", "stop", "iai-mcp-daemon.service"] for c in calls), calls


# ---------------------------------------------------------------------------
# Test 13: logs dispatches tail (macOS) or journalctl (Linux)
# ---------------------------------------------------------------------------


def test_logs_macos_invokes_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda argv, **k: (calls.append(list(argv)) or type("R", (), {"returncode": 0})()),
    )
    rc = cli_mod.main(["daemon", "logs", "-n", "50"])
    assert rc == 0
    assert any(c and c[0] == "tail" for c in calls), calls


def test_logs_linux_invokes_journalctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda argv, **k: (calls.append(list(argv)) or type("R", (), {"returncode": 0})()),
    )
    rc = cli_mod.main(["daemon", "logs", "-n", "100"])
    assert rc == 0
    assert any(
        c[:5] == ["journalctl", "--user", "-u", "iai-mcp-daemon.service", "-n"]
        for c in calls
    ), calls


# ---------------------------------------------------------------------------
# Idempotency: install + install does not error
# ---------------------------------------------------------------------------


def test_install_twice_is_idempotent(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    assert cli_mod.main(["daemon", "install", "--yes"]) == 0
    assert cli_mod.main(["daemon", "install", "--yes"]) == 0
    assert cli_mod.LAUNCHD_TARGET.exists()


def test_uninstall_twice_is_idempotent(
    fake_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    assert cli_mod.main(["daemon", "uninstall", "--yes"]) == 0
    assert cli_mod.main(["daemon", "uninstall", "--yes"]) == 0


# ---------------------------------------------------------------------------
# Help output sanity
# ---------------------------------------------------------------------------


def test_daemon_help_lists_all_subcommands(
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(["daemon", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for sub in (
        "install",
        "uninstall",
        "start",
        "stop",
        "status",
        "logs",
        "force-rem",
        "pause",
        "resume",
        "configure",
    ):
        assert sub in out, f"missing {sub} in daemon --help output"


# ---------------------------------------------------------------------------
# Test 14: force-rem message passes the real daemon validator
# ---------------------------------------------------------------------------


def test_force_rem_message_passes_daemon_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: the force-rem CLI request must satisfy the daemon's
    validator (requires ts to be a str).  Reverting the ts field makes this
    test fail with "ts must be a string"."""
    from iai_mcp.concurrency import _validate_socket_message

    captured: dict = {}

    def fake_send(req, *, timeout=None):
        captured["req"] = req
        return {"ok": True, "reason": "rem_queued"}

    monkeypatch.setattr(cli_mod, "_send_socket_request", fake_send)

    rc = cli_mod.main(["daemon", "force-rem"])
    assert rc == 0
    req = captured["req"]
    assert req["type"] == "force_rem"
    ok, err = _validate_socket_message(req)
    assert ok is True and err is None, f"validator rejected: {err}"


# ---------------------------------------------------------------------------
# daemon stop (DMN-02): self-issued SIGTERM->bounded-wait->SIGKILL escalation
# (path B). The shipped mechanism MUST match what the hermetic wedged-loop
# test proves. All signals + launchctl are mocked: NEVER the real daemon,
# NEVER a real PID.
# ---------------------------------------------------------------------------


def _patch_stop_collaborators(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pid,
    alive_sequence,
):
    """Wire a shared ordered call-log across subprocess.run + os.kill +
    a scripted _is_pid_alive, and stub LifecycleLock.read -> {"pid": pid}.

    Returns the call-log list of tagged tuples in chronological order:
      ("run", argv_list)         -- a subprocess.run (launchctl) invocation
      ("kill", pid, signum)      -- an os.kill invocation
    `alive_sequence` is consumed one item per _is_pid_alive call (after the
    last item is reached, the final value sticks).
    """
    import signal as _signal
    import iai_mcp.lifecycle_lock as lifecycle_lock

    calls: list = []

    def _fake_run(argv, **kwargs):
        calls.append(("run", list(argv)))
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    def _fake_kill(target_pid, signum):
        calls.append(("kill", target_pid, signum))

    seq = list(alive_sequence)
    idx = {"i": 0}

    def _fake_is_pid_alive(target_pid):
        i = idx["i"]
        if i < len(seq):
            val = seq[i]
            idx["i"] = i + 1
        else:
            val = seq[-1] if seq else False
        return val

    # Patch where cmd_daemon_stop binds them (it does a lazy
    # `from iai_mcp.lifecycle_lock import LifecycleLock, _is_pid_alive`,
    # so patching the module attributes is authoritative).
    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(cli_mod.os, "kill", _fake_kill)
    monkeypatch.setattr(lifecycle_lock, "_is_pid_alive", _fake_is_pid_alive)
    monkeypatch.setattr(
        lifecycle_lock.LifecycleLock,
        "read",
        lambda self: ({"pid": pid} if pid is not None else None),
    )
    # No-op the best-effort sentinel write (decoupled from the kill path).
    monkeypatch.setattr(
        cli_mod,
        "logger",
        cli_mod.logger,  # leave logger; we just don't want real state I/O
    )

    return calls, _signal


def test_stop_bootout_precedes_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    # PID dies immediately after SIGTERM (first poll sees it dead).
    calls, sig = _patch_stop_collaborators(
        monkeypatch, pid=4242, alive_sequence=[True, False],
    )
    rc = cli_mod.cmd_daemon_stop(object())
    assert rc == 0

    # bootout (KeepAlive disable) must occur BEFORE the SIGTERM signal.
    bootout_idx = next(
        i for i, c in enumerate(calls)
        if c[0] == "run" and "bootout" in c[1]
    )
    sigterm_idx = next(
        i for i, c in enumerate(calls)
        if c[0] == "kill" and c[2] == sig.SIGTERM
    )
    assert bootout_idx < sigterm_idx, calls
    # SIGTERM targets the lockfile PID, not a bogus one.
    assert ("kill", 4242, sig.SIGTERM) in calls


def test_stop_escalates_to_sigkill_when_pid_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    # Tiny bound + interval so the poll loop spins out fast.
    monkeypatch.setenv("IAI_DAEMON_STOP_TIMEOUT_S", "0.05")
    monkeypatch.setenv("IAI_DAEMON_STOP_POLL_S", "0.01")
    # PID stays alive through every poll -> escalation fires.
    calls, sig = _patch_stop_collaborators(
        monkeypatch, pid=5151, alive_sequence=[True],  # sticks True
    )
    rc = cli_mod.cmd_daemon_stop(object())
    assert rc == 0

    bootout_idx = next(
        i for i, c in enumerate(calls)
        if c[0] == "run" and "bootout" in c[1]
    )
    sigkill_idx = next(
        i for i, c in enumerate(calls)
        if c[0] == "kill" and c[2] == sig.SIGKILL
    )
    # KeepAlive disabled BEFORE the SIGKILL.
    assert bootout_idx < sigkill_idx, calls
    assert ("kill", 5151, sig.SIGKILL) in calls
    # SIGTERM still issued before the escalation.
    assert ("kill", 5151, sig.SIGTERM) in calls


def test_stop_no_sigkill_when_pid_dies_during_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setenv("IAI_DAEMON_STOP_TIMEOUT_S", "1.0")
    monkeypatch.setenv("IAI_DAEMON_STOP_POLL_S", "0.01")
    # alive at the pre-SIGTERM gate, then dead at the first poll -> no SIGKILL.
    calls, sig = _patch_stop_collaborators(
        monkeypatch, pid=6262, alive_sequence=[True, False],
    )
    rc = cli_mod.cmd_daemon_stop(object())
    assert rc == 0

    assert ("kill", 6262, sig.SIGTERM) in calls
    assert not any(
        c[0] == "kill" and c[2] == sig.SIGKILL for c in calls
    ), calls


def test_stop_lockfile_absent_bootout_only_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls, sig = _patch_stop_collaborators(
        monkeypatch, pid=None, alive_sequence=[True],
    )
    rc = cli_mod.cmd_daemon_stop(object())
    assert rc == 0

    # bootout still removes KeepAlive...
    assert any(c[0] == "run" and "bootout" in c[1] for c in calls), calls
    # ...but NO os.kill against a bogus PID.
    assert not any(c[0] == "kill" for c in calls), calls


def test_stop_sentinel_failure_does_not_block_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls, sig = _patch_stop_collaborators(
        monkeypatch, pid=7373, alive_sequence=[True, False],
    )

    # Force the sentinel write to raise; the kill path must still run.
    import iai_mcp.daemon_state as daemon_state

    def _boom():
        raise OSError("sentinel disk full")

    monkeypatch.setattr(daemon_state, "load_state", _boom)

    rc = cli_mod.cmd_daemon_stop(object())
    assert rc == 0
    # Despite the sentinel failure, bootout + SIGTERM still happened.
    assert any(c[0] == "run" and "bootout" in c[1] for c in calls), calls
    assert ("kill", 7373, sig.SIGTERM) in calls


def test_stop_linux_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls: list = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
    # os.kill must NOT be touched on Linux.
    killed: list = []
    monkeypatch.setattr(cli_mod.os, "kill", lambda *a: killed.append(a))

    rc = cli_mod.cmd_daemon_stop(object())
    assert rc == 0
    assert ["systemctl", "--user", "stop", cli_mod.SERVICE_NAME] in calls
    assert killed == [], killed


def test_start_rebootstraps_booted_out_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls: list = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

    rc = cli_mod.cmd_daemon_start(object())
    assert rc == 0

    bootstrap_idx = next(
        i for i, c in enumerate(calls) if "bootstrap" in c
    )
    kickstart_idx = next(
        i for i, c in enumerate(calls) if "kickstart" in c
    )
    # A booted-out job is re-registered (bootstrap) before kickstart.
    assert bootstrap_idx < kickstart_idx, calls
