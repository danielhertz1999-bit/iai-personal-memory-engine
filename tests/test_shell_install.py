"""pytest wrapper for the platform-specific shell tests.

Runs tests/shell/test_launchd_install.sh on macOS and
tests/shell/test_systemd_install.sh on Linux WHEN the env var
`IAI_MCP_RUN_SHELL_INSTALL_TESTS=1` is set. CI sets this env var on the
correct runner; local dev does NOT (the scripts perform a real launchctl
bootstrap / systemctl --user enable cycle which would install the daemon
on the developer's machine and produce a persistent background process).

When the env var is unset, the actual-execution tests skip but the static
verification tests still run (executable bit, skip branches, C4 invariants
referenced in script source).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest


SHELL_DIR = Path(__file__).resolve().parent / "shell"
LAUNCHD_SCRIPT = SHELL_DIR / "test_launchd_install.sh"
SYSTEMD_SCRIPT = SHELL_DIR / "test_systemd_install.sh"
RUN_SHELL = os.environ.get("IAI_MCP_RUN_SHELL_INSTALL_TESTS") == "1"


def _bash_available() -> bool:
    return shutil.which("bash") is not None


@pytest.mark.skipif(not RUN_SHELL, reason="set IAI_MCP_RUN_SHELL_INSTALL_TESTS=1 to run real launchctl bootstrap test")
@pytest.mark.skipif(not LAUNCHD_SCRIPT.exists(), reason="launchd shell test missing")
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only")
def test_launchd_install_idempotency() -> None:
    """C4 + Pitfall 5 + DAEMON-10 idempotency end-to-end on the host."""
    result = subprocess.run(
        ["bash", str(LAUNCHD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"launchd shell test FAILED:\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    # Either PASS or SKIP is acceptable (skip happens when user has a
    # pre-existing plist we won't clobber).
    assert "PASS" in result.stdout or "SKIP" in result.stdout, result.stdout


@pytest.mark.skipif(not RUN_SHELL, reason="set IAI_MCP_RUN_SHELL_INSTALL_TESTS=1 to run real systemctl --user enable test")
@pytest.mark.skipif(not SYSTEMD_SCRIPT.exists(), reason="systemd shell test missing")
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-only")
def test_systemd_install_idempotency() -> None:
    """C4 + Pitfall 5 + DAEMON-10 idempotency end-to-end on the host."""
    result = subprocess.run(
        ["bash", str(SYSTEMD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"systemd shell test FAILED:\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "PASS" in result.stdout or "SKIP" in result.stdout, result.stdout


@pytest.mark.skipif(not LAUNCHD_SCRIPT.exists(), reason="launchd shell test missing")
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_launchd_script_skips_on_non_macos_platform() -> None:
    """Self-skip branch verification (always-runnable smoke test).

    Invokes bash with `uname` reporting Linux via env override is not
    portable, so we instead verify the SKIP branch executes correctly when
    the script source contains the right guard. On non-macOS hosts, running
    the script directly should exit 0 with `SKIP: not macOS` printed.
    """
    if platform.system() == "Darwin":
        pytest.skip("on Darwin -- this asserts the non-Darwin skip branch")
    result = subprocess.run(
        ["bash", str(LAUNCHD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "SKIP: not macOS" in result.stdout


@pytest.mark.skipif(not SYSTEMD_SCRIPT.exists(), reason="systemd shell test missing")
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_systemd_script_skips_on_non_linux_platform() -> None:
    """Self-skip branch verification for the systemd script."""
    if platform.system() == "Linux":
        pytest.skip("on Linux -- this asserts the non-Linux skip branch")
    result = subprocess.run(
        ["bash", str(SYSTEMD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "SKIP: not Linux" in result.stdout


def test_shell_scripts_are_executable() -> None:
    """Both scripts must have the executable bit so CI can invoke directly."""
    import os
    if LAUNCHD_SCRIPT.exists():
        assert os.access(LAUNCHD_SCRIPT, os.X_OK), (
            f"{LAUNCHD_SCRIPT} not executable"
        )
    if SYSTEMD_SCRIPT.exists():
        assert os.access(SYSTEMD_SCRIPT, os.X_OK), (
            f"{SYSTEMD_SCRIPT} not executable"
        )


def test_shell_scripts_have_skip_branch() -> None:
    """Cross-platform skip branch must exist in both scripts (Plan 04-05 AC)."""
    if LAUNCHD_SCRIPT.exists():
        text = LAUNCHD_SCRIPT.read_text()
        assert "SKIP: not macOS" in text, "launchd script missing macOS skip branch"
    if SYSTEMD_SCRIPT.exists():
        text = SYSTEMD_SCRIPT.read_text()
        assert "SKIP: not Linux" in text, "systemd script missing Linux skip branch"


def test_shell_scripts_check_c4_invariant() -> None:
    """Both scripts must verify C4 cleanup of all 3 state files."""
    for script in (LAUNCHD_SCRIPT, SYSTEMD_SCRIPT):
        if not script.exists():
            continue
        text = script.read_text()
        assert "C4" in text, f"{script.name} missing C4 reference"
        assert ".lock" in text, f"{script.name} does not check lock file removal"
        assert ".daemon.sock" in text or "SOCK" in text, (
            f"{script.name} does not check socket file removal"
        )
        assert ".daemon-state.json" in text or "STATE" in text, (
            f"{script.name} does not check state file removal"
        )
