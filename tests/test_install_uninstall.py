"""pytest verifying scripts/install.sh + scripts/uninstall.sh.

All tests run with DRY_RUN=1 (short-circuits real launchctl + kill + rm calls)
+ IAI_TEST_SKIP_BUILD=1 (short-circuits venv/pip/npm in install.sh) so the
developer's actual ~/Library/LaunchAgents/ + ~/.iai-mcp/lancedb are NEVER
touched during pytest runs.

Test matrix:
  - A: install dry-run succeeds + DRY_RUN message present
  - B: install dry-run idempotent (twice in a row, both rc=0)
  - C: uninstall dry-run succeeds
  - D: uninstall dry-run idempotent
  - E: plist template sed substitution (PYTHON_PATH + HOME) — POSIX-portable
  - F: uninstall --purge-state dry-run skips state-file rm
  - G: install.sh syntax (bash -n) valid
  - H: uninstall.sh syntax (bash -n) valid

Tests E, G, H run on any POSIX OS. Tests A-D, F invoke the LaunchAgent block
which gates on `uname == Darwin`; on Linux/CI they exit 0 with a "non-Darwin"
warn line, so they STILL run cross-platform but exercise the skip branch.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "scripts" / "install.sh"
UNINSTALL_SH = REPO / "scripts" / "uninstall.sh"
PLIST_TEMPLATE = REPO / "scripts" / "com.iai-mcp.daemon.plist.template"


def _bash_available() -> bool:
    return shutil.which("bash") is not None


def _dry_run_env() -> dict[str, str]:
    """Env for invocations that must NOT mutate the developer's machine."""
    return {**os.environ, "DRY_RUN": "1", "IAI_TEST_SKIP_BUILD": "1"}


@pytest.fixture(autouse=True)
def _scripts_exist() -> None:
    """Skip all tests if the scripts haven't been created yet (TDD safety)."""
    if not INSTALL_SH.exists():
        pytest.skip(f"{INSTALL_SH} missing — run Task 1 first")
    if not UNINSTALL_SH.exists():
        pytest.skip(f"{UNINSTALL_SH} missing — run Task 2 first")


# ---------------------------------------------------------------------------
# A. install.sh dry-run succeeds + DRY_RUN message present
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
@pytest.mark.skipif(platform.system() != "Darwin", reason="DRY_RUN message only emitted on Darwin")
def test_install_dry_run_succeeds() -> None:
    """install.sh with DRY_RUN=1 + IAI_TEST_SKIP_BUILD=1 exits 0 + emits the
    em-dash-bearing 'DRY_RUN=1 — skipping launchctl calls' marker that
    section 6 prints when uname == Darwin."""
    result = subprocess.run(
        ["bash", str(INSTALL_SH)],
        env=_dry_run_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"install.sh DRY_RUN failed:\n--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    # Message text is a contract — note the em-dash (—), not a hyphen.
    assert "DRY_RUN=1 — skipping launchctl calls" in result.stdout, (
        f"missing DRY_RUN marker in stdout:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# B. install.sh dry-run idempotent
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_install_dry_run_idempotent() -> None:
    """Running install.sh twice in a row with DRY_RUN=1 + IAI_TEST_SKIP_BUILD=1
    must both succeed (rc=0). Idempotency is the core install.sh contract."""
    env = _dry_run_env()
    for attempt in (1, 2):
        result = subprocess.run(
            ["bash", str(INSTALL_SH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"install.sh DRY_RUN attempt {attempt} failed:\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}\n"
        )


# ---------------------------------------------------------------------------
# C. uninstall.sh dry-run succeeds
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_dry_run_succeeds() -> None:
    """uninstall.sh with DRY_RUN=1 exits 0 cleanly with no real launchctl/kill/rm."""
    result = subprocess.run(
        ["bash", str(UNINSTALL_SH)],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"uninstall.sh DRY_RUN failed:\n--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    # The "done" terminator confirms the script reached the end without abort.
    assert "iai-mcp uninstalled" in result.stdout, (
        f"uninstall.sh stdout missing terminator:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# D. uninstall.sh dry-run idempotent
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_dry_run_idempotent() -> None:
    """Running uninstall.sh twice in a row must always succeed."""
    env = {**os.environ, "DRY_RUN": "1"}
    for attempt in (1, 2):
        result = subprocess.run(
            ["bash", str(UNINSTALL_SH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"uninstall.sh DRY_RUN attempt {attempt} failed:\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}\n"
        )


# ---------------------------------------------------------------------------
# E. plist template sed substitution (POSIX-portable, runs on any OS)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
@pytest.mark.skipif(not shutil.which("sed"), reason="sed unavailable")
def test_install_renders_template_with_substitutions() -> None:
    """The same `sed -e "s|{PYTHON_PATH}|...|g" -e "s|{HOME}|...|g"` invocation
    that install.sh section 6 uses must produce a plist with both placeholders
    substituted (and zero residue)."""
    if not PLIST_TEMPLATE.exists():
        pytest.skip(f"{PLIST_TEMPLATE} missing — Wave 1 (07.1-01) not complete")

    fake_python = "/fake/path/.venv/bin/python"
    fake_home = "/tmp/iai-fake-home-test-7-1-03"

    result = subprocess.run(
        [
            "sed",
            "-e", f"s|{{PYTHON_PATH}}|{fake_python}|g",
            "-e", f"s|{{HOME}}|{fake_home}|g",
            str(PLIST_TEMPLATE),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, f"sed failed: {result.stderr}"
    rendered = result.stdout

    # Both substitutions landed.
    assert fake_python in rendered, "PYTHON_PATH not substituted"
    assert fake_home in rendered, "HOME not substituted"

    # No placeholder residue.
    assert "{PYTHON_PATH}" not in rendered, "{PYTHON_PATH} placeholder remains"
    assert "{HOME}" not in rendered, "{HOME} placeholder remains"

    # Sanity check that the rendered output is a plausible plist.
    assert "<plist version=\"1.0\">" in rendered
    assert "com.iai-mcp.daemon" in rendered


# ---------------------------------------------------------------------------
# F. uninstall.sh --purge-state dry-run
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_purge_state_dry_run() -> None:
    """`uninstall.sh --purge-state` with DRY_RUN=1 must skip the actual rm
    of ~/.iai-mcp/.daemon.sock + .daemon-state.json + .lock and emit a
    'skipping rm of state files' marker so the test can verify the gate fired."""
    result = subprocess.run(
        ["bash", str(UNINSTALL_SH), "--purge-state"],
        env={**os.environ, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"uninstall.sh --purge-state DRY_RUN failed:\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n"
    )
    assert "skipping rm of state files" in result.stdout, (
        f"purge-state DRY_RUN gate did not fire:\n{result.stdout}"
    )
    # Verify the developer's actual state files (if any) were not touched.
    # (We cannot assert they DON'T exist — they may exist legitimately —
    # but the DRY_RUN message above is sufficient evidence rm was skipped.)


# ---------------------------------------------------------------------------
# G. install.sh syntax valid
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_install_sh_syntax_valid() -> None:
    """`bash -n scripts/install.sh` must exit 0 (parse-only, no side effects)."""
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (
        f"install.sh has syntax errors:\n--- STDERR ---\n{result.stderr}\n"
    )


# ---------------------------------------------------------------------------
# H. uninstall.sh syntax valid
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _bash_available(), reason="bash unavailable")
def test_uninstall_sh_syntax_valid() -> None:
    """`bash -n scripts/uninstall.sh` must exit 0 (parse-only, no side effects)."""
    result = subprocess.run(
        ["bash", "-n", str(UNINSTALL_SH)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (
        f"uninstall.sh has syntax errors:\n--- STDERR ---\n{result.stderr}\n"
    )
