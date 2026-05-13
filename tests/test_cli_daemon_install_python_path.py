"""tests: regression-lock for `iai-mcp daemon install`
sys.executable substitution into launchd plist + systemd user unit.

Locks the contract that `_render_launchd_plist` and `_render_systemd_unit`
substitute `sys.executable` in place of the template `/usr/local/bin/python3`
and `/usr/bin/python3` placeholders. Without this contract, the daemon
runs under whatever `python3` happens to be first on PATH at launchd /
systemd invocation, which on macOS is typically the SIP-protected
`/usr/local/bin/python3` -- different from the venv Python where iai-mcp
and its dependencies live.

VERIFY result (planner + executor 2026-05-01): production code already
does the substitution. `src/iai_mcp/cli.py::_render_launchd_plist`
calls `text.replace("/usr/local/bin/python3", sys.executable)`, and
`_render_systemd_unit` calls
`text.replace("/usr/bin/python3", sys.executable)`. The plist template
at `deploy/launchd/com.iai-mcp.daemon.plist` carries
`<string>/usr/local/bin/python3</string>` inside `ProgramArguments`, and
`deploy/systemd/iai-mcp-daemon.service` carries
`ExecStart=/usr/bin/python3 -m iai_mcp.daemon`. Production-code change
for this plan is ZERO LINES; this file is a regression lock so a future
refactor that hardcodes the path will fail these tests.

Test 3 (`test_install_warns_when_sys_executable_lacks_psutil`) verified
during Sub-step 1: `cmd_daemon_install` (cli.py 268-362) does NOT carry a
`subprocess.run([sys.executable, "-c", "import psutil"])` probe today.
Per 07.14-CONTEXT.md ("only if gap-driven patch is needed: ... defer
adding such a row to a future phase. Do NOT add it speculatively in
07.14"), the WARN-on-missing-psutil contract is xfail-marked: the
contract is documented for a future plan to enforce, but adding the
probe speculatively is out of scope.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pytest


def _make_install_args(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace matching `cmd_daemon_install` args."""
    defaults = dict(dry_run=True, yes=True)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_install_uses_sys_executable_macos(monkeypatch):
    """`_render_launchd_plist` substitutes `/usr/local/bin/python3` with
    the absolute path of `sys.executable` of the invoking interpreter.

    Scoping note: we patch `iai_mcp.cli.sys.executable` (NOT global
    `sys.executable`) so the override is local to the cli module's `sys`
    reference and does not leak to other modules during pytest collection.
    """
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert f"<string>{fake_python}</string>" in rendered, (
        f"plist did not substitute sys.executable; rendered text:\n{rendered[:500]}"
    )
    assert "<string>/usr/local/bin/python3</string>" not in rendered, (
        "plist still contains the unsubstituted /usr/local/bin/python3 placeholder"
    )


def test_install_uses_sys_executable_linux(monkeypatch):
    """`_render_systemd_unit` substitutes `/usr/bin/python3` with
    `sys.executable`.

    Verifies both that the substituted path appears AND that the original
    `/usr/bin/python3 -m iai_mcp.daemon` ExecStart line is fully replaced
    (not just shadowed by an additional line).
    """
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_systemd_unit

    rendered = _render_systemd_unit()
    assert f"{fake_python} -m iai_mcp.daemon" in rendered or (
        f"{fake_python}" in rendered and "iai_mcp.daemon" in rendered
    ), f"systemd unit did not substitute sys.executable; rendered:\n{rendered[:500]}"
    assert "/usr/bin/python3 -m iai_mcp.daemon" not in rendered, (
        "systemd unit still contains the unsubstituted /usr/bin/python3 placeholder"
    )


# ============================================================================
# Test 3 -- xfail per 07.14-CONTEXT.md deferral
# ============================================================================
# Sub-step 1 verification (executor 2026-05-01): cmd_daemon_install
# (src/iai_mcp/cli.py lines 268-362) does NOT contain a
# `subprocess.run([sys.executable, "-c", "import psutil"])` probe today.
#
# Per 07.14-CONTEXT.md "only if gap-driven patch is needed: ...
# defer adding such a row to a future phase. Do NOT add it speculatively
# in 07.14".
#
# This xfail documents the contract for a future plan that adds the
# probe. If/when the probe lands, the xfail will flip to xpass and the
# developer un-marks it. `strict=False` so an xpass does not fail the
# suite during the transition.
# ============================================================================


# plist invariants -----------------------------


def test_plist_keepalive_is_crashed_only(monkeypatch):
    """Plist KeepAlive uses {"Crashed": true} only -- NOT SuccessfulExit=false.

    lifecycle model: graceful exit 0 on HIBERNATION must
    NOT trigger respawn (so the daemon stays dead until wrapper
    kickstart fires). Crashed=true respawns only on non-zero exit
    (the LifecycleLockConflict path); SuccessfulExit=false would
    create a respawn loop because exit 0 is now the steady state.
    """
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    # Crashed-only block must be present.
    assert "<key>Crashed</key>" in rendered
    # Legacy SuccessfulExit=false must be GONE.
    assert "<key>SuccessfulExit</key>" not in rendered, (
        "removed SuccessfulExit=false from the plist. Its presence "
        "would create a respawn loop because exit 0 is now the steady state."
    )


def test_plist_lifecycle_env_vars_present(monkeypatch):
    """The plist defines LIFECYCLE_* + sleep-quarantine env vars.

    cadence knobs become production-tunable via the plist
    EnvironmentVariables block.
    """
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert "<key>LIFECYCLE_DROWSY_AFTER_SEC</key>" in rendered
    assert "<key>LIFECYCLE_SLEEP_HEARTBEAT_IDLE_SEC</key>" in rendered
    assert "<key>LIFECYCLE_HIBERNATE_AFTER_SEC</key>" in rendered
    assert "<key>IAI_MCP_SLEEP_QUARANTINE_TTL_HOURS</key>" in rendered


def test_plist_legacy_env_vars_removed(monkeypatch):
    """Legacy env vars from the RSS-watchdog + idle_watcher era are gone."""
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert "<key>IAI_MCP_RSS_RESTART_THRESHOLD_MB</key>" not in rendered, (
        "RSS-watchdog removed in Task 1.4; env var must be gone "
        "from the plist."
    )
    assert "<key>IAI_DAEMON_IDLE_SHUTDOWN_SECS</key>" not in rendered
    assert "<key>IAI_MCP_SKIP_STARTUP_OPTIMIZE</key>" not in rendered


@pytest.mark.xfail(
    reason=(
        "psutil-availability probe NOT in cmd_daemon_install today. "
        "Adding speculatively is deferred per 07.14-CONTEXT.md "
        '("only if gap-driven patch is needed: ... defer adding such a '
        'row to a future phase"). This xfail documents the contract for '
        "a future plan."
    ),
    strict=False,
)
def test_install_warns_when_sys_executable_lacks_psutil(
    monkeypatch, capsys, tmp_path,
):
    """When the venv-resolved Python lacks `psutil`, install emits a WARN
    (not FAIL) with a hint to install psutil + re-run.

    NOTE: deferred per CONTEXT.md -- xfail until a future plan adds
    the psutil-availability probe to `cmd_daemon_install`.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    # Simulate `import psutil` failing under the target Python.
    real_run = subprocess.run

    def _fake_run(cmd, **kwargs):
        # Match: subprocess.run([sys.executable, "-c", "import psutil"], ...)
        if (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-c"
            and cmd[2] == "import psutil"
        ):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("subprocess.run", _fake_run)

    from iai_mcp.cli import cmd_daemon_install

    rc = cmd_daemon_install(_make_install_args(dry_run=True, yes=True))
    err = capsys.readouterr().err
    # WARN != FAIL: install proceeds (rc == 0) but stderr carries the hint.
    assert rc == 0, f"install must NOT fail on missing psutil; got rc={rc}"
    err_lower = err.lower()
    assert "psutil" in err_lower
    assert "iai-mcp daemon install" in err_lower
    assert "re-run" in err_lower
