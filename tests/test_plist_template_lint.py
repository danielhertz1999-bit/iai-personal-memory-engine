"""Lint + structural assertions for the LaunchAgent plist template.

The template ``scripts/com.iai-mcp.daemon.plist.template`` is rendered by
``scripts/install.sh``: ``{PYTHON_PATH}`` and ``{HOME}`` are
substituted, then the result is written to
``~/Library/LaunchAgents/com.iai-mcp.daemon.plist`` and registered with
``launchctl load -w``.

These tests guard the *template itself*:

  * ``test_template_renders_to_valid_plist`` — substitute the placeholders
    with realistic values, write to a tmp file, run ``plutil -lint``, and
    assert exit 0 + ``OK`` in stdout.
  * ``test_template_has_required_keys`` — string-level presence of every
    required field (Sockets, RunAtLoad, SockPathMode=384, KeepAlive,
    IAI_MCP_LAUNCHD_MANAGED).
  * ``test_template_does_not_have_RunAtLoad_true`` — regression trap: the
    legacy ``deploy/launchd/com.iai-mcp.daemon.plist`` uses
    ``<key>RunAtLoad</key><true/>`` which defeats socket activation; we
    must NOT reintroduce that pattern in the new template.

The whole module skips on non-Darwin hosts (``plutil`` is macOS-only).
"""
from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="plutil is macOS-only",
)

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "com.iai-mcp.daemon.plist.template"


def test_template_renders_to_valid_plist(tmp_path: Path) -> None:
    """Rendered plist (post-substitution) passes plutil -lint."""
    template_text = TEMPLATE.read_text()
    rendered = template_text.replace(
        "{PYTHON_PATH}", "/usr/bin/python3"
    ).replace("{HOME}", "/tmp/iai-fake-home")
    rendered_path = tmp_path / "com.iai-mcp.daemon.plist"
    rendered_path.write_text(rendered)

    result = subprocess.run(
        ["plutil", "-lint", str(rendered_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"plutil -lint FAILED on rendered template:\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "OK" in result.stdout, result.stdout


def test_template_has_required_keys() -> None:
    """All required fields present (string-level, no regex)."""
    text = TEMPLATE.read_text()
    required_markers = [
        "<key>Sockets</key>",
        "<key>RunAtLoad</key>",
        "<false/>",
        "<key>SockPathMode</key>",
        "<integer>384</integer>",
        "<key>KeepAlive</key>",
        "IAI_MCP_LAUNCHD_MANAGED",
    ]
    missing = [m for m in required_markers if m not in text]
    assert not missing, f"template missing required markers: {missing}"


def test_template_does_not_have_RunAtLoad_true() -> None:
    """Regression trap: legacy deploy/launchd plist's <true/> bug must NOT appear.

    The legacy ``deploy/launchd/com.iai-mcp.daemon.plist`` uses
    ``<key>RunAtLoad</key><true/>`` which defeats socket activation
    (eager spawn at user login = no listener pre-bind). The
    template MUST use ``<false/>`` so launchd defers spawn until the
    first incoming connection on the pre-bound socket.
    """
    text = TEMPLATE.read_text()
    match = re.search(r"<key>RunAtLoad</key>\s*<true/>", text)
    assert match is None, (
        "REGRESSION: template contains <key>RunAtLoad</key>...<true/> which "
        "defeats socket activation. Use <false/> instead."
    )
