"""Brand-surface gate for the `iai` rebrand.

Tests in this module assert:
  - The user CLI logo spells IAI CLI (the `MCP` wordmark became `CLI`)
  - The tagline reads `iai-cli ·` (not `iai-mcp ·`)
  - The doctor header reads `iai doctor` (not `IAI-MCP Doctor`)
  - The consent banner header reads `iai Sleep Daemon` (not `IAI-MCP Sleep Daemon`)
  - Internal identifiers (package import, module attributes) are unchanged

All tests are hermetic: no daemon, no real store, no real HOME.
"""
from __future__ import annotations

import contextlib
import io

import pytest


# ---------------------------------------------------------------------------
# Task 1 — Logo + tagline (BRAND-01)
# ---------------------------------------------------------------------------


def test_logo_has_no_mcp_columns(monkeypatch):
    """BRAND-01: the M-column block signature is absent (the wordmark is CLI, not MCP).

    The M-column in ANSI Shadow font starts with the substring `███╗   ███╗`
    (line 0 of the old `IAI MCP` art). Its absence proves the `MCP` wordmark
    was replaced. Box-drawing chars must still be present (logo still renders
    the `IAI CLI` figure).
    """
    monkeypatch.setenv("NO_COLOR", "1")

    from iai_mcp.iai_cli import _print_logo

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_logo()
    output = buf.getvalue()

    assert "███╗   ███╗" not in output, (
        "M-column signature found — the `MCP` wordmark was not replaced in the logo"
    )
    assert "█" in output, "No box-drawing chars found — logo appears empty after edit"


def test_tagline_reads_iai_cli(monkeypatch):
    """BRAND-01: tagline reads `iai-cli · terminal memory for your agent` (no `iai-mcp ·`)."""
    monkeypatch.setenv("NO_COLOR", "1")

    from iai_mcp.iai_cli import _print_logo

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_logo()
    output = buf.getvalue()

    assert "iai-cli · terminal memory for your agent" in output, (
        f"Tagline not found in logo output. Got:\n{output!r}"
    )
    assert "iai-mcp ·" not in output, (
        "Old `iai-mcp ·` tagline still present in logo output"
    )


def test_brand_internals_intact():
    """BRAND-03: the rename did not break module imports or key attributes."""
    import iai_mcp  # package import must remain `iai_mcp`
    from iai_mcp import iai_cli  # noqa: F401  module importable

    # Core helpers still present
    assert callable(iai_cli._color), "_color helper missing"
    assert callable(iai_cli._print_logo), "_print_logo missing"


# ---------------------------------------------------------------------------
# Task 2 — Doctor header + consent banner (BRAND-03)
# ---------------------------------------------------------------------------


def test_doctor_header_reads_iai(monkeypatch, tmp_path):
    """BRAND-03: print_checklist emits `iai doctor` header, NOT `IAI-MCP Doctor`."""
    monkeypatch.setenv("NO_COLOR", "1")
    # Hermetic: no real store needed for a header assert
    from iai_mcp.doctor import CheckResult, print_checklist

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # Empty list is fine — header prints before the results loop
        print_checklist([])
    output = buf.getvalue()

    assert "iai doctor" in output, (
        f"Doctor header missing `iai doctor`. Got:\n{output!r}"
    )
    assert "IAI-MCP Doctor" not in output, (
        "Old `IAI-MCP Doctor` header still present in doctor output"
    )


def test_consent_banner_header_reads_iai():
    """BRAND-03: CONSENT_BANNER contains `iai Sleep Daemon`, not `IAI-MCP Sleep Daemon`."""
    from iai_mcp.cli import CONSENT_BANNER

    assert "iai Sleep Daemon" in CONSENT_BANNER, (
        f"Consent banner header missing `iai Sleep Daemon`. Banner:\n{CONSENT_BANNER!r}"
    )
    assert "IAI-MCP Sleep Daemon" not in CONSENT_BANNER, (
        "Old `IAI-MCP Sleep Daemon` header still present in consent banner"
    )
