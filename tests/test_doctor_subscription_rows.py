"""doctor rows for Claude subscription credentials + anthropic
SDK absence. Validates check_o + check_p behavior under each documented
status (PASS, WARN, FAIL).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_valid_creds(creds_path: Path, sub_type: str = "max") -> None:
    """Write a credentials.json file with the modern claudeAiOauth schema."""
    expires_at_ms = int(
        (datetime.now(tz=timezone.utc) + timedelta(days=365)).timestamp() * 1000
    )
    creds_path.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-stub",
            "refreshToken": "sk-ant-ort01-stub",
            "expiresAt": expires_at_ms,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": sub_type,
            "rateLimitTier": f"default_claude_{sub_type}_5x",
        }
    }))


# ---------------------------------------------------------------------------
# check_o — subscription credentials presence + validity
# ---------------------------------------------------------------------------


def test_check_o_pass_on_valid_subscription(tmp_path, monkeypatch):
    """a valid subscription credentials.json with inference scope
    flips check_o to PASS."""
    from iai_mcp import claude_cli
    from iai_mcp.doctor import check_o_subscription_credentials

    creds = tmp_path / ".credentials.json"
    _write_valid_creds(creds, sub_type="pro_max")
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    result = check_o_subscription_credentials()
    assert result.status == "PASS"
    assert result.passed is True
    assert "(o)" in result.name
    assert "pro_max" in result.detail


def test_check_o_warn_when_credentials_missing(tmp_path, monkeypatch):
    """missing credentials.json -> WARN (advisory, not FAIL).
    Daemon falls back to Tier-0; no LLM critic, no nightly insight."""
    from iai_mcp import claude_cli
    from iai_mcp.doctor import check_o_subscription_credentials

    monkeypatch.setattr(
        claude_cli, "CREDENTIALS_PATH", tmp_path / "missing.json",
    )

    result = check_o_subscription_credentials()
    assert result.status == "WARN"
    assert result.passed is True  # WARN is advisory, does not fail doctor exit
    assert "credentials_file_missing" in result.detail


def test_check_o_warn_when_credentials_expired_and_no_refresh_token(
    tmp_path, monkeypatch,
):
    """expired accessToken AND missing refreshToken -> WARN. The
    next claude -p call has no way to recover; surface the cause before
    it bites. (Expired accessToken WITH refreshToken is healthy --
    the CLI refreshes transparently.)"""
    from iai_mcp import claude_cli
    from iai_mcp.doctor import check_o_subscription_credentials

    creds = tmp_path / ".credentials.json"
    expired_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=1)).timestamp() * 1000
    )
    creds.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-stub",
            # No refreshToken -- the gate must fail closed.
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
            "subscriptionType": "max",
        }
    }))
    monkeypatch.setattr(claude_cli, "CREDENTIALS_PATH", creds)

    result = check_o_subscription_credentials()
    assert result.status == "WARN"
    assert "credentials_expired" in result.detail


# ---------------------------------------------------------------------------
# check_p — anthropic SDK absent
# ---------------------------------------------------------------------------


def test_check_p_warn_when_sdk_importable(monkeypatch):
    """stale install where `anthropic` site-packages still resolves
    -> WARN. Advisory only; daemon does not USE the SDK, but operator
    should `pip uninstall anthropic` to keep the venv clean."""
    from iai_mcp.doctor import check_p_anthropic_sdk_absent

    # Fake an importable anthropic module so the check sees it.
    fake_module = type(sys)("anthropic")
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    result = check_p_anthropic_sdk_absent()
    assert result.status == "WARN"
    assert result.passed is True
    assert "leftover" in result.detail.lower() or "pip uninstall" in result.detail.lower()


def test_check_p_pass_when_sdk_absent(monkeypatch):
    """clean install where `import anthropic` raises
    ImportError -> PASS. The expected state."""
    from iai_mcp.doctor import check_p_anthropic_sdk_absent

    # Remove anthropic from sys.modules if present, then patch the import
    # mechanism so even a fresh import fails.
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _raise_for_anthropic(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_raise_for_anthropic):
        result = check_p_anthropic_sdk_absent()
    assert result.status == "PASS"
    assert result.passed is True


# ---------------------------------------------------------------------------
# Wire-in: both rows are present in run_diagnosis() output
# ---------------------------------------------------------------------------


def test_run_diagnosis_includes_o_and_p_rows():
    """run_diagnosis() includes (o) + (p) rows in the documented
    position (after m/n, before z)."""
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    o_rows = [n for n in names if n.startswith("(o)")]
    p_rows = [n for n in names if n.startswith("(p)")]
    assert len(o_rows) == 1, f"expected exactly one (o) row, got {o_rows}"
    assert len(p_rows) == 1, f"expected exactly one (p) row, got {p_rows}"

    o_idx = names.index(o_rows[0])
    p_idx = names.index(p_rows[0])
    z_idx = next((i for i, n in enumerate(names) if n.startswith("(z)")), -1)
    n_idx = next((i for i, n in enumerate(names) if n.startswith("(n)")), -1)

    assert n_idx < o_idx < p_idx, "expected (n) < (o) < (p) ordering"
    if z_idx >= 0:
        assert p_idx < z_idx, "(z) must remain the last row"
