"""Phase 07.10 W3 / Plan 05: doctor `check_h_crypto_file_state` + top-of-output hint.

Locks the executable spec for the 8th doctor check row + the migration
remediation hint that prints at the very top of doctor's output when the
file-missing-but-Keychain-entry-exists state is detected (Phase 07.10 D-12).

Detection matrix:
| file present + valid | keyring entry | doctor output       |
| yes                  | any           | PASS                |
| no                   | yes           | WARN + top-of-output hint pointing at `iai-mcp crypto migrate-to-file` |
| no                   | no/error      | PASS (clean fresh-install state)                                      |
| yes (malformed)      | any           | FAIL: prints the file's CryptoKeyError message                        |

These tests run independently of the existing `test_doctor_checklist.py`
fixtures (no daemon socket, no lock file): they only exercise
`check_h_crypto_file_state` directly + the top-of-output hint helper.
"""
from __future__ import annotations

import io
import os
import secrets
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------- check_h_crypto_file_state

def test_check_h_pass_when_file_present_and_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-12 case 1 — valid 0o600 32-byte key file → PASS.

    File-backend resolution honors `IAI_MCP_STORE`; pointing it at tmp_path
    makes the lazy `_key_file_path()` return `tmp_path/.crypto.key`. No
    keyring touch on the file-present branch.
    """
    from iai_mcp.doctor import check_h_crypto_file_state

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    result = check_h_crypto_file_state()
    assert result.status == "PASS", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is True
    assert ".crypto.key" in result.detail


def test_check_h_warn_when_file_missing_and_keyring_has_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-12 case 2 — file absent BUT keyring has a key → WARN with migrate-to-file hint.

    Monkeypatches the LOCAL `keyring.get_password` import inside the check
    so the test does not actually probe the user's macOS Keychain.
    """
    from iai_mcp.doctor import check_h_crypto_file_state

    # File absent: nothing at tmp_path/.crypto.key.
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert not (tmp_path / ".crypto.key").exists()

    # Pretend a Keychain entry exists.
    import keyring as _keyring

    fake_b64 = "Zm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmE="  # 32-byte plausible base64url

    def fake_get(service: str, username: str) -> str | None:
        return fake_b64

    monkeypatch.setattr(_keyring, "get_password", fake_get)

    result = check_h_crypto_file_state()
    assert result.status == "WARN", f"unexpected status={result.status} detail={result.detail}"
    assert "migrate-to-file" in result.detail.lower()
    # WARN must NOT report failure — it does not flip exit code to 1.
    assert result.passed is True


def test_check_h_pass_when_file_missing_and_no_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-12 case 3 — file absent AND no Keychain entry → PASS (clean fresh install).

    Detail mentions both `crypto init` and `IAI_MCP_CRYPTO_PASSPHRASE`
    so a fresh-install user has actionable guidance.
    """
    from iai_mcp.doctor import check_h_crypto_file_state

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert not (tmp_path / ".crypto.key").exists()

    # Simulate "no Keychain entry": get_password returns None.
    import keyring as _keyring

    def fake_get(service: str, username: str) -> str | None:
        return None

    monkeypatch.setattr(_keyring, "get_password", fake_get)

    result = check_h_crypto_file_state()
    assert result.status == "PASS", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is True
    # Detail should point fresh-install users at `crypto init` or the passphrase env.
    detail_l = result.detail.lower()
    assert "init" in detail_l or "passphrase" in detail_l


def test_check_h_pass_when_keyring_backend_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-12 case 3b — file absent AND keyring NoKeyringError → PASS (clean fresh install).

    Linux servers without a Secret Service backend should be treated the
    same as 'no Keychain entry detected' — not a failure, not a warning.
    """
    from iai_mcp.doctor import check_h_crypto_file_state

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    assert not (tmp_path / ".crypto.key").exists()

    import keyring as _keyring
    import keyring.errors as _keyring_errors

    def raise_no_backend(service: str, username: str) -> str | None:
        raise _keyring_errors.NoKeyringError("no backend available (test-stub)")

    monkeypatch.setattr(_keyring, "get_password", raise_no_backend)

    result = check_h_crypto_file_state()
    assert result.status == "PASS", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is True


def test_check_h_fail_when_file_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-12 case 4 — file exists but has wrong length → FAIL with `wrong length` in detail."""
    from iai_mcp.doctor import check_h_crypto_file_state

    key_path = tmp_path / ".crypto.key"
    # Wrong length: 31 bytes instead of 32.
    key_path.write_bytes(b"\x00" * 31)
    os.chmod(key_path, 0o600)

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    result = check_h_crypto_file_state()
    assert result.status == "FAIL", f"unexpected status={result.status} detail={result.detail}"
    assert result.passed is False
    assert "wrong length" in result.detail.lower() or "malformed" in result.detail.lower()


# ---------------------------------------------------------------- top-of-output hint helper

def test_format_top_of_output_hint_emits_line_when_check_h_warns() -> None:
    """D-12 — when a WARN row for check_h is present, the helper emits a `> hint:` line
    that names `migrate-to-file` so the user sees the fix BEFORE the row-by-row print.
    """
    from iai_mcp.doctor import CheckResult, _format_top_of_output_hint

    results = [
        CheckResult("(a) daemon process alive", True, "PID 12345 (iai_mcp.daemon)", status="PASS"),
        CheckResult(
            "(h) crypto key file state",
            True,
            "crypto key file missing at /tmp/x/.crypto.key, but a Keychain entry was found.\n"
            "  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key.",
            status="WARN",
        ),
    ]

    hint = _format_top_of_output_hint(results)
    assert hint is not None, "WARN row for check_h must produce a hint"
    assert hint.startswith("> hint:"), f"hint must be prefixed with `> hint:`, got: {hint!r}"
    assert "migrate-to-file" in hint, f"hint must name migrate-to-file, got: {hint!r}"


def test_format_top_of_output_hint_returns_none_when_no_warn() -> None:
    """No WARN row → no hint."""
    from iai_mcp.doctor import CheckResult, _format_top_of_output_hint

    results = [
        CheckResult("(a) daemon process alive", True, "PID 12345 (iai_mcp.daemon)", status="PASS"),
        CheckResult("(h) crypto key file state", True, "key file present", status="PASS"),
    ]

    assert _format_top_of_output_hint(results) is None


# ---------------------------------------------------------------- run_diagnosis includes check_h

def test_run_diagnosis_includes_check_h(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """D-12 wire-in -- `run_diagnosis()` includes the check_h crypto-key row.

    Originally a positional assertion (8th row); rewritten to name-based
    lookup so subsequent doctor-row additions (Phase 10.4 added m + n)
    do not regress this contract. The (h) and (i) rows must both be
    present in the returned list.

    Uses IAI_MCP_STORE pointing at tmp_path and a valid key file so check_h
    returns PASS without hitting the user's real keyring or filesystem.
    """
    from iai_mcp.doctor import run_diagnosis

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    # Other checks may FAIL in this environment (no daemon running) -- that's
    # fine, we only assert (h) and (i) are present by name.
    results = run_diagnosis()
    h_rows = [r for r in results if "(h)" in r.name and "crypto" in r.name.lower()]
    assert len(h_rows) == 1, (
        f"expected exactly one (h) crypto row in run_diagnosis(); "
        f"got {len(h_rows)} from {[r.name for r in results]}"
    )
    i_rows = [r for r in results if "(i)" in r.name and "lance" in r.name.lower()]
    assert len(i_rows) == 1, (
        f"expected exactly one (i) lance versions row in run_diagnosis(); "
        f"got {len(i_rows)} from {[r.name for r in results]}"
    )


# ---------------------------------------------------------------- cmd_doctor wire-in (advisor-driven)

def test_cmd_doctor_prints_hint_at_top_when_check_h_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """D-12 wire-in pin (advisor) — cmd_doctor MUST call _format_top_of_output_hint
    BEFORE print_checklist so the hint appears at the very top of stdout.

    Rationale: helper-level tests verify the helper produces the right string,
    and run_diagnosis() returns 8 rows — but neither verifies that cmd_doctor
    actually wires the helper into the print path. A future refactor that
    drops the 3-line `if hint is not None: print(hint); print()` block in
    cmd_doctor would not break any other test in this file. This test pins
    the placement-at-top guarantee.

    Strategy: monkeypatch `doctor.run_diagnosis` to return a synthetic 8-row
    list with one WARN row (avoids mocking daemon-state/socket/lock/store/lsof
    simultaneously). Capture stdout and assert the `> hint:` line index is
    BEFORE the row-by-row checklist header.
    """
    import argparse

    from iai_mcp import doctor as _doctor

    synthetic = [
        _doctor.CheckResult("(a) daemon process alive", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(b) socket file fresh", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(c) lock file healthy", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(d) no orphan iai_mcp.core procs", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(e) daemon state file valid", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(f) lancedb store readable", True, "synthetic", status="PASS"),
        _doctor.CheckResult("(g) no dup binders", True, "synthetic", status="PASS"),
        _doctor.CheckResult(
            "(h) crypto key file state",
            True,
            (
                "crypto key file missing at /tmp/.crypto.key, but a Keychain entry was found.\n"
                "  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key."
            ),
            status="WARN",
        ),
    ]
    monkeypatch.setattr(_doctor, "run_diagnosis", lambda: synthetic)

    args = argparse.Namespace(apply=False, yes=False)
    rc = _doctor.cmd_doctor(args)

    captured = capsys.readouterr().out

    hint_idx = captured.find("> hint:")
    header_idx = captured.find("IAI-MCP Doctor")
    assert hint_idx >= 0, f"expected `> hint:` line in stdout, got:\n{captured!r}"
    assert header_idx >= 0, f"expected checklist header in stdout, got:\n{captured!r}"
    assert hint_idx < header_idx, (
        f"hint (idx {hint_idx}) must appear BEFORE checklist header (idx {header_idx})\n"
        f"stdout was:\n{captured}"
    )
    # The hint must name the actionable command.
    assert "migrate-to-file" in captured[: header_idx], (
        f"hint must name `migrate-to-file` ABOVE the checklist header; "
        f"top-of-output region was: {captured[:header_idx]!r}"
    )
    # Exit code: WARN does NOT flip to 1 (advisory only); rc must be 0.
    assert rc == 0, f"WARN rows must not change exit code; got rc={rc}"


# ---------------------------------------------------------------- CheckResult back-compat

def test_check_result_three_arg_constructor_still_works() -> None:
    """Phase 07.10 (Rule 1 deviation): adding `status` to CheckResult must NOT
    break existing tests that construct it with 3 positional args
    (test_doctor_checklist.py uses the 3-arg form ~14 times).
    """
    from iai_mcp.doctor import CheckResult

    r_pass = CheckResult("(x) example", True, "ok")
    assert r_pass.passed is True
    assert r_pass.detail == "ok"
    # Default status must be derived from `passed` so legacy 3-arg construction
    # produces a sensible value.
    assert r_pass.status in ("PASS", "FAIL")
    assert r_pass.status == "PASS"

    r_fail = CheckResult("(y) example", False, "broken")
    assert r_fail.status == "FAIL"
