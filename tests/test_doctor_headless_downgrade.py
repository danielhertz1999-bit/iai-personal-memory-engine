"""/ D- tests for headless-mode doctor downgrade.

On a headless host (VPS, Linux with no display server), `(n) HID idle
source` and `(b) socket file fresh` produce noisy FAIL rows that don't
reflect a genuine fault. This suite pins the new behavior:

  1. `is_headless()` returns True on Linux only when DISPLAY and
     WAYLAND_DISPLAY are both unset.
  2. On macOS the auto-detect is SUPPRESSED (Quartz never sets DISPLAY);
     only the explicit `--headless` flag (force=True) flips the bit.
  3. `_apply_headless_downgrade()` mutates `(b)` and `(n)` FAIL -> WARN
     in place; other rows are untouched; when not headless it is a no-op.
  4. The argparse `doctor` subcommand accepts `--headless`.
"""
from __future__ import annotations

import pytest


def test_is_headless_linux_no_display_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux + DISPLAY/WAYLAND_DISPLAY both unset -> auto-detect fires."""
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=False) is True


def test_is_headless_linux_with_display_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux + DISPLAY set -> auto-detect does NOT fire."""
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=False) is False


def test_is_headless_macos_no_display_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS + DISPLAY/WAYLAND_DISPLAY unset -> NO auto-downgrade.

    This is the macOS regression guard: Quartz desktops never set DISPLAY
    or WAYLAND_DISPLAY, so a naive auto-detect would fire on every Mac
    desktop. The Linux gate prevents that.
    """
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=False) is False


def test_is_headless_macos_with_force_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `--headless` flag always wins, even on macOS."""
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=True) is True


def test_apply_headless_downgrade_mutates_b_and_n() -> None:
    """In headless mode, FAIL rows for (b) and (n) become WARN.

    Other rows (PASS or unrelated FAIL) are left untouched. The downgrade
    is in-place: same list, same CheckResult identities, only `passed` and
    `status` flipped.
    """
    from iai_mcp.doctor import CheckResult, _apply_headless_downgrade

    results = [
        CheckResult(
            name="(a) daemon process alive",
            passed=True,
            detail="PID 1 (iai_mcp.daemon)",
            status="PASS",
        ),
        CheckResult(
            name="(b) socket file fresh",
            passed=False,
            detail="present but unreachable (timeout/refused)",
            status="FAIL",
        ),
        CheckResult(
            name="(n) HID idle source",
            passed=False,
            detail="HIDIdleTime: unavailable; no idle source",
            status="FAIL",
        ),
        CheckResult(
            name="(z) AVX2 CPU support",
            passed=False,
            detail="this host lacks AVX2 -- LanceDB cannot load",
            status="FAIL",
        ),
    ]

    out = _apply_headless_downgrade(results, headless=True)

    # Identity check: in-place mutation, not a new list.
    assert out is results

    by_name = {r.name: r for r in out}

    b = by_name["(b) socket file fresh"]
    assert b.passed is True, f"(b) should now pass (WARN); got {b.passed}"
    assert b.status == "WARN", f"(b) status should be WARN; got {b.status!r}"
    assert "unreachable" in b.detail, (
        f"(b) detail must survive the downgrade; got {b.detail!r}"
    )

    n = by_name["(n) HID idle source"]
    assert n.passed is True, f"(n) should now pass (WARN); got {n.passed}"
    assert n.status == "WARN", f"(n) status should be WARN; got {n.status!r}"

    # (a) PASS row unchanged.
    a = by_name["(a) daemon process alive"]
    assert a.passed is True and a.status == "PASS"

    # (z) FAIL row OUTSIDE the downgrade allowlist stays FAIL.
    z = by_name["(z) AVX2 CPU support"]
    assert z.passed is False, f"(z) must stay FAIL; got passed={z.passed}"
    assert z.status == "FAIL", f"(z) must stay FAIL; got status={z.status!r}"


def test_apply_headless_downgrade_noop_when_not_headless() -> None:
    """When headless=False the post-process pass returns rows untouched."""
    from iai_mcp.doctor import CheckResult, _apply_headless_downgrade

    results = [
        CheckResult(
            name="(b) socket file fresh",
            passed=False,
            detail="present but unreachable",
            status="FAIL",
        ),
    ]

    out = _apply_headless_downgrade(results, headless=False)

    assert out is results
    b = out[0]
    assert b.passed is False, f"(b) must stay FAIL; got passed={b.passed}"
    assert b.status == "FAIL", f"(b) must stay FAIL; got status={b.status!r}"


def test_cli_doctor_accepts_headless_flag() -> None:
    """`iai-mcp doctor --headless` is a recognized argparse flag.

    Default (no flag) -> `args.headless is False`.
    With `--headless` -> `args.headless is True`.
    """
    from iai_mcp.cli import _build_parser

    parser = _build_parser()

    ns_with = parser.parse_args(["doctor", "--headless"])
    assert ns_with.headless is True, (
        f"expected headless=True with --headless; got {ns_with.headless!r}"
    )

    ns_default = parser.parse_args(["doctor"])
    assert ns_default.headless is False, (
        f"expected headless=False by default; got {ns_default.headless!r}"
    )
