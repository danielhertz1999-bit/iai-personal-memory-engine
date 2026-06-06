"""Regression guard: 3-turn sanity for total_session_cost.

CI-runnable guard for bench/total_session_cost.py. The full 10-turn script
runs ad-hoc; this test exercises the shape contracts and the
minimal-vs-standard invariant at CI speed.

Acceptance contracts:
  - minimal total <= standard total (sanity; if not, regressed somewhere)
  - per_turn list has exactly 10 entries (fixed script)
  - counter mode honest-disclosed in JSON (anthropic-count-tokens |
    tiktoken-cl100k-proxy | heuristic-char4)
  - reference-gate failure flips passed=False

See:
- bench/total_session_cost.py — the harness under guard
- bench/tokens.py — 3-tier counter fallback pattern reused here
"""
from __future__ import annotations

import pytest


def test_total_session_cost_reports_per_turn():
    """script is the fixed 10-turn sequence."""
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="minimal")

    assert "per_turn" in out
    assert isinstance(out["per_turn"], list)
    assert len(out["per_turn"]) == 10, (
        f"D5-08 script has 10 turns; got {len(out['per_turn'])}"
    )
    assert out["total_tokens"] == sum(out["per_turn"])
    assert out["adapter"] == "iai-mcp"
    assert out["wake_depth"] == "minimal"


def test_total_session_cost_minimal_le_standard():
    """Invariant: wake_depth=minimal must not cost more than
    wake_depth=standard over the same 10-turn script. If this fails,
    the lazy session-start work regressed.
    """
    from bench.total_session_cost import run_total_session_cost

    minimal = run_total_session_cost(wake_depth="minimal")
    standard = run_total_session_cost(wake_depth="standard")

    assert minimal["total_tokens"] <= standard["total_tokens"], (
        f"minimal {minimal['total_tokens']} > standard {standard['total_tokens']}"
        " — regression"
    )


def test_total_session_cost_counter_mode_disclosed():
    """Honesty: every JSON output must name the counter mode
    used so downstream reports can flag non-official numbers."""
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="minimal")
    assert out["mode"] in (
        "anthropic-count-tokens",
        "tiktoken-cl100k-proxy",
        "heuristic-char4",
        "injected",
    )


def test_total_session_cost_fails_when_above_ref():
    """When the reference-adapter number is explicitly lower than IAI's,
    the comparative gate flips passed=False. Tests supply an
    impossibly-low ref so the assertion is host-independent.
    """
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="standard", mempalace_ref=1)
    assert out["passed"] is False
    assert out["refs"]["mempalace"] == 1


def test_total_session_cost_passes_without_refs():
    """When no reference numbers supplied, passed=True is the degenerate
    answer (the bench still records IAI totals for the report to pick
    up). Honest-disclosure about ref absence lives in the report prose."""
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="minimal")
    assert out["passed"] is True
    assert out["refs"] == {}


def test_total_session_cost_main_exits_int():
    """CLI entry-point returns 0 or 1 (bench CI contract)."""
    from bench import total_session_cost

    code = total_session_cost.main(argv=["--wake-depth", "minimal"])
    assert code in (0, 1)


def test_total_session_cost_injected_counter():
    """Test-only counter injection: caller can pass a deterministic
    token-count function so the test is not hostage to the proxy
    tokeniser's drift."""
    from bench.total_session_cost import run_total_session_cost

    def _fixed(text: str) -> int:
        return max(1, len(text))  # 1-char-per-token for deterministic checks

    out = run_total_session_cost(
        wake_depth="minimal", count_tokens_fn=_fixed,
    )
    assert out["mode"] == "injected"
    assert out["total_tokens"] >= 10  # at least 1/turn * 10 turns
