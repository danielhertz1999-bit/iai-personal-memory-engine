"""Tests for the Phase-1 benchmark harnesses (D-15, OPS-01/02/04).

All tests inject `count_tokens_fn` where applicable so no live Anthropic API
calls happen in CI. The actual Anthropic integration is exercised only when
`ANTHROPIC_API_KEY` is set and the CLIs are run directly by hand.
"""
from __future__ import annotations

from bench.tokens import FRESH_LIMIT, STEADY_LIMIT, run_token_bench
from bench.verbatim import ACCURACY_FLOOR, run_verbatim_bench
from iai_mcp.store import MemoryStore


# ---------------------------------------------------------- bench/tokens.py


def test_tokens_steady_pass(tmp_path):
    """Injected counter at 2500 tokens -> both steady_ok and fresh_ok pass."""
    store = MemoryStore(path=tmp_path)
    res = run_token_bench(store=store, n_runs=3, count_tokens_fn=lambda t: 2500)
    assert res["steady_ok"] is True
    assert res["fresh_ok"] is True
    assert all(w == 2500 for w in res["warm"])
    assert res["mode"] == "injected"
    assert res["limits"]["steady"] == STEADY_LIMIT
    assert res["limits"]["fresh"] == FRESH_LIMIT


def test_tokens_steady_fail(tmp_path):
    """3500 tok > STEADY_LIMIT -> steady_ok False, fails."""
    store = MemoryStore(path=tmp_path)
    res = run_token_bench(store=store, n_runs=3, count_tokens_fn=lambda t: 3500)
    assert res["steady_ok"] is False


def test_tokens_fresh_fail(tmp_path):
    """Fresh prompt at 9000 (> FRESH_LIMIT) triggers fresh_ok=False.

    We flip counts via an iterator: first call (fresh) returns 9000, subsequent
    warm calls return 2500. Demonstrates the boundary.
    """
    store = MemoryStore(path=tmp_path)
    counts = iter([9000, 2500, 2500, 2500])

    def _counter(_text: str) -> int:
        return next(counts)

    res = run_token_bench(store=store, n_runs=3, count_tokens_fn=_counter)
    assert res["fresh_ok"] is False   # 9000 > 8000
    assert res["steady_ok"] is True   # warm still under 3000


def test_tokens_tiktoken_fallback_mode(tmp_path, monkeypatch):
    """No ANTHROPIC_API_KEY but tiktoken installed -> mode == tiktoken-cl100k-proxy."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = MemoryStore(path=tmp_path)
    res = run_token_bench(store=store, n_runs=3)
    assert res["mode"] == "tiktoken-cl100k-proxy"
    # Payload on an empty store has no L0/L1/L2/rich_club content, so the warm
    # prompt is literally ".", which tiktoken counts as a single token.
    # Fresh adds the 1k-chars-tail so remains well under FRESH_LIMIT.
    assert res["steady_ok"] is True
    assert res["fresh_ok"] is True


def test_tokens_char4_fallback_mode(tmp_path, monkeypatch):
    """No ANTHROPIC_API_KEY and no tiktoken -> mode == heuristic-char4."""
    import builtins

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("tiktoken not available in this scenario")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    store = MemoryStore(path=tmp_path)
    res = run_token_bench(store=store, n_runs=3)
    assert res["mode"] == "heuristic-char4"
    assert res["steady_ok"] is True


def test_tokens_fresh_prompt_is_larger_than_warm(tmp_path):
    """Sanity: the fresh prompt differs from the warm prompt (has the 1k tail)."""
    store = MemoryStore(path=tmp_path)
    seen_texts: list[str] = []

    def _capture(text: str) -> int:
        seen_texts.append(text)
        return 100

    run_token_bench(store=store, n_runs=1, count_tokens_fn=_capture)
    # First call was the fresh prompt; second was the warm prompt.
    assert len(seen_texts) == 2
    assert len(seen_texts[0]) > len(seen_texts[1])


# -------------------------------------------------------- bench/verbatim.py


def test_verbatim_passes_small_n(tmp_path):
    """Small-N smoke test: pinned records recall at >= 0.99 accuracy."""
    store = MemoryStore(path=tmp_path)
    res = run_verbatim_bench(
        store=store, n_records=10, session_gap=2, noise_per_session=2
    )
    assert res["accuracy"] >= ACCURACY_FLOOR
    assert res["passed"] is True
    assert res["hits_exact"] == 10


def test_verbatim_returns_floor_constant(tmp_path):
    """The harness exposes its pass/fail threshold so verifiers can assert it."""
    store = MemoryStore(path=tmp_path)
    res = run_verbatim_bench(
        store=store, n_records=5, session_gap=1, noise_per_session=1
    )
    assert res["floor"] == ACCURACY_FLOOR
    assert res["floor"] == 0.99


def test_verbatim_counts_exact_matches(tmp_path):
    """hits_exact <= n_records and accuracy = hits_exact / n_records."""
    store = MemoryStore(path=tmp_path)
    res = run_verbatim_bench(
        store=store, n_records=5, session_gap=1, noise_per_session=1
    )
    assert res["hits_exact"] <= res["n_records"]
    assert res["accuracy"] == res["hits_exact"] / res["n_records"]
