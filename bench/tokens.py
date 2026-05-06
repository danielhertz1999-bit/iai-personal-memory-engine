"""bench/tokens.py -- / benchmark harness.

Measures session-start token budget three ways, preferring the most accurate
source available at runtime:

1. Anthropic `count_tokens` API (best). Used when ANTHROPIC_API_KEY is set.
   Gives an honest billable-token count that includes Anthropic-side overhead
   and exact tokeniser output. Model: claude-sonnet-4-5. This is the only mode
   whose numbers are safe to publish (PROJECT.md: "honest mode-by-mode
   benchmarks, not headline numbers").

2. tiktoken cl100k_base fallback. OpenAI's tokeniser shipped with the tiktoken
   package -- runs fully offline, no network, no key. It under-counts Claude by
   ~5-10% on English and over-counts by ~10-15% on Cyrillic (GPT-4 tokeniser
   packs multibyte differently). Acceptable for local dev and CI; the JSON
   output always records mode so downstream dashboards can reject non-API
   numbers from public charts.

3. char/4 heuristic. Used only when both 1 and 2 are unavailable (e.g. minimal
   CI image without tiktoken installed). Very rough; adequate only for sanity
   checks on the order of magnitude.

Thresholds:
- (steady warm-cache): <= STEADY_LIMIT (3000 tokens) on every warm run
- (first fresh session): <= FRESH_LIMIT (8000 tokens)

Exit codes:
- 0: both steady_ok and fresh_ok
- 1: at least one failed

JSON output format (one line to stdout):
    {"fresh": int, "warm": [int, ...], "steady_ok": bool, "fresh_ok": bool,
     "mode": "anthropic-count-tokens" | "tiktoken-cl100k-proxy" |
             "heuristic-char4" | "injected",
     "limits": {"steady": 3000, "fresh": 8000}}
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable

from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import SessionStartPayload, assemble_session_start
from iai_mcp.store import MemoryStore

# budget targets
STEADY_LIMIT = 3000   # warm-cache steady-state
FRESH_LIMIT = 8000    # first-fresh-session (cache populate premium)


def _anthropic_count_tokens(text: str) -> int:
    """Use Anthropic count_tokens API. Raises if key absent or call fails."""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": text}],
    )
    return int(resp.input_tokens)


def _tiktoken_count(text: str) -> int:
    """Offline tiktoken cl100k_base as a proxy for Claude's tokeniser.

    Raises ImportError if tiktoken not installed -- caller falls through to
    the char/4 heuristic in that case.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _char4_count(text: str) -> int:
    """Last-resort char/4 heuristic. Reasonable for English prose, bad for CJK."""
    return max(1, len(text) // 4)


def _payload_to_prompt(payload: SessionStartPayload) -> str:
    """Flatten the session-start payload to a single prompt string.

    Mirrors the TypeScript wrapper's buildCachedSystemPrompt shape so the
    counted prompt is faithful to what Anthropic actually receives.

    D5-02: at wake_depth=minimal, the legacy l0/l1/l2/rich_club
    fields are empty and the payload is three pointer handles. Include them
    alongside legacy segments so both modes flatten to a representative
    prompt string for counting.
    """
    parts: list[str] = []
    if payload.l0:
        parts.append(f"# L0 identity\n{payload.l0}")
    if payload.l1:
        parts.append(f"# L1 critical facts\n{payload.l1}")
    for segment in payload.l2:
        parts.append(f"# L2 community\n{segment}")
    if payload.rich_club:
        parts.append(f"# Global rich-club\n{payload.rich_club}")
    # / 05-06: lazy session-start wire payload.
    # Under wake_depth=minimal the wire is the compact handle alone
    # (the 3 legacy pointer fields stay on the dataclass for back-compat
    # callers but are NOT serialised to the wire).
    # Under standard/deep the wire is the Phase-1 eager L0/L1/L2/rich_club
    # plus the 3 legacy pointer fields, matching the pre-05-06 baseline.
    # The compact handle is carried on the dataclass under standard/deep
    # too so opt-in callers may read it, but it does NOT add to the wire
    # (that would inflate the standard baseline).
    compact = getattr(payload, "compact_handle", "")
    wake_depth = getattr(payload, "wake_depth", "minimal")
    if wake_depth == "minimal":
        if compact:
            parts.append(compact)
    else:
        lazy = [
            s for s in (
                getattr(payload, "identity_pointer", ""),
                getattr(payload, "brain_handle", ""),
                getattr(payload, "topic_cluster_hint", ""),
            ) if s
        ]
        if lazy:
            parts.append(" ".join(lazy))
    return "\n\n".join(parts)


def _fresh_prompt(payload: SessionStartPayload) -> str:
    """the first fresh-session request pays the cache-populate premium.

    Simulated here by padding the cached prefix with ~1000 tokens of dynamic
    tail content (D-10 dynamic reserve). Anthropic's count_tokens will return
    the sum of both parts in one call.
    """
    prompt = _payload_to_prompt(payload)
    tail = "dynamic tail content " * 125  # ~2500 chars ~ 625 tokens heuristic
    return f"{prompt}\n\n{tail}" if prompt else tail


def run_token_bench(
    store: MemoryStore | None = None,
    n_runs: int = 3,
    count_tokens_fn: Callable[[str], int] | None = None,
    wake_depth: str = "minimal",
) -> dict:
    """Run the token benchmark.

    Parameters:
        store: optional MemoryStore override (tests pass an isolated tmp_path store).
        n_runs: how many warm-cache repeats to measure (OPS-01 steady-state needs
                at least 3 consecutive samples).
        count_tokens_fn: optional token-counter injection (test-only); overrides both
                the Anthropic API and the heuristic fallback.
        wake_depth: TOK-11 — selects session-start payload mode.
                Default ``minimal`` measures the lazy <=30-tok handle; pass
                ``standard`` for the Phase-1 eager dump baseline; ``deep`` for
                the ≤2000-tok expanded rich_club.

    Returns a dict with keys described in the module docstring.
    """
    s = store if store is not None else MemoryStore()
    records_count = s.db.open_table("records").count_rows()
    if records_count > 0:
        _graph, assignment, rc = build_runtime_graph(s)
        payload = assemble_session_start(
            s, assignment, rc, profile_state={"wake_depth": wake_depth},
        )
    else:
        # Empty-store fallback: mint a representative compact handle so the
        # warm-prompt count reflects the wire payload shape even before any
        # record is written. Mirrors session.assemble_session_start at
        # wake_depth=minimal.
        from iai_mcp.handle import encode_compact_handle
        from uuid import uuid4

        _compact = encode_compact_handle("", str(uuid4())[:8], "none", 0)
        payload = SessionStartPayload(
            l0="",
            l1="",
            l2=[],
            rich_club="",
            total_cached_tokens=max(1, len(_compact) // 4),
            total_dynamic_tokens=1000,
            compact_handle=_compact,
            wake_depth=wake_depth,
        )

    counter: Callable[[str], int]
    mode: str
    if count_tokens_fn is not None:
        counter = count_tokens_fn
        mode = "injected"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        counter = _anthropic_count_tokens
        mode = "anthropic-count-tokens"
    else:
        # Prefer tiktoken over char/4 -- it actually tokenises the text and
        # tracks Claude within ~10% across English + Cyrillic.
        try:
            import tiktoken  # noqa: F401
            counter = _tiktoken_count
            mode = "tiktoken-cl100k-proxy"
        except ImportError:
            counter = _char4_count
            mode = "heuristic-char4"

    warm_prompt = _payload_to_prompt(payload) or "."
    fresh_prompt = _fresh_prompt(payload)
    fresh = int(counter(fresh_prompt))
    warm = [int(counter(warm_prompt)) for _ in range(n_runs)]

    fresh_ok = fresh <= FRESH_LIMIT
    steady_ok = all(w <= STEADY_LIMIT for w in warm)

    return {
        "fresh": fresh,
        "warm": warm,
        "steady_ok": steady_ok,
        "fresh_ok": fresh_ok,
        "mode": mode,
        "limits": {"steady": STEADY_LIMIT, "fresh": FRESH_LIMIT},
        "payload_cached_tokens": payload.total_cached_tokens,
        "payload_dynamic_tokens": payload.total_dynamic_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="bench.tokens",
        description=(
            "OPS-01/OPS-02 session-start token bench. TOK-11 added "
            "--wake-depth for measuring the lazy <=30-tok payload vs Phase-1 "
            "eager dump vs the deep variant."
        ),
    )
    parser.add_argument(
        "--wake-depth",
        choices=("minimal", "standard", "deep"),
        default="minimal",
        help="Session-start payload mode (default: minimal per D5-02).",
    )
    args = parser.parse_args(argv)
    result = run_token_bench(wake_depth=args.wake_depth)
    print(json.dumps(result))
    return 0 if (result["steady_ok"] and result["fresh_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
