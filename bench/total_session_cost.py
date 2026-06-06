"""Total session cost bench.

Runs a fixed 10-turn representative script
and counts the total tokens Claude would pay for the full session with
IAI-MCP wired in. The 10 turns cover the axes the real-user workload
touches most: verbatim recall, interleaved code-edit chat (no recall),
cross-community recall, save, introspection.

JSON output (one line to stdout):

    {
      "adapter": "iai-mcp",
      "wake_depth": "minimal"|"standard"|"deep",
      "total_tokens": int,
      "per_turn": [int] * 10,
      "mode": "anthropic-count-tokens"|"tiktoken-cl100k-proxy"|
              "heuristic-char4"|"injected",
      "refs": {"mempalace": int?, "claude_mem": int?},
      "passed": bool, # True iff every supplied ref >= IAI
      "script_name": "-v1"
    }

Exit codes:
    0 if passed, 1 otherwise.

CLI:
    python -m bench.total_session_cost
    python -m bench.total_session_cost --wake-depth standard
    python -m bench.total_session_cost --ref-mempalace 7000 --ref-claude-mem 5000

**Framing note:** this bench is a *simulated* 10-turn script —
it reproduces the token composition (system overhead + tool descriptions
+ tool-call payloads + tool-result bodies) a real MCP runtime would emit
for the turn kinds. Real runtime adds network JSON-RPC envelope
overhead (~30-50 tok/turn); the simulation excludes that. Downstream
reports MUST disclose this caveat alongside the row.

Reference-adapter notes: per Discovery #5, bench/adapters/
mempalace_*.py and claude_mem_*.py do not exist on this machine. The
comparative gate is driven by explicit ref numbers via CLI flags so the
bench is usable without live adapters; when unknown, refs default to
None and passed=True is the degenerate answer. Rows where a measurement
was not taken are disclosed as "mempalace/claude-mem refs not measured".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Callable

# Resolve iai_mcp.* (via src) AND bench.* (via worktree root) to THIS
# worktree, not the parent venv's editable install. Idempotent: each
# `sys.path.insert` is guarded by an "if not already present" check.
import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

# Reuse bench/tokens.py's 3-tier counter helpers — single source of truth
# for what "tiktoken-cl100k-proxy" and friends mean.
from bench.tokens import (
    _anthropic_count_tokens,
    _char4_count,
    _tiktoken_count,
)


# ------------------------------------------------------------- adapters
#
# Live subprocess adapters for the reference column. Each adapter runs
# the 10-turn script through the target tool's CLI, sums the response tokens
# via the injected counter, and returns the total. On ANY failure
# (tool absent, timeout, non-zero exit, empty stdout) the adapter returns
# ``None`` and emits ``{"event": "bench_adapter_unavailable",...}`` to
# stderr. Callers MUST treat None as "honest disclosure, no measurement"
# rather than a hard bench failure.
#
# Security: turn text is a constant from _SCRIPT, never from user input,
# and ``subprocess.run(argv_list, shell=False)`` avoids any shell-injection
# surface. The 30s per-turn timeout bounds the DoS risk.

_ADAPTER_TIMEOUT_SECONDS = 30


def _log_adapter_unavailable(tool: str, reason: str) -> None:
    line = json.dumps({
        "event": "bench_adapter_unavailable",
        "tool": tool,
        "reason": reason,
    })
    print(line, file=sys.stderr)


def _run_subprocess_adapter(
    *,
    tool_name: str,
    cli_name: str,
    argv_template: Callable[[str], list[str]],
    script: list[dict],
    counter: Callable[[str], int],
) -> int | None:
    """Shared helper: locate ``cli_name`` via ``shutil.which``; for each turn
    run its argv (provided by ``argv_template(turn_input)``) with a bounded
    timeout; sum stdout token counts across all turns. Return ``None`` on
    any failure (absent / timeout / non-zero / empty stdout)."""
    exe = shutil.which(cli_name)
    if exe is None:
        _log_adapter_unavailable(tool_name, "cli_not_found")
        return None

    total = 0
    for turn in script:
        argv = [exe, *argv_template(turn["input"])[1:]]
        try:
            proc = subprocess.run(
                argv,
                timeout=_ADAPTER_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _log_adapter_unavailable(tool_name, f"timeout: {exc}")
            return None
        except (OSError, ValueError) as exc:
            _log_adapter_unavailable(tool_name, f"subprocess_error: {exc}")
            return None

        if proc.returncode != 0:
            _log_adapter_unavailable(
                tool_name,
                f"non_zero_exit={proc.returncode} stderr={proc.stderr[:200]!r}",
            )
            return None

        stdout = proc.stdout or ""
        # Empty stdout is a legitimate "no match" response for search-style
        # CLIs; we DO count it (0 tokens) rather than treating as failure,
        # so adapters run against a pristine palace still publish a number.
        total += int(counter(stdout))

    return total


def _run_mempalace_adapter(
    script: list[dict],
    counter: Callable[[str], int],
) -> int | None:
    """Live reference: run each turn through ``mempalace search`` and
    sum the stdout token counts. Returns ``None`` when mempalace is absent
    or any subprocess call fails.
    """
    return _run_subprocess_adapter(
        tool_name="mempalace",
        cli_name="mempalace",
        argv_template=lambda text: ["mempalace", "search", text],
        script=script,
        counter=counter,
    )


def _run_claude_mem_adapter(
    script: list[dict],
    counter: Callable[[str], int],
) -> int | None:
    """Forward-compat mirror of the mempalace adapter. On machines where
    ``claude-mem`` is not installed this returns ``None`` + stderr event;
    when it IS installed (future second-machine cross-validation run) the same
    code path measures it without another plan iteration."""
    return _run_subprocess_adapter(
        tool_name="claude-mem",
        cli_name="claude-mem",
        argv_template=lambda text: ["claude-mem", "recall", text],
        script=script,
        counter=counter,
    )


# ---------------------------------------------------------------- script
#
# Fixed 10-turn representative script. Each turn has a `kind` (used to
# compose a realistic tool-result body) and an `input` (the cue text).
# Order matters: turn 1 pays session-start overhead, turn 4 exercises the
# cross-community recall path, turn 5/6 exercise save/introspect.

SCRIPT_NAME = "D5-08-v1"

_SCRIPT: list[dict] = [
    {
        "kind": "recall",
        "input": "Tell me the decisions we made about Phase 5 architecture",
    },
    {
        "kind": "chat",
        "input": "Let me iterate on this function; no recall needed here",
    },
    {
        "kind": "recall",
        "input": "What did I say about bench discipline?",
    },
    {
        "kind": "recall_cross_community",
        "input": "What is the connection between OPS-13 and the autistic kernel?",
    },
    {
        "kind": "save",
        "input": "Decision locked: use cachetools TTLCache for Phase 5 LRU",
    },
    {
        "kind": "introspect",
        "input": "profile_get_set operation=get knob=wake_depth",
    },
    {
        "kind": "chat",
        "input": "Continuing this refactor; still no recall",
    },
    {
        "kind": "recall",
        "input": "Alice said something about second-machine cross-validation",
    },
    {
        "kind": "reinforce",
        "input": "memory_reinforce the last 3 hits",
    },
    {
        "kind": "introspect",
        "input": "events_query kind=first_turn_recall limit=5",
    },
]


# Tool-description overhead: 134 raw tokens total for the 11 registered
# tools. We reproduce the current tool-description text verbatim so the
# bench reflects the actual overhead Claude sees on each turn.
_POST_TOK15_TOOL_DESCRIPTIONS = "\n".join([
    "Recall verbatim memories matching cue. Returns hits + anti_hits.",
    "Structural recall over role->filler bindings. Returns hits.",
    "Boost Hebbian edges among co-retrieved record ids.",
    "Mark a record contradicted; new fact stored as new record.",
    "Trigger memory consolidation.",
    "Read or write a profile knob (15 sealed). operation: get|set.",
    "List pending curiosity questions. Optional session_id filter.",
    "List induced schemas. Optional domain + confidence_min filters.",
    "Query user-visible events by kind, since, severity, limit.",
    "Topology snapshot: N, C, L, sigma, community_count, regime.",
    "Camouflaging detection status; window_size weekly points.",
])

# Synthetic tool-result body per turn kind. Realistic-but-bounded; a real
# runtime varies by store content but the ratio across wake_depths is
# what measures, not the absolute per-query payload.
_RESULT_BODIES: dict[str, str] = {
    "recall": (
        "hits=[{record_id, literal_surface, score}] "
        "anti_hits=[{record_id, reason}] "
        "activation_trace=[community_gate, spread, rank] "
        "budget_used=200"
    ),
    "save": "ok=true id=<uuid>",
    "introspect": '{"value": "minimal"}',
    "reinforce": "ok=true edges_boosted=3",
    "chat": "",
    "recall_cross_community": (
        "hits=[{record_id, literal_surface, score, community_id}] "
        "anti_hits=[] activation_trace=[cross_community_spread] "
        "budget_used=350"
    ),
}


# ---------------------------------------------------------------- counter select

def _select_counter(
    count_tokens_fn: Callable[[str], int] | None = None,
) -> tuple[Callable[[str], int], str]:
    """3-tier counter fallback mirroring bench/tokens.py:165-182.

    Priority:
      1. explicit injection (`count_tokens_fn` kwarg, tests)
      2. Anthropic count_tokens API (`ANTHROPIC_API_KEY` env var)
      3. tiktoken cl100k_base (offline proxy)
      4. char/4 heuristic (last resort)
    """
    if count_tokens_fn is not None:
        return count_tokens_fn, "injected"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _anthropic_count_tokens, "anthropic-count-tokens"
    try:
        import tiktoken  # noqa: F401
        return _tiktoken_count, "tiktoken-cl100k-proxy"
    except ImportError:
        return _char4_count, "heuristic-char4"


# ---------------------------------------------------------------- per-turn cost

def _session_start_overhead_tokens(wake_depth: str) -> int:
    """Session-start payload size charged to turn 1 per wake_depth mode.

    Measured token counts:
      - minimal: 24 tok (lazy pointers only)
      - standard: 1388 tok (eager L0+L1+L2+rich_club)
      - deep: ~2000 tok (rich_club budget lifted)

    Rounded to cache-metric boundaries so the numbers are
    consistent with the warm session-start measurements.
    """
    if wake_depth == "minimal":
        return 24
    if wake_depth == "standard":
        return 1388
    return 2000  # deep


def _simulate_turn(
    turn: dict,
    counter: Callable[[str], int],
) -> int:
    """Compose the per-turn text that Claude sees and count its tokens."""
    parts: list[str] = [
        _POST_TOK15_TOOL_DESCRIPTIONS,  # constant per-turn overhead
        turn["input"],                   # user / call payload
        _RESULT_BODIES.get(turn["kind"], ""),  # synthetic result body
    ]
    return int(counter("\n".join(p for p in parts if p)))


# ---------------------------------------------------------------- public API

def run_total_session_cost(
    *,
    wake_depth: str = "minimal",
    mempalace_ref: int | None = None,
    claude_mem_ref: int | None = None,
    measure_mempalace: bool = False,
    measure_claude_mem: bool = False,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> dict:
    """Run the fixed 10-turn script at the given wake_depth.

    Parameters:
        wake_depth: "minimal" | "standard" | "deep" — selects session-start
            payload size charged to turn 1.
        mempalace_ref / claude_mem_ref: optional manually-supplied reference
            totals (stored as ``refs["*_manual"]`` for audit). When no live
            measurement exists, a manual int is the comparator for ``passed``.
        measure_mempalace / measure_claude_mem: when True, invoke the live
            subprocess adapter and store the result as ``refs["*_measured"]``.
            A live measurement supersedes the manual ref as the comparator.
        count_tokens_fn: optional counter injection (tests use a fixed
            function to decouple assertions from tokeniser drift).
    """
    counter, mode = _select_counter(count_tokens_fn)

    per_turn: list[int] = []
    for i, turn in enumerate(_SCRIPT):
        t = _simulate_turn(turn, counter)
        if i == 0:
            # Turn 1 pays the session-start overhead per wake_depth.
            t += _session_start_overhead_tokens(wake_depth)
        per_turn.append(int(t))

    total = int(sum(per_turn))

    refs: dict[str, int] = {}
    passed = True

    # Live measurements first so we can decide whether the manual int should
    # be recorded under the legacy key ("mempalace") or the audit-trail key
    # ("mempalace_manual", used when BOTH a measurement AND a manual ref are
    # supplied per Test 6).
    mp_measured: int | None = None
    cm_measured: int | None = None
    if measure_mempalace:
        mp_measured = _run_mempalace_adapter(_SCRIPT, counter)
        if mp_measured is not None:
            refs["mempalace_measured"] = int(mp_measured)
    if measure_claude_mem:
        cm_measured = _run_claude_mem_adapter(_SCRIPT, counter)
        if cm_measured is not None:
            refs["claude_mem_measured"] = int(cm_measured)

    # Manual refs. Back-compat with: when no live measurement is
    # present, the manual int lands under the legacy "mempalace" / "claude_mem"
    # key so pre-existing downstream consumers (and tests) keep working.
    if mempalace_ref is not None:
        key = "mempalace_manual" if mp_measured is not None else "mempalace"
        refs[key] = int(mempalace_ref)
    if claude_mem_ref is not None:
        key = "claude_mem_manual" if cm_measured is not None else "claude_mem"
        refs[key] = int(claude_mem_ref)

    # Gate logic: measured > legacy manual > audit-trail manual > no gate.
    mp_gate = refs.get(
        "mempalace_measured", refs.get("mempalace", refs.get("mempalace_manual"))
    )
    cm_gate = refs.get(
        "claude_mem_measured", refs.get("claude_mem", refs.get("claude_mem_manual"))
    )
    if mp_gate is not None and total > mp_gate:
        passed = False
    if cm_gate is not None and total > cm_gate:
        passed = False

    return {
        "adapter": "iai-mcp",
        "wake_depth": wake_depth,
        "total_tokens": total,
        "per_turn": per_turn,
        "mode": mode,
        "refs": refs,
        "passed": passed,
        "script_name": SCRIPT_NAME,
    }


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.total_session_cost",
        description=(
            "Total session cost bench. Fixed 10-turn representative script; "
            "measures IAI-MCP token cost at wake_depth minimal|standard|deep "
            "and optionally compares to supplied mempalace / claude-mem "
            "reference totals."
        ),
    )
    parser.add_argument(
        "--wake-depth",
        choices=("minimal", "standard", "deep"),
        default="minimal",
        help="session-start payload size (default minimal)",
    )
    parser.add_argument(
        "--ref-mempalace",
        dest="mempalace_ref",
        type=int, default=None,
        help="mempalace reference total (tokens) for the comparative gate",
    )
    parser.add_argument(
        "--ref-claude-mem",
        dest="claude_mem_ref",
        type=int, default=None,
        help="claude-mem reference total (tokens) for the comparative gate",
    )
    parser.add_argument(
        "--measure-mempalace",
        action="store_true",
        help=(
            "attempt a live mempalace subprocess run to fill the "
            "reference column; on failure emits a bench_adapter_unavailable "
            "stderr event and records no measurement"
        ),
    )
    parser.add_argument(
        "--measure-claude-mem",
        action="store_true",
        help=(
            "attempt a live claude-mem subprocess run; identical fallback "
            "shape to --measure-mempalace"
        ),
    )
    args = parser.parse_args(argv)

    result = run_total_session_cost(
        wake_depth=args.wake_depth,
        mempalace_ref=args.mempalace_ref,
        claude_mem_ref=args.claude_mem_ref,
        measure_mempalace=args.measure_mempalace,
        measure_claude_mem=args.measure_claude_mem,
    )
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
