"""Quality-floor guard for MCP tool descriptions.

Complements tests/test_tool_description_budget.py — which enforces the
token CEILING (<=30 cl100k tok per description, <=330 total) — with a
quality FLOOR + structural completeness check that lifts the Glama Tool
Definition Quality Score (TDQS) of the iai-mcp server from C 2.42 toward
B (>=3.0).

The existing budget test stays at 30/330 caps per user directive
2026-05-11 (cap raise rejected). This file uses the FACT that the budget
regex captures only the FIRST `description:` after each `name:` — so
per-param descriptions inside `inputSchema.properties.*` and sibling
fields (annotations, outputSchema) are INVISIBLE to the budget regex.
That headroom is what lets us lift Glama's Behavior + Completeness +
Parameters dimensions without raising the cap.

Five floor tests:
  1. test_each_description_has_usage_or_behavior_marker
     — every top-level desc contains >=1 marker from MARKERS.
  2. test_camouflaging_status_defines_what_is_detected
     — camouflaging_status desc contains "detect" + one of CAM_KEYWORDS.
  3. test_every_tool_has_annotations_block
     — every tool entry has an `annotations: {` block at column 4.
  4. test_every_tool_has_output_schema_block
     — every tool entry has an `outputSchema: {` block at column 4.
  5. test_no_top_level_description_exceeds_30_cl100k_tokens
     — redundant with the budget test, by design (friendly per-tool
     failure message lives in THIS file, not the budget test).

This file is self-contained: it re-declares the extractor + brace-balance
helpers to match the convention seen in test_tool_description_budget.py
and test_tool_schema_python_parity.py (no cross-test imports).
"""
from __future__ import annotations

import re
from pathlib import Path


TOOLS_TS = Path(__file__).resolve().parent.parent / "mcp-wrapper" / "src" / "tools.ts"

# Permissive Usage-or-Behavior marker set (lowercase substring match).
# "use" alone is intentionally NOT in the set (would match "useful", "usually").
# "returns" is included because every description naturally states what it
# returns — forcing every tool to also say "use when" would burn tokens we
# cannot spare on memory_capture's 1-tok headroom.
MARKERS = frozenset({
    "use when",
    "use for",
    "returns",
    "read-only",
    "mutates",
    "idempotent",
    "prefer",
    "detect",
})

# camouflaging_status disambiguation: the term "camouflaging" alone scored
# 1.7/5 on Glama because it was ambiguous to LLM tool-discovery. The desc
# must define WHAT is detected — formality/register trajectory.
CAM_KEYWORDS = frozenset({
    "formality",
    "register",
    "behavioral",
    "pattern",
    "anomaly",
})


# -------------------------------------------------------- token counter (tiered)
def _tok(text: str) -> int:
    """3-tier fallback counter matching bench/tokens.py shape.

    Mirrors test_tool_description_budget.py:_tok exactly so the redundant
    30-tok ceiling test in this file reports identical numbers.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4) if text else 0


# ------------------------------------------------------- description extractor
def _extract_top_level_descriptions() -> list[tuple[str, str]]:
    """Return list of (tool_name, description) for the 12 tool-level descriptions.

    Copy of test_tool_description_budget.py:_extract_top_level_descriptions
    (kept inline; no cross-test imports per convention).
    """
    text = TOOLS_TS.read_text()
    name_re = re.compile(r'name:\s*"([^"]+)"', re.MULTILINE)
    out: list[tuple[str, str]] = []
    positions = [(m.group(1), m.end()) for m in name_re.finditer(text)]
    for tool_name, pos in positions:
        region = text[pos:]
        next_name = name_re.search(region)
        end = next_name.start() if next_name else len(region)
        region = region[:end]
        concat_re = re.compile(
            r'description:\s*('
            r'"(?:[^"\\]|\\.)*"'
            r'(?:\s*\+\s*"(?:[^"\\]|\\.)*")*'
            r')',
            re.MULTILINE,
        )
        m = concat_re.search(region)
        if not m:
            continue
        literal = m.group(1)
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', literal)
        desc = "".join(parts)
        desc = desc.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        out.append((tool_name, desc))
    return out


# ----------------------------------------------------- brace-balance helpers
# Copied from tests/test_tool_schema_python_parity.py:_balance_braces /
# _TOOL_NAME_LINE so this file stays self-contained.

_TOOL_NAME_LINE = re.compile(
    r"^  (?P<name>[a-zA-Z_][a-zA-Z0-9_]*):\s*\{",
    re.MULTILINE,
)


def _balance_braces(text: str, start_idx: int) -> int:
    """Given an index pointing at an opening `{`, return the index of the
    matching closing `}` (exclusive end + 1 = start of next char).
    """
    assert text[start_idx] == "{", f"expected '{{' at {start_idx}"
    depth = 0
    i = start_idx
    in_str: str | None = None
    while i < len(text):
        ch = text[i]
        if in_str is None:
            if ch == '"' or ch == "'":
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        else:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        i += 1
    raise AssertionError(f"unbalanced braces starting at {start_idx}")


def _enumerate_tool_blocks() -> list[tuple[str, int, int]]:
    """Return list of (tool_name, open_brace_idx, close_brace_idx_exclusive)
    for every tool entry inside `toolSchemas`. Uses _balance_braces against
    the `_TOOL_NAME_LINE` regex (4-space-indent siblings live inside this
    span)."""
    text = TOOLS_TS.read_text()
    blocks: list[tuple[str, int, int]] = []
    for m in _TOOL_NAME_LINE.finditer(text):
        open_idx = m.end() - 1
        close_idx = _balance_braces(text, open_idx)
        blocks.append((m.group("name"), open_idx, close_idx))
    return blocks


# ------------------------------------------------------------------- tests
def test_each_description_has_usage_or_behavior_marker() -> None:
    """Every top-level description must contain >=1 marker from MARKERS.

    Lifts Glama D2 Usage Guidelines (2/5 -> 3-4/5) and D3 Behavior
    dimensions across the 12-tool surface.
    """
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, str]] = []
    for name, desc in descs:
        lower = desc.lower()
        if not any(marker in lower for marker in MARKERS):
            offenders.append((name, desc[:80]))
    assert not offenders, (
        "Quality-floor violation: descriptions missing a Usage-or-Behavior "
        f"marker (one of {sorted(MARKERS)}):\n"
        + "\n".join(f"  {n}: {d!r}" for n, d in offenders)
    )


def test_camouflaging_status_defines_what_is_detected() -> None:
    """camouflaging_status was the load-bearing 1.7/5 Glama bottleneck —
    the term alone is ambiguous to LLM tool-discovery. The desc must say
    'detect' + one of {formality, register, behavioral, pattern, anomaly}."""
    descs = dict(_extract_top_level_descriptions())
    assert "camouflaging_status" in descs, (
        "camouflaging_status tool not found in toolSchemas"
    )
    desc = descs["camouflaging_status"].lower()
    assert "detect" in desc, (
        "camouflaging_status desc must contain 'detect' to lift the "
        f"1.7/5 Glama ambiguity score; got: {descs['camouflaging_status']!r}"
    )
    assert any(kw in desc for kw in CAM_KEYWORDS), (
        f"camouflaging_status desc must contain at least one of "
        f"{sorted(CAM_KEYWORDS)} (defines WHAT is detected); "
        f"got: {descs['camouflaging_status']!r}"
    )


def test_every_tool_has_annotations_block() -> None:
    """Each of the 12 tool entries must declare a sibling `annotations: {`
    block at column 4 (4-space indent — same depth as inputSchema).

    Per-tool brace-balance scan: slice each tool's span and check for the
    `annotations:` substring. Lifts Glama D3 Behavior out of the 1-2/5 band.
    """
    text = TOOLS_TS.read_text()
    blocks = _enumerate_tool_blocks()
    assert len(blocks) == 12, (
        f"expected 12 tool entries, found {len(blocks)}: {[b[0] for b in blocks]}"
    )
    annotations_re = re.compile(r"^    annotations:\s*\{", re.MULTILINE)
    total = len(annotations_re.findall(text))
    missing: list[str] = []
    for name, open_idx, close_idx in blocks:
        span = text[open_idx:close_idx]
        if not annotations_re.search(span):
            missing.append(name)
    assert not missing and total == 12, (
        f"Quality-floor violation: tools missing `annotations: {{` block at "
        f"column 4 (found {total}, expected 12); missing tools: "
        f"{sorted(missing)}"
    )


def test_every_tool_has_output_schema_block() -> None:
    """Each of the 12 tool entries must declare a sibling `outputSchema: {`
    block at column 4. Lifts Glama D6 Completeness out of the 1-3/5 band.
    """
    text = TOOLS_TS.read_text()
    blocks = _enumerate_tool_blocks()
    assert len(blocks) == 12, (
        f"expected 12 tool entries, found {len(blocks)}: {[b[0] for b in blocks]}"
    )
    output_re = re.compile(r"^    outputSchema:\s*\{", re.MULTILINE)
    total = len(output_re.findall(text))
    missing: list[str] = []
    for name, open_idx, close_idx in blocks:
        span = text[open_idx:close_idx]
        if not output_re.search(span):
            missing.append(name)
    assert not missing and total == 12, (
        f"Quality-floor violation: tools missing `outputSchema: {{` block at "
        f"column 4 (found {total}, expected 12); missing tools: "
        f"{sorted(missing)}"
    )


def test_no_top_level_description_exceeds_30_cl100k_tokens() -> None:
    """Redundant with tests/test_tool_description_budget.py:test_each_tool_
    description_le_30_tokens by design — makes THIS file fail with a clear
    per-tool message if Task 2 accidentally pushes a description over 30
    tok, without making the executor scroll to the budget test to figure
    out which tool broke.
    """
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, int, str]] = []
    for name, desc in descs:
        n = _tok(desc)
        if n > 30:
            offenders.append((name, n, desc[:80]))
    assert not offenders, (
        "Quality-floor 30-tok ceiling violation (redundant with "
        "test_tool_description_budget.py; this message is the friendly "
        "per-tool variant):\n"
        + "\n".join(f"  {n}: {t} tok -- {d!r}" for n, t, d in offenders)
    )
