"""RED-state test scaffold. Tasks 2-5 turn these GREEN.

Covers / D5-07: MCP tool description budget audit.
- Each of the 11 tools in mcp-wrapper/src/tools.ts has description ≤30 raw tok.
- Total description budget ≤330 raw tok.
- Exactly 11 tools present.

Reads mcp-wrapper/src/tools.ts as text and regex-extracts the `description:`
string literals (TypeScript source, not a compiled artefact).
"""
from __future__ import annotations

import re
from pathlib import Path


TOOLS_TS = Path(__file__).resolve().parent.parent / "mcp-wrapper" / "src" / "tools.ts"


# -------------------------------------------------------- token counter (tiered)
def _tok(text: str) -> int:
    """3-tier fallback counter matching bench/tokens.py shape.

    Tests are self-contained so they do not import bench.* at collect time.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4) if text else 0


# ------------------------------------------------------- description extractor
_DESC_BLOCK_RE = re.compile(
    r"description:\s*"
    r'"((?:[^"\\]|\\.)*)"',
    re.MULTILINE,
)


def _extract_top_level_descriptions() -> list[tuple[str, str]]:
    """Return list of (tool_name, description) for the 11 tool-level descriptions.

    Strategy: walk the file, find each block starting at `name: "..."` and look
    ahead for the NEXT `description: "..."` inside the same tool schema block.
    Ignores description fields nested under inputSchema.properties.* by keying
    off the tool name that immediately precedes the description.
    """
    text = TOOLS_TS.read_text()
    # Find every (name, description) pair where description immediately follows name.
    # Pattern: name: "<tool>", ... description: "<desc>" (description may span
    # adjacent lines as string concatenation with `+`). Keep conservative.
    name_re = re.compile(r'name:\s*"([^"]+)"', re.MULTILINE)
    out: list[tuple[str, str]] = []
    positions = [(m.group(1), m.end()) for m in name_re.finditer(text)]
    for tool_name, pos in positions:
        # Only accept the FIRST description after this name up until the next
        # name-property or end-of-block marker "inputSchema:".
        region = text[pos:]
        # Cut at next occurrence of `name:` to avoid leaking into next tool.
        next_name = name_re.search(region)
        end = next_name.start() if next_name else len(region)
        region = region[:end]
        # Look for top-level description (the first description in this region
        # is the tool's own; subsequent ones under inputSchema.properties are
        # nested and we skip them). Handle multi-line TS concatenation:
        #   description:\n          "part1" +\n          "part2",
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
        # Concatenate parts — extract each quoted string.
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', literal)
        desc = "".join(parts)
        # Unescape common TS escapes for accurate token count.
        desc = desc.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        out.append((tool_name, desc))
    return out


# ------------------------------------------------------------------- tests
def test_tool_count_unchanged_at_12():
    """raised the hot-surface from 11 to 12 by adding memory_capture."""
    descs = _extract_top_level_descriptions()
    assert len(descs) == 12, (
        f"expected 12 tool descriptions, found {len(descs)}: {[n for n, _ in descs]}"
    )


def test_each_tool_description_le_30_tokens():
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, int, str]] = []
    for name, desc in descs:
        n = _tok(desc)
        if n > 30:
            offenders.append((name, n, desc[:80]))
    assert not offenders, (
        " violation: some descriptions exceed 30 tokens:\n"
        + "\n".join(f"  {n}: {t} tok -- {d!r}" for n, t, d in offenders)
    )


def test_total_tool_descriptions_le_330_tokens():
    descs = _extract_top_level_descriptions()
    total = sum(_tok(d) for _, d in descs)
    assert total <= 330, (
        f" violation: total description budget {total} tok > 330\n"
        + "\n".join(f"  {n}: {_tok(d)} tok" for n, d in descs)
    )
