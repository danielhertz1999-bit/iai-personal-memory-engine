"""V3-02 parity guard: every params.get/[] key consumed by core.dispatch
for an MCP-advertised tool MUST appear as an inputSchema.properties entry
in mcp-wrapper/src/tools.ts. New dispatch additions without a schema
entry fail this test loudly with file:line + missing keys.

Pattern analog: tests/test_constitutional_guards.py (file walk + regex/
AST -> offenders list -> assert empty).

Plan 07.13-03 ground truth: the per-method audit table in
internal architecture spec (section "Authoritative
`params.get/[]` audit per dispatch method").
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE_PY = REPO / "src" / "iai_mcp" / "core.py"
TOOLS_TS = REPO / "mcp-wrapper" / "src" / "tools.ts"


# Mirror of mcp-wrapper/src/tools.ts:20-33 TOOL_NAMES.
# Update in lockstep with that constant.
TOOL_NAMES: list[str] = [
    "memory_recall",
    "memory_recall_structural",
    "memory_reinforce",
    "memory_contradict",
    "memory_capture",
    "memory_consolidate",
    "profile_get_set",
    "curiosity_pending",
    "schema_list",
    "events_query",
    "topology",
    "camouflaging_status",
]

# profile_get_set is a wrapper schema; the Python dispatcher exposes two
# distinct branches (profile_get, profile_set). The wrapper schema's
# "operation" + "knob" + "value" properties cover both branches; the test
# maps profile_get_set -> union(profile_get keys, profile_set keys).
PROFILE_DISPATCH_BRANCHES: dict[str, list[str]] = {
    "profile_get_set": ["profile_get", "profile_set"],
}

# Wrapper-only properties: keys advertised by the TS wrapper that have no
# direct params.get/[] analog in the Python dispatch (the wrapper translates
# them client-side). These are NOT expected on the Python side, so the
# parity test (which checks Python keys ⊆ TS keys) tolerates them
# automatically — they simply make the TS set larger.
#
# Documented for clarity only:
# - profile_get_set.operation: wrapper splits get/set client-side via
#   invokeTool switch (mcp-wrapper/src/tools.ts:299-310); never reaches
#   bridge.call as a key.


# ---------------------------------------------------------------------------
# Python-side helper: AST-walk core.dispatch's `if method == "..."` chain
# ---------------------------------------------------------------------------

def _extract_python_keys(module_ast: ast.Module, dispatch_method: str) -> set[str]:
    """Walk the dispatch function's if-chain. Find every body whose guard is
    `method == "<dispatch_method>"`. Within that body collect every
    `params["..."]` Subscript access and every `params.get("...", ...)`
    Call. Return the union of literal-string keys.

    Notes:
      - We collect BOTH `params["..."]` (REQUIRED accesses) and
        `params.get("...", ...)` (OPTIONAL accesses); the parity check
        treats the union as "every key the dispatch may consume".
      - Non-literal keys (e.g. dynamic `params[some_var]`) are skipped;
        if a dispatch branch ever does that, the parity test cannot
        enforce contract on the dynamic name and a manual review is
        required (none today; verified 2026-04-30).
    """
    keys: set[str] = set()
    dispatch_fn = next(
        (
            n for n in module_ast.body
            if isinstance(n, ast.FunctionDef) and n.name == "dispatch"
        ),
        None,
    )
    assert dispatch_fn is not None, "core.dispatch function not found"

    for node in ast.walk(dispatch_fn):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if not (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "method"
            and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Eq)
            and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and isinstance(t.comparators[0].value, str)
        ):
            continue
        if t.comparators[0].value != dispatch_method:
            continue

        for sub in node.body:
            for n in ast.walk(sub):
                # params["key"]
                if (
                    isinstance(n, ast.Subscript)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "params"
                ):
                    slc = n.slice
                    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                        keys.add(slc.value)
                # params.get("key", ...)
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "params"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)
                ):
                    keys.add(n.args[0].value)
    return keys


# ---------------------------------------------------------------------------
# TS-side helper: regex over tools.ts toolSchemas object
# ---------------------------------------------------------------------------

# Robust per-tool block regex. Handles BOTH:
#   (a) memory_consolidate-style single-line empty:
#         memory_consolidate: { ..., inputSchema: { type: "object", properties: {} }, },
#   (b) multi-line full schema with required[] and nested properties.
#
# Strategy: locate the tool name at column-2 (toolSchemas top-level), then
# brace-balance forward to find the matching closing brace of the tool
# entry. Within that span, locate `inputSchema:` and balance again to find
# its matching close. Within that, locate `properties:` and balance to find
# the properties block. Property names are top-level keys at the first
# nesting level inside the properties block.
#
# We do NOT use a full TS parser — forbids new abstractions. The
# brace-balance approach handles both single-line and multi-line variants
# without a dependency.

_TOOL_NAME_LINE = re.compile(
    r"^  (?P<name>[a-zA-Z_][a-zA-Z0-9_]*):\s*\{",
    re.MULTILINE,
)


def _balance_braces(text: str, start_idx: int) -> int:
    """Given an index pointing at an opening `{`, return the index of the
    matching closing `}` (exclusive end + 1 = start of next char). Naive
    brace counter; tolerates strings only insofar as `tools.ts` does not
    embed unbalanced braces inside string literals (verified — all string
    literals in the file are simple text descriptions).
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


def _find_block(text: str, key: str, search_from: int, search_to: int) -> tuple[int, int]:
    """Find `<key>: {` within text[search_from:search_to] and return the
    (open_brace_idx, close_brace_idx_exclusive) pair via brace-balancing.

    Returns (-1, -1) if the key is not found.
    """
    pattern = re.compile(rf"\b{re.escape(key)}\s*:\s*\{{")
    m = pattern.search(text, search_from, search_to)
    if not m:
        return -1, -1
    # The opening brace is the last char of the match.
    open_idx = m.end() - 1
    close_idx = _balance_braces(text, open_idx)
    return open_idx, close_idx


# Property names inside `properties: { ... }` are the top-level keys
# (one nesting level deep). We extract them by brace-balancing each
# top-level entry.
_PROP_KEY_LINE = re.compile(
    r"^\s*(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\{",
    re.MULTILINE,
)


def _extract_property_keys(properties_block: str) -> set[str]:
    """Given the *contents* of a `properties: { ... }` block (without the
    outer braces), return the set of top-level property keys.

    Walks the block character-by-character at depth 0, locating each
    `key: {` match at depth 0 (so nested object properties don't leak
    into the set).
    """
    keys: set[str] = set()
    i = 0
    depth = 0
    in_str: str | None = None
    while i < len(properties_block):
        ch = properties_block[i]
        if in_str is None:
            if ch == '"' or ch == "'":
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 0 and (ch.isalpha() or ch == "_"):
                # Try to match `key: {` or `key: <type>` at depth 0.
                m = re.match(
                    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*",
                    properties_block[i:],
                )
                if m:
                    keys.add(m.group(1))
                    i += m.end()
                    continue
        else:
            if ch == "\\" and i + 1 < len(properties_block):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        i += 1
    return keys


def _extract_ts_keys(ts_text: str, tool_name: str) -> set[str]:
    """Find toolSchemas[<tool_name>] block; return the set of
    inputSchema.properties keys.

    Handles BOTH single-line empty `properties: {}` and full multi-line
    schemas via brace-balancing. No TS parser dependency.
    """
    # Locate the tool name at column-2 inside the toolSchemas object.
    for m in _TOOL_NAME_LINE.finditer(ts_text):
        if m.group("name") != tool_name:
            continue
        tool_open = m.end() - 1
        tool_close = _balance_braces(ts_text, tool_open)
        # Find inputSchema: { ... } within the tool block.
        is_open, is_close = _find_block(
            ts_text, "inputSchema", tool_open + 1, tool_close,
        )
        if is_open == -1:
            raise AssertionError(
                f"tool {tool_name!r}: inputSchema block not found"
            )
        # Find properties: { ... } within inputSchema.
        props_open, props_close = _find_block(
            ts_text, "properties", is_open + 1, is_close,
        )
        if props_open == -1:
            raise AssertionError(
                f"tool {tool_name!r}: properties block not found"
            )
        # Slice the *contents* of the properties block (between the braces).
        props_block = ts_text[props_open + 1 : props_close - 1]
        return _extract_property_keys(props_block)
    raise AssertionError(
        f"tool {tool_name!r} not found in {TOOLS_TS}; update TOOL_NAMES "
        f"mirror in tests/test_tool_schema_python_parity.py"
    )


# ---------------------------------------------------------------------------
# Sanity-check tests for the helpers themselves (catch regex/AST drift)
# ---------------------------------------------------------------------------

def test_ts_extractor_finds_known_tool() -> None:
    """Sanity: _extract_ts_keys returns a non-empty set for a tool we know
    has multiple properties (memory_capture).
    """
    keys = _extract_ts_keys(TOOLS_TS.read_text(), "memory_capture")
    assert keys, "memory_capture schema parsed as empty — regex broken"
    # memory_capture has at least: text, cue, tier, session_id, role.
    assert "text" in keys, keys
    assert "cue" in keys, keys
    assert "tier" in keys, keys
    assert "role" in keys, keys


def test_ts_extractor_handles_empty_properties() -> None:
    """Sanity: _extract_ts_keys returns an empty set for a tool whose
    inputSchema has `properties: {}` (single-line or multi-line).

    Pre-Plan-07.13-03: memory_consolidate had `properties: {}` (empty).
    Post-Plan-07.13-03: memory_consolidate has `properties: { session_id: {...} }`.
    Either way, the extractor must not crash; pre-fix it returns empty,
    post-fix it returns {"session_id"}. We assert it returns a set; the
    parity test enforces the post-fix content.
    """
    # topology has empty properties in both pre- and post-fix states.
    keys = _extract_ts_keys(TOOLS_TS.read_text(), "topology")
    assert keys == set(), keys


def test_python_extractor_finds_known_method() -> None:
    """Sanity: _extract_python_keys returns a non-empty set for a method
    we know reads multiple params (memory_recall).
    """
    core_ast = ast.parse(CORE_PY.read_text())
    keys = _extract_python_keys(core_ast, "memory_recall")
    # memory_recall reads at least: cue, cue_embedding, session_id,
    # budget_tokens, language (per PATTERNS audit).
    assert "cue" in keys, keys
    assert "cue_embedding" in keys, keys
    assert "session_id" in keys, keys
    assert "budget_tokens" in keys, keys
    assert "language" in keys, keys


# ---------------------------------------------------------------------------
# The parity assertion (V3-02 acceptance gate)
# ---------------------------------------------------------------------------

def test_ts_schema_advertises_every_python_param_key() -> None:
    """V3-02 parity: for each MCP-advertised tool, every params.get/[] key
    the Python dispatcher reads MUST appear as an inputSchema.properties
    entry in mcp-wrapper/src/tools.ts.

    Mismatches = TS schema is hiding a parameter that strict-validating
    hosts will refuse to send.
    """
    core_ast = ast.parse(CORE_PY.read_text())
    ts_text = TOOLS_TS.read_text()

    offenders: list[str] = []
    for tool_name in TOOL_NAMES:
        ts_keys = _extract_ts_keys(ts_text, tool_name)
        # profile_get_set: union both dispatch branches.
        dispatch_methods = PROFILE_DISPATCH_BRANCHES.get(
            tool_name, [tool_name],
        )
        py_keys: set[str] = set()
        for m in dispatch_methods:
            py_keys |= _extract_python_keys(core_ast, m)
        missing = py_keys - ts_keys
        if missing:
            offenders.append(
                f"  {tool_name}: TS schema missing keys "
                f"{sorted(missing)} (Python dispatch reads them; "
                f"advertise as optional inputSchema.properties entries)"
            )

    assert not offenders, (
        "V3-02 schema/dispatch parity broken:\n"
        + "\n".join(offenders)
        + f"\n\nFiles to fix:\n  TS schema: {TOOLS_TS}\n  Python dispatch: {CORE_PY}"
    )
