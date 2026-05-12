"""Pin camouflaging_status outputSchema to the actual response shape.

End-to-end sweep of all 12 MCP tools surfaced one wrong-name mismatch:

  tools.ts declares     | core.dispatch returns
  --------------------- | ----------------------
  detected              | detected
  formality_trend       | trajectory_slope        ← name mismatch
  anomaly_score         | current_mean            ← name mismatch
  camouflaging_relaxation | sample_count          ← schema missing
                        | camouflaging_relaxation

Source of truth (no doubt):
  - src/iai_mcp/camouflaging.py:71-92 returns
    {detected, trajectory_slope, current_mean, sample_count}
  - src/iai_mcp/core.py:694-704 injects camouflaging_relaxation onto
    that dict.

This test parses the camouflaging_status outputSchema block in
tools.ts and asserts the declared property keys match the actual
five-key response set. RED before the fix, GREEN after.

Structural test — no daemon spawn, no socket. Re-uses the same
regex/brace-balance convention as test_mcp_tools_description_quality.py
to stay self-contained.
"""
from __future__ import annotations

import re
from pathlib import Path


TOOLS_TS = Path(__file__).resolve().parent.parent / "mcp-wrapper" / "src" / "tools.ts"

EXPECTED_RESPONSE_KEYS = frozenset({
    "detected",
    "trajectory_slope",
    "current_mean",
    "sample_count",
    "camouflaging_relaxation",
})

# Names tools.ts USED to declare that are NOT in the actual response.
# Pinning these out keeps the regression from sneaking back via a copy-
# paste from older planning docs.
FORBIDDEN_KEYS = frozenset({
    "formality_trend",
    "anomaly_score",
})


def _extract_camouflaging_status_outputschema(text: str) -> str:
    """Return the slice of tools.ts that holds the camouflaging_status
    outputSchema { ... } block.

    Strategy: locate the `camouflaging_status: {` opening, then within
    that entry walk to `outputSchema: {`, then brace-balance to the
    matching close. Returns the inner string between `{` and matching
    `}` (exclusive).
    """
    entry_marker = "camouflaging_status: {"
    entry_start = text.find(entry_marker)
    assert entry_start != -1, (
        "camouflaging_status entry not found in tools.ts — file "
        "structure must have changed; update this test."
    )

    schema_marker = "outputSchema: {"
    schema_start = text.find(schema_marker, entry_start)
    assert schema_start != -1, (
        "outputSchema block not found inside camouflaging_status entry "
        "in tools.ts."
    )

    open_brace = schema_start + len(schema_marker) - 1
    depth = 0
    for i in range(open_brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    raise AssertionError(
        "outputSchema block in camouflaging_status is not brace-balanced."
    )


def _declared_property_keys(schema_body: str) -> set[str]:
    """Pull top-level property names from `properties: { ... }`.

    Returns the set of bare identifiers (keys) declared one level
    inside the `properties:` block. Skips nested objects.
    """
    props_marker = "properties: {"
    props_start = schema_body.find(props_marker)
    assert props_start != -1, (
        "outputSchema for camouflaging_status has no properties block."
    )

    open_brace = props_start + len(props_marker) - 1
    depth = 0
    end = None
    for i in range(open_brace, len(schema_body)):
        ch = schema_body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "properties block in outputSchema not brace-balanced."

    body = schema_body[open_brace + 1:end]
    # Match top-level keys only (depth==0 inside body). We treat each
    # `<ident>: {` or `<ident>: <scalar>` at depth 0 as one declared key.
    keys: set[str] = set()
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "{":
            depth += 1
            i += 1
            continue
        if ch == "}":
            depth -= 1
            i += 1
            continue
        if depth == 0 and ch.isalpha():
            m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*:", body[i:])
            if m:
                keys.add(m.group(1))
                i += m.end()
                continue
        i += 1
    return keys


def test_tools_ts_exists() -> None:
    assert TOOLS_TS.is_file(), f"missing {TOOLS_TS}"


def test_camouflaging_status_outputschema_matches_actual_response() -> None:
    """The set of property keys declared in tools.ts MUST equal the
    actual response shape from camouflaging.detect_camouflaging() +
    core.py:701 injection."""
    text = TOOLS_TS.read_text(encoding="utf-8")
    schema_body = _extract_camouflaging_status_outputschema(text)
    declared = _declared_property_keys(schema_body)

    missing_from_schema = EXPECTED_RESPONSE_KEYS - declared
    extras_in_schema = declared - EXPECTED_RESPONSE_KEYS

    assert not missing_from_schema, (
        f"camouflaging_status outputSchema is missing these keys that "
        f"camouflaging.detect_camouflaging() + core.py:701 actually return: "
        f"{sorted(missing_from_schema)}. "
        f"Source of truth: src/iai_mcp/camouflaging.py:87-92 and "
        f"src/iai_mcp/core.py:701."
    )
    assert not extras_in_schema, (
        f"camouflaging_status outputSchema declares these keys that are "
        f"NEVER in the response: {sorted(extras_in_schema)}. "
        f"Source of truth: src/iai_mcp/camouflaging.py:87-92 and "
        f"src/iai_mcp/core.py:701."
    )


def test_camouflaging_status_outputschema_has_no_legacy_names() -> None:
    """formality_trend and anomaly_score are legacy names declared in
    tools.ts but never produced by the Python side. They must not
    appear in the published contract."""
    text = TOOLS_TS.read_text(encoding="utf-8")
    schema_body = _extract_camouflaging_status_outputschema(text)
    declared = _declared_property_keys(schema_body)

    leaks = declared & FORBIDDEN_KEYS
    assert not leaks, (
        f"camouflaging_status outputSchema still declares legacy field "
        f"names that no Python code path returns: {sorted(leaks)}. "
        f"Remove them from tools.ts."
    )
