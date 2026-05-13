""" subagent delegation context (Task 3, ).

Parent session exposes a JSON blob containing the 4-segment session-start
payload (L0, L1, L2, rich-club) plus per-component hashes (for delta
encoding) and a proxy-tools schema listing the 5 Phase-1 memory tools the
subagent may invoke via the parent.

The subagent inherits the parent's session cache; it does NOT re-load the
graph from scratch. This matches the Claude Code subagent-context feature
request (#20304).

Constitutional note: the 3 MCP surface tools (curiosity_pending,
schema_list, events_query) are user-introspection surfaces and are NOT
included in SUBAGENT_HOT_TOOLS. Subagents receive the 5 memory tools; user
introspection stays with the parent session.
"""
from __future__ import annotations


# The 5 memory tools exposed to subagents (hot surface). 's
# new user-introspection tools are intentionally excluded.
SUBAGENT_HOT_TOOLS: tuple[str, ...] = (
    "memory_recall",
    "memory_reinforce",
    "memory_contradict",
    "memory_consolidate",
    "profile_get_set",
)


def subagent_proxy_tools() -> list[dict]:
    """Return a list of tool stubs advertised to the subagent.

    Each stub carries `name` + `proxied_via`; the subagent invokes its
    parent's MCP bridge with the tool name, and the parent forwards the call
    to the Python core.
    """
    return [
        {"name": name, "proxied_via": "parent_session"}
        for name in SUBAGENT_HOT_TOOLS
    ]


def serialize_session_for_subagent(
    store,
    assignment,
    rich_club,
) -> dict:
    """Build a JSON-safe dict for subagent spawn.

    Returns:
        {
          "l0": str,
          "l1": str,
          "l2": list[str],
          "rich_club": str,
          "hashes": {"l0": str, "l1": str, "l2": str, "rich_club": str},
          "proxy_tools": [{"name": ..., "proxied_via": "parent_session"}, ...],
        }
    """
    from iai_mcp.delta import build_delta
    from iai_mcp.session import assemble_session_start

    payload = assemble_session_start(store, assignment, rich_club)
    payload_dict = {
        "l0": payload.l0,
        "l1": payload.l1,
        "l2": list(payload.l2),
        "rich_club": payload.rich_club,
    }
    _delta, hashes = build_delta({}, payload_dict)
    return {
        "l0": payload_dict["l0"],
        "l1": payload_dict["l1"],
        "l2": payload_dict["l2"],
        "rich_club": payload_dict["rich_club"],
        "hashes": hashes,
        "proxy_tools": subagent_proxy_tools(),
    }
