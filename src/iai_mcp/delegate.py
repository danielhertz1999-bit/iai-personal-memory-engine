from __future__ import annotations


SUBAGENT_HOT_TOOLS: tuple[str, ...] = (
    "memory_recall",
    "memory_reinforce",
    "memory_contradict",
    "memory_consolidate",
    "profile_get_set",
)


def subagent_proxy_tools() -> list[dict]:
    return [
        {"name": name, "proxied_via": "parent_session"}
        for name in SUBAGENT_HOT_TOOLS
    ]


def serialize_session_for_subagent(
    store,
    assignment,
    rich_club,
) -> dict:
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
