from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from .test_socket_server_dispatch import short_socket_paths  # noqa: F401
from .test_socket_backward_compat_stdio import (
    _spawn_stdio_core,
    _stdio_call,
    _terminate,
)

_EMPTY_DIGEST_EXPECTED = {
    "rem_cycles_completed": 0,
    "episodes_processed": 0,
    "schemas_induced_tier0": 0,
    "claude_call_used": False,
    "quota_used_pct": 0.0,
    "main_insight_text": None,
    "sigma_observed": None,
    "s5_drift_alerts": [],
    "daemon_uptime_hours": 0,
    "timed_out_cycles": 0,
}

@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from iai_mcp import daemon_state

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path

def test_fresh_spawn_no_rem_yields_zeroed_default(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    save_state({})

    response: dict = {"hits": []}
    _inject_overnight_digest(response)

    assert "overnight_digest" in response, (
        "deterministic contract: key must be present even with no pending digest"
    )
    assert response["overnight_digest"] == _EMPTY_DIGEST_EXPECTED, (
        f"zeroed default mismatch: got {response['overnight_digest']!r}"
    )

def test_rem_cycle_pending_yields_populated_digest(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    now = datetime.now(timezone.utc)
    save_state({
        "pending_digest": {
            "rem_cycles_completed": 4,
            "episodes_processed": 10,
            "schemas_induced_tier0": 3,
            "claude_call_used": True,
            "quota_used_pct": 0.003,
            "main_insight_text": "today's unifying insight",
            "sigma_observed": 1.2,
            "s5_drift_alerts": [],
            "daemon_uptime_hours": 8,
            "timed_out_cycles": 0,
        },
        "last_digest_shown_at": (now - timedelta(hours=20)).isoformat(),
    })

    response: dict = {"hits": []}
    _inject_overnight_digest(response)

    assert "overnight_digest" in response
    dig = response["overnight_digest"]
    assert dig["rem_cycles_completed"] == 4
    assert dig["claude_call_used"] is True
    assert dig["main_insight_text"] == "today's unifying insight"

def test_stdio_and_socket_recall_top_level_keys_identical(short_socket_paths, tmp_path):
    from iai_mcp.store import MemoryStore
    from .test_socket_server_dispatch import _send_jsonrpc, _with_socket_server

    _, sock_path, _ = short_socket_paths

    params = {"cue": "test", "budget_tokens": 100}

    async def _runner(sock_path, store):
        return await _send_jsonrpc(sock_path, "memory_recall", params)

    socket_resp = asyncio.run(
        _with_socket_server(sock_path, MemoryStore(), _runner)
    )

    proc = _spawn_stdio_core()
    try:
        stdio_resp = _stdio_call(proc, "memory_recall", params)
    finally:
        _terminate(proc)

    assert "result" in socket_resp, socket_resp
    assert "result" in stdio_resp, stdio_resp

    socket_result = socket_resp["result"]
    stdio_result = stdio_resp["result"]
    assert isinstance(socket_result, dict), socket_result
    assert isinstance(stdio_result, dict), stdio_result

    assert "overnight_digest" in socket_result, (
        f"socket result missing overnight_digest: keys={sorted(socket_result)}"
    )
    assert "overnight_digest" in stdio_result, (
        f"stdio result missing overnight_digest: keys={sorted(stdio_result)}"
    )
    assert set(socket_result.keys()) == set(stdio_result.keys()), (
        f"top-level key sets differ:\n"
        f"  socket={sorted(socket_result.keys())}\n"
        f"  stdio ={sorted(stdio_result.keys())}\n"
        f"  diff(s-t)={sorted(set(socket_result) - set(stdio_result))}\n"
        f"  diff(t-s)={sorted(set(stdio_result) - set(socket_result))}"
    )
