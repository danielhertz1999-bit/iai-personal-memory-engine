"""H4: dedicated session-start-assembly latency test with 20 large live files.

REQ-6: asserts that p95 of the session-start assembly path (standard wake_depth)
stays < 100 ms even with 20 large synthetic live files on disk (each 500+ event
lines so the maxlen=500 deque actually streams a long file).

Also asserts that read_pending_live_events is ACTUALLY invoked during assembly
(spy/monkeypatch), so the test fails if the live-merge is silently skipped.

SAFETY: HOME, IAI_MCP_STORE, IAI_DAEMON_SOCKET_PATH all redirected to tmp.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def iai_home_latency(tmp_path, monkeypatch):
    """Isolate HOME + store so all paths resolve to tmp."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    yield tmp_path


def _make_large_live_file(deferred_dir: Path, session_id: str, n_events: int = 550) -> Path:
    """Write a live file with n_events event lines (> 500 so deque tail actually streams)."""
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp/latency-test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    base = datetime(2026, 5, 31, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(n_events):
        ev = {
            "text": f"session {session_id} event {i} content for latency test",
            "role": "user",
            "tier": "episodic",
            "ts": (base + timedelta(seconds=i)).isoformat(),
        }
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.perf
def test_session_payload_assembly_latency_with_live_files(iai_home_latency, monkeypatch):
    """H4: p95 of assembly < 100 ms with 20 large live files; helper must fire >= 1 time.

    Builds 20 synthetic live files each with 550+ event lines (exceeds the
    maxlen=500 deque cap so the helper actually streams each file).  Runs the
    standard wake_depth assembly path 30 times, measures p95 wall-time, and
    asserts it is below 100 ms.

    Critically: monkeypatches a counting spy on read_pending_live_events and
    asserts call_count >= 1 so the test FAILS if the live-merge is silently
    skipped — preventing a false-green where the fast path returns before
    invoking the helper.
    """
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.session import assemble_session_start
    from iai_mcp.store import MemoryStore
    import iai_mcp.capture as _cap_mod

    store = MemoryStore(path=iai_home_latency)

    # Build 20 large synthetic live files.
    deferred_dir = iai_home_latency / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        _make_large_live_file(deferred_dir, f"lat-session-{i:02d}", n_events=550)

    # Spy on read_pending_live_events.
    call_count = [0]
    original = _cap_mod.read_pending_live_events

    def spy(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_cap_mod, "read_pending_live_events", spy)

    # Pre-warm (not measured): first call imports modules, JIT-compiles, etc.
    assignment = CommunityAssignment()
    rich_club: list = []
    assemble_session_start(
        store, assignment, rich_club,
        profile_state={"wake_depth": "standard"},
    )
    call_count[0] = 0  # Reset after pre-warm

    # Measure 30 iterations.
    N = 30
    durations: list[float] = []
    for _ in range(N):
        t0 = time.monotonic()
        assemble_session_start(
            store, assignment, rich_club,
            profile_state={"wake_depth": "standard"},
        )
        durations.append(time.monotonic() - t0)

    durations.sort()
    p95_ms = durations[int(N * 0.95)] * 1000
    p95_idx = int(N * 0.95)
    p95_ms = durations[p95_idx] * 1000

    # Assert helper was actually invoked (not silently skipped).
    assert call_count[0] >= 1, (
        f"read_pending_live_events must be called during standard assembly; "
        f"got call_count={call_count[0]}.  The live-merge may have been silently "
        f"skipped, producing a false-green latency result."
    )

    # Assert p95 latency is below 100 ms.
    assert p95_ms < 100.0, (
        f"p95 assembly latency {p95_ms:.1f} ms >= 100 ms with 20 large live files. "
        f"The bounded deque read must keep this well under the 100 ms budget."
    )
