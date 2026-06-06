"""Regression: drain_active_live_captures → store → recent_user_turns end-to-end.

Contract:
- A synthetic session-B.live.jsonl file (header + role:user turns including a
  unique nonce) is processed by drain_active_live_captures() on a tmp store.
- After the drain, recent_user_turns(session_id=B) returns the nonce turn.
- provenance[0].session_id == B's real uuid.
- drain_deferred_captures does NOT process the.live.jsonl file (correct skip).
- A role:user turn under the min-length threshold is skipped (not silently lost
  into a bad count).

This is the authoritative E2E regression for the live-drain path. The test
confirms that:
  1. drain_active_live_captures correctly processes.live.jsonl files.
  2. The drained records are flushed and visible via recent_user_turns.
  3. Session attribution (provenance[0].session_id) is correct.
  4. drain_deferred_captures (ended-file path) still skips.live.jsonl.

SAFETY: uses tmp_path only, never ~/.iai-mcp/ or the live daemon socket.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)


# ---------------------------------------------------------------------------
# Fixture: isolated HOME
# ---------------------------------------------------------------------------


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """Redirect HOME to tmp_path so no test touches ~/.iai-mcp."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-live-drain-e2e-passphrase")
    # Pin the store root so MemoryStore() lands in the isolated tmp dir,
    # not ~/.iai-mcp which the live daemon may be holding.
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _write_live_file(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
) -> Path:
    """Write a synthetic.live.jsonl file with a header and given events."""
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T04:45:00.000000+00:00",
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps(ev, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Core E2E: drain → store → recent_user_turns
# ---------------------------------------------------------------------------


def test_drain_active_live_nonce_surfaces(iai_home):
    """drain_active_live_captures processes a.live.jsonl file; nonce surfaces
    via recent_user_turns(session_id=B) with the real uuid in provenance."""
    from iai_mcp.capture import drain_active_live_captures

    B_SESSION = "7173b585-291f-43e4-96b7-80f3a45e9e14"
    NONCE = "e7k9p cross-session ambient nonce live drain e2e"
    A_SESSION = "aaaabbbb-cccc-dddd-eeee-ffffffffffff"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        B_SESSION,
        [
            {
                "text": NONCE,
                "cue": f"session {B_SESSION} turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:43.000000+00:00",
            },
            {
                "text": "ok",  # below MIN_CAPTURE_LEN=12 — skipped by capture_turn
                "cue": f"session {B_SESSION} turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:44.000000+00:00",
            },
        ],
    )

    store = _open_store()

    # B's live file is not drained by drain_deferred_captures (ended-file path).
    from iai_mcp.capture import drain_deferred_captures
    ended_counts = drain_deferred_captures(store)
    assert ended_counts["events_inserted"] == 0, (
        f"drain_deferred must skip .live.jsonl: {ended_counts}"
    )

    # drain_active_live_captures should process B's file when called from A.
    counts = drain_active_live_captures(store, exclude_session_id=A_SESSION)
    assert counts["events_inserted"] == 1, (
        f"Expected 1 inserted (nonce), got: {counts}"
    )
    assert counts["events_skipped"] == 1, (
        f"Expected 1 skipped (too-short 'ok'), got: {counts}"
    )

    # recent_user_turns with session_id=B must return the nonce.
    turns = store.recent_user_turns(50, session_id=B_SESSION)
    assert len(turns) >= 1, (
        f"recent_user_turns(session_id={B_SESSION!r}) returned {len(turns)} turns; "
        "expected >= 1 after drain"
    )
    texts = [t.literal_surface for t in turns]
    assert any(NONCE in (t or "") for t in texts), (
        f"Nonce not found in recent_user_turns for session {B_SESSION!r}; "
        f"got: {texts!r}"
    )

    # provenance[0].session_id must be the real B uuid, not the caller's A id.
    for t in turns:
        if NONCE in (t.literal_surface or ""):
            prov = (t.provenance or [{}])[0]
            assert prov.get("session_id") == B_SESSION, (
                f"provenance[0].session_id must be {B_SESSION!r}, "
                f"got {prov.get('session_id')!r}"
            )


def test_drain_active_excludes_own_session(iai_home):
    """exclude_session_id prevents a session from draining its own live file."""
    from iai_mcp.capture import drain_active_live_captures

    OWN_SESSION = "ownown00-0000-0000-0000-000000000000"
    NONCE = "own session excluded from live drain test marker xyz"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        OWN_SESSION,
        [
            {
                "text": NONCE,
                "cue": "own session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:50.000000+00:00",
            }
        ],
    )

    store = _open_store()
    counts = drain_active_live_captures(store, exclude_session_id=OWN_SESSION)
    assert counts["events_inserted"] == 0, (
        f"Own session's live file must not be drained (exclude_session_id): {counts}"
    )
    turns = store.recent_user_turns(50, session_id=OWN_SESSION)
    assert len(turns) == 0, (
        f"No turns for own session expected; got {len(turns)}"
    )


def test_drain_active_idempotent_with_offset(iai_home):
    """Calling drain_active_live_captures twice on the same file is idempotent.

    The drain-offset sidecar tracks how many event lines were processed.
    A second call with no new lines must return events_inserted=0.
    """
    from iai_mcp.capture import drain_active_live_captures

    B_SESSION = "bbbbbbb0-0000-0000-0000-000000000000"
    NONCE = "idempotent drain test nonce for offset sidecar live file path"
    A_SESSION = "aaaaaaa0-0000-0000-0000-000000000000"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(
        deferred_dir,
        B_SESSION,
        [
            {
                "text": NONCE,
                "cue": "test turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:46:00.000000+00:00",
            }
        ],
    )

    store = _open_store()
    first = drain_active_live_captures(store, exclude_session_id=A_SESSION)
    assert first["events_inserted"] == 1, f"First drain: {first}"

    # Second call — no new lines added, offset is at EOF.
    second = drain_active_live_captures(store, exclude_session_id=A_SESSION)
    assert second["events_inserted"] == 0, (
        f"Second drain must be idempotent (offset at EOF): {second}"
    )

    # Still exactly one turn in the store.
    turns = store.recent_user_turns(50, session_id=B_SESSION)
    assert len(turns) == 1, f"Expected 1 turn after two drains; got {len(turns)}"
