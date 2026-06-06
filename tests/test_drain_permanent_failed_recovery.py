"""Drain reliability — no silent memory loss from permanent-failed path.

test_tem_import_guard: requires the tem import in store.insert to be
    wrapped with try/except ImportError.
test_capture_turn_inserts_in_clean_env: smoke check that the current env works.
test_drain_permanent_failed_reingests: requires the
    iai-mcp drain-permanent-failed recovery path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from iai_mcp.capture import capture_turn
from tests.conftest_recall import make_tmp_store


def test_tem_import_guard(tmp_path, monkeypatch):
    """Regression gate: unguarded tem import causes insert to fail;
    once the try/except ImportError guard is in place, insert must succeed.

    Pre-fix behavior: sys.modules["iai_mcp.tem"] = None forces
    ImportError inside store.insert() → capture_turn returns
    {"status": "skipped", "reason": startswith "insert-failed:"}.

    Post-fix behavior: the guard catches ImportError,
    structure_hv stays b"" (valid pre-migration sentinel), insert proceeds,
    capture_turn returns {"status": "inserted"}.

    This test asserts the POST-FIX expectation, so it is RED today
    and will flip GREEN when the guard lands.
    """
    store = make_tmp_store(tmp_path)

    original = sys.modules.get("iai_mcp.tem", _SENTINEL := object())
    sys.modules["iai_mcp.tem"] = None  # type: ignore[assignment]
    try:
        result = capture_turn(
            store,
            cue="tem guard test cue",
            text="tem import guard regression test text phase59 unique content",
            tier="episodic",
            session_id="sess-tem-guard",
            role="user",
        )
        # Without the guard, status == "skipped" (insert-failed).
        # This assertion requires the guard to be in place.
        assert result["status"] == "inserted", (
            f"store.insert must survive ImportError from iai_mcp.tem "
            f"when the guard is present; got status={result['status']!r} "
            f"reason={result.get('reason')!r}. "
            "The tem import must be wrapped in try/except ImportError: pass."
        )
    finally:
        if original is _SENTINEL:
            sys.modules.pop("iai_mcp.tem", None)
        else:
            sys.modules["iai_mcp.tem"] = original  # type: ignore[assignment]


def test_capture_turn_inserts_in_clean_env(tmp_path):
    """Smoke: capture_turn works in the current (clean) environment.

    GREEN today — pins current-env health. If this fails, the venv itself
    is broken and must be fixed before can proceed.
    """
    store = make_tmp_store(tmp_path)
    result = capture_turn(
        store,
        cue="clean env smoke test",
        text="clean env smoke test phrase phase59 unique content abc123",
        tier="episodic",
        session_id="sess-clean-env",
        role="user",
    )
    assert result["status"] == "inserted", (
        f"capture_turn failed in clean env: {result!r}"
    )


def test_drain_permanent_failed_reingests(tmp_path, monkeypatch):
    """drain-permanent-failed re-ingests a terminal file.

    Writes a .permanent-failed-<ts>.jsonl with one genuine user event into a
    tmp deferred-captures dir; asserts the recovery path
    renames + re-ingests the file and the genuine line lands in the store.

    Requires drain_permanent_failed_files() / the
    iai-mcp drain-permanent-failed command.

    SAFETY: all paths are under tmp_path, never ~/.iai-mcp/.
    """
    store = make_tmp_store(tmp_path)

    # Build a fake deferred-captures directory with one permanent-failed file.
    captures_dir = tmp_path / "deferred-captures"
    captures_dir.mkdir()

    session_id = "sess-recovery-test"
    genuine_text = "recovery test phrase zzz7777 should land in store"
    event = {
        "type": "user",
        "message": {"role": "user", "content": genuine_text},
        "session_id": session_id,
    }
    pf_file = captures_dir / f".permanent-failed-20260530T120000.jsonl"
    pf_file.write_text(json.dumps(event) + "\n", encoding="utf-8")

    # Point capture machinery at the tmp dir.
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))

    # Forward-ref: import INSIDE test body so collection does not fail
    # before the command ships.
    try:
        from iai_mcp.capture import drain_permanent_failed_files  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pytest.xfail(reason="drain_permanent_failed_files not yet implemented")

    drained = drain_permanent_failed_files(store, deferred_dir=captures_dir)  # type: ignore[call-arg]

    # After recovery, the genuine line must be in the store.
    records = store.all_records()
    matching = [r for r in records if genuine_text in r.literal_surface]
    assert matching, (
        f"REQ-3: permanent-failed re-drain must ingest genuine turn into store; "
        f"'{genuine_text}' not found. drain result: {drained!r}"
    )
