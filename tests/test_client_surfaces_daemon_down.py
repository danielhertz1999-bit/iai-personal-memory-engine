"""RED scaffolds for real Python CLI surfaces (cmd_capture / cmd_last / cmd_recall)
with the daemon DOWN — function-level tests using monkeypatch.

These are the FUNCTION-LEVEL scaffolds: the daemon socket call is forced to fail
via monkeypatch on `iai_mcp.cli._send_jsonrpc_request`, so the direct store path
(not yet wired) is exercised. The GENUINE subprocess versions live in
test_cli_subprocess_daemon_down.py.

All tests are xfail(strict=True) until the corresponding CLI surface wiring lands.
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_store_with_drained_turn(store_root: Path, text: str) -> None:
    """Insert a turn directly into the tmp store (simulating a drained turn).

    A drained turn is in the SQLite store but NOT in.live.jsonl and NOT in
    the bank — the live-layer fallback in cmd_last cannot see it.
    """
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(store_root)
    try:
        rng = np.random.RandomState(seed=77)
        vec = rng.randn(EMBED_DIM).tolist()
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "h2-session", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for cmd_* invocations."""
    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# Test 1: cmd_capture — daemon down → writes DIRECT to store
# ---------------------------------------------------------------------------


def test_cmd_capture_daemon_down_writes_direct_to_store(
    hermetic_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cmd_capture with daemon down writes directly to the Hippo store.

    Forces the daemon socket call to return None (dead socket), then invokes
    cmd_capture and asserts:
    (1) the command returns 0 (success), not the current hard-fail code 1;
    (2) the captured turn is present in the tmp Hippo store.
    """
    import iai_mcp.cli as _cli_mod

    # Force daemon socket call to return None (daemon down).
    monkeypatch.setattr(_cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    from iai_mcp.iai_cli import cmd_capture

    args = _make_args(
        text="h2 capture probe text",
        session_id="h2-cap-session",
    )
    rc = cmd_capture(args)

    assert rc == 0, (
        f"cmd_capture with daemon down must return 0 (success), got {rc}; "
        "the direct-write fallback is not yet wired"
    )

    # Verify the turn landed in the tmp store.
    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        records = store.all_records()
        surfaces = [r.literal_surface or "" for r in records]
        assert any("h2 capture probe text" in s for s in surfaces), (
            "captured turn not found in the tmp Hippo store after daemon-down cmd_capture"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2: cmd_last — daemon down → returns STORE-backed drained turns
# ---------------------------------------------------------------------------


def test_cmd_last_daemon_down_returns_store_backed(
    hermetic_store: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """cmd_last with daemon down returns store-backed drained turns.

    Seeds a drained turn in the tmp store (NOT in.live.jsonl), forces the
    daemon socket to fail, then invokes cmd_last and asserts:
    (1) returns 0;
    (2) stdout contains the drained turn's text (store-backed, not live-only).
    """
    import iai_mcp.cli as _cli_mod

    drained_text = "h2 last drained store turn text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    monkeypatch.setattr(_cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    from iai_mcp.iai_cli import cmd_last

    args = _make_args(n=10, session=None)
    rc = cmd_last(args)

    assert rc == 0, (
        f"cmd_last with daemon down must return 0, got {rc}"
    )

    captured = capsys.readouterr()
    assert drained_text in captured.out, (
        f"drained store turn not in cmd_last stdout;\n"
        f"stdout={captured.out!r}\n"
        "The live-layer fallback cannot see drained turns — must hit the store."
    )


# ---------------------------------------------------------------------------
# Test 3: cmd_recall — daemon down → STORE-backed degraded result (not bank)
# ---------------------------------------------------------------------------


def test_cmd_recall_daemon_down_returns_store_backed_degraded(
    hermetic_store: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """cmd_recall with daemon down returns STORE-backed degraded result (not bank).

    Seeds a drained turn in the tmp store (NOT in the bank), forces the daemon
    socket to fail AND prevents the bank-recall subprocess from running, then
    invokes cmd_recall and asserts:
    (1) returns 0;
    (2) stdout contains the distinctive turn text (store-backed degraded, not bank).
    """
    import iai_mcp.cli as _cli_mod
    import iai_mcp.embed as _embed_mod

    drained_text = "h2 recall store backed degraded probe text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    monkeypatch.setattr(_cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    # The daemon-independent recall path constructs its own embedder via the
    # funnel. In this hermetic tmp-HOME env a real construct would miss the
    # model cache (network/slow). Stub the funnel to RAISE so the path routes
    # to the bypass-safe store-backed recency degrade — the path this test asserts.
    def _no_construct_funnel(_store):
        raise RuntimeError("hermetic: no embedder construct in this degrade test")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _no_construct_funnel)

    # Prevent bank-recall subprocess from running (it cannot find the drained turn).
    import subprocess as _subprocess_mod
    import iai_mcp.iai_cli as _iai_cli_mod

    def _no_bank_subprocess(*args, **kwargs):
        raise AssertionError(
            "cmd_recall must use the direct store degraded path, "
            "not the bank-recall subprocess"
        )

    monkeypatch.setattr(_iai_cli_mod.subprocess, "run", _no_bank_subprocess)

    from iai_mcp.iai_cli import cmd_recall

    args = _make_args(cue="h2 recall store backed degraded", limit=5)
    rc = cmd_recall(args)

    assert rc == 0, (
        f"cmd_recall with daemon down must return 0 (store-backed degraded), got {rc}"
    )

    captured = capsys.readouterr()
    assert drained_text in captured.out, (
        f"drained store turn not in cmd_recall stdout;\n"
        f"stdout={captured.out!r}\n"
        "The bank-recall subprocess cannot see this turn — must be store-backed."
    )
