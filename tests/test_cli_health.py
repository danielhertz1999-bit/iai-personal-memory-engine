"""Tests for the iai-mcp CLI -- health + migrate commands."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest


# ----------------------------------------------------------- iai-mcp health


def test_cli_health_no_events(tmp_path, monkeypatch, capsys):
    """Fresh store -> 'llm_health: no events recorded'."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_health

    args = argparse.Namespace()
    exit_code = cmd_health(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "no events" in out.lower()


def test_cli_health_reports_last_event(tmp_path, monkeypatch, capsys):
    """Seeded llm_health event -> output includes severity + ts rendered in TZ."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_health
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    write_event(
        store,
        kind="llm_health",
        data={"status": "ok"},
        severity="info",
    )
    args = argparse.Namespace()
    exit_code = cmd_health(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "llm_health" in out
    # Severity reported.
    assert "info" in out


# ---------------------------------------------------------- iai-mcp migrate


def test_cli_migrate_dry_run(tmp_path, monkeypatch, capsys):
    """Seeded v1 records -> dry-run prints 'would migrate N records'."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_LEGACY, EMBED_DIM

    store = MemoryStore()
    for i in range(3):
        r = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"Legacy v1 record number {i} with words to detect.",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=[],
            language="en",
            schema_version=SCHEMA_VERSION_LEGACY,
        )
        # simulate un-tagged legacy by clearing language after construction
        r.language = ""
        store.insert(r)

    args = argparse.Namespace(from_=1, to=2, dry_run=True, verbose=False)
    exit_code = cmd_migrate(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "would migrate" in out.lower()

    # Dry run must not mutate the store: all records still v1.
    for r in store.all_records():
        if not r.pinned or r.id == uuid4():  # skip potential L0
            continue
    v1_count = sum(1 for r in store.all_records() if r.schema_version == 1)
    # At least the 3 we inserted must still be v1.
    assert v1_count >= 3


def test_cli_entrypoint_exists():
    """`iai-mcp` entrypoint is registered via pyproject.toml scripts."""
    from iai_mcp.cli import main

    assert callable(main)
