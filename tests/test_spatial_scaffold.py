"""End-to-end regression suite for the spatial-scaffold.

This file pins the integration + heuristic + dry-run + fail-loud contracts:

    smoke: fresh-store schema carries wing/room/drawer.
    SpatialTagger.tag is correct for the three tiers
                    (Desktop marker, default-wings allowlist, fallback) and
                    returns (None, None, None) on empty source_path.
    positive: insert with IAI_MCP_SPATIAL_AUTO_TAG=true + DRY_RUN=false
                    + a Desktop-marker source_path stamps the columns on the
                    stored row.
    negative: insert with IAI_MCP_SPATIAL_AUTO_TAG unset (default
                    False) leaves wing/room/drawer NULL even when the
                    provenance carries a tier-1 source_path.
    malformed AUTO_TAG / DRY_RUN values raise ValueError
                    naming the offending env var (parametrized, 4 sub-cases).
    dry-run = True emits one `spatial_tag_pass` event with
                    the inferred tuple AND keeps the record / row NULL.

Synthetic stores use tmp_path with user_id='alice'. Fixture seed values use
'alice' / 'bob' / lorem-style labels -- never 'Alice' (the project convention).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp.daemon import SpatialConfig, _load_spatial_config
from iai_mcp.events import query_events
from iai_mcp.spatial_tagger import SpatialTagger
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction stays isolated from the user's real store. Defensively
# wipe every IAI_MCP_SPATIAL_* env var so each test starts from defaults
# (auto_tag=False, default_wing="general", dry_run=True under pytest via
# PYTEST_CURRENT_TEST). Tests that need overrides re-set after this
# fixture.
@pytest.fixture(autouse=True)
def _isolate_iai_spatial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_SPATIAL_AUTO_TAG",
        "IAI_MCP_SPATIAL_DEFAULT_WING",
        "IAI_MCP_SPATIAL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# Minimal _make_record helper.
# All 18 kwargs match MemoryRecord's @dataclass signature -- do not paraphrase.
def _make_record(
    *,
    embed_dim: int,
    provenance: list[dict] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface="alice prefers tea over coffee",
        aaak_index="",
        embedding=[0.01] * embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=provenance if provenance is not None else [],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )


def _make_store(tmp_path: Path) -> MemoryStore:
    """Build a per-test MemoryStore rooted at tmp_path with alice as user."""
    return MemoryStore(str(tmp_path / "store"), user_id="alice")


# ---------------------------------------------------------------------------
# Test 1: smoke: fresh-store schema carries the spatial columns
# ---------------------------------------------------------------------------


# Column presence is fully pinned by `test_spatial_columns.py`. This
# one-liner duplicates the column-presence assertion so a developer reading
# this file alone can see the acceptance lines exercised here.
def test_R1_fresh_store_has_wing_room_drawer_columns(tmp_path: Path) -> None:
    """Smoke: a freshly-opened MemoryStore exposes the three spatial
    columns. Full migration semantics live in test_spatial_columns."""
    s = _make_store(tmp_path)
    names = s.db.open_table(RECORDS_TABLE).schema.names
    assert "wing" in names, names
    assert "room" in names, names
    assert "drawer" in names, names


# ---------------------------------------------------------------------------
# Test 2: SpatialTagger heuristic correctness across tiers
# ---------------------------------------------------------------------------


# SpatialTagger.tag returns the expected (wing, room, drawer)
# tuple for each tier (Desktop marker, allowlist, fallback) AND for the
# absent-signal branch. Pure function -- no MemoryStore needed.
def test_R2_spatial_tagger_path_heuristic_correctness() -> None:
    """SpatialTagger.tag returns the expected tuples for the
    three wing tiers plus the absent-signal branch."""
    # Tier 1: Desktop marker -- the component AFTER `Desktop` is the wing.
    wing, room, drawer = SpatialTagger.tag(
        None, "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
    )
    assert wing == "IAI-MCP", wing
    assert room == "iai_mcp", room
    assert drawer == "store", drawer

    # Tier 2: default-wings allowlist -- first matching component wins.
    # No `Desktop/` in the path; `Documents` is in DEFAULT_WINGS.
    wing2, room2, drawer2 = SpatialTagger.tag(
        None, "/Users/alice/Documents/notes/today.md",
    )
    assert wing2 == "Documents", wing2
    assert room2 == "notes", room2
    assert drawer2 == "today", drawer2

    # Tier 3: env-configured default -- path has neither Desktop nor any
    # allowlisted component, so the default_wing kwarg is the fallback.
    wing3, room3, drawer3 = SpatialTagger.tag(
        None, "/var/log/syslog", default_wing="general",
    )
    assert wing3 == "general", wing3
    assert room3 == "log", room3
    assert drawer3 == "syslog", drawer3

    # Absent-signal: empty / None / whitespace source_path -> all None.
    # Crucially the default_wing is NOT substituted here (contract).
    for empty in (None, "", "   "):
        w, r, d = SpatialTagger.tag(None, empty, default_wing="general")
        assert w is None, (empty, w)
        assert r is None, (empty, r)
        assert d is None, (empty, d)


# ---------------------------------------------------------------------------
# Test 3: positive: insert with auto_tag=true populates the row columns
# ---------------------------------------------------------------------------


# Positive path: with IAI_MCP_SPATIAL_AUTO_TAG=true AND
# IAI_MCP_SPATIAL_DRY_RUN=false explicit override (pytest-aware default is
# True), inserting a record whose provenance carries a tier-1
# Desktop-marker source_path stamps wing/room/drawer on the stored row.
def test_R3_insert_path_auto_tag_populates_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive: auto_tag=true + dry_run=false + source_path -> row
    columns populated with the SpatialTagger inference."""
    monkeypatch.setenv("IAI_MCP_SPATIAL_AUTO_TAG", "true")
    monkeypatch.setenv("IAI_MCP_SPATIAL_DRY_RUN", "false")

    s = _make_store(tmp_path)
    rec = _make_record(
        embed_dim=s._embed_dim,
        provenance=[{
            "source_path": "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cue": "alice testing R3 positive path",
            "session_id": "s1",
        }],
    )
    s.insert(rec)

    # In-memory record received the public-attribute mutation.
    assert getattr(rec, "wing", None) == "IAI-MCP"
    assert getattr(rec, "room", None) == "iai_mcp"
    assert getattr(rec, "drawer", None) == "store"

    # Stored row reflects the same inference via _to_row's getattr fallback.
    df = s.db.open_table(RECORDS_TABLE).to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["wing"] == "IAI-MCP", row["wing"]
    assert row["room"] == "iai_mcp", row["room"]
    assert row["drawer"] == "store", row["drawer"]


# ---------------------------------------------------------------------------
# Test 4: negative: auto_tag=false (default) -> columns remain NULL
# ---------------------------------------------------------------------------


# Negative path: default operator posture (auto_tag=False)
# is a no-op -- even with a tier-1 source_path on the record the columns
# stay NULL and no event fires. Pins the "unconfigured deployment stays
# byte-identical" contract.
def test_R3_insert_path_skip_when_auto_tag_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative: auto_tag=false (default) -> row columns NULL AND no
    spatial_tag_pass event emitted."""
    # Explicitly set to "false" so the assertion does not depend on the
    # autouse delenv (defensive against future env-var changes).
    monkeypatch.setenv("IAI_MCP_SPATIAL_AUTO_TAG", "false")

    s = _make_store(tmp_path)
    rec = _make_record(
        embed_dim=s._embed_dim,
        provenance=[{
            "source_path": "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cue": "alice testing R3 negative path",
            "session_id": "s1",
        }],
    )
    s.insert(rec)

    # In-memory record has no spatial attrs set by the helper.
    assert getattr(rec, "wing", None) is None
    assert getattr(rec, "room", None) is None
    assert getattr(rec, "drawer", None) is None

    # Stored row carries NULL spatial columns.
    df = s.db.open_table(RECORDS_TABLE).to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["wing"] is None or (
        isinstance(row["wing"], float) and row["wing"] != row["wing"]
    ), row["wing"]
    assert row["room"] is None or (
        isinstance(row["room"], float) and row["room"] != row["room"]
    ), row["room"]
    assert row["drawer"] is None or (
        isinstance(row["drawer"], float) and row["drawer"] != row["drawer"]
    ), row["drawer"]

    # Zero spatial_tag_pass events emitted (auto_tag=False short-circuits
    # before the event branch).
    events = query_events(s, kind="spatial_tag_pass")
    assert events == [], events


# ---------------------------------------------------------------------------
# Test 5: dry-run preserves NULL columns but emits the event with tuple
# ---------------------------------------------------------------------------


# dry_run=True is the shadow-deploy mode -- the helper
# computes the tuple, emits the spatial_tag_pass event with the inferred
# (wing, room, drawer) AND dry_run_mode=True body keys, but NEVER mutates
# the in-memory record or the stored row. Three-source assertion:
# in-memory, stored row, event payload.
def test_R5_dry_run_no_mutation_but_event_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto_tag=true + dry_run=true -> record + row stay
    NULL but exactly one spatial_tag_pass event with the inferred tuple is
    emitted."""
    monkeypatch.setenv("IAI_MCP_SPATIAL_AUTO_TAG", "true")
    monkeypatch.setenv("IAI_MCP_SPATIAL_DRY_RUN", "true")

    s = _make_store(tmp_path)
    rec = _make_record(
        embed_dim=s._embed_dim,
        provenance=[{
            "source_path": "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cue": "alice testing R5 dry-run",
            "session_id": "s1",
        }],
    )
    s.insert(rec)

    # 1) In-memory record un-mutated.
    assert getattr(rec, "wing", None) is None
    assert getattr(rec, "room", None) is None
    assert getattr(rec, "drawer", None) is None

    # 2) Stored row NULL (no mutation propagated through _to_row).
    df = s.db.open_table(RECORDS_TABLE).to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert row["wing"] is None or (
        isinstance(row["wing"], float) and row["wing"] != row["wing"]
    ), row["wing"]
    assert row["room"] is None or (
        isinstance(row["room"], float) and row["room"] != row["room"]
    ), row["room"]
    assert row["drawer"] is None or (
        isinstance(row["drawer"], float) and row["drawer"] != row["drawer"]
    ), row["drawer"]

    # 3) Event payload carries the inference + dry_run_mode=True.
    events = query_events(s, kind="spatial_tag_pass")
    assert len(events) == 1, f"expected exactly 1 event, got {len(events)}"
    body = events[0]["data"]
    assert body["wing"] == "IAI-MCP", body
    assert body["room"] == "iai_mcp", body
    assert body["drawer"] == "store", body
    assert body["dry_run_mode"] is True, body
    assert body["record_id"] == str(rec.id), body
    assert body["source_path"] == (
        "/Users/alice/Desktop/IAI-MCP/src/iai_mcp/store.py"
    ), body


# ---------------------------------------------------------------------------
# Test 6: malformed AUTO_TAG / DRY_RUN values raise ValueError naming var
# ---------------------------------------------------------------------------


# Every malformed bool env var raises ValueError naming the
# offending variable so operators can act. DEFAULT_WING is string-typed
# (empty -> default) and does NOT fail-loud, so it is not in the matrix.
@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_SPATIAL_AUTO_TAG", "banana"),
        ("IAI_MCP_SPATIAL_AUTO_TAG", "maybe"),
        ("IAI_MCP_SPATIAL_DRY_RUN", "banana"),
        ("IAI_MCP_SPATIAL_DRY_RUN", "maybe"),
    ],
)
def test_R4_invalid_env_var_raises_ValueError_naming_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Fail-loud: every malformed bool knob raises ValueError
    naming the offending env var so operators can act."""
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_spatial_config()


def test_R4_defaults_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults: with no env overrides _load_spatial_config
    returns defaults (auto_tag=False, default_wing='general',
    dry_run=True under PYTEST_CURRENT_TEST)."""
    for var in (
        "IAI_MCP_SPATIAL_AUTO_TAG",
        "IAI_MCP_SPATIAL_DEFAULT_WING",
        "IAI_MCP_SPATIAL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = _load_spatial_config()
    assert isinstance(cfg, SpatialConfig)
    assert cfg.auto_tag is False
    assert cfg.default_wing == "general"
    # PYTEST_CURRENT_TEST is set by pytest -> pytest-aware default fires.
    assert cfg.dry_run is True


if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
