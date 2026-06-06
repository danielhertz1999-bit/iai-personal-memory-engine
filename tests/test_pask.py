"""Regression tests for the Pask teach-back loop.

The tests pin the following acceptance contracts:

    - verify_hit_set behaves correctly on edge cases -- empty input
        returns has_contradictions=False with the trivial summary; a
        synthetic 2-record store with a `contradicts` edge between them
        returns has_contradictions=True with the exact (src, dst) pair;
        a 3-record store with zero contradicts edges returns
        has_contradictions=False.

    - memory_recall via core.dispatch attaches `pask_teachback` to the
        response under default config (PASK_ENABLED=true).

    - memory_recall emits a `pask_teachback_pass` event with the
        expected payload shape (hit_count, has_contradictions,
        contradiction_count, dry_run_mode).

    - malformed env vars fail loud with ValueError naming the offending
        variable (parametrized across the 2 IAI_MCP_PASK_* env vars +
        multiple bad-value shapes). Call `_load_pask_config()` DIRECTLY
        (not via dispatch) because dispatch's outer try/except swallows
        the ValueError per the Pask-is-non-critical guard in core.py.

    - IAI_MCP_PASK_DRY_RUN=true emits the event with dry_run_mode=true
        in the payload; IAI_MCP_PASK_ENABLED=false suppresses both the
        `pask_teachback` response key AND the `pask_teachback_pass` event.

Pattern mirrors tests/test_dmn_meta.py and
tests/test_phase11_7_spatial_scaffold.py. Synthetic stores use tmp_path
with user_id='alice'. Fixture seed values use 'alice' / 'bob' / lorem-
style labels -- never 'Alice' (the project convention).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.core import dispatch
from iai_mcp.daemon import PaskConfig, _load_pask_config
from iai_mcp.events import query_events
from iai_mcp.pask_teachback import verify_hit_set
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Autouse fixture: pin IAI_MCP_STORE to tmp_path so per-test MemoryStore
# construction never touches the user's real ~/.iai-mcp store. Defensively
# wipe both IAI_MCP_PASK_* env vars so each test starts from the defaults
# (enabled=True, dry_run=True under pytest via PYTEST_CURRENT_TEST).
# Tests that need overrides re-set after this fixture.
@pytest.fixture(autouse=True)
def _isolate_iai_pask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_PASK_ENABLED",
        "IAI_MCP_PASK_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)


# Build a minimal MemoryRecord with a per-record orthogonal-ish embedding
# axis so the pattern_separation_gate inside store.insert cannot collapse
# two synthetic records as near-duplicates. Same shape as the
# helper in tests/test_dmn_meta.py; literal_surface defaults
# to an 'alice prefers tea' lorem-style label that satisfies the project convention
# (no 'Alice' in fixture data).
def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    community_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
        community_id=community_id,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )


def _make_store(tmp_path: Path) -> MemoryStore:
    """Build a per-test MemoryStore rooted at tmp_path."""
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


def _seed_orthogonal_records(
    store: MemoryStore, labels: list[str],
) -> list[uuid.UUID]:
    """Insert one record per label with orthogonal embedding axes.

    Returns inserted ids in insertion order. Distinct orthogonal-ish
    embedding axes (one cell per record) keep pattern_separation_gate
    from collapsing the seeded rows as near-duplicates (mirrors helper invariant).
    """
    embed_dim = store._embed_dim
    ids: list[uuid.UUID] = []
    for i, surface in enumerate(labels):
        emb = [0.0] * embed_dim
        emb[i % embed_dim] = 1.0
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=surface,
            embedding=emb,
        )
        store.insert(rec)
        ids.append(rec.id)
    return ids


# ---------------------------------------------------------------------------
# Test 1: verify_hit_set edge cases (empty input)
# ---------------------------------------------------------------------------


def test_verify_hit_set_empty_input_no_contradictions(tmp_path: Path) -> None:
    """Acceptance (edge case): verify_hit_set(store, []) returns the
    trivial happy-path dict -- has_contradictions=False, empty pairs list,
    hit_count=0, and a teachback_summary that explicitly mentions the
    zero count so operators reading the response can disambiguate "no
    contradictions found" from "no hits to check".
    """
    store = _make_store(tmp_path)

    result = verify_hit_set(store, [])

    assert isinstance(result, dict)
    assert result["has_contradictions"] is False, (
        f"empty input must report has_contradictions=False; got {result!r}"
    )
    assert result["contradiction_pairs"] == [], (
        f"empty input must report empty contradiction_pairs; got {result!r}"
    )
    assert result["hit_count"] == 0, (
        f"hit_count must echo len(hit_record_ids); got {result!r}"
    )
    assert isinstance(result["teachback_summary"], str)
    # Summary must mention the (zero) count so operators reading the
    # response can disambiguate "no contradictions" from "no hits".
    assert "0" in result["teachback_summary"], (
        f"teachback_summary must mention hit_count=0; "
        f"got {result['teachback_summary']!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: verify_hit_set detects a contradicts edge between hits
# ---------------------------------------------------------------------------


def test_verify_hit_set_detects_contradicts_edge(tmp_path: Path) -> None:
    """Acceptance: a synthetic store with 2 records joined by ONE
    contradicts edge must surface that pair when both UUIDs are in the
    hit set. The (src, dst) tuple in contradiction_pairs is the
    canonicalised sorted pair (boost_edges sorts in-place), so we check
    membership using a sorted-tuple match rather than equality on a
    fixed orientation. Summary must include the WARNING marker that the
    teach-back layer relies on for downstream LLM consumption.
    """
    store = _make_store(tmp_path)
    ids = _seed_orthogonal_records(
        store, ["alice topic alpha", "bob topic beta"],
    )
    rec_a, rec_b = ids[0], ids[1]

    # Seed a single contradicts edge between the two records.
    store.boost_edges(
        [(rec_a, rec_b)], delta=1.0, edge_type="contradicts",
    )

    result = verify_hit_set(store, [rec_a, rec_b])

    assert result["has_contradictions"] is True, (
        f"a contradicts edge between two hit ids must surface; "
        f"got {result!r}"
    )
    assert result["hit_count"] == 2
    assert len(result["contradiction_pairs"]) == 1, (
        f"exactly one contradicts edge was seeded; "
        f"got {result['contradiction_pairs']!r}"
    )

    # boost_edges canonicalises the (src, dst) tuple to sorted order, so
    # the surfaced pair is sorted(str(a), str(b)) regardless of which
    # orientation we passed in. Match on the sorted tuple to be order-
    # independent.
    surfaced = result["contradiction_pairs"][0]
    expected_sorted = tuple(sorted([str(rec_a), str(rec_b)]))
    assert tuple(sorted(surfaced)) == expected_sorted, (
        f"surfaced contradicts pair must match the seeded UUIDs "
        f"(order-independent); got {surfaced!r}, expected sorted "
        f"{expected_sorted!r}"
    )

    # Summary must carry the WARNING marker -- the teach-back string is
    # injected into the next-turn LLM context and the downstream agent
    # keys on this marker to flag a contradicts surface.
    assert "WARNING" in result["teachback_summary"], (
        f"teachback_summary must carry WARNING marker on a contradicts "
        f"hit; got {result['teachback_summary']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: verify_hit_set returns consistent when no contradicts edges
# ---------------------------------------------------------------------------


def test_verify_hit_set_no_edges_returns_consistent(tmp_path: Path) -> None:
    """Acceptance (negative-case symmetry to test 2): a synthetic
    store with 3 records and ZERO contradicts edges must report
    has_contradictions=False with an empty pairs list, even when all
    three ids are in the hit set. Guards against false positives from
    e.g. a hebbian edge bleeding into the contradicts filter.
    """
    store = _make_store(tmp_path)
    ids = _seed_orthogonal_records(
        store,
        ["alice topic alpha", "bob topic beta", "alice and bob agree on tea"],
    )

    # No edges seeded between any of the three. (Optionally: seed a
    # hebbian edge to prove the filter keys on edge_type, not just
    # presence -- the verify_hit_set body's `edge_type = 'contradicts'`
    # SQL filter is what we are asserting against.)
    store.boost_edges(
        [(ids[0], ids[1])], delta=1.0, edge_type="hebbian",
    )

    result = verify_hit_set(store, ids)

    assert result["has_contradictions"] is False, (
        f"no contradicts edges must report has_contradictions=False "
        f"(a hebbian edge must NOT bleed into the filter); got {result!r}"
    )
    assert result["contradiction_pairs"] == []
    assert result["hit_count"] == 3
    assert "consistent" in result["teachback_summary"].lower(), (
        f"clean-pass summary must mention 'consistent'; "
        f"got {result['teachback_summary']!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: memory_recall response carries pask_teachback dict
# ---------------------------------------------------------------------------


def test_memory_recall_payload_contains_teachback(tmp_path: Path) -> None:
    """Acceptance: invoking memory_recall via core.dispatch attaches a
    `pask_teachback` key to the response dict under default config
    (PASK_ENABLED=true). The empty-store branch returns hit_count=0 with
    has_contradictions=False -- exactly the trivial-happy-path shape
    from test 1, surfaced through the live dispatch path. This is the
    one assertion that proves the core.py integration block fires.
    """
    store = _make_store(tmp_path)

    response = dispatch(
        store,
        "memory_recall",
        {"cue": "alice prefers tea", "session_id": "test-session"},
    )

    assert isinstance(response, dict)
    assert "pask_teachback" in response, (
        f"memory_recall response must carry 'pask_teachback' key under "
        f"PASK_ENABLED=true (default); got keys {sorted(response.keys())!r}"
    )
    teachback = response["pask_teachback"]
    assert isinstance(teachback, dict)
    # Shape contract from pask_teachback.verify_hit_set return dict.
    assert teachback["has_contradictions"] is False
    assert teachback["contradiction_pairs"] == []
    assert teachback["hit_count"] == 0
    assert isinstance(teachback["teachback_summary"], str)


# ---------------------------------------------------------------------------
# Test 5: dispatch emits pask_teachback_pass event with payload
# ---------------------------------------------------------------------------


def test_memory_recall_emits_pask_event_with_dry_run_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: memory_recall emits exactly one
    `pask_teachback_pass` event whose payload carries hit_count,
    has_contradictions, contradiction_count, AND dry_run_mode echoed
    from the live PaskConfig. Explicit
    IAI_MCP_PASK_DRY_RUN=true overrides the pytest-aware default
    (which would already be True) and proves the env var flows through
    `_load_pask_config` into the event body verbatim.
    """
    monkeypatch.setenv("IAI_MCP_PASK_DRY_RUN", "true")

    store = _make_store(tmp_path)

    response = dispatch(
        store,
        "memory_recall",
        {"cue": "alice prefers tea", "session_id": "test-session"},
    )

    # Sanity: response carries the teachback dict (proves the gate
    # opened and the event-emit code path is reachable).
    assert "pask_teachback" in response

    events = query_events(
        store, kind="pask_teachback_pass", limit=10,
    )
    assert len(events) >= 1, (
        f"memory_recall must emit at least one pask_teachback_pass "
        f"event under PASK_ENABLED=true; got {len(events)}"
    )
    body = events[0]["data"]
    # Shape contract from core.py:write_event(pask_teachback_pass,...).
    assert "hit_count" in body
    assert "has_contradictions" in body
    assert "contradiction_count" in body
    assert "dry_run_mode" in body
    # dry_run_mode echoes the env var verbatim.
    assert body["dry_run_mode"] is True, (
        f"dry_run_mode must echo IAI_MCP_PASK_DRY_RUN=true; got {body!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: PASK_ENABLED=false suppresses BOTH response key AND event
# ---------------------------------------------------------------------------


def test_pask_disabled_skips_response_key_and_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: IAI_MCP_PASK_ENABLED=false closes the gate inside
    core.py -- the `pask_teachback` response key is absent AND zero
    `pask_teachback_pass` events fire. Both halves of the gate are
    asserted in one test because they share a single env-var trigger
    and any divergence between the two surfaces (key present, no event;
    or vice-versa) is a bug.
    """
    monkeypatch.setenv("IAI_MCP_PASK_ENABLED", "false")

    store = _make_store(tmp_path)

    response = dispatch(
        store,
        "memory_recall",
        {"cue": "alice prefers tea", "session_id": "test-session"},
    )

    assert "pask_teachback" not in response, (
        f"PASK_ENABLED=false must suppress the pask_teachback response "
        f"key; got keys {sorted(response.keys())!r}"
    )

    events = query_events(
        store, kind="pask_teachback_pass", limit=10,
    )
    assert len(events) == 0, (
        f"PASK_ENABLED=false must suppress ALL pask_teachback_pass "
        f"event emits; got {len(events)}"
    )


# ---------------------------------------------------------------------------
# Test 7: every malformed env var fails loud with ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        # IAI_MCP_PASK_ENABLED -- vocab miss.
        ("IAI_MCP_PASK_ENABLED", "not-a-bool"),
        ("IAI_MCP_PASK_ENABLED", "maybe"),
        ("IAI_MCP_PASK_ENABLED", "perhaps"),
        # IAI_MCP_PASK_DRY_RUN -- vocab miss.
        ("IAI_MCP_PASK_DRY_RUN", "bogus"),
        ("IAI_MCP_PASK_DRY_RUN", "maybe"),
        ("IAI_MCP_PASK_DRY_RUN", "kinda"),
    ],
)
def test_env_var_fail_loud_parametrized(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    """Acceptance: every IAI_MCP_PASK_* env var with a malformed
    value raises ValueError whose message names the offending env var
    (so operators can grep the traceback). Call `_load_pask_config()`
    DIRECTLY because dispatch's outer try/except in core.py
    swallows the ValueError per the Pask-is-non-critical guard.
    """
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ValueError) as excinfo:
        _load_pask_config()

    # ValueError message must name the offending env var so operators
    # can locate the misconfiguration without reading source.
    assert env_var in str(excinfo.value), (
        f"ValueError must name the offending env var {env_var!r}; "
        f"got {excinfo.value!r}"
    )


# Default-parse sanity: with no overrides, _load_pask_config returns
# the defaults (enabled=True, dry_run=True under pytest via
# PYTEST_CURRENT_TEST). Separate from the parametrized fail-loud
# test so a future default change surfaces in exactly one
# assertion site. Mirrors test_dmn_config_defaults_under_pytest in
# tests/test_dmn_meta.py.
def test_pask_config_defaults_under_pytest() -> None:
    """Defaults: enabled=True, dry_run=True (pytest-aware via
    PYTEST_CURRENT_TEST)."""
    cfg = _load_pask_config()
    assert isinstance(cfg, PaskConfig)
    assert cfg.enabled is True
    # Under pytest the pytest-aware default flips dry_run to True
    # (reused -- production default is False).
    assert cfg.dry_run is True
