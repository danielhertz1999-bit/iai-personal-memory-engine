"""Phase 07.9 W4 / — pipeline._find_anti_hits defensive UUID parse.

Pre-fix: a single malformed src/dst value in the edges table aborts
``_find_anti_hits`` at the inner ``UUID(lid)`` call, which in turn
aborts the post-rank stage of ``_recall_core`` for any recall whose
top hit is a contradicts-edge endpoint of the corrupted row. One bad
edge poisons every recall that touches the contradicting hit until
the row is repaired.

Post-fix: ``_find_anti_hits`` filters edge rows whose src/dst cannot be
parsed as UUID before walking, with structured-log observability per
skip; the inner ``UUID(lid)`` is still wrapped defensively for mid-
iteration corruption. Anti-hits is an enrichment signal — degrading
to "no anti-hits" on corruption is always preferred over crashing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.pipeline import _find_anti_hits
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _make_record(rid: UUID, surface: str = "topic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
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
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _add_edge_row(
    store: MemoryStore,
    *,
    src: str,
    dst: str,
    edge_type: str = "contradicts",
    weight: float = 1.0,
) -> None:
    """Direct LanceDB insert for the edges table — used to inject rows
    that the high-level store APIs would normally validate away."""
    tbl = store.db.open_table("edges")
    tbl.add([{
        "src": src,
        "dst": dst,
        "edge_type": edge_type,
        "weight": float(weight),
        "updated_at": datetime.now(timezone.utc),
    }])


def _make_hit(rid: UUID, surface: str = "primary topic") -> MemoryHit:
    return MemoryHit(
        record_id=rid,
        score=0.9,
        reason="test_hit",
        literal_surface=surface,
        adjacent_suggestions=[],
    )


# --------------------------------------------------------------------------- W4 tests


def test_malformed_dst_does_not_crash_and_valid_anti_surfaces(store, caplog):
    """W4 / a contradicts edge with dst='not-a-uuid' is filtered
    + logged; the valid contradicts edge still surfaces as an anti-hit."""
    rid_hit = uuid4()
    rid_anti = uuid4()
    store.insert(_make_record(rid_hit, "primary topic"))
    store.insert(_make_record(rid_anti, "anti topic"))

    # One valid contradicts edge AND one with malformed dst.
    _add_edge_row(store, src=str(rid_hit), dst=str(rid_anti),
                  edge_type="contradicts", weight=1.0)
    _add_edge_row(store, src=str(rid_hit), dst="not-a-uuid",
                  edge_type="contradicts", weight=1.0)

    # MemoryGraph isn't actually consulted in _find_anti_hits per the
    # current implementation (it walks the edges table directly), but
    # the signature requires it. A minimal MemoryGraph satisfies the
    # type contract.
    from iai_mcp.graph import MemoryGraph
    graph = MemoryGraph()

    hit = _make_hit(rid_hit)

    with caplog.at_level(logging.WARNING, logger="iai_mcp.pipeline"):
        anti = _find_anti_hits([hit], store, graph, k=3, records_cache=None)

    # Recall did NOT crash. The valid anti-hit surfaced.
    assert len(anti) == 1, (
        f"expected 1 valid anti-hit; got {len(anti)} "
        f"(records: {[h.record_id for h in anti]})"
    )
    assert anti[0].record_id == rid_anti

    # Log captures the skip event for observability.
    assert any(
        "anti_hits_skip_malformed_edge" in r.getMessage()
        for r in caplog.records
    ), f"expected log line; got {[r.getMessage() for r in caplog.records]}"


def test_malformed_src_filtered_at_upstream_step(store, caplog):
    """W4 / a contradicts edge with src='not-a-uuid' is also
    filtered at the upstream pre-walk step. ``linked`` set never
    sees the bad value and the inner UUID(lid) call is never reached."""
    rid_hit = uuid4()
    rid_anti = uuid4()
    store.insert(_make_record(rid_hit))
    store.insert(_make_record(rid_anti))

    # Valid edge + malformed src.
    _add_edge_row(store, src=str(rid_hit), dst=str(rid_anti),
                  edge_type="contradicts", weight=1.0)
    _add_edge_row(store, src="zzz-bad-src", dst=str(rid_hit),
                  edge_type="contradicts", weight=1.0)

    from iai_mcp.graph import MemoryGraph
    graph = MemoryGraph()
    hit = _make_hit(rid_hit)

    with caplog.at_level(logging.WARNING, logger="iai_mcp.pipeline"):
        anti = _find_anti_hits([hit], store, graph, k=3, records_cache=None)

    # The valid anti-hit still surfaces.
    assert len(anti) == 1
    assert anti[0].record_id == rid_anti
    # Upstream filter logged the skip; inner-lid log did NOT fire.
    assert any(
        "anti_hits_skip_malformed_edge" in r.getMessage()
        for r in caplog.records
    )
    assert not any(
        "anti_hits_skip_malformed_lid" in r.getMessage()
        for r in caplog.records
    ), "upstream filter must remove bad rows before the inner UUID(lid) call"


def test_no_contradicts_edges_returns_empty_clean(store):
    """W4 / control: a hit with no contradicts edges still
    returns [] without crashing. (No regression from the defensive
    filter on the all-clean path.)"""
    rid_hit = uuid4()
    store.insert(_make_record(rid_hit))

    from iai_mcp.graph import MemoryGraph
    graph = MemoryGraph()
    hit = _make_hit(rid_hit)

    anti = _find_anti_hits([hit], store, graph, k=3, records_cache=None)
    assert anti == []
