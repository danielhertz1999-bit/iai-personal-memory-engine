"""— core-side HIPPEA fallback cascade tests.

Closes the N=1k cross-process LRU gap that flagged as
known: the daemon's cascade populates the daemon's LRU, but the MCP
core runs in a different process and ``snapshot_warm_ids()`` returns
``[]`` for the core's first recall. Solution is a synchronous helper
(Task 1) plus a one-time-per-session call site in
``_first_turn_recall_hook`` (Task 2). The daemon's LRU is untouched.

Covered contracts:

    Task 1 — helper (compute_core_side_warm_snapshot):
        T1.1  helper exists and is synchronous
        T1.2  returns list[UUID] with length <= max_records
        T1.3  returns [] when no salient communities (cold fallback)
        T1.4  read-only against store across 5 invocations
        T1.5  does NOT mutate the daemon-side _warm_lru
        T1.6  respects the top-K salient community ranking
        T1.7  C3 guard — no anthropic import in the module
        T1.8  performance floor — <100 ms on N=1000 records

    Task 2 — wiring (_first_turn_recall_hook fallback):
        T2.1  _CORE_WARM_LRU module-level TTLCache present
        T2.2  _CORE_CASCADE_FIRED_PER_SESSION module-level set present
        T2.3  empty daemon snapshot + first call -> cascade fires
        T2.4  second call same session -> cascade is NOT fired again (idempotent)
        T2.5  non-empty daemon snapshot -> core fallback is NOT fired
        T2.6  compute_core_side_warm_snapshot raising is silently swallowed
        T2.7  regression fence — helper does not touch recall accuracy
        T2.8  response carries warm_lru_source observability field
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock
from uuid import UUID, uuid4

import pytest

from iai_mcp import hippea_cascade
from iai_mcp.store import MemoryStore


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


@pytest.fixture(autouse=True)
def _reset_daemon_lru():
    hippea_cascade._warm_lru.clear()
    yield
    hippea_cascade._warm_lru.clear()


@pytest.fixture(autouse=True)
def _reset_core_state():
    """clear _CORE_WARM_LRU and _CORE_CASCADE_FIRED_PER_SESSION
    between tests so idempotency assertions are deterministic."""
    from iai_mcp import core as _core

    # May not exist yet in RED phase — skip gracefully.
    lru = getattr(_core, "_CORE_WARM_LRU", None)
    fired = getattr(_core, "_CORE_CASCADE_FIRED_PER_SESSION", None)
    if lru is not None:
        lru.clear()
    if fired is not None:
        fired.clear()
    yield
    if lru is not None:
        lru.clear()
    if fired is not None:
        fired.clear()


def _make_assignment_with_communities(*community_ids):
    """Minimal CommunityAssignment-shaped object with deterministic mid_regions.
    Each community maps to an empty list — tests that need records inject
    them via store seeding + monkeypatching `_top_n_records_by_centrality`."""
    class _A:
        def __init__(self, mid):
            self.mid_regions = mid
            self.top_communities = list(mid.keys())

    return _A({cid: [] for cid in community_ids})


# --------------------------------------------------------------------------- Task 1


def test_compute_core_side_warm_snapshot_exists_and_is_sync():
    assert hasattr(hippea_cascade, "compute_core_side_warm_snapshot")
    fn = hippea_cascade.compute_core_side_warm_snapshot
    assert not inspect.iscoroutinefunction(fn)


def test_compute_core_side_warm_snapshot_respects_max_records(
    store, monkeypatch
):
    c1, c2, c3 = uuid4(), uuid4(), uuid4()
    assignment = _make_assignment_with_communities(c1, c2, c3)
    # Inject top-K selection so we don't depend on real event history.
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1, c2, c3],
    )
    # Inject centrality-sorted record ids (more than max_records total).
    fake_ids = [uuid4() for _ in range(60)]

    def _per_c(_s, _a, cid, n):
        # Distribute across 3 communities, each returns n items from fake_ids.
        return fake_ids[:n]

    monkeypatch.setattr(hippea_cascade, "_top_n_records_by_centrality", _per_c)

    result = hippea_cascade.compute_core_side_warm_snapshot(
        store, assignment, top_k=3, max_records=50,
    )
    assert isinstance(result, list)
    assert len(result) <= 50
    assert all(isinstance(r, UUID) for r in result)


def test_compute_core_side_warm_snapshot_empty_when_no_salient(store, monkeypatch):
    assignment = _make_assignment_with_communities()
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [],
    )
    result = hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    assert result == []


def test_compute_core_side_warm_snapshot_is_read_only(store, monkeypatch):
    c1 = uuid4()
    assignment = _make_assignment_with_communities(c1)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1],
    )
    monkeypatch.setattr(
        hippea_cascade, "_top_n_records_by_centrality",
        lambda *a, **kw: [],
    )
    # 5 invocations in a row should not mutate any store state reachable via
    # public getters. MemoryStore has no general-purpose accessor count, so
    # we assert on records table count_rows before/after instead.
    before = store.db.open_table("records").count_rows()
    for _ in range(5):
        hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    after = store.db.open_table("records").count_rows()
    assert before == after


def test_compute_core_side_warm_snapshot_does_not_touch_daemon_lru(
    store, monkeypatch
):
    c1 = uuid4()
    assignment = _make_assignment_with_communities(c1)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1],
    )
    monkeypatch.setattr(
        hippea_cascade, "_top_n_records_by_centrality",
        lambda *a, **kw: [uuid4() for _ in range(5)],
    )
    assert len(hippea_cascade._warm_lru) == 0
    hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    # The sync helper is opportunistic for the *caller*'s LRU; it must not
    # quietly write into the daemon's process-local LRU.
    assert len(hippea_cascade._warm_lru) == 0


def test_compute_core_side_warm_snapshot_honours_topk_ranking(store, monkeypatch):
    c_top = uuid4()
    c_mid = uuid4()
    c_low = uuid4()
    assignment = _make_assignment_with_communities(c_top, c_mid, c_low)
    # Salience picks c_top and c_mid (top 2 of 3).
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c_top, c_mid],
    )
    calls: list[UUID] = []

    def _per_c(_s, _a, cid, n):
        calls.append(cid)
        return []

    monkeypatch.setattr(hippea_cascade, "_top_n_records_by_centrality", _per_c)
    hippea_cascade.compute_core_side_warm_snapshot(
        store, assignment, top_k=2, max_records=10,
    )
    assert c_top in calls
    assert c_mid in calls
    assert c_low not in calls


def test_hippea_cascade_module_has_no_anthropic_import():
    source = Path(hippea_cascade.__file__).read_text()
    assert "import anthropic" not in source
    assert "ANTHROPIC_API_KEY" not in source
    assert " from anthropic" not in source


def test_compute_core_side_warm_snapshot_is_fast(store, monkeypatch):
    """Pure salience + per-record store.get — should stay well under 100 ms
    even on N=1000 scale. We stub the salience + centrality layers so the
    timing reflects the orchestration alone (the real formulas are covered
    by test_hippea_cascade.py)."""
    import time

    c1 = uuid4()
    assignment = _make_assignment_with_communities(c1)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1],
    )
    monkeypatch.setattr(
        hippea_cascade, "_top_n_records_by_centrality",
        lambda *a, **kw: [uuid4() for _ in range(50)],
    )
    t0 = time.perf_counter()
    result = hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100
    assert len(result) == 50


# --------------------------------------------------------------------------- Task 2


def test_core_warm_lru_module_level_ttlcache():
    from iai_mcp import core as _core

    assert hasattr(_core, "_CORE_WARM_LRU")
    # The attribute is a cachetools TTLCache instance; its dict-like shape
    # is what the fallback code relies on.
    lru = _core._CORE_WARM_LRU
    assert hasattr(lru, "__setitem__")
    assert hasattr(lru, "__getitem__")
    # Exposed constants per plan: maxsize=50, ttl=300.
    assert getattr(lru, "maxsize", None) == 50


def test_core_cascade_fired_per_session_module_level_set():
    from iai_mcp import core as _core

    assert hasattr(_core, "_CORE_CASCADE_FIRED_PER_SESSION")
    assert isinstance(_core._CORE_CASCADE_FIRED_PER_SESSION, set)


def _invoke_first_turn_hook(session_id="sess-a", cue="hello"):
    """Drive _first_turn_recall_hook with minimal params + a patched
    consume_first_turn so the idempotency flag doesn't block the call."""
    from iai_mcp import core as _core

    response: dict = {}
    params = {"session_id": session_id, "cue": cue}

    # Build a fake store that survives the retrieve path without LanceDB
    # round-trips (saves ~seconds per test case).
    store = mock.MagicMock()
    store.get = mock.MagicMock(return_value=None)

    with mock.patch("iai_mcp.daemon_state.consume_first_turn", return_value=True), \
         mock.patch("iai_mcp.daemon_state.load_state", return_value={}):
        with mock.patch(
            "iai_mcp.retrieve.recall",
            return_value=mock.MagicMock(hits=[], budget_used=0, anti_hits=[]),
        ), mock.patch(
            "iai_mcp.retrieve.build_runtime_graph",
            return_value=(None, _make_assignment_with_communities(), None),
        ):
            _core._first_turn_recall_hook(response, params=params, store=store)
    return response


def test_empty_daemon_snapshot_triggers_core_cascade():
    from iai_mcp import core as _core

    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(3)],
    ) as css:
        _invoke_first_turn_hook(session_id="sess-empty")
        assert css.call_count == 1
        assert "sess-empty" in _core._CORE_CASCADE_FIRED_PER_SESSION


def test_same_session_does_not_refire_cascade():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(3)],
    ) as css:
        _invoke_first_turn_hook(session_id="sess-idem")
        _invoke_first_turn_hook(session_id="sess-idem")
        _invoke_first_turn_hook(session_id="sess-idem")
        assert css.call_count == 1


def test_non_empty_daemon_snapshot_skips_core_cascade():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[uuid4()]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[],
    ) as css:
        _invoke_first_turn_hook(session_id="sess-daemon-warm")
        assert css.call_count == 0


def test_core_cascade_failure_is_silent():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        side_effect=RuntimeError("boom"),
    ):
        response = _invoke_first_turn_hook(session_id="sess-bad-cascade")
    # Hook must complete; response must carry a first_turn_recall dict even
    # with no hits. Silent-fail is the contract.
    assert "first_turn_recall" in response


def test_m04_regression_fence_cascade_is_read_only():
    """Running the fallback multiple times does not alter the cold recall
    path's hit list. The cascade populates an LRU for observability; the
    authoritative ``retrieve.recall(...)`` still runs and owns the answer."""
    observed_results = []

    def _recall_side_effect(**kw):
        r = mock.MagicMock(hits=[mock.MagicMock(record_id=uuid4())], budget_used=10, anti_hits=[])
        observed_results.append(r)
        return r

    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(5)],
    ), mock.patch(
        "iai_mcp.retrieve.recall", side_effect=_recall_side_effect,
    ), mock.patch(
        "iai_mcp.retrieve.build_runtime_graph",
        return_value=(None, _make_assignment_with_communities(), None),
    ):
        from iai_mcp import core as _core

        for sess in ("s1", "s2", "s3"):
            resp = {}
            params = {"session_id": sess, "cue": "x"}
            store = mock.MagicMock()
            store.get = mock.MagicMock(return_value=None)
            with mock.patch(
                "iai_mcp.daemon_state.consume_first_turn", return_value=True
            ), mock.patch(
                "iai_mcp.daemon_state.load_state", return_value={}
            ):
                _core._first_turn_recall_hook(resp, params=params, store=store)
    # Every session invoked recall exactly once — cascade did not steal
    # or duplicate invocations.
    assert len(observed_results) == 3


def test_response_carries_warm_lru_source():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(2)],
    ):
        response = _invoke_first_turn_hook(session_id="sess-obs")
    assert "first_turn_recall" in response
    assert "warm_lru_source" in response["first_turn_recall"]
    assert response["first_turn_recall"]["warm_lru_source"] in (
        "daemon", "core_fallback", "none",
    )
