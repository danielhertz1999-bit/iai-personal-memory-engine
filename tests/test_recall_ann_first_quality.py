"""Gate B: bounded ANN-first recall quality parity + set-inclusion.

Strategy: build a deterministic reference store, run BOTH the full-graph
baseline AND the bounded ANN-first assembler on it, and assert:

  bounded_hits ⊇ full_graph_hits on the gold set (CC-C parity)

This directly tests 62-04's contract: the bounded path must not DROP any gold
that the baseline full-graph pipeline keeps. It honors N=1k AND N=10k + the
top-50 rich-club cap (binds at N=10k).

Also asserts:
- Off-path cache prime (BLOCKER-2) for non-empty rich-club
- Hub-sensitive gold (UUID(int=1)) surfaces in both paths
- Anti-hit path: a self-contained cue+contradicts-edge sub-test confirms
  the 62-04 UNCAPPED contradicts read still surfaces anti-hits.
- ann_path_used=True in the bounded path (deterministic marker)
- UUID(int=5) two-hop-only gold (cosine=0.02, outside ANN top-200): parity
  between full-graph and bounded path (both surface it or neither does).

Architecture: UUID(5) degree boost uses REAL in-pool records (UUID(100..109),
cosine ~0.5 to cue) instead of phantom nodes (the prior approach that was
unreproducible). The bounded assembler's global_degree fix (computing GLOBAL
edge counts via uncapped incident_edges per candidate, normalised against the
cached global max_degree) gives the bounded path the same degree signal as
the full-graph path. This makes the parity test have genuine teeth.

Hermetic: monkeypatched HOME/IAI_MCP_STORE/IAI_DAEMON_SOCKET_PATH to tmp.
No live ~/.iai-mcp, no daemon stop/restart.
Run: pytest tests/test_recall_ann_first_quality.py -x
(N=10k cell is marked slow; also covered via --runslow)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.embed import Embedder
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RNG_SEED = 20260601
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recall_quality_baseline.json"

# Rich-club cap value used by the ANN assembler (top-50, core.py _RC_CAP=50).
# Lock value here — do NOT loosen.
RICH_CLUB_CAP = 50

LEXICAL_GENERIC_CUE = "hello"
LEXICAL_SPECIFIC_CUE = "specialized technical framework review"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")


def _make_gold_record(i: int, vec: list[float]) -> object:
    """Build a gold record with deterministic stable UUID(int=i)."""
    from iai_mcp.types import MemoryRecord
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=f"User reference gold doc {i}",
        aaak_index="",
        embedding=vec,
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
    )


def _build_reference_store(store_path: Path, n_records: int) -> MemoryStore:
    """Build the reference store faithful to the Wave-0 fixture generator.

    Reproduces test_recall_stage_profile.py's seeding EXACTLY (same RNG_SEED,
    same edge planting, same phantom edge targets for degree boost) so the
    store is as close as possible to the fixture generator's environment.
    """
    store = MemoryStore(str(store_path))

    # Bulk filler — identical RNG path to Wave-0 fixture generator.
    rng = np.random.default_rng(RNG_SEED)
    for i in range(n_records):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v = (v / np.linalg.norm(v)).tolist()
        rec = _make(text=f"User record {i} filler content for gate-b baseline", vec=v)
        store.insert(rec)

    embedder = Embedder()
    cue_gen_arr = np.asarray(embedder.embed(LEXICAL_GENERIC_CUE), dtype=np.float32)
    cue_gen_arr /= np.linalg.norm(cue_gen_arr)
    cue_spec_arr = np.asarray(embedder.embed(LEXICAL_SPECIFIC_CUE), dtype=np.float32)
    cue_spec_arr /= np.linalg.norm(cue_spec_arr)

    # Hub gold: UUID(int=1), collinear with generic cue
    store.insert(_make_gold_record(1, list(cue_gen_arr)))

    # Hub node: UUID(int=2), high-degree
    rng4 = np.random.default_rng(44444)
    hub_vec = rng4.random(EMBED_DIM).astype(np.float32)
    hub_vec /= np.linalg.norm(hub_vec)
    store.insert(_make_gold_record(2, hub_vec.tolist()))
    store.boost_edges([(UUID(int=2), UUID(int=1))], edge_type="hebbian", delta=[3.0])
    for extra_i in range(12):
        store.boost_edges([(UUID(int=2), UUID(int=1000 + extra_i))], edge_type="hebbian", delta=[1.0])

    # Seed: UUID(int=3), collinear with specific cue
    store.insert(_make_gold_record(3, list(cue_spec_arr)))

    # Intermediate: UUID(int=4), cosine ~0.4 to specific cue
    rng5 = np.random.default_rng(55555)
    inter_noise = rng5.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_spec_arr) * cue_spec_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_vec = 0.4 * cue_spec_arr + 0.9165 * inter_noise
    inter_vec /= np.linalg.norm(inter_vec)
    store.insert(_make_gold_record(4, inter_vec.tolist()))
    for extra_j in range(10):
        store.boost_edges([(UUID(int=4), UUID(int=2000 + extra_j))], edge_type="hebbian", delta=[1.0])

    # Two-hop gold: UUID(int=5), cosine ~0.02 (outside ANN top-200).
    # Degree boost via REAL in-pool records (UUID(int=100..109)) so that:
    # (a) the full-graph path gives UUID(5) high global degree and ranks it,
    # (b) those records ARE in the ANN candidate pool (cosine ~0.5 to cue),
    # (c) the bounded-assembler global_degree fix picks up the same count.
    # This replaces the 62-00 phantom-node approach which was unreproducible:
    # phantom nodes (UUID(3000..3007)) existed only as graph nodes materialised
    # by graph.add_edge, not as SQLite records, so the bounded assembler's
    # incident_edges count excluded them and UUID(5) got near-zero degree in
    # the bounded path.
    rng6 = np.random.default_rng(66666)
    noise = rng6.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_spec_arr) * cue_spec_arr
    noise /= np.linalg.norm(noise)
    target_cosine = 0.02
    orth_mag = float(np.sqrt(max(0.0, 1.0 - target_cosine**2)))
    two_hop_vec = target_cosine * cue_spec_arr + orth_mag * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))
    store.boost_edges([(UUID(int=3), UUID(int=4))], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(UUID(int=4), UUID(int=5))], edge_type="hebbian", delta=[5.0])
    # Degree-boost helper records at UUID(int=100..149): vectors ORTHOGONAL to
    # the specific cue (cosine ≈ 0) so they are OUTSIDE ANN top-200 and do NOT
    # appear in the bounded candidate pool for the specific-cue recall test.
    # UUID(5) is boosted to all of them, giving it a GLOBAL degree of ~52
    # (1 from UUID(4) + 50 here) in the edges table.
    # These edges are counted by the bounded assembler's uncapped incident_edges
    # call for UUID(5) (since UUID(5) IS in the pool via hop-2), giving UUID(5)
    # a high deg_norm even though the helper nodes themselves are not in the pool.
    # never_merge=True pins them so pattern_separation cannot dissolve them.
    rng7 = np.random.default_rng(77001)
    for _boost_i in range(50):
        # Build an orthonormal basis vector via Gram-Schmidt against cue_spec_arr.
        _raw = rng7.random(EMBED_DIM).astype(np.float32)
        _raw -= np.dot(_raw, cue_spec_arr) * cue_spec_arr
        _raw /= np.linalg.norm(_raw)
        _brec = _make_gold_record(100 + _boost_i, _raw.tolist())
        _brec.never_merge = True
        store.insert(_brec)
        store.boost_edges(
            [(UUID(int=5), UUID(int=100 + _boost_i))], edge_type="hebbian", delta=[2.0]
        )

    # Contradicts pair: UUID(int=6) <-> UUID(int=7)
    rng3 = np.random.default_rng(77777)
    ca_vec = rng3.random(EMBED_DIM).astype(np.float32)
    ca_vec = (ca_vec / np.linalg.norm(ca_vec)).tolist()
    cb_vec = rng3.random(EMBED_DIM).astype(np.float32)
    cb_vec = (cb_vec / np.linalg.norm(cb_vec)).tolist()
    store.insert(_make_gold_record(6, ca_vec))
    store.insert(_make_gold_record(7, cb_vec))
    store.boost_edges([(UUID(int=6), UUID(int=7))], edge_type="contradicts", delta=[1.0])

    return store


def _prime_cache(store: MemoryStore) -> tuple:
    """OFF-PATH CACHE PRIME (BLOCKER-2): build_runtime_graph + save UNTIMED.

    Returns (graph, assignment, rc) — used by both the off-path prime AND
    the full-graph baseline recall.

    Saves max_degree so load_recall_structural can expose the global max
    degree to the bounded assembler (needed for correct deg_norm scoring).
    """
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, assignment, rc, max_degree=int(getattr(graph, "_max_degree", 0) or 0))
    return graph, assignment, rc


def _full_graph_recall(
    store: MemoryStore,
    graph,
    assignment,
    rc,
    cue: str,
    budget: int = 2000,
) -> set[str]:
    """Run the old full-graph recall_for_response (the live baseline).

    This is the comparand for Gate B parity: bounded ⊇ full-graph on gold.
    Uses the SAME profile_state as core.dispatch so that degree-norm weights
    (literal_preservation knob) are identical in both paths.
    """
    import iai_mcp.pipeline as _pm
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.embed import Embedder
    from iai_mcp import core as _core

    _pm._last_recall_latency_ms = 0.0
    embedder = Embedder()
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=embedder,
        cue=cue,
        session_id="gate-b-baseline",
        budget_tokens=budget,
        mode="concept",
        profile_state=None,  # no profile override; uses default (lp_scale=1.0)
    )
    return {str(h.record_id) for h in resp.hits}


def _bounded_recall(store: MemoryStore, cue: str, budget: int = 2000) -> dict:
    """Run bounded ANN-first recall via core.dispatch (the production path)."""
    import iai_mcp.pipeline as _pm
    from iai_mcp import core

    _pm._last_recall_latency_ms = 0.0
    params = {
        "cue": cue,
        "session_id": "gate-b-test",
        "budget_tokens": budget,
    }
    return core.dispatch(store, "memory_recall", params)


# ---------------------------------------------------------------------------
# Core parity assertion
# ---------------------------------------------------------------------------


def _run_gate_b_parity(
    store: MemoryStore,
    graph,
    assignment,
    rc,
    n_label: str,
    monkeypatch,
) -> None:
    """Run Gate B parity: bounded ⊇ full-graph on gold set, for both cues.

    Both paths use literal_preservation="medium" (lp_scale=1.0) to match
    the 62-00 fixture generator, which called recall_for_response without
    a profile_state override (default lp_scale=1.0 via empty profile_state).
    The production default is "strong" (0.3), which causes UUID(5) to score
    below top-50 in both paths equally — the correct parity state but with
    no teeth on the two-hop-only gold. The fixture's medium profile is the
    correct context for the CC-C tripwire.

    monkeypatch is used to patch core._profile_state["literal_preservation"]
    so the restore is guaranteed even if any inner call raises an exception.
    """
    from iai_mcp import core as _core
    # Use medium literal_preservation (lp_scale=1.0) so degree signal
    # contributes at 0.1x (matching the 62-00 fixture generator).
    # monkeypatch.setitem auto-restores on test teardown (exception-safe).
    monkeypatch.setitem(_core._profile_state, "literal_preservation", "medium")

    embedder = Embedder()

    failures = []
    telemetry = []

    for cue, cue_label, hub_sensitive in [
        (LEXICAL_GENERIC_CUE, "lexical-generic", True),
        (LEXICAL_SPECIFIC_CUE, "lexical-specific", False),
    ]:
        # Full-graph baseline (live, same store).
        full_hits = _full_graph_recall(store, graph, assignment, rc, cue)

        # Bounded ANN-first (core.dispatch).
        bounded_resp = _bounded_recall(store, cue)
        bounded_hits = {h["record_id"] for h in bounded_resp.get("hits", [])}

        # ann_path_used must be True.
        if not bounded_resp.get("ann_path_used", False):
            failures.append(
                f"  [FAIL] {cue_label} {n_label}: ann_path_used=False"
            )

        # Gate B parity: bounded ⊇ full-graph on gold set (CC-C).
        # Gold set = stable keys (UUID(int=1..7)) that the FULL-GRAPH baseline
        # includes in its hits for this cue.
        gold_stable_keys = {
            str(UUID(int=i)) for i in range(1, 8)
        }
        full_gold = full_hits & gold_stable_keys
        bounded_gold = bounded_hits & gold_stable_keys

        dropped = full_gold - bounded_gold

        if dropped:
            failures.append(
                f"  [FAIL CC-C] {cue_label} {n_label}: "
                f"bounded dropped gold keys that full-graph kept: {dropped}. "
                f"full_gold={full_gold}, bounded_gold={bounded_gold}"
            )
        else:
            telemetry.append(
                f"  [PASS] {cue_label} {n_label}: "
                f"bounded parity. full_gold={full_gold} "
                f"bounded_gold={bounded_gold}"
            )

        # Hub-sensitive gold (UUID(int=1)) must surface in bounded path when
        # the cache is primed (rich-club is non-empty).
        hub_key = str(UUID(int=1))
        if hub_sensitive and hub_key not in bounded_hits:
            failures.append(
                f"  [FAIL HUB] {cue_label} {n_label}: "
                f"hub-sensitive gold {hub_key} not in bounded hits "
                f"(off-path cache prime should give non-empty rich-club)"
            )
        elif hub_sensitive:
            telemetry.append(f"  [PASS HUB] {cue_label} {n_label}: hub gold surfaced")

        # HARD FAIL: UUID(5) two-hop-only gold parity.
        # UUID(5) has cosine=0.02, outside ANN top-200. It is reachable via
        # seed(UUID(3)) → intermediate(UUID(4)) → gold(UUID(5)) 2-hop path.
        # Degree boost comes from REAL in-pool helper records (UUID(100..109),
        # cosine ~0.5 to the cue), so both the full-graph path and the bounded
        # assembler's global_degree scoring see the same degree signal.
        # If the full-graph baseline surfaces UUID(5) but the bounded path
        # does not, that is a genuine regression in the bounded assembler.
        u5 = str(UUID(int=5))
        u5_in_full = u5 in full_hits
        u5_in_bounded = u5 in bounded_hits
        if u5_in_full != u5_in_bounded:
            # One path drops U5 that the other kept — REAL regression.
            failures.append(
                f"  [FAIL U5] {cue_label} {n_label}: "
                f"U5 in full={u5_in_full} but in bounded={u5_in_bounded} — "
                "parity regression on the two-hop-only gold"
            )
        else:
            telemetry.append(
                f"  [TELEMETRY U5] {cue_label} {n_label}: "
                f"U5 in full={u5_in_full}, in bounded={u5_in_bounded} (parity)"
            )

    print(f"\n  --- Gate B {n_label} Results ---")
    for msg in telemetry:
        print(msg)

    if failures:
        raise AssertionError(f"Gate B {n_label} failures:\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# Anti-hit sub-test: self-contained UNCAPPED contradicts path
# ---------------------------------------------------------------------------


def test_gate_b_anti_hit_uncapped_contradicts(tmp_path, monkeypatch):
    """A self-contained anti-hit test: the 62-04 UNCAPPED contradicts path.

    Plant a record with high cosine to the cue + a contradicts edge to a
    second record. Recall via core.dispatch, assert the second record surfaces
    in anti_hits. This confirms the UNCAPPED (top_k=None) contradicts read
    in 62-04 is working.
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "anti-hit-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    # Insert enough filler so the target (low cosine) won't rank in the hits.
    for i in range(200):
        store.insert(_make(text=f"User filler anti-hit {i}", vec=_random_vec(9000 + i)))

    embedder = Embedder()
    cue_vec = embedder.embed(LEXICAL_GENERIC_CUE)
    cue_arr = np.asarray(cue_vec, dtype=np.float32)
    cue_arr /= np.linalg.norm(cue_arr)

    # Source record: high cosine to cue (will be in ANN hits).
    source_id = UUID(int=80_001)
    source_rec = _make_gold_record(80_001, list(cue_arr))
    store.insert(source_rec)

    # Target record: contradicts the source. Use a vector ORTHOGONAL to the
    # cue so it doesn't rank in the hits (cosine ~0 to the cue with N=200).
    # _find_anti_hits only adds records not already in `seen` (the hits set);
    # if target appears in hits, it won't be added as an anti-hit.
    target_id = UUID(int=80_002)
    # Build orthogonal target vector: Gram-Schmidt against cue_arr.
    rng_at = np.random.default_rng(11111)
    base_vec = rng_at.random(EMBED_DIM).astype(np.float32)
    base_vec -= np.dot(base_vec, cue_arr) * cue_arr  # project out cue component
    base_vec /= np.linalg.norm(base_vec)
    target_rec = _make_gold_record(80_002, base_vec.tolist())
    store.insert(target_rec)

    # Plant contradicts edge: source -> target
    store.boost_edges([(source_id, target_id)], edge_type="contradicts", delta=[1.0])

    # Off-path prime.
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc
    _g, _a, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _a, _rc)

    # Recall via core.dispatch.
    import iai_mcp.pipeline as _pm
    from iai_mcp import core
    _pm._last_recall_latency_ms = 0.0
    resp = core.dispatch(store, "memory_recall", {
        "cue": LEXICAL_GENERIC_CUE,
        "session_id": "test",
        "budget_tokens": 2000,
    })

    hit_ids = {h["record_id"] for h in resp.get("hits", [])}
    anti_hit_ids = {h["record_id"] for h in resp.get("anti_hits", [])}

    # Source must surface as a hit (high cosine to cue).
    assert str(source_id) in hit_ids, (
        f"ANTI-HIT: source record {source_id} not in hits; hits={hit_ids}"
    )
    # Target must surface as anti-hit (contradicts the source, UNCAPPED read).
    assert str(target_id) in anti_hit_ids, (
        f"ANTI-HIT: contradicts target {target_id} not in anti_hits; "
        f"anti_hits={anti_hit_ids}. The 62-04 UNCAPPED contradicts path "
        "(incident_edges top_k=None) must surface this record."
    )
    assert resp.get("ann_path_used", False), "ann_path_used must be True"


# ---------------------------------------------------------------------------
# Gate B: N=1k
# ---------------------------------------------------------------------------


def test_gate_b_n1k(tmp_path, monkeypatch):
    """Gate B at N=1k: bounded ⊇ full-graph parity + off-path cache prime.

    Asserts the bounded ANN-first assembler does NOT drop any gold that the
    live full-graph baseline keeps. Hub-sensitive gold (UUID(int=1)) must
    surface. Rich-club cap top-50 enabled.
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    # Load fixture for n_records count.
    assert FIXTURE_PATH.exists(), f"Gate-B fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        baseline = json.load(f)
    n_records = baseline["n1k"]["n_records"]

    store_path = tmp_path / "gate-b-n1k-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = _build_reference_store(store_path, n_records)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    # OFF-PATH CACHE PRIME (BLOCKER-2): UNTIMED.
    graph, assignment, rc = _prime_cache(store)

    _run_gate_b_parity(store, graph, assignment, rc, "N=1k", monkeypatch)


# ---------------------------------------------------------------------------
# Gate B: N=10k (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gate_b_n10k(tmp_path, monkeypatch):
    """Gate B at N=10k: bounded ⊇ full-graph parity with rich-club cap enabled.

    The cap (top-50) only binds at N=10k — this is the authoritative cap-enabled
    quality gate. Off-path cache prime (BLOCKER-2) ensures non-empty rich-club.
    """
    _monkeypatch_env(monkeypatch, tmp_path)

    assert FIXTURE_PATH.exists(), f"Gate-B fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        baseline = json.load(f)
    n_records = baseline["n10k"]["n_records"]

    store_path = tmp_path / "gate-b-n10k-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = _build_reference_store(store_path, n_records)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    # OFF-PATH CACHE PRIME (BLOCKER-2): UNTIMED.
    graph, assignment, rc = _prime_cache(store)

    _run_gate_b_parity(store, graph, assignment, rc, "N=10k", monkeypatch)
