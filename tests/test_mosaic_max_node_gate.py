"""Max-node cosine community gate.

Centroid-cosine is structurally brittle in high-dim space (Fortunato 2010
*Phys Reports* DOI:10.1016/j.physrep.2009.11.002, Mucha 2010 *Science*
DOI:10.1126/science.1184819). Max-node cosine evaluates a community by
its strongest constituent member, mathematically immune to centroid drift
and partition fragmentation.

These tests pin the gate contract on `pipeline._community_gate`:

1. **drift witness** — fixture where centroid-cosine picks community B
   but the correct community is A (which holds a single perfect-match
   member diluted by 10 orthogonal siblings). Max-node MUST pick A.
2. **fragmentation robustness** — 1-record-per-community geometry; the
   cos-1.0 record's community MUST be in top-3 regardless of how many
   sibling communities are emitted by Leiden.
3. **backwards-compat** — `member_embeddings=None` falls back to the
   centroid path bit-for-bit; existing `_community_gate` callers that
   don't yet pass the kwarg stay green.
4. **determinism** — same cue + same assignment + same member_embeddings
   over 5 calls -> identical UUID order; tie-break by lexical UUID str.
5. **perf gate** — N=5000 members across 100 communities, mean wall time
   < 5 ms (vectorized matmul budget; max-node adds one
   `np.maximum.reduceat` over the per-member scores).
"""
from __future__ import annotations

import time
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment, _compute_centroid
from iai_mcp.pipeline import _community_gate
from iai_mcp.types import EMBED_DIM


# --------------------------------------------------------------- fixtures ---


def _unit_axis(i: int, dim: int = EMBED_DIM) -> list[float]:
    """Return the i-th unit basis vector in `dim`-d space."""
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _build_drift_fixture() -> tuple[
    CommunityAssignment, dict[UUID, list[float]], list[float], UUID, UUID
]:
    """B* primary witness: centroid-cosine picks the WRONG community.

    Community A:
      - 1 member whose embedding == cue (cos = 1.0)
      - 10 members orthogonal to the cue (cos ~ 0)
      => centroid(A) is the mean of those 11 vectors. The dominant axis
         is the cue's axis (1.0 component) plus the 10 orthogonal axes
         (1.0 each on their own dimensions). After mean+normalize the
         cue-axis weight is ~1/11 of the unit-norm centroid, giving
         centroid(A). cue ~ 1/sqrt(11) ~ 0.30.

    Community B:
      - 11 members ALL on axis 1, each at cosine 0.5 to the cue.
        (cue is at axis 0; B members are 0.5*axis_0 + 0.5*sqrt(3)*axis_1
        normalized, so cos(B_member, cue) = 0.5 exactly.)
      => centroid(B) = the mean of 11 identical vectors = the vector
         itself. centroid(B). cue = 0.5 exactly.

    Therefore centroid-cosine: B (0.5) > A (~0.30). Picks B first.

    Max-node-cosine:
      - max over A members: 1.0 (the perfect-match member)
      - max over B members: 0.5
    Picks A first.

    Returns (assignment, member_embeddings, cue, comm_A_id, comm_B_id).
    """
    dim = EMBED_DIM
    # Cue = axis 0 unit vector.
    cue = _unit_axis(0, dim)

    # Community A members.
    a_perfect = _unit_axis(0, dim)  # cos to cue = 1.0
    a_orthogonals = [_unit_axis(2 + i, dim) for i in range(10)]  # axes 2..11

    # Community B members: all the same vector (0.5*axis_0 + sqrt(3)/2*axis_1).
    # cos(B_member, cue) = 0.5; centroid(B) = that vector; centroid(B).cue=0.5.
    b_template = [0.0] * dim
    b_template[0] = 0.5
    b_template[1] = (3.0 ** 0.5) / 2.0  # sqrt(3)/2
    b_members = [list(b_template) for _ in range(11)]

    # Allocate UUIDs.
    a_member_ids = [uuid4() for _ in range(11)]
    b_member_ids = [uuid4() for _ in range(11)]
    comm_A = uuid4()
    comm_B = uuid4()

    # member_embeddings: id -> vector
    member_embeddings: dict[UUID, list[float]] = {}
    for mid, vec in zip(a_member_ids, [a_perfect, *a_orthogonals]):
        member_embeddings[mid] = vec
    for mid, vec in zip(b_member_ids, b_members):
        member_embeddings[mid] = vec

    # Centroids — use the same helper the assignment dataclass uses so we
    # match the actual production geometry.
    centroid_A = _compute_centroid([a_perfect, *a_orthogonals])
    centroid_B = _compute_centroid(b_members)

    mid_regions = {
        comm_A: a_member_ids,
        comm_B: b_member_ids,
    }
    node_to_community = {}
    for mid in a_member_ids:
        node_to_community[mid] = comm_A
    for mid in b_member_ids:
        node_to_community[mid] = comm_B

    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids={comm_A: centroid_A, comm_B: centroid_B},
        modularity=0.0,
        backend="leiden-test-drift-witness",
        top_communities=[comm_A, comm_B],
        mid_regions=mid_regions,
    )
    return assignment, member_embeddings, cue, comm_A, comm_B


def _build_one_per_community(
    n: int,
) -> tuple[CommunityAssignment, dict[UUID, list[float]]]:
    """1-record-per-community geometry on distinct primary axes (orthogonal).

    Mirrors `test_recall_community_gate_diagnostic._build_one_record_per_community`
    but skips the store/graph boilerplate — `_community_gate` reads only
    `assignment` + (optionally) `member_embeddings`.
    """
    dim = EMBED_DIM
    member_ids = [uuid4() for _ in range(n)]
    comm_ids = [uuid4() for _ in range(n)]
    node_to_community: dict[UUID, UUID] = {}
    centroids: dict[UUID, list[float]] = {}
    mid_regions: dict[UUID, list[UUID]] = {}
    member_embeddings: dict[UUID, list[float]] = {}
    for i in range(n):
        vec = _unit_axis(i, dim)
        node_to_community[member_ids[i]] = comm_ids[i]
        centroids[comm_ids[i]] = list(vec)  # single-member centroid == member
        mid_regions[comm_ids[i]] = [member_ids[i]]
        member_embeddings[member_ids[i]] = vec
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test-fragment",
        top_communities=comm_ids[:3],
        mid_regions=mid_regions,
    )
    return assignment, member_embeddings


# --------------------------------------------------------------- tests ----


def test_max_node_gate_finds_correct_community_when_centroid_drifts():
    """B* primary witness: max-node picks A; centroid would have picked B.

    Fortunato 2010 + Mucha 2010 in NLM deep research:
    centroid-cosine fails when a community contains a strong specific
    member diluted by orthogonal siblings (the high-dim hubness problem).
    Max-node-cosine evaluates the community by its strongest responder
    and recovers the correct ranking.
    """
    assignment, member_embeddings, cue, comm_A, comm_B = _build_drift_fixture()

    # --- sanity: confirm the centroid path would pick B (the wrong one)
    # so this test is genuinely a drift witness, not a no-op.
    centroid_order = _community_gate(cue, assignment, top_n=2)
    assert centroid_order[0] == comm_B, (
        "Drift-fixture invariant violated: centroid-cosine should pick "
        f"comm_B first (centroid(B).cue=0.5, centroid(A).cue~0.30). "
        f"Got centroid_order[0]={centroid_order[0]}. Recompute fixture."
    )

    # --- B* assertion: max-node path picks A.
    max_node_order = _community_gate(
        cue, assignment, top_n=2, member_embeddings=member_embeddings,
    )
    assert max_node_order[0] == comm_A, (
        "B* contract violated: max-node-cosine MUST pick comm_A first "
        "(max(A members . cue) = 1.0 vs max(B members . cue) = 0.5). "
        f"Got max_node_order={max_node_order}. "
        "If this fails, _community_gate did NOT switch to max-node when "
        "member_embeddings was passed."
    )


def test_max_node_gate_robust_to_fragmentation():
    """Max-node ranks the cos-1.0 community in top-3 regardless of partition size.

    NLM deep research F2: "Whether custom_leiden fragments the graph into
    5000 communities or groups it into 5, the correct record's community
    will always pass the gate."

    With 50 single-member communities on orthogonal axes and a cue at
    axis 5, the cos-1.0 community (containing the axis-5 member) MUST be
    in the top-3 returned communities.
    """
    n = 50
    assignment, member_embeddings = _build_one_per_community(n)
    # Identify the community containing the cos-1.0 record (axis 5).
    target_member: UUID | None = None
    for mid, emb in member_embeddings.items():
        if emb[5] == 1.0 and all(
            emb[i] == 0.0 for i in range(EMBED_DIM) if i != 5
        ):
            target_member = mid
            break
    assert target_member is not None, "fixture lookup failed"
    target_comm = assignment.node_to_community[target_member]

    cue = _unit_axis(5)
    top3 = _community_gate(
        cue, assignment, top_n=3, member_embeddings=member_embeddings,
    )
    assert target_comm in top3, (
        "Fragmentation-robustness violated: the community holding the "
        "cos-1.0 record is NOT in top-3. With 1-record-per-community "
        "geometry on orthogonal axes, max-node-cosine MUST pick the "
        "axis-aligned community first. "
        f"Got top3={top3}, target_comm={target_comm}."
    )
    # Stronger property: it's the TOP-1 (cos 1.0 dwarfs the others which
    # all tie at 0.0).
    assert top3[0] == target_comm, (
        f"Max-node top-1 should be the cos-1.0 community; got {top3[0]}."
    )


def test_max_node_gate_backwards_compat_without_member_embeddings():
    """Omitting `member_embeddings` reproduces the centroid-cosine path.

    Critical for the existing diagnostic tests in
    `test_recall_community_gate_diagnostic.py` that construct
    `CommunityAssignment` directly without a records_cache.
    """
    n = 50
    assignment, _ = _build_one_per_community(n)
    cue = _unit_axis(5)

    out_no_kwarg = _community_gate(cue, assignment, top_n=5)
    out_explicit_none = _community_gate(
        cue, assignment, top_n=5, member_embeddings=None,
    )
    assert out_no_kwarg == out_explicit_none, (
        "Backwards-compat broken: omitting member_embeddings vs passing "
        "None must yield bit-identical results.\n"
        f"omit:     {out_no_kwarg}\n"
        f"explicit: {out_explicit_none}"
    )
    # And the centroid path should still return SOMETHING (5 UUIDs).
    assert len(out_no_kwarg) == 5


def test_max_node_gate_determinism():
    """Same inputs over 5 calls -> identical UUID order.

    Tie-break must be stable lexical UUID string (matches the existing
    centroid path's stable-sort by (-score, UUID-str)).
    """
    n = 30
    assignment, member_embeddings = _build_one_per_community(n)
    # Cue equidistant to many communities so ties exercise tie-break.
    cue = _unit_axis(7)

    runs = [
        _community_gate(
            cue, assignment, top_n=5, member_embeddings=member_embeddings,
        )
        for _ in range(5)
    ]
    first = runs[0]
    for i, r in enumerate(runs[1:], start=1):
        assert r == first, (
            f"Determinism violated at run {i}: {r} != run0 {first}"
        )

    # Tie-break property: among the post-top-1 communities (all cosine 0.0
    # to the orthogonal cue), the order is by ascending UUID str.
    # The top-1 is the axis-7 community (cos 1.0); ranks 2..5 are
    # cos-0.0 communities sorted by stable secondary criterion.
    tied = first[1:]  # exclude the unambiguous top-1
    tied_strs = [str(u) for u in tied]
    assert tied_strs == sorted(tied_strs), (
        "Stable tie-break must be ascending UUID-str within a tied score "
        f"bucket. Got: {tied_strs}, expected: {sorted(tied_strs)}"
    )


def test_max_node_gate_perf_under_5ms_at_n5000():
    """N=5000 members across 100 communities: mean wall time < 5 ms.

    Conservative budget. vectorized centroid-cosine target was
    0.1 ms; max-node adds one matmul over member-stack (5000x384) plus
    a per-community max. On a modern laptop this is <1 ms; we assert
    5 ms to absorb cold-cache / CI noise.

     hard constraint: wall-time impact <= 5x centroid path.
    """
    rng = np.random.default_rng(seed=42)
    dim = EMBED_DIM
    n_communities = 100
    members_per_community = 50  # total = 5000
    member_ids: list[UUID] = []
    # Production caller (`_recall_core`) builds the dict with ndarray
    # values (one cast per record at records_cache build); the gate
    # then runs `np.stack` for sub-millisecond stacking. List-of-list
    # values trigger a 30x-slower per-float cast on np.asarray. This
    # perf test exercises the production-realistic shape.
    member_embeddings: dict[UUID, np.ndarray] = {}
    mid_regions: dict[UUID, list[UUID]] = {}
    centroids: dict[UUID, list[float]] = {}
    node_to_community: dict[UUID, UUID] = {}
    top_communities: list[UUID] = []
    for c_idx in range(n_communities):
        comm_id = uuid4()
        top_communities.append(comm_id)
        mid_regions[comm_id] = []
        member_vecs: list[list[float]] = []
        for _ in range(members_per_community):
            v = rng.standard_normal(dim).astype(np.float32)
            n_v = float(np.linalg.norm(v))
            if n_v > 0:
                v = v / n_v
            else:
                v = np.zeros(dim, dtype=np.float32)
            m_id = uuid4()
            member_ids.append(m_id)
            mid_regions[comm_id].append(m_id)
            node_to_community[m_id] = comm_id
            member_embeddings[m_id] = v  # ndarray, like production
            member_vecs.append(v.tolist())
        centroids[comm_id] = _compute_centroid(member_vecs)
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test-perf",
        top_communities=top_communities[:7],
        mid_regions=mid_regions,
    )
    cue = rng.standard_normal(dim).astype(np.float32)
    cn = float(np.linalg.norm(cue))
    cue = (cue / cn).tolist() if cn > 0 else cue.tolist()

    # Warm-up call (allocate buffers; first call is always slowest).
    _community_gate(
        cue, assignment, top_n=3, member_embeddings=member_embeddings,
    )

    # Measure 20 runs.
    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _community_gate(
            cue, assignment, top_n=3, member_embeddings=member_embeddings,
        )
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / n_runs) * 1000.0
    assert mean_ms < 5.0, (
        f"Max-node gate perf regression: mean wall time {mean_ms:.3f} ms "
        f"exceeds 5.0 ms budget at N=5000 members over 100 communities. "
        f"Total {n_runs} runs took {elapsed*1000:.1f} ms."
    )
