"""S4 viability -- on-read consistency + monotropic proactive checks (, ).

 constitutional:
- (e) on-read consistency: runs inside `pipeline_recall` on top-K returned
  records. Pairwise cosine with ART vigilance ρ_s4=0.97 + `contradicts`
  edge lookup. Emits `s4_contradiction` events. Populates
  `RecallResponse.hints`.
- (f) monotropic proactive: only fires when profile.monotropism_depth[domain]
  > 0.7 AND new_record.detail_level >= 4. Scans within-domain only.
  Performance guard: if domain > 100 records, skip with warning event.

 addition:
- `run_offline_pass(store)` -- new entry point, CALLED by the daemon /
  session_exit hook. Currently runs `sigma.compute_and_emit(store)` only;
  future plans append more offline-pass items here. Failures emit
  `kind="s4_error"` and never crash the pass.

Explicitly forbidden ( negative assertions):
- NO `daily_scan` function (Ashby Requisite Variety violation).
- NO `session_exit_sweep` function (Anderson activation-based violation).

All detected contradictions go through `events.write_event` -- no .jsonl files
.
"""
from __future__ import annotations

from uuid import UUID

import numpy as np

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryHit, MemoryRecord


# (e) vigilance: 0.97 for near-duplicate contradiction detection.
# Stricter than write-path ρ=0.95: we only flag VERY close matches.
S4_VIGILANCE_RHO = 0.97

# (f) performance guard: skip when domain has > this many records.
MONOTROPIC_MAX_PAIRWISE = 100

# (f) monotropism-depth threshold.
S4_MONOTROPIC_THETA = 0.7


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 on zero-norm inputs."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def on_read_check(
    store: MemoryStore,
    hits: list[MemoryHit],
    session_id: str,
) -> list[dict]:
    """(e) on-read consistency check.

    Two detection paths, both run per-retrieval on the top-K hits:

    1. `contradicts`-edge authoritative: any pair of hits connected by an
       existing `contradicts` edge is flagged regardless of cosine. This is
       the definitive route -- the user (or a prior S4 run) already said
       "these two disagree", so we surface it every time they co-retrieve.

    2. Cosine + tag-polarity heuristic: pairs with cosine >= ρ_s4 (0.97) AND
       conflicting polarity tags ({positive,negative} or {asserted,retracted})
       are flagged as `info`-severity. or can replace this
       with NLI-based semantic contradiction.

    Returns a list of hint dicts; each dict is shaped per
    RecallResponse.hints contract. Also writes one `s4_contradiction` event
    per detected pair to the LanceDB events table .

    note: `on_read_check_batch` is the variant. It accepts
    an optional `records_cache` kwarg so pipeline_recall can reuse the cache
    it already built at stage 1 (zero extra store.get calls). This function
    is preserved as the back-compat / ad-hoc caller API (retrieve.recall
    still calls it; no records_cache available there).
    """
    if len(hits) < 2:
        return []

    hint_list: list[dict] = []

    # Load records for the hit ids. Missing records are skipped silently -- a
    # recent store.delete could race us.
    records: dict[UUID, MemoryRecord] = {}
    for h in hits:
        rec = store.get(h.record_id)
        if rec is not None:
            records[h.record_id] = rec
    if len(records) < 2:
        return []

    # Load contradicts edges among these records. We precompute the set of
    # (sorted src,dst) pairs so the pairwise loop below is O(1) lookup.
    contradict_pairs: set[tuple[str, str]] = set()
    try:
        edges_df = store.db.open_table("edges").to_pandas()
    except Exception:
        edges_df = None
    if edges_df is not None and not edges_df.empty:
        contradict_df = edges_df[edges_df["edge_type"] == "contradicts"]
        hit_ids = {str(h.record_id) for h in hits}
        for _, row in contradict_df.iterrows():
            src = row["src"]
            dst = row["dst"]
            if src in hit_ids and dst in hit_ids:
                contradict_pairs.add(tuple(sorted([src, dst])))

    # Pairwise scan across hit records.
    hit_records = list(records.values())
    for i in range(len(hit_records)):
        for j in range(i + 1, len(hit_records)):
            a = hit_records[i]
            b = hit_records[j]
            key = tuple(sorted([str(a.id), str(b.id)]))
            sim = _cosine(a.embedding, b.embedding)

            # Path 1: explicit edge is authoritative.
            if key in contradict_pairs:
                hint = {
                    "kind": "s4_contradiction",
                    "severity": "warning",
                    "source_ids": [str(a.id), str(b.id)],
                    "text": (
                        f"inconsistency: records have a contradicts edge; "
                        f"review {a.id}, {b.id}"
                    ),
                    "similarity": sim,
                }
                hint_list.append(hint)
                write_event(
                    store,
                    kind="s4_contradiction",
                    data={
                        "source_ids": list(key),
                        "similarity": sim,
                        "mechanism": "contradicts_edge",
                    },
                    severity="warning",
                    session_id=session_id,
                    source_ids=[a.id, b.id],
                )
                continue

            # Path 2: cosine + polarity-tag heuristic.
            if sim >= S4_VIGILANCE_RHO:
                a_tags = set(a.tags or [])
                b_tags = set(b.tags or [])
                polarity_conflict = (
                    ("positive" in a_tags and "negative" in b_tags)
                    or ("negative" in a_tags and "positive" in b_tags)
                    or ("asserted" in a_tags and "retracted" in b_tags)
                    or ("retracted" in a_tags and "asserted" in b_tags)
                )
                if polarity_conflict:
                    hint = {
                        "kind": "s4_contradiction",
                        "severity": "info",
                        "source_ids": [str(a.id), str(b.id)],
                        "text": (
                            f"inconsistency: near-duplicate ({sim:.3f}) with "
                            f"conflicting polarity tags"
                        ),
                        "similarity": sim,
                    }
                    hint_list.append(hint)
                    write_event(
                        store,
                        kind="s4_contradiction",
                        data={
                            "source_ids": list(key),
                            "similarity": sim,
                            "mechanism": "tag_polarity",
                        },
                        severity="info",
                        session_id=session_id,
                        source_ids=[a.id, b.id],
                    )
    return hint_list


def on_read_check_batch(
    store: MemoryStore,
    hits: list[MemoryHit],
    session_id: str,
    records_cache: "dict[UUID, MemoryRecord] | None" = None,
) -> list[dict]:
    """: batched variant of on_read_check.

    Semantically identical to on_read_check (returns the same hint-shape list,
    emits the same events). The ONLY difference is the record-loading step:

    - If `records_cache` is provided, use it directly. ZERO store.get calls.
    - Otherwise, do ONE `store.all_records()` call instead of N `store.get()`
      calls. ZERO per-hit round-trips either way.

    The pairwise contradiction-detection loop, the polarity-tag heuristic, the
    vigilance threshold (S4_VIGILANCE_RHO), and the event-emission logic are
    byte-for-byte equivalent to on_read_check.

    Why this is the perf-critical surface ( SC-6):
    Pre-fix: pipeline_recall built records_cache at stage 1, then s4.on_read_check
             called `store.get(h.record_id)` per hit -- every call is a full
             to_pandas() scan (~140ms each at N=100 on executor hardware).
    Post-fix: pipeline_recall passes records_cache through; s4 does zero extra
             round-trips. Saves ~140ms per hit x N hits per recall.
    """
    if len(hits) < 2:
        return []

    hint_list: list[dict] = []

    # Load records via cache (preferred) or one batched fallback.
    records: dict[UUID, MemoryRecord] = {}
    if records_cache is not None:
        for h in hits:
            rec = records_cache.get(h.record_id)
            if rec is not None:
                records[h.record_id] = rec
    else:
        all_recs = store.all_records()
        by_id = {r.id: r for r in all_recs}
        for h in hits:
            rec = by_id.get(h.record_id)
            if rec is not None:
                records[h.record_id] = rec
    if len(records) < 2:
        return []

    # Load contradicts edges among these records. One edges.to_pandas() scan
    # (same as on_read_check).
    contradict_pairs: set[tuple[str, str]] = set()
    try:
        edges_df = store.db.open_table("edges").to_pandas()
    except Exception:
        edges_df = None
    if edges_df is not None and not edges_df.empty:
        contradict_df = edges_df[edges_df["edge_type"] == "contradicts"]
        hit_ids = {str(h.record_id) for h in hits}
        for _, row in contradict_df.iterrows():
            src = row["src"]
            dst = row["dst"]
            if src in hit_ids and dst in hit_ids:
                contradict_pairs.add(tuple(sorted([src, dst])))

    # Pairwise scan -- identical logic to on_read_check.
    hit_records = list(records.values())
    for i in range(len(hit_records)):
        for j in range(i + 1, len(hit_records)):
            a = hit_records[i]
            b = hit_records[j]
            key = tuple(sorted([str(a.id), str(b.id)]))
            sim = _cosine(a.embedding, b.embedding)

            # Path 1: explicit edge is authoritative.
            if key in contradict_pairs:
                hint = {
                    "kind": "s4_contradiction",
                    "severity": "warning",
                    "source_ids": [str(a.id), str(b.id)],
                    "text": (
                        f"inconsistency: records have a contradicts edge; "
                        f"review {a.id}, {b.id}"
                    ),
                    "similarity": sim,
                }
                hint_list.append(hint)
                write_event(
                    store,
                    kind="s4_contradiction",
                    data={
                        "source_ids": list(key),
                        "similarity": sim,
                        "mechanism": "contradicts_edge",
                    },
                    severity="warning",
                    session_id=session_id,
                    source_ids=[a.id, b.id],
                )
                continue

            # Path 2: cosine + polarity-tag heuristic.
            if sim >= S4_VIGILANCE_RHO:
                a_tags = set(a.tags or [])
                b_tags = set(b.tags or [])
                polarity_conflict = (
                    ("positive" in a_tags and "negative" in b_tags)
                    or ("negative" in a_tags and "positive" in b_tags)
                    or ("asserted" in a_tags and "retracted" in b_tags)
                    or ("retracted" in a_tags and "asserted" in b_tags)
                )
                if polarity_conflict:
                    hint = {
                        "kind": "s4_contradiction",
                        "severity": "info",
                        "source_ids": [str(a.id), str(b.id)],
                        "text": (
                            f"inconsistency: near-duplicate ({sim:.3f}) with "
                            f"conflicting polarity tags"
                        ),
                        "similarity": sim,
                    }
                    hint_list.append(hint)
                    write_event(
                        store,
                        kind="s4_contradiction",
                        data={
                            "source_ids": list(key),
                            "similarity": sim,
                            "mechanism": "tag_polarity",
                        },
                        severity="info",
                        session_id=session_id,
                        source_ids=[a.id, b.id],
                    )
    return hint_list


def monotropic_proactive_check(
    store: MemoryStore,
    new_record: MemoryRecord,
    profile_state: dict,
    session_id: str,
) -> list[dict]:
    """(f) monotropic proactive check.

    Three gates (all must pass):

    1. `profile_state["monotropism_depth"][domain] > θ_deep` (0.7). The user's
       autistic profile indicates DEEP focus in this domain -- we're willing
       to spend cycles checking for near-duplicates.
    2. `new_record.detail_level >= 4`. Shallow records (detail 1-3) don't
       warrant the pairwise scan.
    3. `new_record` carries a `domain:<name>` tag. Records without a domain
       tag are excluded (nothing to compare against).

    Performance guard: if the domain has > MONOTROPIC_MAX_PAIRWISE records,
    skip the scan and emit a `s4_monotropic_skip` warning event. The scan is
    O(N) cosine comparisons; 100 is a reasonable ceiling.

    Rule 1 deviation: if `profile_state["monotropism_depth"]` is not a dict
    (type drift), degrade silently to empty hints (no exception).
    """
    md = profile_state.get("monotropism_depth", {})
    if not isinstance(md, dict):
        return []  # profile_state wrongly typed -- degrade silently

    # Locate the record's domain tag ("domain:coding", "domain:gardening", ...)
    domain_tag: str | None = next(
        (t for t in (new_record.tags or []) if t.startswith("domain:")),
        None,
    )
    if domain_tag is None:
        return []

    # Gate 1: monotropism depth must exceed θ_deep.
    domain_name = domain_tag.split(":", 1)[1]
    depth = md.get(domain_name, 0.0)
    if depth <= S4_MONOTROPIC_THETA:
        return []

    # Gate 2: detail_level must be >= 4.
    if new_record.detail_level < 4:
        return []

    # Load same-domain records (excluding the new record itself).
    same_domain = [
        r for r in store.all_records()
        if (r.tags or []) and domain_tag in r.tags and r.id != new_record.id
    ]

    # Performance guard: skip + warn above ceiling.
    if len(same_domain) > MONOTROPIC_MAX_PAIRWISE:
        write_event(
            store,
            kind="s4_monotropic_skip",
            data={
                "domain": domain_tag,
                "count": len(same_domain),
                "record_id": str(new_record.id),
            },
            severity="warning",
            domain=domain_tag,
            session_id=session_id,
        )
        return []

    hints: list[dict] = []
    for r in same_domain:
        sim = _cosine(new_record.embedding, r.embedding)
        if sim >= S4_VIGILANCE_RHO:
            hint = {
                "kind": "s4_monotropic_contradiction",
                "severity": "info",
                "source_ids": [str(new_record.id), str(r.id)],
                "text": (
                    f"monotropic near-duplicate in {domain_tag}: sim={sim:.3f}"
                ),
                "similarity": sim,
            }
            hints.append(hint)
            write_event(
                store,
                kind="s4_monotropic_contradiction",
                data={
                    "domain": domain_tag,
                    "source_ids": [str(new_record.id), str(r.id)],
                    "similarity": sim,
                },
                severity="info",
                domain=domain_tag,
                session_id=session_id,
                source_ids=[new_record.id, r.id],
            )
    return hints


def run_offline_pass(store: MemoryStore) -> dict:
    """: S4 offline-pass entry point.

    Called by the daemon's offline cycle (or by session_exit / cron).
    Currently runs ONE check: `sigma.compute_and_emit(store)` -- which writes
    `kind=sigma_observation` (developmental / healthy / insufficient_data) OR
    `kind=sigma_drift` (mid_life_drift) and (in developmental phase) bumps the
    Hebbian rate via a `profile_updated` event.

    Failures are caught and emitted as `kind="s4_error"`; the pass does NOT
    crash. This mirrors the diagnostic discipline of `on_read_check`:
    S4 work is observation, never blocks reads or writes.

    Returns a dict with the per-step outcome:
      {"sigma": <snapshot dict or {"error": "..."}>}
    """
    from iai_mcp import sigma  # local import; sigma is heavy (networkx)

    out: dict = {}
    try:
        out["sigma"] = sigma.compute_and_emit(store)
    except Exception as exc:  # noqa: BLE001 - diagnostic catch-all
        try:
            write_event(
                store,
                kind="s4_error",
                data={"step": "sigma", "error": repr(exc)},
                severity="warning",
            )
        except Exception:
            pass
        out["sigma"] = {"error": repr(exc)}
    return out
