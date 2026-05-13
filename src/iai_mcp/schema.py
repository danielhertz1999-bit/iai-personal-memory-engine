"""Schema induction (LEARN-03, , ) -- Task 3.

 (scheduling): dual-path schema surfacing.
- Primary: batch induction inside the heavy sleep cycle. Tier-1 Haiku
  extraction when `should_call_llm` permits, Tier-0 cooccurrence + TF-IDF
  fallback otherwise.
- Secondary: entropy-gated provisional schemas surfaced during
  `pipeline_recall` when score distribution entropy > 0.8 bits AND the
  cohesive community has >= 2 shared tags.

 (thresholds, autism-aware):
- Auto-induct when co_occurrence >= 5 AND confidence >= 0.85.
- User-approval flag at co_occurrence in [3, 5) AND confidence in [0.65, 0.85).
- Below: discard.
- Exceptions preserved as first-class records (never absorbed).
- Abstraction level: concrete (Dawson-Mottron Raven's preference).

Schema records are first-class hubs:
- tier="semantic", detail_level=3 -> never_decay=True.
- schema_instance_of edges from evidence -> schema never decay.
- pipeline routing can prioritise schema records when pattern
  matches.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID, uuid4

from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT


# ---------------------------------------------------------------- constants

AUTO_INDUCT_COOCCURRENCE: int = 5
AUTO_INDUCT_CONFIDENCE: float = 0.85
USER_APPROVAL_COOCCURRENCE: int = 3
USER_APPROVAL_CONFIDENCE: float = 0.65
MAX_EVIDENCE_PER_SCHEMA: int = 50
PROVISIONAL_ENTROPY_MIN: float = 0.8


# ---------------------------------------------------------------- candidate


@dataclass
class SchemaCandidate:
    """One schema candidate surfaced by induce_schemas_*."""

    pattern: str
    confidence: float
    evidence_count: int
    evidence_ids: list[UUID] = field(default_factory=list)
    domain: str | None = None
    exceptions: list[UUID] = field(default_factory=list)
    status: str = "auto"   # "auto" | "pending_user_approval"


# ---------------------------------------------------------------- Tier-0 induction


def _tag_cooccurrence(records: Iterable) -> dict:
    """Bucket records by tag-pair frequency. Returns {frozenset(pair): [record_ids]}.

    -04 -A: accepts either ``list[MemoryRecord]`` (back-compat;
    used by external callers passing dataclass instances) or an iterable of
    projected ``dict`` rows from ``store.iter_record_columns(["id", "tags_json"])``.

    Dispatch is duck-typed: items with a ``.tags`` attribute are treated as
    MemoryRecord; items without are treated as dict rows. This keeps both
    surfaces alive while migrating the production path off ``all_records()``.

    For dict rows, ``tags_json`` is parsed defensively (mirrors the W3
    pattern in ``sleep._tier0_schema_surfacing`` — corrupted rows contribute
    zero counts but do not crash). The ``id`` field arrives as a string from
    LanceDB and is converted to ``UUID`` here so callers always see
    ``list[UUID]`` evidence_ids regardless of which input shape was passed.
    """
    pairs: dict = {}
    for r in records:
        # Dispatch on duck-typing: MemoryRecord has .tags + .id attributes;
        # dict rows have ["tags_json"] + ["id"] keys.
        if hasattr(r, "tags"):
            # MemoryRecord path (back-compat for external/test callers).
            raw_tags = r.tags or []
            rid = r.id
        else:
            # Dict-row path (-A migrated production path). Defensive parse:
            # malformed tags_json contributes zero pairs but does not raise.
            tags_raw = r.get("tags_json") or "[]"
            try:
                raw_tags = json.loads(tags_raw) if tags_raw else []
            except (TypeError, json.JSONDecodeError):
                raw_tags = []
            id_raw = r.get("id")
            if id_raw is None:
                continue
            # iter_record_columns yields id as a string; convert to UUID at
            # the boundary so SchemaCandidate.evidence_ids stays list[UUID].
            try:
                rid = UUID(id_raw) if isinstance(id_raw, str) else id_raw
            except (ValueError, AttributeError):
                continue

        tags = [
            t for t in raw_tags
            if not t.startswith("raw:") and not t.startswith("domain:")
        ]
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                key = frozenset([tags[i], tags[j]])
                pairs.setdefault(key, []).append(rid)
    return pairs


def induce_schemas_tier0(store: MemoryStore) -> list[SchemaCandidate]:
    """ Tier-0 path: tag cooccurrence + TF-IDF; no LLM.

    Returns a list of SchemaCandidate. Each candidate passes the gate:
    - status="auto"               -> count >= 5 AND confidence >= 0.85
    - status="pending_user_approval" -> count in [3,5) AND confidence in [0.65, 0.85)

    -04 -A: streams via ``store.iter_record_columns(
    ["id", "tags_json"], batch_size=1024)`` instead of ``store.all_records()``.
    Encrypted columns (literal_surface, provenance_json,
    profile_modulation_gain_json) are NEVER read on this path; the W5 cipher
    cache is short-circuited entirely. On the 8105-record production store
    this saves ~16210 AES-GCM operations + ~14.5 MB literal_surface
    materialisation per ``run_heavy_consolidation`` invocation, and unblocks
    the W4 ≤1 ``all_records()`` invariant on the heavy cycle.

    Single-pass record-count tally: count_total is incremented inside the
    iterator loop and the ``< CLUSTER_MIN_SIZE`` floor is checked afterwards.
    Mirrors the pattern in ``sleep._tier0_schema_surfacing`` (W3).
    """
    rows = list(store.iter_record_columns(["id", "tags_json"], batch_size=1024))
    if len(rows) < 3:
        return []

    pair_counts = _tag_cooccurrence(rows)
    candidates: list[SchemaCandidate] = []
    for pair, evidence in pair_counts.items():
        count = len(evidence)
        # Heuristic confidence: saturates toward 1.0 at 10+ evidence records.
        confidence = min(1.0, count / 10.0)
        pattern = f"tags:{'+'.join(sorted(pair))}"
        if count >= AUTO_INDUCT_COOCCURRENCE and confidence >= AUTO_INDUCT_CONFIDENCE:
            status = "auto"
        elif (
            USER_APPROVAL_COOCCURRENCE <= count < AUTO_INDUCT_COOCCURRENCE
            and confidence >= USER_APPROVAL_CONFIDENCE
        ):
            status = "pending_user_approval"
        else:
            continue
        candidates.append(
            SchemaCandidate(
                pattern=pattern,
                confidence=confidence,
                evidence_count=count,
                evidence_ids=list(evidence[:MAX_EVIDENCE_PER_SCHEMA]),
                status=status,
            )
        )
    return candidates


# ---------------------------------------------------------------- Tier-1 w/ D-GUARD


def induce_schemas_tier1(
    store: MemoryStore,
    budget: BudgetLedger,
    rate: RateLimitLedger,
    llm_enabled: bool = True,
) -> list[SchemaCandidate]:
    """ Tier-1 path: Haiku extraction gated by D-GUARD ladder.

    When should_call_llm returns False (any ladder step), emit an
    llm_health event and delegate to `induce_schemas_tier0`.

    scope: the Tier-1 branch is reserved; wires the
    actual anthropic.batches.create call. This function's contract is: on
    allow, call budget.record_spend and emit llm_health; then fall back to
    tier0 (because real Batch output is a deliverable). The
    effective_tier in the event is "tier0" regardless until .
    """
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ok, reason = should_call_llm(
        budget=budget, rate=rate,
        llm_enabled=llm_enabled, has_api_key=has_key,
        estimated_usd=0.005,
    )
    if not ok:
        write_event(
            store,
            kind="llm_health",
            data={
                "component": "schema_induction",
                "tier": "fallback",
                "reason": reason,
            },
            severity="warning",
        )
        return induce_schemas_tier0(store)

    # Tier-1 eligible -- scaffold only (wires real Batch API).
    try:
        import anthropic  # noqa: F401 -- lazy import, raise-only if missing
        budget.record_spend(0.002, kind="schema_induction")
        write_event(
            store,
            kind="llm_health",
            data={
                "component": "schema_induction",
                "tier": "haiku",
                "note": "wires real Batch API; 02-03 scaffolds only",
            },
            severity="info",
        )
    except Exception as e:
        write_event(
            store,
            kind="llm_health",
            data={"component": "schema_induction", "error": str(e)},
            severity="critical",
        )
    return induce_schemas_tier0(store)


# ---------------------------------------------------------------- persist


def _majority_language(evidence_ids: list[UUID], store: MemoryStore) -> str:
    """Return the plurality ISO-639-1 language tag among evidence records.

    Constitutional fix: schema hubs must carry the
    language of their source evidence, not a hardcoded 'en'. A user whose
    records are Russian would otherwise get schemas tagged 'en' and fail
    their own language='ru' filter at retrieval.

    Algorithm:
        - Fetch each evidence record via store.get (skip missing/deleted ones).
        - Collect their language fields (skip empty/None).
        - Return max(set(langs), key=langs.count). Tie-break is deterministic
          given a stable input list order: max with key=list.count returns
          the first element from the set iteration whose count is the
          maximum, and Python's set iteration on strings follows insertion
          order in CPython >= 3.7 for the distinct-values pattern used here
          because we build the distinct set from a list iteration.
        - Fallback 'en' when evidence is empty or all records are missing.

    Tie-break policy: when two languages are tied, the one whose first
    occurrence appears EARLIEST in evidence_ids wins. Matches
    default 'en' when no signal is available (least-surprise).
    """
    langs: list[str] = []
    for eid in evidence_ids:
        rec = store.get(eid)
        if rec is None:
            continue
        if rec.language:
            langs.append(rec.language)
    if not langs:
        return "en"
    # Deterministic tie-break: iterate langs in order, pick the first whose
    # count is the max. max(set(langs), key=langs.count) is undefined for
    # set ordering, so we use a hand-rolled pass instead.
    best = langs[0]
    best_count = langs.count(best)
    seen: set[str] = {best}
    for lang in langs[1:]:
        if lang in seen:
            continue
        seen.add(lang)
        c = langs.count(lang)
        if c > best_count:
            best = lang
            best_count = c
    return best


def persist_schema(
    store: MemoryStore,
    candidate: SchemaCandidate,
) -> UUID:
    """Insert a schema record + schema_instance_of edges to evidence.

    Schema records carry:
    - tier="semantic", detail_level=3 (never_decay auto-true)
    - tags=["schema", <status>, f"pattern:{pattern}"]
    - s5_trust_score=0.5 (neutral prior; LEARN-06 may raise over time)
    - schema_version=2
    """
    from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
    from iai_mcp.embed import embedder_for_store

    summary = (
        f"Schema: {candidate.pattern} (confidence={candidate.confidence:.2f})"
    )

    # R1 ( + ): pattern dedup. Search for an existing
    # schema record carrying the tag `pattern:{candidate.pattern}` in the
    # semantic tier. If found, reinforce schema_instance_of edges from new
    # evidence onto the existing keeper, emit `schema_reinforced`, and
    # return the existing schema_id. If not found, fall through to the
    # original insert path. Closes the chain-induction bleed: every sleep
    # cycle would otherwise insert a fresh tier="semantic", never_decay
    # row for the same pattern (live store accumulated 7+ duplicates per
    # pattern with degree-bonus shouldering verbatim records out of hits[]).
    pattern_tag = f"pattern:{candidate.pattern}"
    # -04 -B: keeper scan migrated from store.all_records to
    # store.iter_record_columns(["id", "tier", "tags_json"], batch_size=1024).
    # Projection skips encrypted columns (literal_surface, provenance_json,
    # profile_modulation_gain_json) entirely — the W5 cipher cache is
    # short-circuited on this path. Early-exit (`break`) semantics preserved.
    # The matching row's id arrives as a string from LanceDB; we convert to
    # UUID at the boundary so downstream code sees the same type contract as
    # the pre- ``existing_keeper.id`` access pattern.
    existing_keeper_id: UUID | None = None
    try:
        for row in store.iter_record_columns(
            ["id", "tier", "tags_json"], batch_size=1024
        ):
            if row.get("tier") != "semantic":
                continue
            tags_raw = row.get("tags_json") or "[]"
            try:
                tags = json.loads(tags_raw) if tags_raw else []
            except (TypeError, json.JSONDecodeError):
                tags = []
            if pattern_tag in tags:
                id_raw = row.get("id")
                if id_raw is None:
                    continue
                try:
                    existing_keeper_id = (
                        UUID(id_raw) if isinstance(id_raw, str) else id_raw
                    )
                except (ValueError, AttributeError):
                    continue
                break
    except Exception:
        # Defensive: if the scan fails, fall through to the insert path so
        # we never silently lose a schema. Mirrors the diagnostic-write
        # contract used in pipeline.py provenance batching.
        existing_keeper_id = None

    if existing_keeper_id is not None:
        from iai_mcp.store import EDGES_TABLE

        # Reinforce schema_instance_of edges from each new evidence record
        # onto the existing keeper. Reuses the same delta formula as the
        # insert path (max(0.1, candidate.confidence)) for symmetry.
        delta = max(0.1, candidate.confidence)
        new_pairs = [(ev_id, existing_keeper_id) for ev_id in candidate.evidence_ids]
        if new_pairs:
            store.boost_edges(
                new_pairs,
                edge_type="schema_instance_of",
                delta=delta,
            )

        # Compute total_evidence after reinforcement: count
        # `schema_instance_of` edges incident on the keeper. Read via the
        # edges table to avoid trusting any in-memory cache.
        # Note: store.boost_edges canonicalises (src, dst) to a sorted
        # tuple, so the keeper appears in EITHER column depending on the
        # string ordering of the paired evidence UUID. OR-counting both
        # columns gives the true edge-incidence count (no double-count
        # since each edge row has the keeper in exactly one column).
        try:
            edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
            keeper_str = str(existing_keeper_id)
            total_evidence = int(
                ((edges_df["edge_type"] == "schema_instance_of")
                 & ((edges_df["dst"] == keeper_str)
                    | (edges_df["src"] == keeper_str))).sum()
            )
        except Exception:
            total_evidence = len(candidate.evidence_ids)

        write_event(
            store,
            kind="schema_reinforced",
            data={
                "schema_id": str(existing_keeper_id),
                "pattern": candidate.pattern,
                "evidence_added": len(candidate.evidence_ids),
                "total_evidence": total_evidence,
            },
            severity="info",
            source_ids=[existing_keeper_id, *candidate.evidence_ids[:5]],
        )
        return existing_keeper_id

    emb = embedder_for_store(store).embed(summary)
    now = datetime.now(timezone.utc)
    schema_id = uuid4()
    # fix: derive language from the plurality language
    # of the evidence records, not a hardcoded 'en'. Schema hubs for Russian /
    # Japanese / Arabic clusters now carry the correct ISO-639-1 tag so
    # language-filtered retrieval surfaces them as expected.
    derived_language = _majority_language(candidate.evidence_ids, store)
    schema_rec = MemoryRecord(
        id=schema_id,
        tier="semantic",
        literal_surface=summary,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.7,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=False,
        provenance=[
            {
                "ts": now.isoformat(),
                "cue": "schema_induction",
                "session_id": "system",
            }
        ],
        created_at=now,
        updated_at=now,
        tags=[
            "schema",
            candidate.status,
            f"pattern:{candidate.pattern}",
        ],
        language=derived_language,
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    enforce_language_tagged(schema_rec)
    schema_rec.aaak_index = generate_aaak_index(schema_rec)
    store.insert(schema_rec)

    # R3: batch the schema_instance_of edges into ONE boost_edges
    # call (one merge_insert + one tbl.add at most). Previously this loop
    # issued N Lance versions on edges.lance for an N-evidence schema.
    instance_pairs = [(ev_id, schema_id) for ev_id in candidate.evidence_ids]
    if instance_pairs:
        store.boost_edges(
            instance_pairs,
            edge_type="schema_instance_of",
            delta=max(0.1, candidate.confidence),
        )

    write_event(
        store,
        kind="schema_induction_run",
        data={
            "schema_id": str(schema_id),
            "pattern": candidate.pattern,
            "confidence": candidate.confidence,
            "evidence_count": candidate.evidence_count,
            "status": candidate.status,
        },
        severity="info",
        source_ids=[schema_id, *candidate.evidence_ids[:5]],
    )
    return schema_id


# ---------------------------------------------------------------- provisional


def provisional_schemas_for_recall(
    store: MemoryStore,
    hits: list,
    entropy_bits: float,
    records_cache: "dict | None" = None,
) -> list[dict]:
    """ secondary path: surface provisional schema hints on high-entropy recalls.

    Returns a list of hint dicts compatible with RecallResponse.hints, one per
    cohesive tag appearing in >= 2 of the top hits.

    perf: batched all_records() fetch replaces N+1 store.get()
    calls. A single to_pandas() call is still O(total_records) but constant
    per recall, not per-hit. This was a major bottleneck at N=50.

    perf (Rule 1 auto-fix): accept optional `records_cache` so
    pipeline_recall can pass its already-built cache through -- avoids a
    second `store.all_records()` scan per recall (~40ms at N=100). Falls
    back to all_records() if no cache provided (preserves back-compat for
    ad-hoc callers; tests without pipeline_recall still work).
    """
    if entropy_bits < PROVISIONAL_ENTROPY_MIN or len(hits) < 3:
        return []

    # Batch-fetch all records once; hits are typically <=5 so the cost of
    # filtering in-memory dominates over 5 separate store.get() round-trips.
    hit_ids = {h.record_id for h in hits}
    if records_cache is not None:
        # Reuse the cache built at pipeline_recall stage 1. Zero scans.
        by_id = {
            rid: rec for rid, rec in records_cache.items() if rid in hit_ids
        }
    else:
        try:
            all_recs = store.all_records()
        except Exception:
            return []
        by_id = {r.id: r for r in all_recs if r.id in hit_ids}

    tag_count: Counter = Counter()
    for h in hits:
        rec = by_id.get(h.record_id)
        if rec is None:
            continue
        for t in (rec.tags or []):
            if t.startswith("raw:") or t.startswith("domain:"):
                continue
            tag_count[t] += 1

    provisional: list[dict] = []
    for tag, cnt in tag_count.most_common(3):
        if cnt >= 2:
            source_ids: list[str] = []
            for h in hits:
                rec = by_id.get(h.record_id)
                if rec is None:
                    continue
                if tag in (rec.tags or []):
                    source_ids.append(str(h.record_id))
                if len(source_ids) >= 5:
                    break
            provisional.append(
                {
                    "kind": "provisional_schema",
                    "severity": "info",
                    "source_ids": source_ids,
                    "text": f"Potential schema: tag={tag} cnt={cnt}",
                    "provisional": True,
                    "entropy": entropy_bits,
                }
            )
    return provisional
