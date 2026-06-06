# Self-contained module. NO daemon imports at top level --
# every daemon-side coupling is lazy inside method bodies so this module
# stays cheap to import in isolation (tests, CLI fallback, future tooling).
"""ReflectionAgent + MetaAnalyst.

Two coupled mechanisms (Buckner DMN / Andrews-Hanna 2014 + Von Foerster
second-order observer):

* ``ReflectionAgent.synthesize(store, window_hours)`` returns a fresh
  ``MemoryRecord`` (tier="semantic") whose ``literal_surface`` is a
  Tier-0 deterministic summary -- community_id distribution + first-50-
  chars topic labels reused from. NO LLM call in v1.
  Provenance carries ``synthesized_by="dmn_reflection"`` so consolidation
  and downstream consumers can distinguish synthetic from user-captured.

* ``MetaAnalyst.snapshot(store, window_hours)`` reads the events table
  via ``query_events`` and returns a plain dict with daily counts by
  kind plus a delta proxy. Pure read; no mutation, no event emission --
  the SleepStep handler in is the only place that wraps
  this in a ``system_health_report`` write_event call.

Both surfaces are pure functions over (store, window_hours). The
classes exist for namespacing / future-extension hooks only; today they
hold no state. They are safe to instantiate cheaply per call.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_V4


# Helper: trim the topic label to 50 chars + strip trailing whitespace.
# Mirrors user_model._first_50_chars verbatim -- kept local rather than
# imported to preserve the "no daemon imports at top level" invariant
# (user_model.py lazy-imports daemon helpers; importing the helper here
# would not break that, but duplicating a 4-line pure function keeps the
# module's import graph maximally narrow and the dependency arrow
# unidirectional: dmn_reflection -> {types, events, store} only).
def _first_50_chars(s: str) -> str:
    """Return the first 50 chars of ``s`` with trailing whitespace stripped.

    Empty / non-string input returns the empty string so the caller can
    safely use the result as a dict key / list element without a guard.
    """
    if not s or not isinstance(s, str):
        return ""
    return s[:50].rstrip()


class ReflectionAgent:
    """Tier-0 deterministic DMN reflection synthesiser.

    Stateless. ``synthesize`` walks decrypted records in the trailing
    ``window_hours`` window, groups by community_id, picks the top-5
    communities by member count, labels each by the first 50 chars of the
    most-recent representative record's ``literal_surface``, and emits a
    fresh ``MemoryRecord`` at tier="semantic". The returned record is NOT
    inserted -- the SleepStep handler in owns persistence so
    dry-run mode can skip the write while still computing the would-be
    record for diagnostic emission.

    Embedding is a placeholder zero vector of ``store.embed_dim`` floats;
    the next REM consolidation cycle re-embeds via the normal flow
    (consolidator's existing embed pass). The
    schema_version is pinned to v4 (current) and ``community_id=None``
    so the synthesised record is later clustered by the existing
    community-detection pass like any other fresh insert.
    """

    def synthesize(self, store, window_hours: int) -> MemoryRecord:
        # Lazy import kept local so any future daemon-side coupling in
        # events.py does not silently re-couple this module.
        from iai_mcp.events import query_events  # noqa: F401 (kept for symmetry)

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)

        # === records in window =========================================
        # store.all_records() returns decrypted plaintext records; the
        # raw to_pandas() path on RECORDS_TABLE would hand back AES-256-GCM
        # ciphertext for literal_surface (see store.py L2167) so we MUST
        # go through all_records to read the prefix we expose as a topic
        # label. Same constraint reasons through.
        recs = store.all_records()
        in_window: list = []
        for r in recs:
            created = getattr(r, "created_at", None)
            if created is None:
                continue
            # Defensive tz coercion -- a naive datetime on disk would
            # raise on the >= compare with a tz-aware cutoff.
            # aggregator hit the same edge (T-11.6-08 fixture); same fix.
            try:
                if getattr(created, "tzinfo", None) is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError, AttributeError):
                continue
            if created < cutoff:
                continue
            # Exclude prior reflection/digest records from the input set.
            # After community-detection (crisis_recluster), synthetic
            # reflection records can acquire a community_id and appear as
            # community members. Without this exclusion they surface as
            # topic labels (first-50-chars of "Daily reflection: …"),
            # causing nested "Daily reflection: top topics were [Daily
            # reflection: …]" strings in the new reflection. The structured
            # provenance key synthesized_by is the canonical marker; it is
            # written by this class and is more durable than a string-
            # prefix check.
            prov_list = getattr(r, "provenance", None) or []
            if any(
                (p.get("synthesized_by") == "dmn_reflection" if isinstance(p, dict) else False)
                for p in prov_list
            ):
                continue
            in_window.append(r)

        captured_count = len(in_window)

        # === topics from community_id ==================================
        community_to_records: dict[UUID, list] = {}
        for r in in_window:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            community_to_records.setdefault(cid, []).append(r)

        community_counts: Counter = Counter(
            {cid: len(rlist) for cid, rlist in community_to_records.items()}
        )

        # Label each community by its most-recent record's first 50 chars
        # -- explicitly: "first-50-chars of top-1-record's
        # literal_surface per community". "Top-1" by recency matches
        # aggregator's interpretation.
        community_labels: dict[UUID, str] = {}
        for cid, rlist in community_to_records.items():
            top = max(rlist, key=lambda r: getattr(r, "created_at", now))
            community_labels[cid] = _first_50_chars(
                getattr(top, "literal_surface", "")
            )

        # Top-5 communities by member count. Labels with empty strings
        # (e.g. an empty literal_surface) are dropped from the surface
        # list to keep the natural-language summary readable -- they
        # carry no signal a reader could act on.
        top_cids = [cid for cid, _ in community_counts.most_common(5)]
        topics: list[str] = [
            community_labels[cid]
            for cid in top_cids
            if community_labels.get(cid)
        ]

        # === recall count ==============================================
        # The summary string includes "recalled M times". We pull M
        # from the events table over the same window. Best-effort -- if
        # the events table is empty / unavailable the count falls to 0
        # and the summary string still reads naturally ("recalled 0
        # times").
        recall_events = query_events(
            store,
            kind="memory_recall",
            since=cutoff,
            limit=10000,
        )
        recalled_count = len(recall_events)

        # === build literal_surface ====================================
        topics_str = "[" + ", ".join(topics) + "]"
        literal_surface = (
            f"Daily reflection: top topics were {topics_str}; "
            f"captured {captured_count} turns; "
            f"recalled {recalled_count} times."
        )

        # === build MemoryRecord =======================================
        # provenance is a list[dict] on the dataclass; the store
        # serialises it to the provenance_json column (see store.py
        # _row_from_record). The provenance_json column will contain
        # synthesized_by='dmn_reflection' after serialisation; downstream
        # consolidation can distinguish synthetic from user-captured rows.
        provenance_entry: dict[str, Any] = {
            "synthesized_by": "dmn_reflection",
            "window_hours": int(window_hours),
            "topics": list(topics),
            "captured_count": int(captured_count),
            "recalled_count": int(recalled_count),
            "ts": now.isoformat(),
        }

        # Embedding placeholder per: zero vector at the store's
        # configured dim. Next REM consolidation cycle re-embeds with
        # the real model. structure_hv stays empty (pre-bind sentinel,
        # per types.py STRUCTURE_HV_BYTES rule).
        embed_dim = int(store.embed_dim)
        embedding = [0.0] * embed_dim

        # detail_level=1 (default low) so the synthesised record does
        # NOT auto-flip never_decay=True (types.py __post_init__ sets
        # never_decay when detail_level >= 3). Synthetic reflections
        # are expected to be replaced daily and should be subject to
        # the normal FSRS decay schedule like any other semantic row.
        return MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=literal_surface,
            aaak_index="",
            embedding=embedding,
            community_id=None,
            centrality=0.5,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[provenance_entry],
            created_at=now,
            updated_at=now,
            language="en",
            tags=[],
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_V4,
            structure_hv=b"",
        )


class MetaAnalyst:
    """Second-order observer over the events table.

    Stateless pure read. ``snapshot`` pulls up to 10000 events via
    ``query_events`` (the existing helper handles decryption, kind
    filtering, time filtering, and tz-aware compares), then counts
    kinds locally in Python so the result shape is decoupled from the
    store scanner.

    Counts tracked (per):

    * ``recall_count`` -- events with kind=="memory_recall"
    * ``capture_count`` -- events with kind=="memory_capture"
    * ``sleep_cycles_count`` -- events with kind=="sleep_step_completed"
      AND payload step name == "COMPACT_RECORDS" (the final NREM step;
      one per full sleep cycle)
    * ``breach_count`` -- events with kind=="essential_variable_breach"
    * ``erasure_count`` -- events with kind=="erasure_agent_pass"

    Plus ``average_record_count_delta`` (best-effort proxy: net
    captures - erasures in the window; 0 when neither fires),
    ``window_hours`` (echo for audit), and ``generated_at`` (ISO
    timestamp, UTC).

    The return type is a plain dict rather than a dataclass per
    ("returns plain dict (not dataclass)") so the snapshot can be
    written into a ``system_health_report`` event's data payload by
     without any (de)serialization hop.
    """

    # snapshot pulls 10000 events to give a generous 24h ceiling. At the
    # steady-state ~hundreds-of-events-per-day rate
    # the daemon emits today this is comfortably over-provisioned; if a
    # future phase scales the event rate past 10k/24h, this constant
    # is the single tunable to bump.
    _QUERY_LIMIT: int = 10000

    def snapshot(self, store, window_hours: int) -> dict[str, Any]:
        # Lazy import for the same defensive reason as ReflectionAgent.synthesize.
        from iai_mcp.events import query_events

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)

        # Pull all events newer than cutoff (single scan; we filter by
        # kind locally so we walk the events table exactly once instead
        # of once per kind).
        events_list = query_events(
            store,
            since=cutoff,
            limit=self._QUERY_LIMIT,
        )

        recall_count = 0
        capture_count = 0
        sleep_cycles_count = 0
        breach_count = 0
        erasure_count = 0

        for ev in events_list:
            kind = ev.get("kind") or ""
            if kind == "memory_recall":
                recall_count += 1
            elif kind == "memory_capture":
                capture_count += 1
            elif kind == "sleep_step_completed":
                # Only the COMPACT_RECORDS step terminates a full sleep
                # cycle. The event payload carries the step name as a
                # plaintext "step" key (sleep_pipeline.py
                # _emit_step_completed). Older events / events from a
                # different emitter that omit the key are skipped, not
                # counted -- silent miscount > silent overcount.
                data = ev.get("data") or {}
                step_name = data.get("step")
                if step_name == "COMPACT_RECORDS":
                    sleep_cycles_count += 1
            elif kind == "essential_variable_breach":
                breach_count += 1
            elif kind == "erasure_agent_pass":
                erasure_count += 1

        # Average record-count delta proxy. Net (captures - erasures) is
        # a coarse but cheap signal of corpus growth in the window. The underlying
        # event stream is not a per-record delta time series -- it is a
        # bag of write events. The proxy maps cleanly: each capture is
        # +1 record, each erasure pass is N erased records, but we lack
        # the per-pass N here (the erasure_agent_pass payload may carry
        # it; we don't depend on the shape since the contract is a
        # single float). Default 0 when neither fires keeps the surface
        # well-defined on an empty store.
        if (capture_count + erasure_count) > 0:
            average_record_count_delta = float(
                capture_count - erasure_count
            )
        else:
            average_record_count_delta = 0.0

        return {
            "recall_count": recall_count,
            "capture_count": capture_count,
            "sleep_cycles_count": sleep_cycles_count,
            "breach_count": breach_count,
            "erasure_count": erasure_count,
            "average_record_count_delta": average_record_count_delta,
            "window_hours": int(window_hours),
            "generated_at": now.isoformat(),
        }
