"""MOSAIC: Memory-Oriented Sparse Aggregative Identification of Communities.

Lineage-tracking subsystem -- community continuity +
prior-aware initialisation.

Symbol names (LineageTracker, LineageEvent, LineageReport, init_partitions,
pick_merge_survivor) are public and stable.

This module owns:

  1. The audit trail of community evolution across a single
     `run_mosaic` invocation -- replaces the post-hoc
     "cosine match >= 0.7" heuristic from `community.py:_map_to_stable_uuids`
     with an explicit event log driven by the algorithm during local move /
     refinement / aggregation.
  2. The initialiser `init_partitions(graph, prior, prior_mode)` --
     entrypoint for `run_mosaic` to either resume from a prior
     partition (seeded) or discard the prior and start fresh (cold for
     crisis_recluster).

Public surface:

  - `EventType` -- Literal of "birth" | "split" | "merge" | "death"
  - `LineageEvent` -- @dataclass(frozen=True), one audit record
  - `LineageReport` -- @dataclass, immutable snapshot of an event tuple
  - `LineageTracker` -- mutable recorder with:
      * `register_prior_birth(uuid, ts)` -- bootstrap a known-prior UUID's
        birth timestamp WITHOUT emitting an event (used by init_partitions).
      * `pick_merge_survivor(candidates)` -- UUID survival policy.
      * `known_uuids()` -- inspection accessor (tests + wiring).
      * Internal `_birth_ts: dict[UUID, datetime]` bookkeeping (the oldest
        timestamp survives merges).
  - `init_partitions(graph, prior, prior_mode)` -- consumed by `run_mosaic`.

UUID survival policy:
  - merge: surviving UUID = OLDEST UUID by birth-event timestamp;
            tie-break = `min(uuid, key=str)`; unknown UUID loses via
            `datetime.max` sentinel.
  - split: largest sub-community keeps the parent UUID; others get fresh.
  - death: retired UUID recorded with timestamp; consumer retains for 30d
            in the bank for `last_seen` queries.
  - birth: fresh `uuid4()`, recorded via `record_birth`.

First-migration degeneracy (warning #3):

  When migrating from legacy `_map_to_stable_uuids` state (no birth-timestamp
  metadata in the prior), `pick_merge_survivor` is functionally LEX-ONLY on
  first migration -- every surviving prior UUID receives the same fake
  `datetime.now(timezone.utc) - timedelta(microseconds=1)` timestamp (per
  `init_partitions` Branch B), so the `(timestamp, str(uuid))` tiebreak
  collapses to lex-only. Oldest-survives activates from run 2 onward as new
  birth events accumulate real timestamps via `record_birth`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph

EventType = Literal["birth", "split", "merge", "death"]

# Sentinel for `pick_merge_survivor` when a candidate UUID has no registered
# birth timestamp. `datetime.max` is the largest possible value, so any
# registered candidate (with a real ts) wins the comparison.
_UNKNOWN_BIRTH_TS: datetime = datetime.max.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LineageEvent:
    """Immutable record of one community-lineage event.

    `frozen=True` makes the dataclass hashable and prevents in-place mutation
    of the audit trail (prevents tampering with the lineage log).
    `child_uuids` is a tuple (not list) so the dataclass stays hashable.
    """

    event_type: EventType
    timestamp: datetime
    parent_uuid: UUID | None
    child_uuids: tuple[UUID, ...]
    member_count: int


@dataclass
class LineageReport:
    """Immutable snapshot of all lineage events produced by a single run.

    Wire shape is intentionally minimal -- one tuple of events. Additional
    fields may be added without breaking existing callers; do NOT iterate by
    index.
    """

    events: tuple[LineageEvent, ...] = field(default_factory=tuple)


class LineageTracker:
    """Mutable recorder of lineage events during a single Leiden run.

    Public surface:

        __init__(self) -> None
        register_prior_birth(self, uuid: UUID, ts: datetime) -> None
        record_birth(self, new: UUID, member_count: int) -> None
        record_split(self, parent: UUID, children: list[UUID],
                     member_count: int) -> None
        record_merge(self, parents: list[UUID], surviving: UUID,
                     member_count: int) -> None
        record_death(self, retired: UUID, member_count: int) -> None
        pick_merge_survivor(self, candidates: list[UUID]) -> UUID
        known_uuids(self) -> set[UUID]
        report(self) -> LineageReport

    Internal state:

        _events: list[LineageEvent] -- append-only event log
        _birth_ts: dict[UUID, datetime] -- UUID -> first-seen timestamp;
                                              used by `pick_merge_survivor`
                                              to score candidates by age.
    """

    def __init__(self) -> None:
        self._events: list[LineageEvent] = []
        self._birth_ts: dict[UUID, datetime] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # -------------------------------------------------------------------

    def register_prior_birth(self, uuid: UUID, ts: datetime) -> None:
        """Bootstrap a known-prior UUID's birth timestamp WITHOUT emitting an
        event. Called by `init_partitions` during seeded-mode setup so the
        survivor-pick policy can score legacy UUIDs by age.

        `setdefault` semantics: the FIRST timestamp registered for a UUID
        wins; later calls are no-ops. This keeps replay deterministic across
        runs that re-bootstrap the same prior state.
        """
        self._birth_ts.setdefault(uuid, ts)

    def pick_merge_survivor(self, candidates: list[UUID]) -> UUID:
        """Choose the UUID that should survive a merge.

        Policy (deterministic, no centroid math):
          1. Sort candidates by `(birth_ts, str(uuid))` ascending.
          2. Return the first element.

        A candidate with no registered `_birth_ts` entry gets the
        `_UNKNOWN_BIRTH_TS = datetime.max` sentinel, so any candidate with a
        real timestamp wins. If ALL candidates are unknown (e.g. on the very
        first migration before any birth event has been recorded), the result
        collapses to pure lex-by-str.
        """

        def key(u: UUID) -> tuple[datetime, str]:
            return (self._birth_ts.get(u, _UNKNOWN_BIRTH_TS), str(u))

        return min(candidates, key=key)

    def known_uuids(self) -> set[UUID]:
        """Inspection accessor -- the set of UUIDs the tracker has stamped
        with a birth timestamp. Returns a snapshot copy; mutating the result
        does not affect the tracker."""
        return set(self._birth_ts.keys())

    # --------------------------------------------------------- recorders

    def record_birth(self, new: UUID, member_count: int) -> None:
        ts = self._now()
        # setdefault so a buggy re-emission cannot overwrite the original ts.
        self._birth_ts.setdefault(new, ts)
        self._events.append(
            LineageEvent(
                event_type="birth",
                timestamp=ts,
                parent_uuid=None,
                child_uuids=(new,),
                member_count=member_count,
            )
        )

    def record_split(
        self, parent: UUID, children: list[UUID], member_count: int
    ) -> None:
        ts = self._now()
        # Each child gets a birth timestamp (setdefault preserves any prior).
        for c in children:
            self._birth_ts.setdefault(c, ts)
        self._events.append(
            LineageEvent(
                event_type="split",
                timestamp=ts,
                parent_uuid=parent,
                child_uuids=tuple(children),
                member_count=member_count,
            )
        )

    def record_merge(
        self,
        parents: list[UUID],
        surviving: UUID,
        member_count: int,
    ) -> None:
        # `parent_uuid` is the surviving UUID; non-surviving parents go into
        # `child_uuids` so the record still hashes and the merge direction is
        # legible: surviving = parent, retired = children. The caller is
        # responsible for invoking `pick_merge_survivor` to choose `surviving`
        # before calling `record_merge`.
        ts = self._now()
        retired = tuple(p for p in parents if p != surviving)
        self._events.append(
            LineageEvent(
                event_type="merge",
                timestamp=ts,
                parent_uuid=surviving,
                child_uuids=retired,
                member_count=member_count,
            )
        )

    def record_death(self, retired: UUID, member_count: int) -> None:
        self._events.append(
            LineageEvent(
                event_type="death",
                timestamp=self._now(),
                parent_uuid=retired,
                child_uuids=(),
                member_count=member_count,
            )
        )

    def report(self) -> LineageReport:
        """Finalise the tracker into an immutable `LineageReport` snapshot.

        The returned `LineageReport.events` is a tuple constructed from the
        current `_events` list -- subsequent mutations to the tracker do NOT
        leak into the snapshot (test
        `test_lineage_report_returns_frozen_snapshot` is the witness).
        """
        return LineageReport(events=tuple(self._events))


# ============================================================================
# -- prior-aware initialisation
# ============================================================================


def init_partitions(
    graph: MemoryGraph,
    prior: CommunityAssignment | None,
    prior_mode: Literal["seeded", "cold"],
) -> tuple[np.ndarray, dict[int, UUID], LineageTracker]:
    """initialisation for `run_mosaic`.

    Returns:
      partition -- np.ndarray[int64] of size N, indexed in the canonical node
                    order (`sorted(graph.iter_nodes(), key=str)`).
      int_to_uuid -- dict mapping algorithmic int label -> community UUID.
      lineage -- `LineageTracker` initialised with birth timestamps for
                    every surviving prior UUID (via `register_prior_birth`)
                    plus `birth` events for any new-node singletons.

    prior_mode behaviour:

      `prior is None` OR `prior_mode == "cold"`:
        Each node is its own singleton with a fresh `uuid4()` community.
        No lineage events recorded -- cold start has no prior identity to
        track births FROM.

      `prior_mode == "seeded"`:
        - Filter `prior.node_to_community` to active leaf UUIDs only (stale
          entries -- leaf UUIDs no longer in `graph.iter_nodes()` -- are
          dropped silently).
        - Each surviving prior community UUID maps to a stable algorithmic
          int label, assigned in canonical node-order.
        - New nodes (in `graph.iter_nodes()` but not in
          `prior.node_to_community`) get fresh singleton ints with fresh
          `uuid4()` community UUIDs and a `birth` event in the lineage
          tracker.
        - Surviving prior community UUIDs get their birth timestamps
          bootstrapped via `register_prior_birth`. Because `CommunityAssignment`
          does not (yet) carry a `birth_timestamps` field, we use
          `datetime.now(timezone.utc) - timedelta(microseconds=1)` as the
          fake birth ts -- a deterministic value strictly less than any
          new-node birth recorded during this run, but identical across all
          prior UUIDs (warning #3: first-migration degeneracy
          collapses `pick_merge_survivor` to lex-only).

    Raises:
      ValueError if `prior_mode` is not one of {"seeded", "cold"}.
    """
    # Validate mode -- silent fallback would hide a programming error.
    if prior_mode not in ("seeded", "cold"):
        raise ValueError(
            f"prior_mode must be 'seeded' or 'cold', got {prior_mode!r}"
        )

    # Canonical node order: sorted by string representation for determinism.
    nodes_sorted: list[UUID] = sorted(graph.iter_nodes(), key=str)
    n = len(nodes_sorted)

    lineage = LineageTracker()

    # Empty-graph short-circuit.
    if n == 0:
        return np.empty(0, dtype=np.int64), {}, lineage

    partition = np.empty(n, dtype=np.int64)
    int_to_uuid: dict[int, UUID] = {}

    # Branch A: cold (or `prior is None`) -- all-singletons with fresh UUIDs.
    # No birth events recorded; cold start has no prior identity to track
    # births FROM (the run's consumers treat the whole report as the
    # birth state).
    if prior is None or prior_mode == "cold":
        for i in range(n):
            partition[i] = i
            int_to_uuid[i] = uuid4()
        return partition, int_to_uuid, lineage

    # Branch B: seeded -- reuse prior community UUIDs for surviving members,
    # fresh singletons for new nodes.
    active_node_set = set(nodes_sorted)
    # Filter stale prior entries (leaf UUIDs no longer in the graph).
    active_priors: dict[UUID, UUID] = {
        leaf: comm
        for leaf, comm in prior.node_to_community.items()
        if leaf in active_node_set
    }

    # First-migration fake birth ts -- `now - 1µs` so any new births recorded
    # during this run land strictly later (deterministic ordering against
    # future merges). All surviving prior UUIDs share this ts, collapsing
    # the survivor tiebreak to lex-only on the first migration (warning #3).
    prior_birth_ts = datetime.now(timezone.utc) - timedelta(microseconds=1)

    # Walk nodes in canonical order, assigning ints. The FIRST node in
    # canonical order that belongs to a given prior community UUID claims the
    # next available int label; subsequent nodes in the same community reuse
    # that int. New nodes (not in `active_priors`) get fresh singletons.
    next_int = 0
    uuid_to_int: dict[UUID, int] = {}

    for i, node in enumerate(nodes_sorted):
        if node in active_priors:
            community_uuid = active_priors[node]
            if community_uuid not in uuid_to_int:
                uuid_to_int[community_uuid] = next_int
                int_to_uuid[next_int] = community_uuid
                # Bootstrap the birth ts -- no event emitted, just bookkeeping
                # so the survivor-pick policy can later score this UUID.
                lineage.register_prior_birth(community_uuid, prior_birth_ts)
                next_int += 1
            partition[i] = uuid_to_int[community_uuid]
        else:
            # New node: fresh singleton with fresh UUID.
            new_uuid = uuid4()
            int_to_uuid[next_int] = new_uuid
            partition[i] = next_int
            # Emit a real `birth` event -- new nodes ARE born this run.
            lineage.record_birth(new_uuid, member_count=1)
            next_int += 1

    return partition, int_to_uuid, lineage
