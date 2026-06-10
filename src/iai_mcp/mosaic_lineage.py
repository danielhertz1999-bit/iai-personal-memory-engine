from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph

EventType = Literal["birth", "split", "merge", "death"]

_UNKNOWN_BIRTH_TS: datetime = datetime.max.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LineageEvent:

    event_type: EventType
    timestamp: datetime
    parent_uuid: UUID | None
    child_uuids: tuple[UUID, ...]
    member_count: int


@dataclass
class LineageReport:

    events: tuple[LineageEvent, ...] = field(default_factory=tuple)


class LineageTracker:

    def __init__(self) -> None:
        self._events: list[LineageEvent] = []
        self._birth_ts: dict[UUID, datetime] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)


    def register_prior_birth(self, uuid: UUID, ts: datetime) -> None:
        self._birth_ts.setdefault(uuid, ts)

    def pick_merge_survivor(self, candidates: list[UUID]) -> UUID:

        def key(u: UUID) -> tuple[datetime, str]:
            return (self._birth_ts.get(u, _UNKNOWN_BIRTH_TS), str(u))

        return min(candidates, key=key)

    def known_uuids(self) -> set[UUID]:
        return set(self._birth_ts.keys())


    def record_birth(self, new: UUID, member_count: int) -> None:
        ts = self._now()
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
        return LineageReport(events=tuple(self._events))


def init_partitions(
    graph: MemoryGraph,
    prior: CommunityAssignment | None,
    prior_mode: Literal["seeded", "cold"],
) -> tuple[np.ndarray, dict[int, UUID], LineageTracker]:
    if prior_mode not in ("seeded", "cold"):
        raise ValueError(
            f"prior_mode must be 'seeded' or 'cold', got {prior_mode!r}"
        )

    nodes_sorted: list[UUID] = sorted(graph.iter_nodes(), key=str)
    n = len(nodes_sorted)

    lineage = LineageTracker()

    if n == 0:
        return np.empty(0, dtype=np.int64), {}, lineage

    partition = np.empty(n, dtype=np.int64)
    int_to_uuid: dict[int, UUID] = {}

    if prior is None or prior_mode == "cold":
        for i in range(n):
            partition[i] = i
            int_to_uuid[i] = uuid4()
        return partition, int_to_uuid, lineage

    active_node_set = set(nodes_sorted)
    active_priors: dict[UUID, UUID] = {
        leaf: comm
        for leaf, comm in prior.node_to_community.items()
        if leaf in active_node_set
    }

    prior_birth_ts = datetime.now(timezone.utc) - timedelta(microseconds=1)

    next_int = 0
    uuid_to_int: dict[UUID, int] = {}

    for i, node in enumerate(nodes_sorted):
        if node in active_priors:
            community_uuid = active_priors[node]
            if community_uuid not in uuid_to_int:
                uuid_to_int[community_uuid] = next_int
                int_to_uuid[next_int] = community_uuid
                lineage.register_prior_birth(community_uuid, prior_birth_ts)
                next_int += 1
            partition[i] = uuid_to_int[community_uuid]
        else:
            new_uuid = uuid4()
            int_to_uuid[next_int] = new_uuid
            partition[i] = next_int
            lineage.record_birth(new_uuid, member_count=1)
            next_int += 1

    return partition, int_to_uuid, lineage
