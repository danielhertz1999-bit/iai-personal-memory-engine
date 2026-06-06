"""Ashby 1956 step-mechanism for the sleep pipeline.

Monitors three essential variables of the memory graph topology at every
sleep-cycle boundary. When any variable breaches its allowable range,
EssentialVariableTracker.check() returns a non-None BreachInfo for that
variable; the caller (sleep_pipeline._run_internal) emits the matching
`essential_variable_breach` event and routes a True transition through
S2Coordinator.set_crisis_mode().

Essential variables:
- rich_club_ratio -- van den Heuvel & Sporns 2011 rich-club coefficient.
                       Floor breach when observed < rich_club_ratio_floor.
                       Below the floor, the topology lacks structural hubs
                       and recall degrades to random-walk noise.
- community_count -- Number of distinct communities from Leiden partition.
                       Ceiling breach when (community_count / total_nodes)
                       > community_count_ceiling_ratio. Above the ceiling,
                       the graph is fragmented.
- edge_density -- 2 * |edges| / (N * (N-1)); the fraction of possible
                       undirected pairs that are connected. Floor breach
                       when observed < edge_density_floor. Below the
                       floor, the graph has insufficient connectivity to
                       support multi-hop retrieval -- sanity check that
                       protects against an empty store self-tripping.

STATELESS DESIGN: no in-memory history; each check() call
re-evaluates against the current snapshot in isolation. Trend detection
across cycles is a deliberate future-phase capability, not a current one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Frozen dataclass -- sleep_pipeline._run_internal builds one of these at
# cycle start from MemoryGraph.rich_club_coefficient() +
# CommunityAssignment.node_to_community + MemoryGraph.node_count() + edge
# count. Locking the shape here prevents an incompatible payload.
@dataclass(frozen=True)
class TopologySnapshot:
    """Single sleep-cycle topology reading used by EssentialVariableTracker.

    Fields:
        rich_club_ratio: MemoryGraph.rich_club_coefficient() output (0.0..1.0).
        community_count: len(set(CommunityAssignment.node_to_community.values())).
        edge_density: 2 * |edges| / (N * (N-1)); 0.0 for N < 2.
        total_nodes: MemoryGraph.node_count(); 0 short-circuits all checks.
    """

    rich_club_ratio: float
    community_count: int
    edge_density: float
    total_nodes: int


# Frozen dataclass -- return payload shape for one breached essential
# variable. Plain Literal[str] for `direction` (not enum) to keep the
# event-payload JSON-serialisable without a `.value` lookup downstream.
@dataclass(frozen=True)
class BreachInfo:
    """Returned by EssentialVariableTracker.check() per breached variable.

    Fields:
        variable_name: one of "rich_club_ratio" / "community_count" / "edge_density".
        observed_value: the actual measured value at snapshot time
                        (for community_count this is the RATIO
                        community_count / total_nodes, not the raw count).
        threshold: the configured floor / ceiling that was breached.
        direction: "floor_breach" when observed < floor;
                        "ceiling_breach" when observed > ceiling.
    """

    variable_name: str
    observed_value: float
    threshold: float
    direction: Literal["floor_breach", "ceiling_breach"]


# EssentialVariableTracker -- 1956 step-mechanism.
# STATELESS by design: the constructor reads three thresholds off
# the SleepOverhaulConfig (or any duck-typed object with the same three
# float attributes) and stores ONLY those floats. No history, no counter,
# no buffer. Each `check(snapshot)` re-evaluates fresh.
class EssentialVariableTracker:
    """1956 step-mechanism -- STATELESS.

    Each `check(snapshot)` call evaluates the snapshot against the three
    configured thresholds and returns a dict[str, BreachInfo | None] with
    exactly the three keys "rich_club_ratio", "community_count", and
    "edge_density". A non-None value means that variable breached its
    allowable range; None means in-bounds.

    The tracker emits NO events and holds NO lifecycle / coordinator
    references. The caller (sleep_pipeline._run_internal) is responsible
    for emitting `essential_variable_breach` and routing the
    S2Coordinator.set_crisis_mode(True, reason) transition.
    """

    def __init__(self, cfg) -> None:  # cfg = SleepOverhaulConfig (duck-typed)
        # Duck-typed cfg: any object with the three threshold attributes
        # works (lets tests pass a bare namespace without importing
        # SleepOverhaulConfig from daemon.py -- keeps this module
        # import-cycle-free).
        self._rich_club_floor: float = float(cfg.rich_club_ratio_floor)
        self._community_count_ceiling_ratio: float = float(
            cfg.community_count_ceiling_ratio
        )
        self._edge_density_floor: float = float(cfg.edge_density_floor)

    def check(self, snapshot: TopologySnapshot) -> dict[str, "BreachInfo | None"]:
        """Evaluate snapshot against the three thresholds.

        Returns a dict with exactly 3 keys: "rich_club_ratio",
        "community_count", "edge_density". Each value is a BreachInfo
        when the variable breached its allowable range, None otherwise.

        Empty-store short-circuit: when `snapshot.total_nodes == 0`, all
        three values are None. An empty graph has no topology to defend,
        and the community_count check would otherwise divide by zero.

        Breach semantics:
        - rich_club_ratio: floor breach when observed < floor.
        - community_count: ceiling breach on the RATIO
                            (community_count / total_nodes) > ceiling.
                            The OBSERVED value reported in BreachInfo is
                            the ratio, NOT the
                            raw count -- the raw count is recoverable
                            from `snapshot.community_count` at the call
                            site.
        - edge_density: floor breach when observed < floor.

        Pure value return -- mutates no `self._*`, emits no events.
        """
        if snapshot.total_nodes == 0:
            return {
                "rich_club_ratio": None,
                "community_count": None,
                "edge_density": None,
            }

        result: dict[str, BreachInfo | None] = {}

        # rich_club_ratio: floor breach when observed value drops below the
        # configured floor. Live baseline 0.10; default floor 0.05.
        if snapshot.rich_club_ratio < self._rich_club_floor:
            result["rich_club_ratio"] = BreachInfo(
                variable_name="rich_club_ratio",
                observed_value=float(snapshot.rich_club_ratio),
                threshold=float(self._rich_club_floor),
                direction="floor_breach",
            )
        else:
            result["rich_club_ratio"] = None

        # community_count: ceiling breach on RATIO (community_count /
        # total_nodes) > configured ceiling. The OBSERVED value reported
        # here is the ratio, not the raw count, so downstream events log
        # the fragmentation directly.
        cc_ratio = snapshot.community_count / snapshot.total_nodes
        if cc_ratio > self._community_count_ceiling_ratio:
            result["community_count"] = BreachInfo(
                variable_name="community_count",
                observed_value=float(cc_ratio),
                threshold=float(self._community_count_ceiling_ratio),
                direction="ceiling_breach",
            )
        else:
            result["community_count"] = None

        # edge_density: floor breach when observed density drops below the
        # configured floor. Sanity check that protects against an empty /
        # near-empty store self-tripping (the total_nodes == 0 guard
        # above handles the strictly empty case).
        if snapshot.edge_density < self._edge_density_floor:
            result["edge_density"] = BreachInfo(
                variable_name="edge_density",
                observed_value=float(snapshot.edge_density),
                threshold=float(self._edge_density_floor),
                direction="floor_breach",
            )
        else:
            result["edge_density"] = None

        return result
