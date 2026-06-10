
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TopologySnapshot:

    rich_club_ratio: float
    community_count: int
    edge_density: float
    total_nodes: int


@dataclass(frozen=True)
class BreachInfo:

    variable_name: str
    observed_value: float
    threshold: float
    direction: Literal["floor_breach", "ceiling_breach"]


class EssentialVariableTracker:

    def __init__(self, cfg) -> None:
        self._rich_club_floor: float = float(cfg.rich_club_ratio_floor)
        self._community_count_ceiling_ratio: float = float(
            cfg.community_count_ceiling_ratio
        )
        self._edge_density_floor: float = float(cfg.edge_density_floor)

    def check(self, snapshot: TopologySnapshot) -> dict[str, "BreachInfo | None"]:
        if snapshot.total_nodes == 0:
            return {
                "rich_club_ratio": None,
                "community_count": None,
                "edge_density": None,
            }

        result: dict[str, BreachInfo | None] = {}

        if snapshot.rich_club_ratio < self._rich_club_floor:
            result["rich_club_ratio"] = BreachInfo(
                variable_name="rich_club_ratio",
                observed_value=float(snapshot.rich_club_ratio),
                threshold=float(self._rich_club_floor),
                direction="floor_breach",
            )
        else:
            result["rich_club_ratio"] = None

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
