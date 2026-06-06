"""IAI-MCP daemon configuration dataclasses and loaders.

Extracted from daemon.py — 10 frozen Config bundles with matching
_load_*_config() functions. Each loader reads env vars on demand (no
import-time cache) so pytest monkeypatch.setenv works between cases.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# ErasureConfig + _load_erasure_config: typed env-var bundle for the
# active-forgetting (Rac1/cofilin-inspired) ErasureAgent step that sits
# between DREAM_DECAY and OPTIMIZE_LANCE in the sleep pipeline.
# Five env vars with documented defaults + fail-loud validation. The helper
# is CALL-ON-DEMAND by design (no module-level cache, no functools.lru_cache):
# daemon boot calls it once for fail-loud validation; per-step handlers
# re-invoke it fresh inside their method bodies so
# `monkeypatch.setenv(...)` + fresh-pipeline test pattern works.
# ---------------------------------------------------------------------------

# Default values. Kept as module-level constants so consumers can reference
# them in docstrings / log messages without re-deriving the magic numbers.
_ERASURE_DEFAULT_CENTRALITY_THRESHOLD: float = 0.02
_ERASURE_DEFAULT_AGE_DAYS: int = 30
_ERASURE_DEFAULT_RETRIEVAL_WINDOW_DAYS: int = 30
_ERASURE_DEFAULT_TOMBSTONE_TTL_SEC: int = 604800  # 7 days

# Boolean-parse vocab for IAI_MCP_ERASURE_DRY_RUN. Case-insensitive match;
# empty-string maps to False explicitly (distinct from absent / None which
# triggers the pytest-aware default below).
_ERASURE_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_ERASURE_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ErasureConfig:
    """Typed bundle for ErasureAgent thresholds + dry-run toggle.

    Frozen for symmetry with downstream code that may hash / cache the value
    per pipeline pass. All five fields originate from env vars validated
    inside `_load_erasure_config()`.
    """

    centrality_threshold: float
    age_days: int
    retrieval_window_days: int
    tombstone_ttl_sec: int
    dry_run: bool


def _load_erasure_config() -> ErasureConfig:
    """Read the 5 IAI_MCP_ERASURE_* env vars and return a typed ErasureConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "true")` then
    construct a fresh `SleepPipeline`; if this helper were cached at import
    time the test would never see the env-var override.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # centrality_threshold: float in [0.0, 1.0]
    raw_centrality = os.environ.get(
        "IAI_MCP_ERASURE_CENTRALITY_THRESHOLD",
        str(_ERASURE_DEFAULT_CENTRALITY_THRESHOLD),
    )
    try:
        centrality_threshold = float(raw_centrality)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_CENTRALITY_THRESHOLD: invalid value "
            f"{raw_centrality!r}, expected float"
        )
    if not (0.0 <= centrality_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_ERASURE_CENTRALITY_THRESHOLD: invalid value "
            f"{raw_centrality!r}, expected float in [0.0, 1.0]"
        )

    # age_days: int > 0
    raw_age_days = os.environ.get(
        "IAI_MCP_ERASURE_AGE_DAYS", str(_ERASURE_DEFAULT_AGE_DAYS)
    )
    try:
        age_days = int(raw_age_days)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_AGE_DAYS: invalid value {raw_age_days!r}, "
            f"expected int"
        )
    if age_days <= 0:
        raise ValueError(
            f"IAI_MCP_ERASURE_AGE_DAYS: invalid value {raw_age_days!r}, "
            f"expected int > 0"
        )

    # retrieval_window_days: int > 0
    raw_retrieval_window = os.environ.get(
        "IAI_MCP_ERASURE_RETRIEVAL_WINDOW_DAYS",
        str(_ERASURE_DEFAULT_RETRIEVAL_WINDOW_DAYS),
    )
    try:
        retrieval_window_days = int(raw_retrieval_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_RETRIEVAL_WINDOW_DAYS: invalid value "
            f"{raw_retrieval_window!r}, expected int"
        )
    if retrieval_window_days <= 0:
        raise ValueError(
            f"IAI_MCP_ERASURE_RETRIEVAL_WINDOW_DAYS: invalid value "
            f"{raw_retrieval_window!r}, expected int > 0"
        )

    # tombstone_ttl_sec: int > 0
    raw_tombstone_ttl = os.environ.get(
        "IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC",
        str(_ERASURE_DEFAULT_TOMBSTONE_TTL_SEC),
    )
    try:
        tombstone_ttl_sec = int(raw_tombstone_ttl)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC: invalid value "
            f"{raw_tombstone_ttl!r}, expected int"
        )
    if tombstone_ttl_sec <= 0:
        raise ValueError(
            f"IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC: invalid value "
            f"{raw_tombstone_ttl!r}, expected int > 0"
        )

    # dry_run: bool — absent → pytest-aware default; present → parse vocab.
    # Per: production default False, pytest default True. The pytest
    # branch is triggered by the standard PYTEST_CURRENT_TEST env var that
    # pytest sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_ERASURE_DRY_RUN")
    if raw_dry_run is None:
        # Absent — apply pytest-aware default.
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _ERASURE_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _ERASURE_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_ERASURE_DRY_RUN: invalid value {raw_dry_run!r}, "
                f"expected one of "
                f"{sorted(_ERASURE_DRY_RUN_TRUE_VALUES | _ERASURE_DRY_RUN_FALSE_VALUES)}"
            )

    return ErasureConfig(
        centrality_threshold=centrality_threshold,
        age_days=age_days,
        retrieval_window_days=retrieval_window_days,
        tombstone_ttl_sec=tombstone_ttl_sec,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# PatSepConfig + _load_patsep_config: typed env-var bundle for the
# pattern_separation_gate. Same call-on-demand discipline as ErasureConfig:
# daemon boot calls it once for fail-loud validation,
# gate re-invokes it fresh inside its body so pytest monkeypatch.setenv
# works.
# ---------------------------------------------------------------------------

# Default values. Kept as module-level constants so consumers can reference
# them in docstrings / log messages without re-deriving the magic numbers.
_PATSEP_DEFAULT_NEAR_DUP_THRESHOLD: float = 0.92
_PATSEP_DEFAULT_LINK_THRESHOLD: float = 0.70
_PATSEP_DEFAULT_LINK_INITIAL_WEIGHT: float = 0.10
_PATSEP_DEFAULT_TOP_K: int = 8

# Boolean-parse vocab for IAI_MCP_PATSEP_DRY_RUN. Case-insensitive match;
# empty-string maps to False explicitly (distinct from absent / None which
# triggers the pytest-aware default below).
_PATSEP_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_PATSEP_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class PatSepConfig:
    """Typed bundle for pattern_separation_gate thresholds + dry-run toggle.

    Frozen for symmetry with downstream code that may hash / cache the value
    per gate invocation. All five fields originate from env vars validated
    inside `_load_patsep_config()`.
    """

    near_dup_threshold: float
    link_threshold: float
    link_initial_weight: float
    top_k: int
    dry_run: bool


def _load_patsep_config() -> PatSepConfig:
    """Read the 5 IAI_MCP_PATSEP_* env vars and return a typed PatSepConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "true")` then
    construct a fresh pattern_separation_gate caller; if this helper were
    cached at import time the test would never see the env-var override.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # near_dup_threshold: float in (0.0, 1.0]
    raw_near_dup = os.environ.get(
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        str(_PATSEP_DEFAULT_NEAR_DUP_THRESHOLD),
    )
    try:
        near_dup_threshold = float(raw_near_dup)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD: invalid value "
            f"{raw_near_dup!r}, expected float"
        )
    if not (0.0 < near_dup_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD: invalid value "
            f"{raw_near_dup!r}, expected float in (0.0, 1.0]"
        )

    # link_threshold: float in (0.0, 1.0]
    raw_link = os.environ.get(
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        str(_PATSEP_DEFAULT_LINK_THRESHOLD),
    )
    try:
        link_threshold = float(raw_link)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_THRESHOLD: invalid value "
            f"{raw_link!r}, expected float"
        )
    if not (0.0 < link_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_THRESHOLD: invalid value "
            f"{raw_link!r}, expected float in (0.0, 1.0]"
        )

    # cross-constraint: link_threshold must be strictly less than near_dup_threshold
    if not (link_threshold < near_dup_threshold):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_THRESHOLD: invalid value {raw_link!r}, "
            f"must be strictly less than IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD "
            f"(got link={link_threshold}, near_dup={near_dup_threshold})"
        )

    # link_initial_weight: float in (0.0, 1.0]
    raw_weight = os.environ.get(
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        str(_PATSEP_DEFAULT_LINK_INITIAL_WEIGHT),
    )
    try:
        link_initial_weight = float(raw_weight)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT: invalid value "
            f"{raw_weight!r}, expected float"
        )
    if not (0.0 < link_initial_weight <= 1.0):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT: invalid value "
            f"{raw_weight!r}, expected float in (0.0, 1.0]"
        )

    # top_k: int in [1, 64]
    raw_top_k = os.environ.get(
        "IAI_MCP_PATSEP_TOP_K", str(_PATSEP_DEFAULT_TOP_K)
    )
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_TOP_K: invalid value {raw_top_k!r}, "
            f"expected int"
        )
    if not (1 <= top_k <= 64):
        raise ValueError(
            f"IAI_MCP_PATSEP_TOP_K: invalid value {raw_top_k!r}, "
            f"expected int in [1, 64]"
        )

    # dry_run: bool — absent → pytest-aware default; present → parse vocab.
    # Per: production default False, pytest default True. The
    # pytest branch is triggered by the standard PYTEST_CURRENT_TEST env var
    # that pytest sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_PATSEP_DRY_RUN")
    if raw_dry_run is None:
        # Absent — apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _PATSEP_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _PATSEP_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_PATSEP_DRY_RUN: invalid value {raw_dry_run!r}, "
                f"expected one of "
                f"{sorted(_PATSEP_DRY_RUN_TRUE_VALUES | _PATSEP_DRY_RUN_FALSE_VALUES)}"
            )

    return PatSepConfig(
        near_dup_threshold=near_dup_threshold,
        link_threshold=link_threshold,
        link_initial_weight=link_initial_weight,
        top_k=top_k,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# S2Config + _load_s2_config: typed env-var bundle for the S2Coordinator
# (module). Same call-on-demand discipline as PatSepConfig:
# daemon boot calls it once for fail-loud validation,
# the coordinator constructor re-invokes it fresh per instantiation so
# pytest monkeypatch.setenv works between cases.
# ---------------------------------------------------------------------------

# Default values. Kept as module-level constants so consumers (and external
# readers) can reference them in docstrings / log messages without
# re-deriving the magic numbers.
_S2_DEFAULT_MIN_INTERVAL_SEC: float = 5.0
_S2_DEFAULT_MAX_RETRY: int = 3

# Boolean-parse vocab for IAI_MCP_S2_DRY_RUN. Case-insensitive match;
# empty-string maps to False explicitly (distinct from absent / None which
# triggers the pytest-aware default below).
_S2_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_S2_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class S2Config:
    """Typed bundle for S2Coordinator runtime knobs.

    Frozen for symmetry with ErasureConfig + PatSepConfig.
    All three fields originate from env vars validated inside
    `_load_s2_config()`.
    """

    min_interval_sec: float
    max_retry: int
    dry_run: bool


def _load_s2_config() -> S2Config:
    """Read the 3 IAI_MCP_S2_* env vars and return a typed S2Config.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip dry_run / min_interval between
    cases; if this helper were cached at import time those overrides would
    never be visible.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # min_interval_sec: float > 0.0 — ring-buffer reverse-direction window.
    raw_min_interval = os.environ.get(
        "IAI_MCP_S2_MIN_INTERVAL_SEC",
        str(_S2_DEFAULT_MIN_INTERVAL_SEC),
    )
    try:
        min_interval_sec = float(raw_min_interval)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_S2_MIN_INTERVAL_SEC: invalid value "
            f"{raw_min_interval!r}, expected float"
        )
    if not (min_interval_sec > 0.0):
        raise ValueError(
            f"IAI_MCP_S2_MIN_INTERVAL_SEC: invalid value "
            f"{raw_min_interval!r}, expected float > 0.0"
        )

    # max_retry: int in [0, 10] — caller-side retry budget on
    # S2OscillationConflict; the coordinator itself never retries.
    raw_max_retry = os.environ.get(
        "IAI_MCP_S2_MAX_RETRY", str(_S2_DEFAULT_MAX_RETRY)
    )
    try:
        max_retry = int(raw_max_retry)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_S2_MAX_RETRY: invalid value {raw_max_retry!r}, "
            f"expected int"
        )
    if not (0 <= max_retry <= 10):
        raise ValueError(
            f"IAI_MCP_S2_MAX_RETRY: invalid value {raw_max_retry!r}, "
            f"expected int in [0, 10]"
        )

    # dry_run: bool — absent → pytest-aware default; present → parse vocab.
    # Per (reused): production default False, pytest default
    # True. The pytest branch is triggered by the standard PYTEST_CURRENT_TEST
    # env var that pytest sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_S2_DRY_RUN")
    if raw_dry_run is None:
        # Absent — apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _S2_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _S2_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_S2_DRY_RUN: invalid value {raw_dry_run!r}, "
                f"expected one of "
                f"{sorted(_S2_DRY_RUN_TRUE_VALUES | _S2_DRY_RUN_FALSE_VALUES)}"
            )

    return S2Config(
        min_interval_sec=min_interval_sec,
        max_retry=max_retry,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# SleepOverhaulConfig + _load_sleep_overhaul_config: typed env-var bundle
# for the sleep-pipeline overhaul (REM/NREM bifurcation +
# cSPW-R cluster replay + step-mechanism + EssentialVariableTracker).
# Same call-on-demand discipline as ErasureConfig, PatSepConfig, and
# S2Config: daemon boot
# calls it once for fail-loud validation; _step_cluster_replay and
# _step_crisis_recluster re-invoke it fresh inside their bodies so
# pytest monkeypatch.setenv works between cases.
# ---------------------------------------------------------------------------

# Rich-club floor: current live 0.10; 0.05 = "halfway to collapse" per
# EssentialVariableTracker crisis threshold.
_SLEEP_OVERHAUL_DEFAULT_RICH_CLUB_RATIO_FLOOR: float = 0.05
# Community-count ceiling ratio: fragmentation breach when |communities| /
# |nodes| exceeds this fraction.
_SLEEP_OVERHAUL_DEFAULT_COMMUNITY_COUNT_CEILING_RATIO: float = 0.9
# Edge-density floor: graph collapse breach when global density drops below
# this value (EssentialVariableTracker floor).
_SLEEP_OVERHAUL_DEFAULT_EDGE_DENSITY_FLOOR: float = 0.001
# cSPW-R cluster replay window in seconds: _step_cluster_replay scans
# recently-touched records inside this rolling window
# (default 5 min mirrors hippocampal sharp-wave-ripple cadence).
_SLEEP_OVERHAUL_DEFAULT_CLUSTER_WINDOW_SEC: int = 300
# Crisis recluster drop quartile: under crisis mode, _step_crisis_recluster
# drops the bottom this-fraction of edges by Hebbian weight before
# re-running community detection.
_SLEEP_OVERHAUL_DEFAULT_CRISIS_DROP_QUARTILE: float = 0.25
# Cluster-replay initial Hebbian weight: the boost_edges delta applied
# per replayed co-occurrence in _step_cluster_replay (locked here as
# a constant; not a tunable env var).
_SLEEP_OVERHAUL_DEFAULT_CLUSTER_REPLAY_INITIAL_WEIGHT: float = 0.05

# Boolean-parse vocab for IAI_MCP_SLEEP_OVERHAUL_DRY_RUN. Case-insensitive
# match; empty-string maps to False explicitly (distinct from absent / None
# which triggers the pytest-aware default below). Mirrors PatSep / S2.
_SLEEP_OVERHAUL_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_SLEEP_OVERHAUL_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class SleepOverhaulConfig:
    """Typed bundle for sleep-overhaul runtime knobs.

    Frozen for symmetry with ErasureConfig + PatSepConfig + S2Config
    All seven fields originate from env vars
    validated inside `_load_sleep_overhaul_config()`.
    """

    rich_club_ratio_floor: float
    community_count_ceiling_ratio: float
    edge_density_floor: float
    cluster_window_sec: int
    crisis_drop_quartile: float
    cluster_replay_initial_weight: float
    dry_run: bool


def _load_sleep_overhaul_config() -> SleepOverhaulConfig:
    """Read the 7 IAI_MCP_* sleep-overhaul env vars and return a typed
    SleepOverhaulConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip the floors / quartile / dry_run
    between cases; if this helper were cached at import time those overrides
    would never be visible inside sleep overhaul (_step_cluster_replay +
    _step_crisis_recluster + EssentialVariableTracker emission path).

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # rich_club_ratio_floor: float in (0.0, 1.0]
    raw_rich_club = os.environ.get(
        "IAI_MCP_RICH_CLUB_RATIO_FLOOR",
        str(_SLEEP_OVERHAUL_DEFAULT_RICH_CLUB_RATIO_FLOOR),
    )
    try:
        rich_club_ratio_floor = float(raw_rich_club)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_RICH_CLUB_RATIO_FLOOR: invalid value "
            f"{raw_rich_club!r}, expected float"
        )
    if not (0.0 < rich_club_ratio_floor <= 1.0):
        raise ValueError(
            f"IAI_MCP_RICH_CLUB_RATIO_FLOOR: invalid value "
            f"{raw_rich_club!r}, expected float in (0.0, 1.0]"
        )

    # community_count_ceiling_ratio: float in (0.0, 1.0]
    raw_comm_ceiling = os.environ.get(
        "IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO",
        str(_SLEEP_OVERHAUL_DEFAULT_COMMUNITY_COUNT_CEILING_RATIO),
    )
    try:
        community_count_ceiling_ratio = float(raw_comm_ceiling)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO: invalid value "
            f"{raw_comm_ceiling!r}, expected float"
        )
    if not (0.0 < community_count_ceiling_ratio <= 1.0):
        raise ValueError(
            f"IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO: invalid value "
            f"{raw_comm_ceiling!r}, expected float in (0.0, 1.0]"
        )

    # edge_density_floor: float in (0.0, 1.0]
    raw_edge_density = os.environ.get(
        "IAI_MCP_EDGE_DENSITY_FLOOR",
        str(_SLEEP_OVERHAUL_DEFAULT_EDGE_DENSITY_FLOOR),
    )
    try:
        edge_density_floor = float(raw_edge_density)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_EDGE_DENSITY_FLOOR: invalid value "
            f"{raw_edge_density!r}, expected float"
        )
    if not (0.0 < edge_density_floor <= 1.0):
        raise ValueError(
            f"IAI_MCP_EDGE_DENSITY_FLOOR: invalid value "
            f"{raw_edge_density!r}, expected float in (0.0, 1.0]"
        )

    # cluster_window_sec: int in [1, 86400] — one second up to 24h.
    raw_window = os.environ.get(
        "IAI_MCP_CLUSTER_WINDOW_SEC",
        str(_SLEEP_OVERHAUL_DEFAULT_CLUSTER_WINDOW_SEC),
    )
    try:
        cluster_window_sec = int(raw_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_CLUSTER_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int"
        )
    if not (1 <= cluster_window_sec <= 86400):
        raise ValueError(
            f"IAI_MCP_CLUSTER_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int in [1, 86400]"
        )

    # crisis_drop_quartile: float in (0.0, 1.0) — strict both ends.
    # Dropping 0% is a no-op crisis; dropping 100% is destruction.
    raw_quartile = os.environ.get(
        "IAI_MCP_CRISIS_DROP_QUARTILE",
        str(_SLEEP_OVERHAUL_DEFAULT_CRISIS_DROP_QUARTILE),
    )
    try:
        crisis_drop_quartile = float(raw_quartile)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_CRISIS_DROP_QUARTILE: invalid value "
            f"{raw_quartile!r}, expected float"
        )
    if not (0.0 < crisis_drop_quartile < 1.0):
        raise ValueError(
            f"IAI_MCP_CRISIS_DROP_QUARTILE: invalid value "
            f"{raw_quartile!r}, expected float in (0.0, 1.0)"
        )

    # cluster_replay_initial_weight: float in (0.0, 1.0]
    raw_replay_weight = os.environ.get(
        "IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT",
        str(_SLEEP_OVERHAUL_DEFAULT_CLUSTER_REPLAY_INITIAL_WEIGHT),
    )
    try:
        cluster_replay_initial_weight = float(raw_replay_weight)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT: invalid value "
            f"{raw_replay_weight!r}, expected float"
        )
    if not (0.0 < cluster_replay_initial_weight <= 1.0):
        raise ValueError(
            f"IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT: invalid value "
            f"{raw_replay_weight!r}, expected float in (0.0, 1.0]"
        )

    # dry_run: bool — absent → pytest-aware default; present → parse vocab.
    # Per (reused via PatSep / S2): production default False,
    # pytest default True. The pytest branch is triggered by the standard
    # PYTEST_CURRENT_TEST env var that pytest sets for the duration of each
    # test.
    raw_dry_run = os.environ.get("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN")
    if raw_dry_run is None:
        # Absent — apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _SLEEP_OVERHAUL_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _SLEEP_OVERHAUL_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_SLEEP_OVERHAUL_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_SLEEP_OVERHAUL_DRY_RUN_TRUE_VALUES | _SLEEP_OVERHAUL_DRY_RUN_FALSE_VALUES)}"
            )

    return SleepOverhaulConfig(
        rich_club_ratio_floor=rich_club_ratio_floor,
        community_count_ceiling_ratio=community_count_ceiling_ratio,
        edge_density_floor=edge_density_floor,
        cluster_window_sec=cluster_window_sec,
        crisis_drop_quartile=crisis_drop_quartile,
        cluster_replay_initial_weight=cluster_replay_initial_weight,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# ReconsolidationConfig + _load_reconsolidation_config: typed env-var bundle
# for schema-bypass + memory-reconsolidation (Tse 2007 mPFC
# schema-fit + Nader 2000 labile-on-retrieval). Same call-on-demand
# discipline as ErasureConfig, PatSepConfig, S2Config, and
# SleepOverhaulConfig: daemon
# boot calls it once for fail-loud validation; insert-time schema-bypass
# tagging and _step_reconsolidation re-invoke it fresh inside their bodies
# so pytest monkeypatch.setenv works between cases. Five env vars:
# SCHEMA_BYPASS_COS_THRESHOLD, LABILE_WINDOW_SEC, RECONSOLIDATION_TIER1,
# RECONSOLIDATION_ERROR_THRESHOLD, RECONSOLIDATION_DRY_RUN.
# ---------------------------------------------------------------------------

# Schema-bypass cosine threshold: Tse 2007 empirical schema-fit cutoff;
# embedding cosine to nearest community centroid ≥ this value marks a record
# as schema-compatible at insert time.
_RECONSOLIDATION_DEFAULT_SCHEMA_BYPASS_COS_THRESHOLD: float = 0.85
# Labile window seconds: Nader 2000 6-hour reconsolidation window; on
# memory_recall hit, record's `labile_until` = now + this many seconds
# (default 21600 = 6h).
_RECONSOLIDATION_DEFAULT_LABILE_WINDOW_SEC: int = 21600
# Tier-1 critic default: locks critic OFF by default in both prod and
# pytest; operator opts in via env. Avoids unbounded LLM spend in REM cycles
# until critic is explicitly enabled.
_RECONSOLIDATION_DEFAULT_TIER1: bool = False
# Reconsolidation error threshold: midpoint [0.0, 1.0] gate for the Tier-1
# critic's prediction_error output; records with err ≥ this fraction get
# their provenance updated (conservative default).
_RECONSOLIDATION_DEFAULT_ERROR_THRESHOLD: float = 0.5
# Production dry-run default: False (mutations land). Under pytest the helper
# flips dry_run to True via the PYTEST_CURRENT_TEST sentinel branch — mirrors
# pytest-aware default.
_RECONSOLIDATION_DEFAULT_DRY_RUN_PRODUCTION: bool = False

# Boolean-parse vocab for the two bool env vars
# (IAI_MCP_RECONSOLIDATION_TIER1 + IAI_MCP_RECONSOLIDATION_DRY_RUN).
# Case-insensitive match; empty-string maps to False explicitly (distinct
# from absent/None which triggers the pytest-aware default for DRY_RUN, and
# the locked-off default for TIER1). Identical vocab to the other bundles.
_RECONSOLIDATION_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_RECONSOLIDATION_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ReconsolidationConfig:
    """Typed bundle for schema-bypass + reconsolidation knobs.

    Frozen for symmetry with ErasureConfig + PatSepConfig + S2Config +
    SleepOverhaulConfig. All five fields
    originate from env vars validated inside `_load_reconsolidation_config()`.
    """

    schema_bypass_cos_threshold: float
    labile_window_sec: int
    reconsolidation_tier1: bool
    reconsolidation_error_threshold: float
    dry_run: bool


def _load_reconsolidation_config() -> ReconsolidationConfig:
    """Read the 5 IAI_MCP_* reconsolidation env vars and return a typed
    ReconsolidationConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip thresholds / tier1 / dry_run
    between cases; if this helper were cached at import time those overrides
    would never be visible inside the schema-bypass + reconsolidation code
    paths.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # schema_bypass_cos_threshold: float in [0.0, 1.0]
    raw_cos = os.environ.get(
        "IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD",
        str(_RECONSOLIDATION_DEFAULT_SCHEMA_BYPASS_COS_THRESHOLD),
    )
    try:
        schema_bypass_cos_threshold = float(raw_cos)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD: invalid value "
            f"{raw_cos!r}, expected float"
        )
    if not (0.0 <= schema_bypass_cos_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD: invalid value "
            f"{raw_cos!r}, expected float in [0.0, 1.0]"
        )

    # labile_window_sec: int > 0
    raw_window = os.environ.get(
        "IAI_MCP_LABILE_WINDOW_SEC",
        str(_RECONSOLIDATION_DEFAULT_LABILE_WINDOW_SEC),
    )
    try:
        labile_window_sec = int(raw_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_LABILE_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int"
        )
    if not (labile_window_sec > 0):
        raise ValueError(
            f"IAI_MCP_LABILE_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int > 0"
        )

    # reconsolidation_tier1: bool — absent → False (: critic off by
    # default in both prod and pytest); present → parse vocab.
    raw_tier1 = os.environ.get("IAI_MCP_RECONSOLIDATION_TIER1")
    if raw_tier1 is None:
        reconsolidation_tier1 = _RECONSOLIDATION_DEFAULT_TIER1
    else:
        normalized_tier1 = raw_tier1.strip().lower()
        if normalized_tier1 in _RECONSOLIDATION_TRUE_VALUES:
            reconsolidation_tier1 = True
        elif normalized_tier1 in _RECONSOLIDATION_FALSE_VALUES:
            reconsolidation_tier1 = False
        else:
            raise ValueError(
                f"IAI_MCP_RECONSOLIDATION_TIER1: invalid value "
                f"{raw_tier1!r}, expected one of "
                f"{sorted(_RECONSOLIDATION_TRUE_VALUES | _RECONSOLIDATION_FALSE_VALUES)}"
            )

    # reconsolidation_error_threshold: float in [0.0, 1.0]
    raw_err = os.environ.get(
        "IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD",
        str(_RECONSOLIDATION_DEFAULT_ERROR_THRESHOLD),
    )
    try:
        reconsolidation_error_threshold = float(raw_err)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD: invalid value "
            f"{raw_err!r}, expected float"
        )
    if not (0.0 <= reconsolidation_error_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD: invalid value "
            f"{raw_err!r}, expected float in [0.0, 1.0]"
        )

    # dry_run: bool — absent → pytest-aware default; present → parse vocab.
    # Per (reused via PatSep / S2 / SleepOverhaul):
    # production default False, pytest default True. The pytest branch is
    # triggered by the standard PYTEST_CURRENT_TEST env var that pytest
    # sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_RECONSOLIDATION_DRY_RUN")
    if raw_dry_run is None:
        # Absent — apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized_dry = raw_dry_run.strip().lower()
        if normalized_dry in _RECONSOLIDATION_TRUE_VALUES:
            dry_run = True
        elif normalized_dry in _RECONSOLIDATION_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_RECONSOLIDATION_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_RECONSOLIDATION_TRUE_VALUES | _RECONSOLIDATION_FALSE_VALUES)}"
            )

    return ReconsolidationConfig(
        schema_bypass_cos_threshold=schema_bypass_cos_threshold,
        labile_window_sec=labile_window_sec,
        reconsolidation_tier1=reconsolidation_tier1,
        reconsolidation_error_threshold=reconsolidation_error_threshold,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# StcConfig + _load_stc_config: typed env-var bundle for the
# Synaptic Tagging-and-Capture (STC) temporal-association feature
# (Frey-Morris 1997 analogue: weak peri-event turns get upgraded from
# semantic -> episodic when a STRONG_EVENT fires inside the buffer's
# rolling time window). Same call-on-demand discipline as ErasureConfig,
# PatSepConfig, S2Config, SleepOverhaulConfig, and ReconsolidationConfig:
# daemon boot calls it once for fail-loud validation;
# (PeriEventBuffer.trigger_stc + the daemon-main singleton-wire site)
# re-invoke fresh inside their bodies so pytest monkeypatch.setenv works
# between cases. Four env vars: PERI_EVENT_BUFFER_SIZE, PERI_EVENT_WINDOW_SEC,
# STC_STRONG_EVENT_TYPES, STC_DRY_RUN.
# ---------------------------------------------------------------------------

# Peri-event ring-buffer size: deque(maxlen=...) of recent capture tuples.
# Default 20 covers ~10-20 min of normal capture rate at the canonical
# 1 turn / 30-60 s cadence.
_STC_DEFAULT_PERI_EVENT_BUFFER_SIZE: int = 20
# Biological reconsolidation window analogue, 30 min default. Frey-Morris
# 1997 STC tagging horizon: weak turns inside this window from a strong
# event get upgraded.
_STC_DEFAULT_PERI_EVENT_WINDOW_SEC: int = 1800
# Event kinds that fire the STC upgrade pass. CSV in env, frozenset[str]
# in config. Tokens are stripped and lowercased on parse; empty tokens
# fail loud (default list).
_STC_DEFAULT_STRONG_EVENT_TYPES: str = "memory_capture,error_trace,user_correction"

# Boolean-parse vocab for IAI_MCP_STC_DRY_RUN. Case-insensitive match;
# empty-string maps to False explicitly (distinct from absent / None which
# triggers the pytest-aware default below). Mirrors PatSep / S2 /
# SleepOverhaul / Reconsolidation.
_STC_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_STC_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class StcConfig:
    """Typed bundle for STC runtime knobs.

    Frozen for symmetry with ErasureConfig + PatSepConfig + S2Config +
    SleepOverhaulConfig + ReconsolidationConfig. All four fields originate
    from env vars validated inside
    `_load_stc_config()`.
    """

    peri_event_buffer_size: int
    peri_event_window_sec: int
    strong_event_types: frozenset[str]
    dry_run: bool


def _load_stc_config() -> StcConfig:
    """Read the 4 IAI_MCP_* STC env vars and return a typed StcConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests (PeriEventBuffer.trigger_stc + the daemon-main singleton-wire
    site) use `monkeypatch.setenv` to flip buffer_size /
    window / strong-event-types / dry_run between cases; if this helper
    were cached at import time those overrides would never be visible
    inside the STC upgrade paths.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # peri_event_buffer_size: int in [1, 1000]
    raw_buffer_size = os.environ.get(
        "IAI_MCP_PERI_EVENT_BUFFER_SIZE",
        str(_STC_DEFAULT_PERI_EVENT_BUFFER_SIZE),
    )
    try:
        peri_event_buffer_size = int(raw_buffer_size)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_BUFFER_SIZE: invalid value "
            f"{raw_buffer_size!r}, expected int"
        )
    if not (1 <= peri_event_buffer_size <= 1000):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_BUFFER_SIZE: invalid value "
            f"{raw_buffer_size!r}, expected int in [1, 1000]"
        )

    # peri_event_window_sec: int in [1, 86400] -- one second up to 24h.
    # Mirrors cluster_window_sec range at L1480 (analog).
    raw_window = os.environ.get(
        "IAI_MCP_PERI_EVENT_WINDOW_SEC",
        str(_STC_DEFAULT_PERI_EVENT_WINDOW_SEC),
    )
    try:
        peri_event_window_sec = int(raw_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int"
        )
    if not (1 <= peri_event_window_sec <= 86400):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int in [1, 86400]"
        )

    # strong_event_types: non-empty CSV -> frozenset of stripped, lowercased
    # tokens. Every post-strip token must be non-empty; empty CSV or
    # all-whitespace input fails loud (fail-loud contract).
    raw_strong = os.environ.get(
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        _STC_DEFAULT_STRONG_EVENT_TYPES,
    )
    tokens = [tok.strip().lower() for tok in raw_strong.split(",")]
    if any(tok == "" for tok in tokens):
        raise ValueError(
            f"IAI_MCP_STC_STRONG_EVENT_TYPES: invalid value {raw_strong!r}, "
            f"expected non-empty comma-separated event-type names"
        )
    strong_event_types = frozenset(tokens)
    if not strong_event_types:
        raise ValueError(
            f"IAI_MCP_STC_STRONG_EVENT_TYPES: invalid value {raw_strong!r}, "
            f"expected non-empty comma-separated event-type names"
        )

    # dry_run: bool -- absent -> pytest-aware default; present -> parse vocab.
    # Per (reused via PatSep / S2 / SleepOverhaul /
    # Reconsolidation): production default False, pytest default True. The
    # pytest branch is triggered by the standard PYTEST_CURRENT_TEST env var
    # that pytest sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_STC_DRY_RUN")
    if raw_dry_run is None:
        # Absent -- apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _STC_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _STC_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_STC_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_STC_DRY_RUN_TRUE_VALUES | _STC_DRY_RUN_FALSE_VALUES)}"
            )

    return StcConfig(
        peri_event_buffer_size=peri_event_buffer_size,
        peri_event_window_sec=peri_event_window_sec,
        strong_event_types=strong_event_types,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# UserModelConfig + _load_user_model_config: typed env-var bundle for the
# user-model + predictive-prefetch. Same
# call-on-demand discipline as ErasureConfig, PatSepConfig, S2Config,
# SleepOverhaulConfig, and ReconsolidationConfig: daemon boot calls it once
# for fail-loud validation; the UserModelAggregator + UserModelPrefetcher
# and _step_user_model_update re-invoke it fresh inside their
# bodies so pytest monkeypatch.setenv works between cases. Four env vars:
# AGGREGATION_WINDOW_DAYS, PREFETCH_TOP_K, PATH, DRY_RUN.
# ---------------------------------------------------------------------------

# Aggregation window days: rolling window the user-model
# aggregator scans for capture turns when re-building the model. Default
# 30 days mirrors a calendar-month behavioural horizon; upper bound 365
# prevents an operator from accidentally requesting a multi-year scan
# that would dominate the REM cycle.
_USER_MODEL_DEFAULT_AGGREGATION_WINDOW_DAYS: int = 30
# Predictive-prefetch top-K: number of model-predicted records the
# prefetcher pulls into the warm LRU at session start. Default 10 keeps
# the warm-up budget within the session-start token target; upper bound
# 100 is a hard ceiling.
_USER_MODEL_DEFAULT_PREFETCH_TOP_K: int = 10
# User-model JSON path: tilde notation stored verbatim in the config
# bundle; consumers expand via os.path.expanduser at the use-site.
# Default lives under ~/.iai-mcp/ alongside the daemon's other state.
_USER_MODEL_DEFAULT_PATH: str = "~/.iai-mcp/user_model.json"
# Production dry-run default: False (mutations land). Under pytest the
# helper flips dry_run to True via the PYTEST_CURRENT_TEST sentinel
# branch -- mirrors pytest-aware default.
_USER_MODEL_DEFAULT_DRY_RUN_PRODUCTION: bool = False

# Boolean-parse vocab for IAI_MCP_USER_MODEL_DRY_RUN. Case-insensitive
# match; empty-string maps to False explicitly (distinct from absent /
# None which triggers the pytest-aware default below). Kept independent
# from the reconsolidation / STC vocabs.
_USER_MODEL_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_USER_MODEL_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class UserModelConfig:
    """Typed bundle for user-model + prefetch knobs.

    Frozen for symmetry with ErasureConfig / PatSepConfig / S2Config /
    SleepOverhaulConfig / ReconsolidationConfig. All four fields originate
    from env vars validated inside
    `_load_user_model_config()`.
    """

    aggregation_window_days: int
    prefetch_top_k: int
    user_model_path: str
    dry_run: bool


def _load_user_model_config() -> UserModelConfig:
    """Read the 4 IAI_MCP_USER_MODEL_* env vars and return a typed
    UserModelConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip the window / top-k / path /
    dry_run between cases; if this helper were cached at import time those
    overrides would never be visible inside the user-model + prefetch code
    paths.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # aggregation_window_days: int in [1, 365]
    raw_window_days = os.environ.get(
        "IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS",
        str(_USER_MODEL_DEFAULT_AGGREGATION_WINDOW_DAYS),
    )
    try:
        aggregation_window_days = int(raw_window_days)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS: invalid value "
            f"{raw_window_days!r}, expected int"
        )
    if not (1 <= aggregation_window_days <= 365):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS: invalid value "
            f"{raw_window_days!r}, expected int in [1, 365]"
        )

    # prefetch_top_k: int in [1, 100]
    raw_top_k = os.environ.get(
        "IAI_MCP_USER_MODEL_PREFETCH_TOP_K",
        str(_USER_MODEL_DEFAULT_PREFETCH_TOP_K),
    )
    try:
        prefetch_top_k = int(raw_top_k)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_PREFETCH_TOP_K: invalid value "
            f"{raw_top_k!r}, expected int"
        )
    if not (1 <= prefetch_top_k <= 100):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_PREFETCH_TOP_K: invalid value "
            f"{raw_top_k!r}, expected int in [1, 100]"
        )

    # user_model_path: str -- empty / absent both fall back to default.
    # Stored verbatim (no expanduser here); consumers expand at use-site.
    # Empty string is treated as "use default" to avoid the footgun where
    # `export IAI_MCP_USER_MODEL_PATH=""` would otherwise persist an
    # empty path.
    raw_path = os.environ.get("IAI_MCP_USER_MODEL_PATH")
    if raw_path is None or raw_path == "":
        user_model_path = _USER_MODEL_DEFAULT_PATH
    else:
        user_model_path = raw_path

    # dry_run: bool -- absent -> pytest-aware default; present -> parse vocab.
    # Per (reused via PatSep / S2 / SleepOverhaul /
    # Reconsolidation / STC): production default False, pytest default True.
    # The pytest branch is triggered by the standard PYTEST_CURRENT_TEST
    # env var that pytest sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_USER_MODEL_DRY_RUN")
    if raw_dry_run is None:
        # Absent -- apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _USER_MODEL_TRUE_VALUES:
            dry_run = True
        elif normalized in _USER_MODEL_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_USER_MODEL_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_USER_MODEL_TRUE_VALUES | _USER_MODEL_FALSE_VALUES)}"
            )

    return UserModelConfig(
        aggregation_window_days=aggregation_window_days,
        prefetch_top_k=prefetch_top_k,
        user_model_path=user_model_path,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# SpatialConfig + _load_spatial_config: typed env-var bundle for the
# spatial-scaffold. Same call-on-demand discipline as
# UserModelConfig. Three env vars:
# AUTO_TAG, DEFAULT_WING, DRY_RUN.
# ---------------------------------------------------------------------------

# Auto-tag toggle: operator opt-in for the SpatialTagger heuristic.
# Default False so an unconfigured deployment never silently routes
# captures into wings; an operator must set the env var to
# explicitly enable spatial tagging.
_SPATIAL_DEFAULT_AUTO_TAG: bool = False
# Default wing: literal fallback string used when SpatialTagger finds no
# matching vocabulary hit or when the operator unsets the override. The
# value "general" mirrors (fallback wing) -- a catch-all
# bucket that survives schema migrations without re-tagging old records.
_SPATIAL_DEFAULT_WING: str = "general"
# Production dry-run default: False (mutations land). Under pytest the
# helper flips dry_run to True via the PYTEST_CURRENT_TEST sentinel
# branch -- mirrors pytest-aware default reused across all config bundles.
_SPATIAL_DEFAULT_DRY_RUN_PRODUCTION: bool = False

# Boolean-parse vocab for IAI_MCP_SPATIAL_AUTO_TAG and
# IAI_MCP_SPATIAL_DRY_RUN. Case-insensitive match; empty-string maps to
# False explicitly (distinct from absent / None which triggers the
# constant / pytest-aware default below). Kept independent from the
# user-model / reconsolidation / STC vocabs.
_SPATIAL_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_SPATIAL_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class SpatialConfig:
    """Typed bundle for spatial-scaffold knobs.

    Frozen for symmetry with ErasureConfig / PatSepConfig / S2Config /
    SleepOverhaulConfig / ReconsolidationConfig / UserModelConfig.
    All three fields originate from env vars validated
    inside `_load_spatial_config()`.
    """

    auto_tag: bool
    default_wing: str
    dry_run: bool


def _load_spatial_config() -> SpatialConfig:
    """Read the 3 IAI_MCP_SPATIAL_* env vars and return a typed
    SpatialConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip auto_tag / default_wing / dry_run
    between cases; if this helper were cached at import time those overrides
    would never be visible inside the spatial code paths.

    Raises ValueError with the offending variable name in the message when
    any value is malformed.
    """
    # auto_tag: bool -- absent -> constant default; present -> parse vocab.
    # Unlike DRY_RUN this is NOT pytest-aware: production and pytest both
    # default to False so a missing operator opt-in cannot accidentally
    # enable spatial routing under tests that do not explicitly set it.
    raw_auto_tag = os.environ.get("IAI_MCP_SPATIAL_AUTO_TAG")
    if raw_auto_tag is None:
        auto_tag = _SPATIAL_DEFAULT_AUTO_TAG
    else:
        normalized = raw_auto_tag.strip().lower()
        if normalized in _SPATIAL_TRUE_VALUES:
            auto_tag = True
        elif normalized in _SPATIAL_FALSE_VALUES:
            auto_tag = False
        else:
            raise ValueError(
                f"IAI_MCP_SPATIAL_AUTO_TAG: invalid value "
                f"{raw_auto_tag!r}, expected one of "
                f"{sorted(_SPATIAL_TRUE_VALUES | _SPATIAL_FALSE_VALUES)}"
            )

    # default_wing: str -- empty / absent both fall back to default.
    # Stored verbatim. Same convention as user_model_path handling: empty
    # string is treated as "use default" to avoid the footgun where
    # `export IAI_MCP_SPATIAL_DEFAULT_WING=""` would otherwise persist
    # an empty wing label.
    raw_wing = os.environ.get("IAI_MCP_SPATIAL_DEFAULT_WING")
    if raw_wing is None or raw_wing == "":
        default_wing = _SPATIAL_DEFAULT_WING
    else:
        default_wing = raw_wing

    # dry_run: bool -- absent -> pytest-aware default; present -> parse vocab.
    # Per (reused via PatSep / S2 / SleepOverhaul /
    # Reconsolidation / STC / UserModel): production default False,
    # pytest default True. The pytest branch is triggered by the standard
    # PYTEST_CURRENT_TEST env var that pytest sets for the duration of
    # each test.
    raw_dry_run = os.environ.get("IAI_MCP_SPATIAL_DRY_RUN")
    if raw_dry_run is None:
        # Absent -- apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _SPATIAL_TRUE_VALUES:
            dry_run = True
        elif normalized in _SPATIAL_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_SPATIAL_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_SPATIAL_TRUE_VALUES | _SPATIAL_FALSE_VALUES)}"
            )

    return SpatialConfig(
        auto_tag=auto_tag,
        default_wing=default_wing,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# DmnConfig + _load_dmn_config: typed env-var bundle for the
# DMN Reflection Agent + Meta-Analyst. Same call-on-demand
# discipline as ErasureConfig, PatSepConfig, S2Config, SleepOverhaulConfig,
# ReconsolidationConfig, StcConfig, UserModelConfig, and SpatialConfig:
# daemon boot calls it once for fail-loud validation; the
# `_step_dmn_reflection` body + ReflectionAgent / MetaAnalyst
# constructors re-invoke it fresh inside their bodies so pytest
# monkeypatch.setenv works between cases. Three env vars:
# REFLECTION_WINDOW_HOURS, META_ANALYST_ENABLED, DRY_RUN.
# ---------------------------------------------------------------------------

# Reflection window hours: trailing window the ReflectionAgent
# scans when synthesizing the daily-narrative engram. Default 24h mirrors
# the Andrews-Hanna autobiographical-recall horizon. Upper bound 720
# (30 days) prevents an operator from accidentally requesting a
# multi-month scan that would dominate the REM cycle.
_DMN_DEFAULT_REFLECTION_WINDOW_HOURS: int = 24
# Meta-Analyst enabled toggle: when False, the DMN_REFLECTION SleepStep
# when False, DMN_REFLECTION still runs ReflectionAgent but skips
# MetaAnalyst.snapshot, so no `system_health_report` event is emitted.
# Default True --
# the Von Foerster second-order observer is the headline feature of the
# phase and should be on out of the box.
_DMN_DEFAULT_META_ANALYST_ENABLED: bool = True
# Production dry-run default: False (mutations land). Under pytest the
# helper flips dry_run to True via the PYTEST_CURRENT_TEST sentinel
# branch -- mirrors pytest-aware default reused across all config bundles.
_DMN_DEFAULT_DRY_RUN_PRODUCTION: bool = False

# Boolean-parse vocab for IAI_MCP_META_ANALYST_ENABLED and
# IAI_MCP_DMN_DRY_RUN. Case-insensitive match; empty-string maps to
# False explicitly (distinct from absent / None which triggers the
# constant / pytest-aware default below). Kept independent from the
# spatial / user-model / reconsolidation / STC vocabs.
_DMN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_DMN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class DmnConfig:
    """Typed bundle for DMN Reflection + Meta-Analyst knobs.

    Frozen for symmetry with ErasureConfig / PatSepConfig / S2Config /
    SleepOverhaulConfig / ReconsolidationConfig / StcConfig /
    UserModelConfig / SpatialConfig. All three fields originate from env
    vars validated inside `_load_dmn_config()`.
    """

    reflection_window_hours: int
    meta_analyst_enabled: bool
    dry_run: bool


def _load_dmn_config() -> DmnConfig:
    """Read the 3 IAI_MCP_DMN_* / IAI_MCP_META_ANALYST_* env vars and
    return a typed DmnConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip reflection_window_hours /
    meta_analyst_enabled / dry_run between cases; if this helper were
    cached at import time those overrides would never be visible inside
    the DMN code paths.

    Raises ValueError with the offending variable name in the message when
    any value is malformed or out of range.
    """
    # reflection_window_hours: int in [1, 720]
    raw_window_hours = os.environ.get(
        "IAI_MCP_DMN_REFLECTION_WINDOW_HOURS",
        str(_DMN_DEFAULT_REFLECTION_WINDOW_HOURS),
    )
    try:
        reflection_window_hours = int(raw_window_hours)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_DMN_REFLECTION_WINDOW_HOURS: invalid value "
            f"{raw_window_hours!r}, expected int"
        )
    if not (1 <= reflection_window_hours <= 720):
        raise ValueError(
            f"IAI_MCP_DMN_REFLECTION_WINDOW_HOURS: invalid value "
            f"{raw_window_hours!r}, expected int in [1, 720]"
        )

    # meta_analyst_enabled: bool -- absent -> constant default True;
    # present -> parse vocab. Unlike DRY_RUN this is NOT pytest-aware:
    # production and pytest both default to True so the toggle behaviour
    # (false -> no health report) requires an explicit opt-out.
    raw_meta_enabled = os.environ.get("IAI_MCP_META_ANALYST_ENABLED")
    if raw_meta_enabled is None:
        meta_analyst_enabled = _DMN_DEFAULT_META_ANALYST_ENABLED
    else:
        normalized = raw_meta_enabled.strip().lower()
        if normalized in _DMN_TRUE_VALUES:
            meta_analyst_enabled = True
        elif normalized in _DMN_FALSE_VALUES:
            meta_analyst_enabled = False
        else:
            raise ValueError(
                f"IAI_MCP_META_ANALYST_ENABLED: invalid value "
                f"{raw_meta_enabled!r}, expected one of "
                f"{sorted(_DMN_TRUE_VALUES | _DMN_FALSE_VALUES)}"
            )

    # dry_run: bool -- absent -> pytest-aware default; present -> parse vocab.
    # Production default False, pytest default True. The pytest branch is
    # triggered by the standard PYTEST_CURRENT_TEST env var that pytest
    # sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_DMN_DRY_RUN")
    if raw_dry_run is None:
        # Absent -- apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _DMN_TRUE_VALUES:
            dry_run = True
        elif normalized in _DMN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_DMN_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_DMN_TRUE_VALUES | _DMN_FALSE_VALUES)}"
            )

    return DmnConfig(
        reflection_window_hours=reflection_window_hours,
        meta_analyst_enabled=meta_analyst_enabled,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# PaskConfig + _load_pask_config: typed env-var bundle for the Pask
# teach-back loop. Same call-on-demand discipline as ErasureConfig,
# PatSepConfig, S2Config, SleepOverhaulConfig, ReconsolidationConfig,
# StcConfig, UserModelConfig, SpatialConfig, and DmnConfig: daemon boot
# calls it once for fail-loud validation; PaskAgent body, sleep step, and
# integration tests re-invoke it fresh so pytest monkeypatch.setenv works
# between cases. Two env vars: PASK_ENABLED, PASK_DRY_RUN.
# ---------------------------------------------------------------------------

# Pask-loop enabled toggle: when False, the PASK_TEACHBACK SleepStep
# becomes a no-op -- no teach-back synthesis, no comprehension probe,
# no `pask_loop_report` event. Default True -- the second-order
# self-modeling probe should be on out of the box. Symmetric with
# _DMN_DEFAULT_META_ANALYST_ENABLED and the other "on by default"
# toggles across config bundles.
_PASK_DEFAULT_ENABLED: bool = True
# Production dry-run default: False (mutations land). Under pytest the
# helper flips dry_run to True via the PYTEST_CURRENT_TEST sentinel
# branch -- mirrors pytest-aware default reused across all config bundles.
_PASK_DEFAULT_DRY_RUN_PRODUCTION: bool = False

# Boolean-parse vocab for IAI_MCP_PASK_ENABLED and IAI_MCP_PASK_DRY_RUN.
# Case-insensitive match; empty-string maps to False explicitly (distinct
# from absent / None which triggers the constant / pytest-aware default
# below). Kept independent from the dmn / spatial / user-model / recon /
# stc vocabs.
_PASK_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_PASK_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class PaskConfig:
    """Typed bundle for FINAL Pask teach-back loop knobs.

    Frozen for symmetry with ErasureConfig / PatSepConfig / S2Config /
    SleepOverhaulConfig / ReconsolidationConfig / StcConfig /
    UserModelConfig / SpatialConfig / DmnConfig. Both fields originate
    from env vars validated inside `_load_pask_config()`.
    """

    enabled: bool
    dry_run: bool


def _load_pask_config() -> PaskConfig:
    """Read the 2 IAI_MCP_PASK_* env vars and return a typed PaskConfig.

    Call-on-demand: every invocation re-reads `os.environ` from scratch.
    No module-level cache, no `functools.lru_cache`, no import-time freeze.
    Tests use `monkeypatch.setenv` to flip enabled / dry_run between
    cases; if this helper were cached at import time those overrides
    would never be visible inside the Pask code paths.

    Raises ValueError with the offending variable name in the message when
    any value is malformed.
    """
    # enabled: bool -- absent -> constant default True; present -> parse vocab.
    # Unlike DRY_RUN this is NOT pytest-aware: production and pytest both
    # default to True so the toggle behaviour (false -> no
    # pask_loop_report) requires an explicit opt-out.
    raw_enabled = os.environ.get("IAI_MCP_PASK_ENABLED")
    if raw_enabled is None:
        enabled = _PASK_DEFAULT_ENABLED
    else:
        normalized = raw_enabled.strip().lower()
        if normalized in _PASK_TRUE_VALUES:
            enabled = True
        elif normalized in _PASK_FALSE_VALUES:
            enabled = False
        else:
            raise ValueError(
                f"IAI_MCP_PASK_ENABLED: invalid value "
                f"{raw_enabled!r}, expected one of "
                f"{sorted(_PASK_TRUE_VALUES | _PASK_FALSE_VALUES)}"
            )

    # dry_run: bool -- absent -> pytest-aware default; present -> parse vocab.
    # Production default False, pytest default True. The pytest branch is
    # triggered by the standard PYTEST_CURRENT_TEST env var that pytest
    # sets for the duration of each test.
    raw_dry_run = os.environ.get("IAI_MCP_PASK_DRY_RUN")
    if raw_dry_run is None:
        # Absent -- apply pytest-aware default (reused).
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _PASK_TRUE_VALUES:
            dry_run = True
        elif normalized in _PASK_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_PASK_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_PASK_TRUE_VALUES | _PASK_FALSE_VALUES)}"
            )

    return PaskConfig(
        enabled=enabled,
        dry_run=dry_run,
    )
