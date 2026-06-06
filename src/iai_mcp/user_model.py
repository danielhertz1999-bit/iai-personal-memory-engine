# Wire format + safe persistence for ~/.iai-mcp/user_model.json.
# The daemon import is LAZY inside _resolve_path (no top-level coupling,
# no circular-import risk).
# save() mirrors daemon_state.save_state -- tempfile + fsync +
# chmod 0o600 + os.replace (POSIX atomic rename). Crash-mid-write
# leaves the prior file intact.
"""UserModel dataclass + load/save/default helpers.

This module is the persisted wire format for the user-model pipeline.
The dataclass is consumed at session-start by the prefetcher and
refreshed by the REM sleep aggregator. This module contains ONLY the
dataclass and its persistence helpers.

Persistence properties:
- First-run load() (no file on disk) returns default().
- save+load round-trip is lossless across every field, including
  ``time_of_day_pattern`` int dict keys which JSON serializes as strings
  and load() coerces back to int.
- Persisted file mode is 0o600 (user-only) -- chmod is applied to the
  temp file BEFORE the atomic rename so the visible file is never
  world-readable, even transiently.
- Crash mid-write leaves the prior file intact (POSIX os.replace
  guarantee on the same filesystem -- temp file lives in the same
  directory as the target).
- Corrupt JSON / unreadable file / malformed timestamp -> return
  default() (self-heal, matches daemon_state.load_state pattern).
- Missing field keys -> default value for that field (forward-compat
  for future field additions).

Example JSON wire format on disk::

    {
      "top_recent_topics": ["python async", "torchhd hdc"],
      "tool_usage_freq": {"memory_recall": 42, "memory_capture": 7},
      "time_of_day_pattern": {"9": 5, "14": 12},
      "recent_projects": [],
      "last_updated": "2026-05-16T08:42:13.512+00:00",
      "aggregation_window_days": 30
    }

Note that ``time_of_day_pattern`` keys are JSON strings ("9", "14") in
the file but ``int`` (9, 14) in the in-memory dataclass -- load()
converts on read.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID


# Lazy import: the daemon module imports many heavy things at the top level
# (native Rust extension, networkx dev-oracle, etc.) and at least one daemon
# code path imports user_model.py back. Keeping the daemon import inside the
# function body breaks the cycle and keeps this module cheap to import in
# isolation (e.g. from tests, from the bank-recall CLI fallback).
def _resolve_path() -> Path:
    """Return the on-disk path for the persisted UserModel.

    Reads ``IAI_MCP_USER_MODEL_PATH`` (with fallback default) via the
    daemon's typed env-var bundle so the rules for empty / absent /
    malformed are exactly the ones used everywhere else in the daemon.
    """
    # Lazy import -- a top-level ``from iai_mcp.daemon import...``
    # would create a cycle with daemon.py and force every importer of
    # user_model.py to pay the daemon's full import cost.
    from iai_mcp.daemon_config import _load_user_model_config

    cfg = _load_user_model_config()
    return Path(os.path.expanduser(cfg.user_model_path))


@dataclass
class UserModel:
    """Persisted snapshot of the user's recent activity profile.

    Fields (filled in by UserModelAggregator from in-store events /
    records over a configurable rolling window):

    - ``top_recent_topics``: short labels per community_id cluster
      (first ~50 chars of the most-representative literal_surface).
    - ``tool_usage_freq``: per-tool call counts.
    - ``time_of_day_pattern``: hour-of-day (0-23) -> query count
      histogram. Keys are real Python ``int``; the JSON file
      represents them as strings and load() converts back.
    - ``recent_projects``: empty in v1; populated once a
      ``project_marker`` event type lands.
    - ``last_updated``: timezone-aware UTC datetime of the last
      aggregator pass.
    - ``aggregation_window_days``: rolling-window size the last
      aggregator pass used (audit trail; mirrors the
      ``aggregation_window_days`` config knob).
    """

    top_recent_topics: list[str] = field(default_factory=list)
    tool_usage_freq: dict[str, int] = field(default_factory=dict)
    time_of_day_pattern: dict[int, int] = field(default_factory=dict)
    recent_projects: list[str] = field(default_factory=list)
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    aggregation_window_days: int = 30
    # Meta-learning: plasticity multiplier for DREAM_DECAY.
    # >1.0 = faster decay (crisis), <1.0 = slower decay (stable), 1.0 = neutral.
    plasticity_gain: float = 1.0
    # Soft knobs that modulate the sealed 11-knob registry.
    # Keys are knob names, values are float multipliers.
    # The system can evolve these without touching the sealed registry.
    soft_knobs: dict[str, float] = field(default_factory=dict)


def default() -> UserModel:
    """Return a fresh UserModel with empty containers.

    ``aggregation_window_days = 30`` mirrors the schema default. If the
    deployed env var ``IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS`` has
    been changed, default() still returns 30 -- the next aggregator
    pass overwrites the field with the configured value, so
    the audit trail in the persisted file always reflects the window
    used to produce the snapshot, not the schema default.
    """
    return UserModel()


def load() -> UserModel:
    """Load the persisted UserModel; self-heal to default() on any
    failure mode.

    Self-heal triggers (return default()):
    - File does not exist (first run).
    - File exists but is unreadable (OSError).
    - File exists but is not valid JSON.
    - File parses but ``last_updated`` is missing / malformed.

    Field-level forward-compat: missing keys in the JSON map to the
    dataclass field default, so a future field addition will not
    invalidate older snapshots on disk.
    """
    path = _resolve_path()
    if not path.exists():
        return default()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Corrupt / unreadable -- self-heal; next save() writes fresh.
        return default()

    if not isinstance(data, dict):
        # JSON valid but not the expected object shape -- self-heal.
        return default()

    # last_updated: must be a parseable ISO timestamp.
    raw_ts = data.get("last_updated")
    try:
        last_updated = datetime.fromisoformat(raw_ts) if raw_ts else (
            datetime.now(timezone.utc)
        )
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return default()

    # time_of_day_pattern: JSON serializes dict keys as str; coerce
    # back to int. Defensive against non-int-castable keys / values.
    raw_tod = data.get("time_of_day_pattern", {}) or {}
    try:
        time_of_day_pattern = {int(k): int(v) for k, v in raw_tod.items()}
    except (TypeError, ValueError):
        time_of_day_pattern = {}

    return UserModel(
        top_recent_topics=list(data.get("top_recent_topics", []) or []),
        tool_usage_freq=dict(data.get("tool_usage_freq", {}) or {}),
        time_of_day_pattern=time_of_day_pattern,
        recent_projects=list(data.get("recent_projects", []) or []),
        last_updated=last_updated,
        aggregation_window_days=int(
            data.get("aggregation_window_days", 30)
        ),
    )


def save(model: UserModel) -> None:
    """Atomically persist ``model`` to the resolved path.

    Structural copy of ``daemon_state.save_state``.

    Semantics:
    - Creates parent dir if missing.
    - Writes to a sibling temp file in the same directory (required so
      os.replace can do an atomic rename on the same filesystem).
    - fsync the file contents before rename so the data is on disk.
    - chmod 0o600 BEFORE the swap so the visible file is never
      world-readable, even transiently.
    - On exception: unlink the temp file so the parent dir does not
      accumulate stragglers.
    """
    path = _resolve_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".user-model.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        payload = {
            "top_recent_topics": list(model.top_recent_topics),
            "tool_usage_freq": dict(model.tool_usage_freq),
            # JSON has no int-key dicts; persist as-is (json.dump will
            # str-ify), load() coerces back to int.
            "time_of_day_pattern": dict(model.time_of_day_pattern),
            "recent_projects": list(model.recent_projects),
            "last_updated": model.last_updated.isoformat(),
            "aggregation_window_days": int(model.aggregation_window_days),
        }
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Aggregator + Prefetcher
# ---------------------------------------------------------------------------
# Read side (UserModelAggregator) and consume side (UserModelPrefetcher) of the
# UserModel document. Both are pure functions over (store, UserModel) -- no I/O
# beyond store reads, no save() side effect. The REM SleepStep handler
# is the only writer.
#
# Topic extraction: NOT NLP. Topics come from the community_id
# distribution over decrypted records; each community is labelled by the first
# 50 chars of the most-recent record's literal_surface. Cheap enough for a
# daily REM pass; surprise feedback tunes the rank on the next pass.
#
# community_id is UUID|None on MemoryRecord. Records with None are skipped
# -- they haven't been clustered yet and have no label to contribute.


# helper: trim the topic label to 50 chars + drop trailing whitespace.
# Hard-cap on what leaks to disk -- the JSON file should never carry an
# entire record's prose, only enough of a prefix to be a readable label.
def _first_50_chars(s: str) -> str:
    """Return the first 50 chars of ``s`` with trailing whitespace stripped.

    Empty / non-string input returns the empty string so the caller can safely
    use the result as a dict key / list element without a guard.
    """
    if not s or not isinstance(s, str):
        return ""
    return s[:50].rstrip()


class UserModelAggregator:
    """REM-phase pass that rebuilds a UserModel from recent events and records.

    Stateless (the class exists purely for namespacing + future-extension
    hooks). The caller (the SleepStep handler) is responsible for persistence;
    ``aggregate()`` returns a fresh UserModel and never touches
    disk on its own.

    Algorithm:
      1. Walk decrypted records (store.all_records); bucket by community_id.
         None community_id => skipped (not yet clustered).
      2. Label each community by ``_first_50_chars`` of the top-1 record by
         ``created_at`` -- the most recent representative is more likely to
         match what the user is currently working on than an old one.
      3. Read ``user_model_surprise`` events in the window; each event boosts
         the matching label's effective count by +1 BEFORE the sort. Labels
         that don't match any current community (out-of-vocab) are ignored.
      4. Sort labels by effective count desc; keep top 10.
      5. Walk all events in the window for tool_usage_freq + hour-of-day
         bucketing. ``data["tool"]`` if present, else ``event["kind"]``.
      6. Walk ``project_marker`` events for ``recent_projects`` (empty if
         none -- v1 ships without a project_marker emitter; the read path
         is forward-compat).
    """

    def aggregate(
        self, store, window_days: int | None = None
    ) -> UserModel:
        # Lazy imports: importing daemon / events at module top level would
        # create a cycle (daemon imports user_model) and force every importer
        # of user_model.py (e.g. tests, bank-recall CLI) to pay the daemon's
        # full native Rust embedder boot cost.
        from iai_mcp.daemon_config import _load_user_model_config
        from iai_mcp.events import query_events

        if window_days is None:
            window_days = _load_user_model_config().aggregation_window_days
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)

        # === topics from community_id ==================================
        # store.all_records() returns decrypted plaintext records; the
        # raw to_pandas() path on RECORDS_TABLE would hand back AES-256-GCM
        # ciphertext for literal_surface, so we MUST go through
        # all_records() here to get decrypted text.
        recs = store.all_records()
        community_to_records: dict[UUID, list] = {}
        for r in recs:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            community_to_records.setdefault(cid, []).append(r)

        community_counts: dict[UUID, int] = {
            cid: len(rlist) for cid, rlist in community_to_records.items()
        }

        # Label each community by its most-recent record's first 50 chars.
        community_labels: dict[UUID, str] = {}
        for cid, rlist in community_to_records.items():
            top = max(
                rlist, key=lambda r: getattr(r, "created_at", now)
            )
            community_labels[cid] = _first_50_chars(
                getattr(top, "literal_surface", "")
            )

        # === surprise boost ====================================
        # Each user_model_surprise event in the window contributes +1 to
        # the label named by ``data["actual_topic"]``. Labels not in the
        # current community vocabulary are ignored (out-of-vocab).
        surprises = query_events(
            store,
            kind="user_model_surprise",
            since=cutoff,
            limit=1000,
        )
        surprise_boost: Counter = Counter()
        for ev in surprises:
            actual = (ev.get("data") or {}).get("actual_topic")
            if isinstance(actual, str):
                surprise_boost[actual] += 1

        effective_counts: dict[UUID, int] = {}
        for cid, base_count in community_counts.items():
            label = community_labels.get(cid, "")
            effective_counts[cid] = base_count + surprise_boost.get(
                label, 0
            )
        sorted_cids = sorted(
            effective_counts.keys(),
            key=lambda c: effective_counts[c],
            reverse=True,
        )
        top_recent_topics = [
            community_labels[c]
            for c in sorted_cids[:10]
            if community_labels.get(c)
        ]

        # === tool_usage_freq + time_of_day_pattern ===================
        events_list = query_events(store, since=cutoff, limit=10000)
        tool_counter: Counter = Counter()
        hour_counter: Counter = Counter()
        for ev in events_list:
            data = ev.get("data") or {}
            tool = data.get("tool")
            key = tool if isinstance(tool, str) else ev.get("kind")
            if isinstance(key, str):
                tool_counter[key] += 1

            ts = ev.get("ts")
            if ts is None:
                continue
            # ts may be a pandas Timestamp (from to_pandas) or a tz-aware /
            # naive datetime. Normalise defensively -- a single malformed
            # row should not crash the whole pass.
            try:
                py = ts.to_pydatetime() if hasattr(
                    ts, "to_pydatetime"
                ) else ts
            except (TypeError, ValueError, AttributeError):
                continue
            if getattr(py, "tzinfo", None) is None:
                try:
                    py = py.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError, AttributeError):
                    continue
            try:
                hour_counter[int(py.hour)] += 1
            except (TypeError, ValueError, AttributeError):
                continue
        tool_usage_freq = dict(tool_counter.most_common(20))
        time_of_day_pattern = dict(hour_counter)

        # === recent_projects =========================================
        project_events = query_events(
            store,
            kind="project_marker",
            since=cutoff,
            limit=1000,
        )
        projects: set[str] = set()
        for ev in project_events:
            p = (ev.get("data") or {}).get("project")
            if isinstance(p, str) and p:
                projects.add(p)
        recent_projects = sorted(projects)

        return UserModel(
            top_recent_topics=top_recent_topics,
            tool_usage_freq=tool_usage_freq,
            time_of_day_pattern=time_of_day_pattern,
            recent_projects=recent_projects,
            last_updated=now,
            aggregation_window_days=window_days,
        )


class UserModelPrefetcher:
    """SessionStart pre-warm. Given a loaded UserModel and the current
    wall-clock, pick top-K records likely to be relevant.

    Scoring for records with non-None community_id:
      - +``(10 - rank_idx) / 10`` if the record's community label appears in
        ``model.top_recent_topics``. Rank 0 = highest weight (1.0), rank 9 =
        lowest (0.1). Ensures the bias scales smoothly with rank.
      - +0.3 if ``record.created_at.hour`` is among the top 3 hours of
        ``model.time_of_day_pattern``. A flat bonus, not rank-weighted --
        the user's habitual hours are a coarse signal.

    Records with score 0 are dropped (no topic match AND no hour match) --
    they'd just dilute the prefetch payload. Sort: score desc, then
    ``created_at`` desc as tiebreak. Returns ``[str(record.id),...]``.

    Empty model (no topics and no hour pattern) returns ``[]`` -- there is
    nothing to predict yet, and the SessionStart integration falls
    through to the normal recall path uncontaminated.
    """

    def prefetch(
        self,
        store,
        model: UserModel,
        top_k: int | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        # lazy import -- see UserModelAggregator.aggregate.
        from iai_mcp.daemon_config import _load_user_model_config

        if top_k is None:
            top_k = _load_user_model_config().prefetch_top_k
        if now is None:
            now = datetime.now(timezone.utc)

        # Empty model => nothing to predict yet.
        if (
            not model.top_recent_topics
            and not model.time_of_day_pattern
        ):
            return []

        topic_rank: dict[str, int] = {
            t: i for i, t in enumerate(model.top_recent_topics)
        }

        # Top-3 hours by count drive the hour-of-day bonus.
        top_hours: set[int] = set()
        if model.time_of_day_pattern:
            sorted_hours = sorted(
                model.time_of_day_pattern.keys(),
                key=lambda h: model.time_of_day_pattern[h],
                reverse=True,
            )
            top_hours = set(sorted_hours[:3])

        recs = store.all_records()
        # Re-derive community labels live (same algorithm as the
        # aggregator) so the prefetcher matches whatever vocabulary the
        # current store actually has, even if the persisted model is
        # slightly stale.
        community_to_records: dict[UUID, list] = {}
        for r in recs:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            community_to_records.setdefault(cid, []).append(r)

        community_labels: dict[UUID, str] = {}
        for cid, rlist in community_to_records.items():
            top = max(
                rlist, key=lambda r: getattr(r, "created_at", now)
            )
            community_labels[cid] = _first_50_chars(
                getattr(top, "literal_surface", "")
            )

        scored: list[tuple[float, datetime, str]] = []
        for r in recs:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            label = community_labels.get(cid, "")
            score = 0.0
            if label and label in topic_rank:
                rank = topic_rank[label]
                score += (10 - rank) / 10.0
            created = getattr(r, "created_at", None)
            if created is not None:
                try:
                    if getattr(created, "tzinfo", None) is None:
                        created = created.replace(tzinfo=timezone.utc)
                    if created.hour in top_hours:
                        score += 0.3
                except (TypeError, ValueError, AttributeError):
                    pass
            if score > 0.0:
                scored.append((score, created or now, str(r.id)))

        # Sort by score desc, then created desc.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [rid for (_, _, rid) in scored[:top_k]]


# Surprise-tracking single-event emit. No in-memory state -- events ARE
# the state. Next aggregate pass reads query_events(kind="user_model_surprise")
# and boosts actual_topic weight by count
# (logic lives in UserModelAggregator.aggregate).
def record_surprise(
    store,
    predicted_topic: str,
    actual_topic: str,
) -> None:
    """Emit a single user_model_surprise event.

    No aggregation buffer, no caching. Each call is one
    event. The next REM aggregation pass picks them up via
    query_events and boosts actual_topic in top_recent_topics.

    Dry-run state is tagged on the event for trajectory diagnostics.
    """
    from iai_mcp.daemon_config import _load_user_model_config
    from iai_mcp.events import write_event

    cfg = _load_user_model_config()
    write_event(
        store,
        "user_model_surprise",
        {
            "predicted_topic": str(predicted_topic),
            "actual_topic": str(actual_topic),
            "dry_run_mode": bool(cfg.dry_run),
        },
        severity="info",
    )
