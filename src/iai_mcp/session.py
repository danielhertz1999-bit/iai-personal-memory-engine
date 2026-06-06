"""Session-start assembler (budget, continuity).

Produces the 4-segment cached prefix that Claude's MCP wrapper places in front
of every request under Anthropic 1h-TTL prompt caching:

    L0 -- pinned identity kernel (always includes the user's L0 record)
    L1 -- critical-facts block (pinned + high-detail records)
    L2[...] -- community summaries (top MAX_TOP_COMMUNITIES=7)
    rich_club -- global hub prefetch (rich-club nodes)

assemble_session_start emits ``kind='session_started'`` with a deterministic
``session_state_hash`` so context-repeat-rate can be computed live.

Budget breakdown:
    L0_BUDGET_TOKENS = 80
    L1_BUDGET_TOKENS = 200
    L2_PER_COMMUNITY_TOKENS = 50 (cap of 7 -> L2 totals ~350 tok)
    RICH_CLUB_BUDGET_TOKENS = 1500
    TOTAL_CACHED_BUDGET = 2000
    (plus ~1000 tok dynamic tail -> steady-state <= 3000)

Tokens are counted via a local `_approx_tokens(text) = max(1, len(text) // 4)`
heuristic that matches Anthropic's documented rough ratio; bench/tokens.py
uses the tiktoken cl100k offline proxy for cross-validation.

 observable: `payload.l0` always contains the user's configured identity
when the pinned L0 record is present, so the verifier can assert identity
continuity on a fresh session open.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from uuid import UUID

from iai_mcp.aaak import generate_aaak_index
from iai_mcp.community import CommunityAssignment
from iai_mcp.handle import decode_compact_handle, encode_compact_handle
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------- render-time hygiene

# ANSI CSI/SGR sequences: ESC [ <params> <final-byte>
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Harness/command XML marker names to strip.
_MARKER_NAMES = (
    "command-name",
    "local-command-stdout",
    "local-command-caveat",
    "command-message",
    "command-args",
    "task-notification",
    "task-id",
)

# Pre-compile one pair of patterns (well-formed then dangling-open) per name.
_MARKER_PATTERNS: list[tuple[re.Pattern, re.Pattern, re.Pattern]] = []
for _name in _MARKER_NAMES:
    _well_formed = re.compile(
        r"<" + re.escape(_name) + r"(?:\s[^>]*)?>.*?</" + re.escape(_name) + r">",
        re.DOTALL,
    )
    # Dangling open: opening angle bracket through end-of-string (no closing >).
    _dangling = re.compile(r"<" + re.escape(_name) + r".*", re.DOTALL)
    # Leftover bare close tag after well-formed strip.
    _close_tag = re.compile(r"</" + re.escape(_name) + r">")
    _MARKER_PATTERNS.append((_well_formed, _dangling, _close_tag))


def _clean_surface(text: str) -> str:
    """Strip ANSI codes and harness XML markers from a literal surface string.

    Applied render-time before the ``[:N]`` truncation slice. The stored
    ``literal_surface`` is never touched — only the in-memory copy used to
    build the brief is cleaned.

    Strip order:
    1. ANSI SGR / CSI escape sequences.
    2. For each harness marker name: remove well-formed ``<name>…</name>``
       blocks first (DOTALL, non-greedy); then remove any dangling
       ``<name…`` opener (with no closing ``>``) through end-of-string.
    3. Collapse internal whitespace runs (including newlines) to a single
       space; strip leading/trailing whitespace.
    """
    if not text:
        return ""
    # 1. ANSI escape sequences.
    text = _ANSI_RE.sub("", text)
    # 2. Harness markers.
    for well_formed, dangling, close_tag in _MARKER_PATTERNS:
        text = well_formed.sub("", text)
        text = dangling.sub("", text)
        text = close_tag.sub("", text)
    # 3. Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------- budgets
L0_BUDGET_TOKENS = 80
L1_BUDGET_TOKENS = 200
L2_PER_COMMUNITY_TOKENS = 50
L2_COMMUNITY_CAP = 7          # community-summary cap
RICH_CLUB_BUDGET_TOKENS = 1500
TOTAL_CACHED_BUDGET = 2000    # L0 + L1 + L2 + rich_club <= this
DYNAMIC_TAIL_TOKENS = 1000    # reserve for per-turn tool results

# Pinned L0 UUID (matches core._seed_l0_identity).
L0_RECORD_UUID = UUID("00000000-0000-0000-0000-000000000001")

# Maximum character length for the rendered session-start markdown payload.
# Used both by the daemon precache writer and by session_refresh_if_stale to
# cap the returned brief before handing it to the caller.
SESSION_START_CACHE_MAX_CHARS: int = 10_000


# --------------------------------------------------------------- data shape


@dataclass
class SessionStartPayload:
    """Cached prefix + metadata (lazy fields for delta encoding).

    `breakpoint_marker` is where the TS wrapper splits stable vs volatile
    content before applying Anthropic `cache_control`. The Python side never
    inserts it into the segment strings -- it's just a sentinel string the TS
    side recognises.

    Three new pointer fields are populated at
    `wake_depth=minimal` (the new default); legacy l0/l1/l2/rich_club left
    empty at minimal mode. `wake_depth` is echoed so the client knows
    which mode produced the payload.
    """

    l0: str = ""
    l1: str = ""
    l2: list[str] = field(default_factory=list)
    rich_club: str = ""
    total_cached_tokens: int = 0
    total_dynamic_tokens: int = 0
    breakpoint_marker: str = "--<cache-breakpoint>--"
    # — lazy session-start fields (<=30 raw tok combined).
    identity_pointer: str = ""       # "<id:{8-hex-of-L0-uuid}>" (~8 tok)
    brain_handle: str = ""           # "<sess:{8-hex} pend:{N}>" (~12 tok)
    topic_cluster_hint: str = ""     # "<topic:{community_label}>" (~8 tok)
    # — single compact handle, ≤16 raw tok target. At
    # `wake_depth=minimal` this supersedes the three legacy pointers above
    # (they are left empty to keep the budget tight); `standard`/`deep`
    # populate BOTH the compact handle and the legacy fields for back-compat.
    compact_handle: str = ""         # "<iai:{16-hex-blake2s}>" (~6-10 raw tok)
    wake_depth: str = "minimal"      # echoed for introspection
    # Render-time most-recent-thread summary. Populated at standard/deep by
    # _recent_thread_segment; left empty at minimal. NOT included in
    # _session_state_hash (changes every session; including it would break
    # the context-repeat signal).
    recent_thread: str = ""


# ---------------------------------------------------------- token counting


def _approx_tokens(text: str) -> int:
    """~4 chars per token heuristic (Anthropic documentation ballpark).

    Minimum 1 for any non-empty text so callers don't divide-by-zero.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ----------------------------------------------------------------- helpers


def _resolve_compact_handle_to_pointers(handle: str) -> tuple[str, str, str] | None:
    """Rebuild the legacy (identity_pointer, brain_handle, topic_cluster_hint)
    triple from a compact ``<iai:HHHHHHHHHHHHHHHH>`` handle minted earlier in
    this process.

     no-info-loss proof: everything the 3-field shape conveyed is
    recoverable from the compact handle via the LRU in ``iai_mcp.handle`` ---
    identity prefix, session prefix, topic label and pending count. Returns
    ``None`` when the handle is malformed OR the LRU has evicted the record,
    mirroring ``decode_compact_handle``'s contract: callers that need strict
    resolution should keep the legacy fields available under
    ``wake_depth=standard`` / ``deep`` as fallback.
    """
    parts = decode_compact_handle(handle)
    if parts is None:
        return None
    identity_pointer = f"<id:{parts[0]}>" if parts[0] else ""
    brain_handle = f"<sess:{parts[1]} pend:{parts[3]}>"
    topic_cluster_hint = f"<topic:{parts[2]}>"
    return identity_pointer, brain_handle, topic_cluster_hint


def _fetch_record(store: MemoryStore, uid: UUID) -> MemoryRecord | None:
    try:
        return store.get(uid)
    except (OSError, KeyError, ValueError, RuntimeError):
        return None


# ----------------------------------------------------------- segment builders


def _l0_segment(store: MemoryStore) -> str:
    """identity kernel -- the pinned L0 record by fixed UUID.

    Returned string shape: "<aaak_index>\n<literal_surface[:200]>". Empty when
    the L0 record hasn't been seeded yet (fresh stores before first core boot).
    """
    rec = _fetch_record(store, L0_RECORD_UUID)
    if rec is None:
        return ""
    aaak = rec.aaak_index or generate_aaak_index(rec)
    # Clean render-time (strip ANSI + harness markers) BEFORE truncating.
    # L0 budget is ~80 tok (~320 chars); leave slack for the aaak line.
    cleaned = _clean_surface(rec.literal_surface)[:200]
    return f"{aaak}\n{cleaned}"


def _l1_segment(store: MemoryStore, max_records: int = 10) -> str:
    """L1 critical-facts block -- pinned records with detail_level >= 4.

    Excludes the L0 record (duplicated in L0 segment). Lines formatted as
    "- <literal_surface[:100]>" so they fit in ~25 tokens each; 10 of them
    saturate the L1_BUDGET_TOKENS ~= 200 tok budget.
    """
    try:
        records = store.all_records()
    except (OSError, RuntimeError, ValueError):
        return ""
    pinned_hi_detail = [
        r for r in records
        if r.pinned and r.detail_level >= 4 and r.id != L0_RECORD_UUID
    ]
    # Deterministic ordering: by detail_level desc, then by created_at asc.
    pinned_hi_detail.sort(
        key=lambda r: (-r.detail_level, r.created_at)
    )
    pinned_hi_detail = pinned_hi_detail[:max_records]
    if not pinned_hi_detail:
        return ""
    # Clean render-time before truncation; skip records that clean to empty.
    lines = []
    for r in pinned_hi_detail:
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        lines.append(f"- {cleaned[:100]}")
    return "\n".join(lines)


def _l2_segments(
    store: MemoryStore,
    assignment: CommunityAssignment,
) -> list[str]:
    """Up to L2_COMMUNITY_CAP (7) community summary lines.

    Each summary samples up to 3 member records from the community's
    mid_regions list and joins them with `|`. Budget guardrail: each line
    is capped at approximately L2_PER_COMMUNITY_TOKENS * 4 chars (=200 chars).

    Empty list when the assignment has no top_communities (fresh/flat case).
    """
    top = list(assignment.top_communities)[:L2_COMMUNITY_CAP]
    if not top:
        return []

    # records_cache: keep the single all_records() call hot (same trick
    # pipeline.py uses -- avoids N+1 store.get scans).
    try:
        records = store.all_records()
    except (OSError, RuntimeError, ValueError):
        return []
    by_uuid = {r.id: r for r in records}

    summaries: list[str] = []
    max_chars = L2_PER_COMMUNITY_TOKENS * 4  # ~200 chars budget per line
    for cid in top:
        members = assignment.mid_regions.get(cid, [])[:3]
        parts: list[str] = []
        for mid in members:
            rec = by_uuid.get(mid)
            if rec is None:
                continue
            # Clean render-time before truncation; skip empty surfaces.
            cleaned = _clean_surface(rec.literal_surface)
            if not cleaned:
                continue
            wing = rec.aaak_index.split("/")[0] if rec.aaak_index else "W:?"
            parts.append(f"{wing}/{cleaned[:40]}")
        if not parts:
            continue
        body = " | ".join(parts)
        line = f"[community {str(cid)[:8]}] {body}"
        if len(line) > max_chars:
            line = line[:max_chars]
        summaries.append(line)
    return summaries


def _rich_club_segment(store: MemoryStore, rich_club: list[UUID]) -> str:
    """Global rich-club summary, truncated to RICH_CLUB_BUDGET_TOKENS.

    Each rich-club node contributes one line "<aaak_index>: <literal_surface[:60]>".
    Lines are added until the running token count would exceed the budget.
    """
    return _rich_club_segment_with_budget(store, rich_club, budget=RICH_CLUB_BUDGET_TOKENS)


def _rich_club_segment_with_budget(
    store: MemoryStore,
    rich_club: list[UUID],
    *,
    budget: int,
) -> str:
    """Rich-club summary with an explicit budget (deep mode).

    Same rendering as `_rich_club_segment`; `budget` replaces the default cap
    so wake_depth=deep can lift the rich_club allotment to ~2000 tok.
    """
    if not rich_club:
        return ""
    try:
        records = store.all_records()
    except (OSError, RuntimeError, ValueError):
        return ""
    by_uuid = {r.id: r for r in records}

    lines: list[str] = []
    running = 0
    for uid in rich_club:
        rec = by_uuid.get(uid)
        if rec is None:
            continue
        # Clean render-time before truncation; skip records that clean to empty.
        cleaned = _clean_surface(rec.literal_surface)
        if not cleaned:
            continue
        aaak = rec.aaak_index or generate_aaak_index(rec)
        line = f"{aaak}: {cleaned[:60]}"
        cost = _approx_tokens(line)
        # Respect running budget -- +1 accounts for the join newline.
        if running + cost + 1 > budget:
            break
        lines.append(line)
        running += cost + 1
    return "\n".join(lines)


def _recent_thread_segment(
    store: MemoryStore,
    *,
    max_records: int = 5,
    pending_live_events: "list | None" = None,
) -> str:
    """Build a short summary of the most-recent work lines from the store.

    Sorts all non-identity records by created_at descending, cleans each
    surface, skips empty results, and returns up to max_records lines
    formatted as "- <cleaned[:120]>". Returns "" when nothing is kept.

    Parameters
    ----------
    pending_live_events:
        OPT-IN: when provided (a list of dicts from
        ``capture.read_pending_live_events``), merges pending conversational
        turns (user + assistant roles) with the store records before sorting.
        Deduplicated against the store idem-tag set AND a seen-pending set.
        When ``None`` (the default), behaves exactly as before.
    """
    try:
        records = store.all_records()
    except (OSError, RuntimeError, ValueError):
        return ""

    # Exclude the pinned L0 identity record (duplicated in Identity section).
    candidates = [r for r in records if r.id != L0_RECORD_UUID]

    if pending_live_events is not None:
        from iai_mcp.capture import _idem_tag as _cap_idem_tag
        from iai_mcp.store import _PendingTurn

        # Build store idem-tag set.
        store_idem_set: set = set()
        for r in candidates:
            for tag in (r.tags or []):
                if tag.startswith("idem:"):
                    store_idem_set.add(tag)

        seen_pending: set = set()
        for ev in pending_live_events:
            role = ev.get("role", "user")
            # Thread segment includes both conversational roles (user + assistant).
            if role not in ("user", "assistant"):
                continue
            ev_session = ev.get("session_id", "-")
            src_uuid = ev.get("source_uuid")
            ts_iso = ev["ts_iso"]
            text = ev.get("text", "")
            idem = _cap_idem_tag(ev_session, role, ts_iso, text, source_uuid=src_uuid)
            if idem in store_idem_set or idem in seen_pending:
                continue
            seen_pending.add(idem)
            candidates.append(_PendingTurn(
                text=text,
                session_id=ev_session,
                ts=ev["ts"],
                idem_tag=idem,
                source_uuid=src_uuid,
                role=role,
            ))

    # Sort newest first.
    candidates.sort(key=lambda r: r.created_at, reverse=True)
    lines: list[str] = []
    for r in candidates:
        if len(lines) >= max_records:
            break
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        lines.append(f"- {cleaned[:120]}")
    return "\n".join(lines)


# ------------------------------------------------------------------ public


def _session_state_hash(payload: SessionStartPayload) -> str:
    """Deterministic SHA-256 over the 4-segment cached prefix.

    Two sessions whose L0 + L1 + L2 + rich_club segments are byte-identical
    produce the SAME session_state_hash -- the "context-repeat" metric.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(payload.l0.encode("utf-8"))
    h.update(b"\x1f")  # ASCII unit separator
    h.update(payload.l1.encode("utf-8"))
    h.update(b"\x1f")
    h.update("\n".join(payload.l2).encode("utf-8"))
    h.update(b"\x1f")
    h.update(payload.rich_club.encode("utf-8"))
    return h.hexdigest()


def _dominant_community_label(assignment: CommunityAssignment) -> str:
    """Return a short (<=8 char) label for the largest community.

    Returns 'none' when no communities exist (fresh or flat assignment). The
    label is the first 8 hex of the dominant community UUID — a stable handle
    that fits in ~3-4 tokens.
    """
    try:
        top = list(assignment.top_communities)
        if not top:
            return "none"
        # top_communities is already ordered by member count (L1).
        return str(top[0])[:8]
    except (TypeError, AttributeError):
        return "none"


def _count_pending_first_turn(store: MemoryStore) -> int:
    """Count open first_turn_pending sessions in daemon_state.

    Returns 0 if daemon_state is missing or malformed (silent fallback). This
    is only cosmetic input to the brain_handle pointer; the minimal payload
    must survive a missing daemon gracefully.
    """
    try:
        from iai_mcp.daemon_state import load_state
        state = load_state()
        pending = state.get("first_turn_pending", {})
        if isinstance(pending, dict):
            return sum(1 for v in pending.values() if v)
        return 0
    except (OSError, json.JSONDecodeError, ImportError, ValueError):
        return 0


def _compose_session_start_payload(
    store: MemoryStore,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    *,
    session_id: str = "-",
    profile_state: dict | None = None,
) -> SessionStartPayload:
    """Emit-free composition path for the session-start cached prefix.

    Returns the same `SessionStartPayload` shape that `assemble_session_start`
    produces, with identical wake_depth branching (minimal / standard / deep),
    but does NOT emit the `session_started` event. Used by the daemon precache
    writer so REM-loop cache writes do not pollute the events table with one
    synthetic session-start per cycle.

    The public entry point that downstream callers should keep using is
    `assemble_session_start` — which wraps this helper and adds the emit.
    """
    from iai_mcp.profile import default_state
    state = profile_state if isinstance(profile_state, dict) else default_state()
    wake_depth = state.get("wake_depth", "minimal")
    if wake_depth not in ("minimal", "standard", "deep"):
        wake_depth = "minimal"  # silent fallback

    if wake_depth == "minimal":
        # Payload will not be Anthropic-cached
        # (<=30 raw tok < Sonnet 4.6 min 2048). DO NOT set cache_control.
        #
        # Collapse the three legacy pointers
        # (identity_pointer + brain_handle + topic_cluster_hint, ~24 raw tok
        # together) into a single `<iai:HHHHHHHHHHHHHHHH>` handle (~6-10 raw
        # tok). The LRU inside `iai_mcp.handle` retains the reverse mapping
        # so downstream code can resolve the handle to its triple.
        #
        # Back-compat contract: the 3 legacy fields stay populated on the
        # dataclass so callers reading the old shape keep working; only
        # ``total_cached_tokens`` is charged for the compact handle (the
        # wire prefix at wake_depth=minimal is the compact handle alone).
        l0_rec = _fetch_record(store, L0_RECORD_UUID)
        identity_short = str(L0_RECORD_UUID)[:8] if l0_rec is not None else ""
        identity_pointer = f"<id:{identity_short}>" if identity_short else ""
        pending = _count_pending_first_turn(store)
        session_short = str(session_id)[:8]
        brain_handle = f"<sess:{session_short} pend:{pending}>"
        topic_label = _dominant_community_label(assignment)
        topic_cluster_hint = f"<topic:{topic_label}>"
        compact_handle = encode_compact_handle(
            identity_short, session_short, topic_label, pending
        )
        cached = _approx_tokens(compact_handle)
        payload = SessionStartPayload(
            l0="",
            l1="",
            l2=[],
            rich_club="",
            total_cached_tokens=cached,
            total_dynamic_tokens=DYNAMIC_TAIL_TOKENS,
            identity_pointer=identity_pointer,
            brain_handle=brain_handle,
            topic_cluster_hint=topic_cluster_hint,
            compact_handle=compact_handle,
            wake_depth="minimal",
        )
    else:
        # standard and deep share the eager assembly path; deep lifts
        # the rich_club budget by re-running the segment with a larger cap.
        l0 = _l0_segment(store)
        l1 = _l1_segment(store)
        l2 = _l2_segments(store, assignment)
        if wake_depth == "deep":
            rc = _rich_club_segment_with_budget(store, rich_club, budget=2000)
        else:
            rc = _rich_club_segment(store, rich_club)

        cached = (
            _approx_tokens(l0)
            + _approx_tokens(l1)
            + sum(_approx_tokens(s) for s in l2)
            + _approx_tokens(rc)
        )

        # New pointers also populated under standard/deep so downstream callers
        # can use them alongside legacy segments if they want.:
        # the compact handle is ALSO minted here so a consumer can opt in to
        # the short form without requiring a wake_depth mode switch.
        l0_rec = _fetch_record(store, L0_RECORD_UUID)
        identity_short = str(L0_RECORD_UUID)[:8] if l0_rec is not None else ""
        identity_pointer = f"<id:{identity_short}>" if identity_short else ""
        pending = _count_pending_first_turn(store)
        session_short = str(session_id)[:8]
        brain_handle = f"<sess:{session_short} pend:{pending}>"
        topic_label = _dominant_community_label(assignment)
        topic_cluster_hint = f"<topic:{topic_label}>"
        compact_handle = encode_compact_handle(
            identity_short, session_short, topic_label, pending
        )

        from iai_mcp.capture import read_pending_live_events
        _pending = read_pending_live_events()
        recent_thread = _recent_thread_segment(store, pending_live_events=_pending)

        payload = SessionStartPayload(
            l0=l0,
            l1=l1,
            l2=l2,
            rich_club=rc,
            total_cached_tokens=cached,
            total_dynamic_tokens=DYNAMIC_TAIL_TOKENS,
            identity_pointer=identity_pointer,
            brain_handle=brain_handle,
            topic_cluster_hint=topic_cluster_hint,
            compact_handle=compact_handle,
            wake_depth=wake_depth,
            recent_thread=recent_thread,
        )

    return payload


def assemble_session_start(
    store: MemoryStore,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    *,
    session_id: str = "-",
    profile_state: dict | None = None,
) -> SessionStartPayload:
    """Assemble the session-start cached prefix.

    Branches on the `wake_depth` profile knob (MCP-12):

    - ``minimal`` (default): produce a ≤30 raw-tok pointer handle (identity,
      brain session, topic cluster). Legacy l0/l1/l2/rich_club emitted empty
      for back-compat with existing TS-wrapper callers.
    - ``standard``: reproduce the 1388-tok eager dump — l0/l1/l2/
      rich_club populated via `_l0_segment`, `_l1_segment`, `_l2_segments`,
      `_rich_club_segment`. New fields emitted empty.
    - ``deep``: same shape as standard but rich_club budget lifted to 2000.
      Populates both the legacy segments and the new pointers.

    Emits ``kind='session_started'`` with a deterministic
    ``session_state_hash`` over the cached prefix. Two consecutive sessions
    whose cached prefix is identical produce the same hash.

    Note: at `wake_depth=minimal` the payload is ≤30 raw tok which is BELOW
    the Sonnet 4.6 / Opus 4.7 cache minimum (2048 / 4096). DO NOT add
    ``cache_control`` to the minimal branch prefix — it would be silently
    ignored by the Anthropic API and waste a breakpoint slot.

    Composition is delegated to `_compose_session_start_payload`; this
    function adds the `session_started` event emit (one event per call,
    carrying `session_state_hash` for trajectory tracking).
    """
    payload = _compose_session_start_payload(
        store,
        assignment,
        rich_club,
        session_id=session_id,
        profile_state=profile_state,
    )

    # Emit kind='session_started' with session_state_hash for trajectory
    # tracking. Diagnostic-only: never block session start on emit failure.
    try:
        from datetime import datetime, timezone
        from iai_mcp.events import write_event
        write_event(
            store,
            kind="session_started",
            data={
                "session_id": session_id,
                "session_state_hash": _session_state_hash(payload),
                "total_cached_tokens": payload.total_cached_tokens,
                "wake_depth": payload.wake_depth,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
            session_id=session_id,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("session_started_event_failed", extra={"err": str(exc)[:80]})

    return payload


def format_payload_as_markdown(payload: "SessionStartPayload | dict") -> str:
    """Render the cached prefix and recent-thread element as clean markdown.

    Section order: Identity, Most recent work (when present), Critical facts,
    Topic communities (one block per L2 segment), Key memories.

    Legacy segment strings (l0/l1/l2/rich_club) are preserved and flow through
    unchanged — no segment architecture is removed. Empty segments are skipped.
    """
    if isinstance(payload, dict):
        l0 = payload.get("l0") or ""
        l1 = payload.get("l1") or ""
        l2 = list(payload.get("l2") or [])
        rich_club = payload.get("rich_club") or ""
        recent_thread = payload.get("recent_thread") or ""
    else:
        l0 = payload.l0
        l1 = payload.l1
        l2 = list(payload.l2)
        rich_club = payload.rich_club
        recent_thread = payload.recent_thread
    blocks: list[str] = []
    if l0:
        blocks.append(f"## Identity\n{l0}")
    if recent_thread:
        blocks.append(f"## Most recent work\n{recent_thread}")
    if l1:
        blocks.append(f"## Critical facts\n{l1}")
    for seg in l2:
        if seg:
            blocks.append(f"## Topic communities\n{seg}")
    if rich_club:
        blocks.append(f"## Key memories\n{rich_club}")
    return "\n\n".join(blocks)


def max_record_created_at(store: MemoryStore) -> str | None:
    """Return the ISO timestamp of the newest non-tombstoned record, or None.

    Runs a single ``SELECT MAX(created_at)`` under the store connection lock
    so it is safe to call from any thread while the daemon holds the store.
    Returns None when the store has no live records.
    """
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()
    return row[0] if row and row[0] else None
