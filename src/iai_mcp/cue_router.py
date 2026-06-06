"""Cue-detection router.

Classifies a memory_recall cue into 'verbatim' or 'concept' mode based on
surface signals (quoted phrases, exact-recall markers, RU starts-with
triggers). Drives mode-dependent retrieval in both pipeline_recall (full
graph path) and retrieve.recall (baseline fallback).

Design notes:
- When the cue signals exact recall, the user wants ONE hit, not 30, so
  verbatim mode is the response shape.
- Episodic and semantic stores have distinguishable retrieval surfaces; the
  cue tells us which store the user is asking for.
- The verbatim-fidelity target is defended at the entrypoint: any
  verbatim-flavoured cue routes to the surface that protects it
  (tier filter + zeroed graph-bonus).

Triggers (compiled once at module load):

  EN (re.IGNORECASE):
    - quoted-phrase: "..." (one pair of straight double quotes around text)
    - european-quote: «...» (one pair of guillemets around text)
    - word-marker: verbatim | exact | quote | quoted | said | wrote
    - day-N: day <digits> (e.g. "day 17", "Day 7")

  RU (case-insensitive, anchored at start-of-cue ^):
    - ru-start-найди-дословно
    - ru-start-точная-цитата
    - ru-start-что-я-сказал
    - ru-start-что-я-писал

Behaviour:
- Any one EN match wins (returned with its label) and the function returns
  ("verbatim", intent, label) immediately.
- Otherwise any one RU match wins (returned with its label).
- No match -> ("concept", intent, None).
- Empty / falsy text -> ("concept", None, None).

The triggered_pattern label is for diagnostic logging (event payloads,
debug traces) and is NOT surfaced on the JSON-RPC response — only the
mode string lives in RecallResponse.cue_mode.

# Intent dimension (orthogonal to mode): identifies the user's retrieval
# intent so Stage 8 can apply mode-conditional score modifiers.
# Currently supports one intent: "historical_verbatim" — user wants the
# ORIGINAL (pre-correction) record, not the latest contradicting record.
# Stage 8 downweights candidates with an incoming `contradicts` edge
# (i.e., a "correction target" whose corrector recorded a contradicts edge
# from itself) so the contradicted (original) record wins on cues like
# "Quote the original ETA wording" / "what was first" / "приведи
# оригинальную формулировку".
"""
from __future__ import annotations

import re

EN_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("quoted-phrase",  re.compile(r'"[^"]+"')),
    ("european-quote", re.compile(r'«[^»]+»')),
    ("word-marker",    re.compile(r'\b(verbatim|exact|quote|quoted|said|wrote)\b', re.IGNORECASE)),
    ("day-N",          re.compile(r'\bday\s+\d+\b', re.IGNORECASE)),
]

RU_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("ru-start-найди-дословно",  re.compile(r'^найди дословно', re.IGNORECASE)),
    ("ru-start-точная-цитата",   re.compile(r'^точная цитата',  re.IGNORECASE)),
    ("ru-start-что-я-сказал",    re.compile(r'^что я сказал',    re.IGNORECASE)),
    ("ru-start-что-я-писал",     re.compile(r'^что я писал',     re.IGNORECASE)),
]

# Historical-verbatim intent: orthogonal to (verbatim, concept) mode.
# When cue signals "original / before / first / earlier / previously"
# (and Russian equivalents), Stage 8 downweights records that are the
# TARGET of a contradicts edge (i.e., the "corrected" original whose
# corrector pointed back to it). Per `store.add_contradicts_edge(original,
# new)` the edge direction is src=original → dst=new (corrector). The
# WRONG/corrector ends up as the dst across the outgoing dict; downweight
# that side so the contradicted (original) ranks above its corrector.
EN_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("historical-en-original",   re.compile(r'\b(original|originally)\b', re.IGNORECASE)),
    ("historical-en-before",     re.compile(r'\bbefore\b', re.IGNORECASE)),
    ("historical-en-first",      re.compile(r'\b(first|initial|initially)\b', re.IGNORECASE)),
    ("historical-en-earlier",    re.compile(r'\bearlier\b', re.IGNORECASE)),
    ("historical-en-previously", re.compile(r'\b(previously|previous)\b', re.IGNORECASE)),
]

RU_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    # Use \b (word boundary), NOT ^ anchor — historical markers appear
    # mid-cue in naturalistic Russian phrasing (Open Question 2 resolution).
    ("historical-ru-original",   re.compile(r'\b(оригинал|оригинальн\w*)\b', re.IGNORECASE)),
    ("historical-ru-snachala",   re.compile(r'\bсначала\b', re.IGNORECASE)),
    ("historical-ru-iznachal",   re.compile(r'\bизначальн\w*\b', re.IGNORECASE)),
    ("historical-ru-ranee",      re.compile(r'\bранее\b', re.IGNORECASE)),
]


def _classify_cue(text: str) -> tuple[str, str | None, str | None]:
    """Return (mode, intent, triggered_pattern) for the given cue.

    mode is "verbatim" if any (EN_TRIGGERS / RU_TRIGGERS) trigger matches,
    else "concept".

    intent is "historical_verbatim" if any
    (EN_HISTORICAL_TRIGGERS / RU_HISTORICAL_TRIGGERS) trigger matches,
    else None. Intent is ORTHOGONAL to mode — a cue can be
    (verbatim, historical_verbatim) like "Quote the original ETA wording"
    or (concept, None) like "what about auth".

    triggered_pattern is the trigger label (string) on a verbatim hit, or
    None when the cue routes to concept (no trigger matched). The verbatim
    label is returned in preference to a historical-only match label.

    Empty / None-ish input returns ("concept", None, None) — defensive
    default so the dispatcher never crashes on a missing cue field.

    """
    if not text:
        return "concept", None, None

    # Pass 1: mode classification (verbatim vs concept).
    mode = "concept"
    label: str | None = None
    for lbl, pat in EN_TRIGGERS:
        if pat.search(text):
            mode = "verbatim"
            label = lbl
            break
    if mode != "verbatim":
        for lbl, pat in RU_TRIGGERS:
            if pat.search(text):
                mode = "verbatim"
                label = lbl
                break

    # Pass 2: intent classification (historical_verbatim vs None).
    # Intent is computed regardless of mode — a concept-mode cue with
    # "original" in it still signals historical-verbatim intent. Stage 8
    # gates the downweight on intent, not mode.
    intent: str | None = None
    for _lbl, pat in EN_HISTORICAL_TRIGGERS:
        if pat.search(text):
            intent = "historical_verbatim"
            break
    if intent is None:
        for _lbl, pat in RU_HISTORICAL_TRIGGERS:
            if pat.search(text):
                intent = "historical_verbatim"
                break

    return mode, intent, label
