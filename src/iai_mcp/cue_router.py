"""cue-detection router.

Classifies a memory_recall cue into 'verbatim' or 'concept' mode based on
surface signals (quoted phrases, exact-recall markers, RU starts-with
triggers). Drives mode-dependent retrieval in both pipeline_recall (full
graph path) and retrieve.recall (baseline fallback).

Constitutional framing:
- Mottron EPF / Bowler TSH / Murray monotropism: when the cue signals exact
  recall, the user wants ONE hit, not 30. Verbatim mode is the response shape.
- McClelland CLS: episodic and semantic stores have distinguishable retrieval
  surfaces; the cue tells us which store the user is asking.
- Beer VSM S1 vs S4: verbatim is operations, schema is intelligence; the
  router separates the two recursion levels at the entrypoint.
- Ashby ultrastability: the North-Star verbatim ≥99% essential variable is
  defended at the entrypoint — any verbatim-flavoured cue routes to the
  surface that protects it (tier filter + zeroed graph-bonus).

Triggers per CONTEXT (compiled once at module load):

  EN (re.IGNORECASE):
    - quoted-phrase  : "..."  (one pair of straight double quotes around text)
    - european-quote : «...»  (one pair of guillemets around text)
    - word-marker    : verbatim | exact | quote | quoted | said | wrote
    - day-N          : day <digits>  (e.g. "day 17", "Day 7")

  RU (case-insensitive, anchored at start-of-cue ^):
    - ru-start-найди-дословно
    - ru-start-точная-цитата
    - ru-start-что-я-сказал
    - ru-start-что-я-писал

Behaviour:
- Any one EN match wins (returned with its label) and the function returns
  ("verbatim", label) immediately.
- Otherwise any one RU match wins (returned with its label).
- No match -> ("concept", None).
- Empty / falsy text -> ("concept", None).

The triggered_pattern label is for diagnostic logging (event payloads,
debug traces) and is NOT surfaced on the JSON-RPC response — only the
mode string lives in RecallResponse.cue_mode.
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


def _classify_cue(text: str) -> tuple[str, str | None]:
    """Return (mode, triggered_pattern) for the given cue.

    mode is "verbatim" if any trigger matches, else "concept".
    triggered_pattern is the trigger label (string) on a verbatim hit, or
    None when the cue routes to concept (no trigger matched).

    Empty / None-ish input returns ("concept", None) — defensive default
    so the dispatcher never crashes on a missing cue field.
    """
    if not text:
        return "concept", None
    for label, pat in EN_TRIGGERS:
        if pat.search(text):
            return "verbatim", label
    for label, pat in RU_TRIGGERS:
        if pat.search(text):
            return "verbatim", label
    return "concept", None
