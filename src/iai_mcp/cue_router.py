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

EN_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("historical-en-original",   re.compile(r'\b(original|originally)\b', re.IGNORECASE)),
    ("historical-en-before",     re.compile(r'\bbefore\b', re.IGNORECASE)),
    ("historical-en-first",      re.compile(r'\b(first|initial|initially)\b', re.IGNORECASE)),
    ("historical-en-earlier",    re.compile(r'\bearlier\b', re.IGNORECASE)),
    ("historical-en-previously", re.compile(r'\b(previously|previous)\b', re.IGNORECASE)),
]

RU_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("historical-ru-original",   re.compile(r'\b(оригинал|оригинальн\w*)\b', re.IGNORECASE)),
    ("historical-ru-snachala",   re.compile(r'\bсначала\b', re.IGNORECASE)),
    ("historical-ru-iznachal",   re.compile(r'\bизначальн\w*\b', re.IGNORECASE)),
    ("historical-ru-ranee",      re.compile(r'\bранее\b', re.IGNORECASE)),
]


def _classify_cue(text: str) -> tuple[str, str | None, str | None]:
    if not text:
        return "concept", None, None

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
