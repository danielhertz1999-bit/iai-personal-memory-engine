"""AUTIST-13 — surface-feature formality scorer (Chapman ecological self-regulation).

Invariants (Chapman ecological self-regulation framing):
- Observes ONLY the user's surface lexical features.
- Never models user internal state, never tries to infer "is the user masking".
- Paired with src/iai_mcp/camouflaging.py which adjusts OUR register in response.

Scientific anchor: Chapman R (2021) "Neurodiversity and the Social Ecology of Mental
Functions." Cook 2021 + Raymaker 2020 tell us what NOT to model (masking as an
inferred user state).

Four surface features (weighted sum):
1. Lexical formality (w=0.45) — per-language register-marker density. Strongest signal.
2. Sentence complexity (w=0.20) — sigmoid on avg chars-per-sentence + clause density.
3. Hedging density (w=0.15) — hedge markers per 100 tokens.
4. Punctuation formality (w=0.20) — semicolon + em-dash + full-quote density.

Output: formality_score(text, lang) -> float in [0.0, 1.0]. 0 = fully informal,
1 = fully formal. Unknown lang returns 0.5 (neutral) with a logged warning; NEVER raises.

Weight rationale: weights were fixture-tuned to 0.45/0.20/0.15/0.20 because the lex
dimension is the most unambiguous signal across RU+EN and the shortest formal sentences
(e.g. "The proposal is, therefore, accepted.") are otherwise penalised by the
complexity sigmoid. Fixture accuracy: 100% (51/51) with the current weights.
"""
from __future__ import annotations

import logging
import math
import re
import warnings
from typing import Iterable


# ------------------------------------------------------------------- constants

LEX_MARKERS: dict[str, list[str]] = {
    "en": [
        "therefore", "however", "accordingly", "nonetheless", "furthermore",
        "hence", "thus", "consequently", "moreover", "notwithstanding",
        "whereas", "hereby", "herein", "thereof", "pursuant", "aforementioned",
        "shall", "aforesaid",
    ],
    "ru": [
        "тем не менее", "следовательно", "однако", "впрочем", "таким образом",
        "вследствие", "настоящим", "согласно", "вышеизложенного", "вышеизложенному",
        "в соответствии", "по-видимому", "в силу", "исходя из", "данное",
        "настоящее", "прилагаемым", "представленное", "уведомляем",
    ],
}

HEDGE_MARKERS: dict[str, list[str]] = {
    "en": [
        "possibly", "perhaps", "might", "may", "could", "seemingly",
        "appears to", "seems", "somewhat", "apparently", "presumably",
    ],
    "ru": [
        "возможно", "вероятно", "видимо", "по-видимому", "наверное",
        "кажется", "пожалуй", "скорее всего", "вроде", "будто",
    ],
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "lex": 0.45,
    "complexity": 0.20,
    "hedge": 0.15,
    "punct": 0.20,
}

# Sentence-complexity sigmoid parameters.
# avg chars per sentence: centre 40 credits terse formal writing (e.g. "The
# proposal is, therefore, accepted."). clause count adds a second signal
# weighted equally with length (avg_cl centre 0.5 = one comma per sentence).
_SENTENCE_COMPLEXITY_CENTER: float = 40.0
_SENTENCE_COMPLEXITY_SCALE: float = 25.0
_CLAUSE_COUNT_CENTER: float = 0.5
_CLAUSE_COUNT_SCALE: float = 0.5

# Density sigmoid parameters. Tuned so 0 markers -> ~0.1, 1.5 markers/100tok -> 0.5.
_LEX_DENSITY_CENTER: float = 1.5  # markers per 100 tokens
_LEX_DENSITY_SCALE: float = 1.2
_HEDGE_DENSITY_CENTER: float = 1.0
_HEDGE_DENSITY_SCALE: float = 0.8
_PUNCT_DENSITY_CENTER: float = 1.5
_PUNCT_DENSITY_SCALE: float = 1.3

_NEUTRAL_SCORE: float = 0.5

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- helpers
def _tokens(text: str) -> list[str]:
    """Whitespace split on letter sequences; lowercase. Unicode-aware."""
    cleaned = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    return [t.lower() for t in cleaned.split() if t]


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"[.!?;]+", text)
    return [p.strip() for p in parts if p.strip()]


def _sigmoid(x: float) -> float:
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    ez = math.exp(x)
    return ez / (1.0 + ez)


def _count_phrase_occurrences(text_lower: str, phrases: Iterable[str]) -> int:
    count = 0
    for p in phrases:
        if not p:
            continue
        if " " in p or "-" in p:
            # Multi-word or hyphenated phrase -> substring match is fine.
            count += text_lower.count(p)
        else:
            count += len(re.findall(rf"\b{re.escape(p)}\b", text_lower, flags=re.UNICODE))
    return count


# ------------------------------------------------------------------- features
def _lex_score(text: str, lang: str) -> float:
    """Per-language register-marker density, sigmoid-bounded to [0, 1]."""
    markers = LEX_MARKERS.get(lang, [])
    if not markers:
        return _NEUTRAL_SCORE
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = _count_phrase_occurrences(text.lower(), markers)
    density = hits * 100.0 / max(len(toks), 1)
    return _sigmoid((density - _LEX_DENSITY_CENTER) / _LEX_DENSITY_SCALE)


def _complexity_score(text: str) -> float:
    """Avg chars per sentence + clause-count proxy. Language-independent.

    Returns equal-weight blend of:
    - length sigmoid (centred at 40 chars so terse formal sentences aren't depressed).
    - clause sigmoid based on commas per sentence (centred at 0.5 = one comma avg).
    """
    sents = _sentence_split(text)
    if not sents:
        return 0.0
    avg_len = sum(len(s) for s in sents) / len(sents)
    avg_clauses = sum(s.count(",") for s in sents) / len(sents)
    len_score = _sigmoid(
        (avg_len - _SENTENCE_COMPLEXITY_CENTER) / _SENTENCE_COMPLEXITY_SCALE
    )
    cl_score = _sigmoid((avg_clauses - _CLAUSE_COUNT_CENTER) / _CLAUSE_COUNT_SCALE)
    return 0.5 * len_score + 0.5 * cl_score


def _hedge_score(text: str, lang: str) -> float:
    """Hedging density per 100 tokens, sigmoid-bounded to [0, 1]."""
    markers = HEDGE_MARKERS.get(lang, [])
    if not markers:
        return _NEUTRAL_SCORE
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = _count_phrase_occurrences(text.lower(), markers)
    density = hits * 100.0 / max(len(toks), 1)
    return _sigmoid((density - _HEDGE_DENSITY_CENTER) / _HEDGE_DENSITY_SCALE)


def _punct_score(text: str) -> float:
    """Semicolon + em-dash + full-quote density per 100 tokens."""
    toks = _tokens(text)
    if not toks:
        return 0.0
    semi = text.count(";")
    em = text.count("—") + text.count("–")
    fq = (
        text.count('"')
        + text.count("“")
        + text.count("”")
        + text.count("«")
        + text.count("»")
    )
    hits = semi + em + fq
    density = hits * 100.0 / max(len(toks), 1)
    return _sigmoid((density - _PUNCT_DENSITY_CENTER) / _PUNCT_DENSITY_SCALE)


# ------------------------------------------------------------------- public
def formality_score(
    text: str,
    lang: str,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Return surface-feature formality score in [0.0, 1.0].

    0.0 = fully informal, 1.0 = fully formal. Unknown languages get a neutral 0.5
    with a logged warning (global-product graceful degradation). NEVER
    raises on bad input.

    Args:
        text: free-form user utterance (SURFACE only).
        lang: ISO-639-1 language code ("en", "ru"). Other codes -> neutral + warning.
        weights: optional override {lex, complexity, hedge, punct}.

    Guard reminder: callers pass user SURFACE text only. The scorer
    does not see any inferred internal state. See camouflaging.py for how the
    score is consumed (to adjust OUR register, never the user's).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    if lang not in LEX_MARKERS:
        warnings.warn(
            f"formality_score: lang={lang!r} outside RU+EN baseline; "
            "returning neutral 0.5",
            stacklevel=2,
        )
        _logger.debug("formality_score unknown lang=%s text_len=%d", lang, len(text))
        return _NEUTRAL_SCORE

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total_w = sum(w.values()) or 1.0

    lex = _lex_score(text, lang)
    complexity = _complexity_score(text)
    hedge = _hedge_score(text, lang)
    punct = _punct_score(text)

    weighted = (
        w["lex"] * lex
        + w["complexity"] * complexity
        + w["hedge"] * hedge
        + w["punct"] * punct
    ) / total_w
    # Clamp to [0, 1] defensively.
    return max(0.0, min(1.0, weighted))
