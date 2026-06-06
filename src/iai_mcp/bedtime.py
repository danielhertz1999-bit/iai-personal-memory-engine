"""-- bedtime wind-down detection.

Dual-gate bedtime suggestion emitter:
  Gate A: wind-down phrase regex match per language (8 languages)
  Gate B: late in learned quiet window (inside OR within 30min of start)

When BOTH gates pass, `detect_wind_down` returns a small dict that `core.py`
injects into `memory_recall` responses as `sleep_suggestion`. Claude (the
LLM in the active session) decides social framing -- our code NEVER hardcodes
user-facing phrasing.

Guards:
- This module does NOT initiate sleep. It only suggests. The only path
  that moves the daemon into SLEEP is `core.handle_initiate_sleep_mode`
  with `consent=True`. No auto-start in this file.
- This module is read-only w.r.t. records. It reads `cue` strings;
  it NEVER mutates `literal_surface`.
- No fcntl, no daemon state mutation. All logic is pure in-process.

Patterns mirror `shield.py`'s 8-language dict style (same language set:
en/ru/ja/ar/de/fr/es/zh per global-product mandate). Latin-script
languages use `\b` word boundaries; CJK / Arabic use character-class
proximity and whitespace-tolerant forms since Unicode `\b` is unreliable
across scripts.

ReDoS-safe: every pattern uses bounded quantifiers only. No nested `(.+)+`
constructs, no `.*.*`. Stress-tested against 10KB of "a"s under 100ms total.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from iai_mcp.quiet_window import BUCKET_MINUTES


# ------------------------------------------------------------ constants

# dual-gate: within this many minutes of the learned quiet-window start
# also counts as "late" (a user who says "good night" 25 minutes before their
# usual quiet window is winding down, not speaking rhetorically).
WIND_DOWN_GATE_MINUTES_BEFORE: int = 30


# ------------------------------------------------------------ per-language regex

# English wind-down phrases. Case-insensitive match.
WIND_DOWN_EN: list[str] = [
    r"\bgood\s*night\b",
    r"\bgoodnight\b",
    r"\bnight[,!.]?\s*$",
    r"\bI'?m\s+(heading|going)\s+to\s+bed\b",
    r"\b(time\s+(to|for)\s+bed|bedtime)\b",
    r"\bI'?m\s+(tired|exhausted|sleepy)\b",
    r"\b(catch\s+you\s+tomorrow|see\s+you\s+tomorrow)\b",
    r"\blet'?s\s+(continue|pick\s+up)\s+tomorrow\b",
    r"\bgoing\s+to\s+sleep\b",
]

# Russian (same 8-language set as shield.py).
WIND_DOWN_RU: list[str] = [
    r"спокойной\s+ночи",
    r"пойду\s+(спать|в\s+постель)",
    r"(я\s+)?(устал|устала|вымотан[аы]?|засыпаю)",
    r"пора\s+(спать|ложиться)",
    r"до\s+завтра",
    r"давай\s+завтра",
    r"ухожу\s+спать",
    r"(окей|ок|ладно),?\s+сплю",
    r"ложусь",
]

# Japanese -- NREM cues + "see you tomorrow". No \b; lookaround on adjacent
# punctuation / kana / CJK characters.
WIND_DOWN_JA: list[str] = [
    r"お\s*や\s*す\s*み(なさい)?",     # おやすみ / おやすみなさい
    r"寝\s*ます",                       # 寝ます
    r"(眠|ねむ)い",                     # 眠い / ねむい
    r"(寝る|ねる)(ね|よ|わ)?",          # 寝る / ねる / 寝るね
    r"また\s*(明日|あした)",            # また明日
    r"(疲|つか)れた",                   # 疲れた / つかれた
    r"ベッド\s*に\s*(入る|はいる)",     # ベッドに入る
]

# Arabic -- RTL script; use direct patterns.
WIND_DOWN_AR: list[str] = [
    r"تصبح\s+على\s+خير",
    r"ليلة\s+سعيدة",
    r"أنا\s+(ذاهب|ذاهبة)\s+(للنوم|إلى\s+النوم)",
    r"أنا\s+(متعب|متعبة|تعبان[ةه]?)",
    r"سأنام",
    r"وقت\s+النوم",
    r"إلى\s+(الغد|اللقاء\s+غدا)",
]

WIND_DOWN_DE: list[str] = [
    r"\bgute\s+nacht\b",
    r"\bgn8\b",
    r"\bich\s+gehe\s+(jetzt\s+)?(ins\s+bett|schlafen)\b",
    r"\b(ich\s+bin\s+)?(müde|kaputt|fertig)\b",
    r"\bschlafenszeit\b",
    r"\bbis\s+morgen\b",
    r"\blass\s+uns\s+morgen\s+weitermachen\b",
]

WIND_DOWN_FR: list[str] = [
    r"\bbonne\s+nuit\b",
    r"\bje\s+(vais|pars)\s+(me\s+coucher|dormir)\b",
    r"\b(je\s+suis\s+)?(fatigu[ée]|[ée]puis[ée])\b",
    r"\b(il\s+est\s+)?l'?heure\s+de\s+(dormir|me\s+coucher)\b",
    r"\b[aà]\s+demain\b",
    r"\bon\s+reprend\s+demain\b",
]

WIND_DOWN_ES: list[str] = [
    r"\bbuenas\s+noches\b",
    r"\bme\s+voy\s+a\s+(dormir|la\s+cama|descansar)\b",
    r"\b(estoy\s+)?(cansad[oa]|agotad[oa])\b",
    r"\bhora\s+de\s+dormir\b",
    r"\bhasta\s+ma[ñn]ana\b",
    r"\bseguimos\s+ma[ñn]ana\b",
]

WIND_DOWN_ZH: list[str] = [
    r"晚\s*安",                         # 晚安
    r"我\s*(要|去)\s*睡\s*(觉|了)",      # 我要睡觉 / 我去睡了
    r"累\s*了",                          # 累了
    r"(该|到)\s*睡\s*(觉)?\s*了",        # 该睡了 / 到睡觉了
    r"明\s*天\s*见",                     # 明天见
    r"明\s*天\s*继\s*续",                # 明天继续
]

# language coverage: exactly the 8 languages shield.py supports.
WIND_DOWN_BY_LANG: dict[str, list[str]] = {
    "en": WIND_DOWN_EN,
    "ru": WIND_DOWN_RU,
    "ja": WIND_DOWN_JA,
    "ar": WIND_DOWN_AR,
    "de": WIND_DOWN_DE,
    "fr": WIND_DOWN_FR,
    "es": WIND_DOWN_ES,
    "zh": WIND_DOWN_ZH,
}

# Pre-compile every pattern once. IGNORECASE is safe on non-Latin scripts
# (lowercasing is identity-preserving for CJK; Cyrillic handles cleanly).
_COMPILED: dict[str, list[re.Pattern]] = {
    lang: [re.compile(p, re.IGNORECASE) for p in pats]
    for lang, pats in WIND_DOWN_BY_LANG.items()
}

# Authoritative language set -- downstream greps against this constant.
WIND_DOWN_LANGUAGES_SUPPORTED: frozenset[str] = frozenset(WIND_DOWN_BY_LANG.keys())


# ------------------------------------------------------------ gate A: phrase match


def detect_wind_down_phrase(cue: str, language: str) -> Tuple[bool, str]:
    """Gate A: does the cue contain a wind-down phrase?

    Policy mirrors shield.py: primary language is always tried; ALSO try
    English regardless of `language` because users cross-lingual mid-sentence
    ("ok, going to sleep" in a Russian conversation is still a wind-down
    signal). We do NOT fall back to any other language beyond EN -- that
    would explode the FPR.

    Returns (matched, matched_pattern). matched_pattern is the source regex
    string (not the compiled object) for audit/logging purposes.
    """
    if not cue:
        return False, ""

    # Primary language (when different from "en").
    for p in _COMPILED.get(language or "", []):
        if p.search(cue):
            return True, p.pattern

    # Always also try EN if we haven't already.
    if language != "en":
        for p in _COMPILED["en"]:
            if p.search(cue):
                return True, p.pattern

    return False, ""


# ------------------------------------------------------------ gate B: late in quiet window


def is_late_in_quiet_window(
    window: Optional[Tuple[int, int]],
    now: datetime,
    tz: ZoneInfo,
) -> bool:
    """Gate B: is `now` inside the quiet window OR within 30min of its start?

    `window` is the (start_bucket, duration_buckets) pair emitted by
    `quiet_window.learn_quiet_window` -- start_bucket is an index into the
    48-bucket local-time day (30min each) and duration is the number of
    buckets. Returns False if no window is set (learn_quiet_window returned
    None, caller should be using the bootstrap 2h-idle trigger instead).

    Wrap-around: a window starting at 22:00 and lasting 8h crosses local
    midnight; "inside" then means `cur >= start_minutes` OR `cur < end_minutes`.
    """
    if not window:
        return False

    start_bucket, duration = window
    try:
        now_local = now.astimezone(tz)
    except (TypeError, ValueError, OverflowError):
        # DST edge or bad tz -- fail closed (don't suggest bedtime on
        # malformed input).
        return False

    cur_minutes = now_local.hour * 60 + now_local.minute
    start_minutes = start_bucket * BUCKET_MINUTES
    end_minutes = (start_bucket + duration) * BUCKET_MINUTES

    # Handle wrap-around midnight explicitly.
    if end_minutes > 24 * 60:
        wrapped_end = end_minutes - 24 * 60
        inside = cur_minutes >= start_minutes or cur_minutes < wrapped_end
    else:
        inside = start_minutes <= cur_minutes < end_minutes

    if inside:
        return True

    # Within 30min of start (cyclic -- a 21:45 cue for a 22:00 window counts).
    minutes_until_start = (start_minutes - cur_minutes) % (24 * 60)
    return 0 <= minutes_until_start <= WIND_DOWN_GATE_MINUTES_BEFORE


# ------------------------------------------------------------ dual-gate detector


def detect_wind_down(
    cue: str,
    language: str,
    state: dict,
    now: datetime,
    tz: ZoneInfo,
) -> Optional[dict]:
    """dual-gate bedtime detector.

    Returns a `sleep_suggestion` dict when BOTH gates pass:
      Gate A: wind-down phrase match (primary lang + EN fallback)
      Gate B: late-in-learned-quiet-window (inside OR within 30min of start)

    Returns None otherwise -- never a partial / fuzzy signal. Downstream
    consumers (`core._inject_sleep_suggestion`) key on the presence of the
    key, so None means the response simply does not carry `sleep_suggestion`.

    Payload shape (small, no PII beyond the matched regex pattern):
        {
            "message_hint": "user_wind_down_detected",
            "matched_pattern": str,
            "quiet_window_start_bucket": int,
            "quiet_window_duration": int,
        }
    """
    matched, pattern = detect_wind_down_phrase(cue, language)
    if not matched:
        return None

    window = state.get("quiet_window") if isinstance(state, dict) else None
    if not window:
        return None
    if not is_late_in_quiet_window(window, now, tz):
        return None

    start_bucket, duration = window
    return {
        "message_hint": "user_wind_down_detected",
        "matched_pattern": pattern,
        "quiet_window_start_bucket": int(start_bucket),
        "quiet_window_duration": int(duration),
    }


__all__ = [
    "WIND_DOWN_AR",
    "WIND_DOWN_BY_LANG",
    "WIND_DOWN_DE",
    "WIND_DOWN_EN",
    "WIND_DOWN_ES",
    "WIND_DOWN_FR",
    "WIND_DOWN_GATE_MINUTES_BEFORE",
    "WIND_DOWN_JA",
    "WIND_DOWN_LANGUAGES_SUPPORTED",
    "WIND_DOWN_RU",
    "WIND_DOWN_ZH",
    "detect_wind_down",
    "detect_wind_down_phrase",
    "is_late_in_quiet_window",
]
