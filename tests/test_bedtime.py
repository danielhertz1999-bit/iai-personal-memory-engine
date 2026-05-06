"""Tests for iai_mcp.bedtime -- Task 1.

Covers 14 behaviours from the plan:
1. English positive -- "good night" / "heading to bed" / "tired"
2. English negative (phrase alone, no dual-gate)
3. Russian positive
4. Japanese positive
5. Arabic positive
6. de/fr/es/zh positive (one phrase per language at minimum)
7. Cross-lingual fallback -- EN always tried; RU NOT tried under language="en"
8. Dual-gate: phrase alone NOT enough (no quiet window -> None)
9. Dual-gate: inside quiet window -> dict
10. Dual-gate: within 30min of start -> dict
11. Dual-gate: 1h before start -> None
12. Fixture-driven corpus: 5 positive + 5 negative per language
13. False positive rate < 10% on phrase-only check across all 8 fixtures
14. ReDoS protection: 10KB input under 100ms total across all patterns
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iai_mcp import bedtime
from iai_mcp.bedtime import (
    WIND_DOWN_BY_LANG,
    WIND_DOWN_GATE_MINUTES_BEFORE,
    WIND_DOWN_LANGUAGES_SUPPORTED,
    detect_wind_down,
    detect_wind_down_phrase,
    is_late_in_quiet_window,
)

UTC = timezone.utc
FIXTURES = Path(__file__).parent / "fixtures" / "bedtime"


# ---------------------------------------------------------------- phrase gate


def test_english_positive() -> None:
    for cue in [
        "good night",
        "I'm heading to bed",
        "I'm tired, going to sleep",
        "catch you tomorrow",
        "it's bedtime",
        "Goodnight!",
    ]:
        matched, pattern = detect_wind_down_phrase(cue, "en")
        assert matched, f"expected EN positive for {cue!r}"
        assert pattern


def test_english_phrase_matches_even_rhetorical() -> None:
    """Phrase alone IS enough for the phrase gate -- the dual gate adds
    the quiet-window filter. This test locks the phrase behaviour in
    isolation so dual-gate tests can differentiate."""
    cue = "the villain said good night and laughed"
    matched, pattern = detect_wind_down_phrase(cue, "en")
    assert matched, "phrase gate alone is intentionally permissive"
    assert "night" in pattern.lower()


def test_russian_positive() -> None:
    for cue in [
        "пойду спать",
        "спокойной ночи",
        "устал, иду в постель",
        "до завтра",
        "пора ложиться",
    ]:
        matched, _ = detect_wind_down_phrase(cue, "ru")
        assert matched, f"expected RU positive for {cue!r}"


def test_japanese_positive() -> None:
    for cue in [
        "おやすみ",
        "おやすみなさい",
        "寝ます",
        "また明日",
        "疲れた",
    ]:
        matched, _ = detect_wind_down_phrase(cue, "ja")
        assert matched, f"expected JA positive for {cue!r}"


def test_arabic_positive() -> None:
    for cue in [
        "تصبح على خير",
        "ليلة سعيدة",
        "أنا متعب سأنام",
    ]:
        matched, _ = detect_wind_down_phrase(cue, "ar")
        assert matched, f"expected AR positive for {cue!r}"


def test_de_fr_es_zh_positive() -> None:
    cases: dict[str, list[str]] = {
        "de": ["gute Nacht", "ich bin müde", "bis morgen"],
        "fr": ["bonne nuit", "je suis fatigué", "à demain"],
        "es": ["buenas noches", "estoy cansado", "hasta mañana"],
        "zh": ["晚安", "我要睡觉", "累了"],
    }
    for lang, cues in cases.items():
        for cue in cues:
            matched, _ = detect_wind_down_phrase(cue, lang)
            assert matched, f"expected {lang.upper()} positive for {cue!r}"


def test_cross_lingual_en_is_fallback_but_ru_is_not() -> None:
    # EN fallback always tried: "good night" under language="ru" still matches.
    matched_en_under_ru, _ = detect_wind_down_phrase("good night", "ru")
    assert matched_en_under_ru, "EN fallback must trigger regardless of language"

    # RU is NOT tried under language="en": a purely Russian cue must NOT match.
    matched_ru_under_en, _ = detect_wind_down_phrase("я пойду спать", "en")
    assert not matched_ru_under_en, (
        "RU phrases must not fall back under language=en"
    )


def test_phrase_empty_cue_no_match() -> None:
    assert detect_wind_down_phrase("", "en") == (False, "")
    assert detect_wind_down_phrase("", "ru") == (False, "")


def test_phrase_unknown_language_still_tries_english() -> None:
    """Language we don't support (e.g. 'ko') must still try EN fallback."""
    matched, _ = detect_wind_down_phrase("good night", "ko")
    assert matched, "EN fallback required for unsupported languages too"


# ---------------------------------------------------------------- quiet-window gate


def _utc(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def test_is_late_no_window() -> None:
    assert is_late_in_quiet_window(None, _utc(2026, 4, 18, 22, 0), UTC) is False


def test_is_late_inside_window() -> None:
    # window = (44, 16) means start at bucket 44 = 22:00, duration 8h.
    # 23:30 local should be inside.
    assert is_late_in_quiet_window(
        (44, 16), _utc(2026, 4, 18, 23, 30), UTC,
    ) is True


def test_is_late_within_30min_of_start() -> None:
    # start 22:00, now 21:45 -> within 30min -> True.
    assert is_late_in_quiet_window(
        (44, 16), _utc(2026, 4, 18, 21, 45), UTC,
    ) is True


def test_is_late_exactly_30min_before_start() -> None:
    # Boundary: 21:30 should still count (within 30min threshold, inclusive).
    assert is_late_in_quiet_window(
        (44, 16), _utc(2026, 4, 18, 21, 30), UTC,
    ) is True


def test_is_late_one_hour_before_start() -> None:
    # start 22:00, now 21:00 -> 60min before -> False.
    assert is_late_in_quiet_window(
        (44, 16), _utc(2026, 4, 18, 21, 0), UTC,
    ) is False


def test_is_late_window_wraps_midnight() -> None:
    # window = (44, 16): 22:00 start + 8h = 06:00 next morning.
    # 02:30 local should be inside (post-midnight part of the window).
    assert is_late_in_quiet_window(
        (44, 16), _utc(2026, 4, 19, 2, 30), UTC,
    ) is True


def test_is_late_outside_window_afternoon() -> None:
    # window = (44, 16): 22:00-06:00. 15:00 afternoon -> outside + not within 30min.
    assert is_late_in_quiet_window(
        (44, 16), _utc(2026, 4, 18, 15, 0), UTC,
    ) is False


# ---------------------------------------------------------------- dual-gate


def test_dual_gate_phrase_alone_not_enough() -> None:
    # Phrase matches but no quiet window set -> None.
    result = detect_wind_down(
        "good night", "en", state={}, now=_utc(2026, 4, 18, 12, 0), tz=UTC,
    )
    assert result is None


def test_dual_gate_no_phrase_inside_window() -> None:
    # Inside window but no phrase match -> None.
    result = detect_wind_down(
        "let me check the code",
        "en",
        state={"quiet_window": (44, 16)},
        now=_utc(2026, 4, 18, 23, 30),
        tz=UTC,
    )
    assert result is None


def test_dual_gate_both_pass_inside_window() -> None:
    result = detect_wind_down(
        "good night",
        "en",
        state={"quiet_window": (44, 16)},
        now=_utc(2026, 4, 18, 23, 30),
        tz=UTC,
    )
    assert result is not None
    assert result["message_hint"] == "user_wind_down_detected"
    assert "night" in result["matched_pattern"].lower()
    assert result["quiet_window_start_bucket"] == 44
    assert result["quiet_window_duration"] == 16


def test_dual_gate_both_pass_30min_before_window() -> None:
    # 21:45 local, window starts 22:00 -> within 30min threshold.
    result = detect_wind_down(
        "good night",
        "en",
        state={"quiet_window": (44, 16)},
        now=_utc(2026, 4, 18, 21, 45),
        tz=UTC,
    )
    assert result is not None
    assert result["quiet_window_start_bucket"] == 44


def test_dual_gate_phrase_but_too_early() -> None:
    # 21:00 local, window starts 22:00 -> 60min too early -> None.
    result = detect_wind_down(
        "good night",
        "en",
        state={"quiet_window": (44, 16)},
        now=_utc(2026, 4, 18, 21, 0),
        tz=UTC,
    )
    assert result is None


# ---------------------------------------------------------------- fixture corpus


_LANGS = sorted(WIND_DOWN_BY_LANG.keys())


@pytest.mark.parametrize("lang", _LANGS)
def test_fixture_corpus(lang: str) -> None:
    fp = FIXTURES / f"{lang}.txt"
    assert fp.exists(), f"fixture file missing: {fp}"
    lines = [
        ln.strip()
        for ln in fp.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert len(lines) >= 10, f"{lang}: expected >=10 fixture lines, got {len(lines)}"

    for line in lines:
        assert "\t" in line, f"{lang}: fixture line missing tab separator: {line!r}"
        sentence, expected = line.rsplit("\t", 1)
        matched, _ = detect_wind_down_phrase(sentence, lang)
        assert matched == (expected == "yes"), (
            f"{lang}: {sentence!r} expected {expected} got {matched}"
        )


def test_fixture_corpus_false_positive_rate_under_10_percent() -> None:
    """Across all 8 languages (80 lines = 40 pos + 40 neg), the phrase-only
    false positive rate MUST be < 10%. The dual gate ratchets this down to
    the target of <5% in practice."""
    fp_count = 0
    neg_total = 0
    for lang in _LANGS:
        fp = FIXTURES / f"{lang}.txt"
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            sentence, expected = line.rsplit("\t", 1)
            if expected == "no":
                neg_total += 1
                matched, _ = detect_wind_down_phrase(sentence, lang)
                if matched:
                    fp_count += 1
    assert neg_total >= 40, f"expected >=40 negative fixtures, got {neg_total}"
    fpr = fp_count / neg_total
    assert fpr < 0.10, (
        f"phrase-only FPR {fpr:.2%} exceeds 10% ceiling "
        f"({fp_count}/{neg_total}). Tighten fixtures or patterns."
    )


# ---------------------------------------------------------------- ReDoS guard


def test_redos_protection_bounded_quantifiers_under_100ms() -> None:
    """All patterns are pre-compiled and use bounded quantifiers.
    10KB of 'a' characters must execute in < 100ms across every pattern."""
    big = "a" * 10240
    deadline = 0.100  # seconds
    total_start = time.monotonic()
    for lang, patterns in bedtime._COMPILED.items():
        for p in patterns:
            t0 = time.monotonic()
            p.search(big)
            if time.monotonic() - t0 > deadline:
                pytest.fail(
                    f"ReDoS suspected: {lang} pattern {p.pattern!r} took "
                    f">{deadline}s on 10KB input"
                )
    total_elapsed = time.monotonic() - total_start
    assert total_elapsed < 1.0, (
        f"combined ReDoS sweep took {total_elapsed:.3f}s (budget 1.0s)"
    )


# ---------------------------------------------------------------- coverage sanity


def test_language_coverage_is_exactly_eight_d11() -> None:
    """wind-down regex must cover exactly the 8 shield.py languages."""
    assert WIND_DOWN_LANGUAGES_SUPPORTED == frozenset(
        {"en", "ru", "ja", "ar", "de", "fr", "es", "zh"},
    )
    assert len(WIND_DOWN_BY_LANG) == 8


def test_gate_minutes_before_is_thirty_d09() -> None:
    """D-09 dual-gate: 30 minutes before quiet-window start counts as late."""
    assert WIND_DOWN_GATE_MINUTES_BEFORE == 30
