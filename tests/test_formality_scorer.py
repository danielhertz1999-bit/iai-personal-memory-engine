"""Plan 03-03 Task 1 RED + Task 2 GREEN — surface-feature formality scorer.

Validates the formality scorer against a RU+EN fixture of ~50 formal/informal pairs.
Constitutional guard: the scorer observes ONLY the user's surface text. There is no
user-internal-state signal anywhere in this test or in the module it tests.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "formality_ru_en_50pairs.json"


def _load_fixture():
    with FIXTURE_PATH.open() as f:
        return json.load(f)


# ------------------------------------------------------------- fixture integrity
def test_fixture_loads_and_has_enough_pairs():
    pairs = _load_fixture()
    assert len(pairs) >= 45, f"expected ~50 pairs, got {len(pairs)}"
    langs = {p["lang"] for p in pairs}
    assert "en" in langs and "ru" in langs


def test_fixture_shape():
    pairs = _load_fixture()
    for p in pairs:
        assert set(p.keys()) >= {"id", "lang", "formal", "informal"}
        assert isinstance(p["formal"], str) and p["formal"].strip()
        assert isinstance(p["informal"], str) and p["informal"].strip()


# ------------------------------------------------------------- scorer contract
def test_formality_score_fixture_accuracy_at_least_85_percent():
    """Formal text must score > informal text on >= 85% of pairs."""
    from iai_mcp.formality import formality_score

    pairs = _load_fixture()
    wins = sum(
        1
        for p in pairs
        if formality_score(p["formal"], p["lang"]) > formality_score(p["informal"], p["lang"])
    )
    accuracy = wins / len(pairs)
    assert accuracy >= 0.85, f"accuracy {accuracy:.2%} ({wins}/{len(pairs)}) below 85% floor"


def test_formality_score_en_formal_anchor():
    from iai_mcp.formality import formality_score

    score = formality_score("The proposal is, therefore, accepted.", "en")
    assert score >= 0.6, f"expected highly formal sentence >= 0.6, got {score:.3f}"


def test_formality_score_en_informal_anchor():
    from iai_mcp.formality import formality_score

    score = formality_score("yo, works for me lol", "en")
    assert score <= 0.3, f"expected clearly informal <= 0.3, got {score:.3f}"


def test_formality_score_unknown_lang_returns_neutral_with_warning():
    """MEMORY.md global-product mandate: unknown lang degrades gracefully."""
    from iai_mcp.formality import formality_score

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        score = formality_score("some test text", "zz")
    assert score == 0.5
    # A warning must have been issued.
    assert any("formality_score" in str(w.message).lower() or "zz" in str(w.message) for w in w_list)


def test_formality_score_unknown_lang_never_raises():
    from iai_mcp.formality import formality_score

    # Must never raise, regardless of the lang string.
    for bad_lang in ("", "zz", "xx", "de", "fr"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = formality_score("test", bad_lang)


def test_formality_score_empty_text_returns_zero():
    from iai_mcp.formality import formality_score

    assert formality_score("", "en") == 0.0
    assert formality_score("   ", "en") == 0.0


def test_formality_score_range_bounded_in_0_1():
    from iai_mcp.formality import formality_score

    pairs = _load_fixture()
    for p in pairs:
        for txt in (p["formal"], p["informal"]):
            s = formality_score(txt, p["lang"])
            assert 0.0 <= s <= 1.0, f"score {s} out of [0, 1] for {txt!r}"
