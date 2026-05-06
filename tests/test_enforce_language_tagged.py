"""Tests for enforce_language_tagged (Plan 02-01, constitutional).

Phase 1's enforce_english_raw gated storage to English-only. amends to
native-language storage: every record carries a language tag; the guard
function only raises if the tag is missing or auto-detection is low confidence.

enforce_english_raw is retained as a backward-compat shim for callers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(text: str, language: str = "", tags: list[str] | None = None) -> MemoryRecord:
    """Build a MemoryRecord with an overridable language tag.

    When language="" we would normally fail __post_init__, but we need to
    exercise the "missing tag" enforcement path. So we set a placeholder
    language="XX" when the caller asks for empty and the guard will fail
    accordingly via its own checks.
    """
    # For tests that probe missing language, pass "XX" (still valid non-empty)
    # and then zero it out on the record after construction.
    actual_lang = language if language else "XX"
    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language=actual_lang,
    )
    if not language:
        # Post-construction: simulate "record missing language" for the guard.
        r.language = ""
    return r


# ---------------------------------------------------- enforce_language_tagged


def test_enforce_language_tagged_accepts_english_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("hello world", language="en")
    enforce_language_tagged(r)  # should not raise


def test_enforce_language_tagged_accepts_russian_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("привет мир", language="ru")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_japanese_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("こんにちは", language="ja")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_arabic_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("مرحبا بالعالم", language="ar")
    enforce_language_tagged(r)


def test_enforce_language_tagged_rejects_missing_language_no_detect():
    """record.language="" without detect=True must raise."""
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("some text", language="")  # simulates un-tagged record
    with pytest.raises(ValueError) as exc:
        enforce_language_tagged(r)
    assert "constitutional" in str(exc.value).lower()


def test_enforce_language_tagged_auto_detect_sets_language():
    """When detect=True and language empty, runs langdetect and mutates record."""
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec(
        "This is a reasonable English sentence with enough words for detection.",
        language="",
    )
    enforce_language_tagged(r, detect=True)
    assert r.language == "en"


def test_enforce_language_tagged_auto_detect_russian():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec(
        "Это осмысленное предложение на русском языке с достаточным количеством слов.",
        language="",
    )
    enforce_language_tagged(r, detect=True)
    assert r.language == "ru"


def test_enforce_language_tagged_empty_text_gets_default_en():
    """Empty literal_surface + detect=True falls through to 'en' default."""
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("", language="")
    enforce_language_tagged(r, detect=True)
    assert r.language == "en"


# ------------------------------------------------ enforce_english_raw shim


def test_enforce_english_raw_still_importable():
    """Backward compat: the Phase-1 guard is still a valid import."""
    from iai_mcp.aaak import enforce_english_raw

    assert callable(enforce_english_raw)


def test_enforce_english_raw_with_language_tag_still_phase1_semantics():
    """The shim preserves semantics: even with language='ru' set,
    untagged Cyrillic literal_surface WITHOUT 'raw:<lang>' tag still raises.

    callers who want native-language storage should call
    `enforce_language_tagged` instead of this shim.
    """
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="ru")
    with pytest.raises(ValueError):
        enforce_english_raw(r)


def test_enforce_english_raw_still_blocks_untagged_cyrillic():
    """Phase 1 behaviour preserved for untagged records (language="")."""
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="")
    with pytest.raises(ValueError) as exc:
        enforce_english_raw(r)
    assert "constitutional" in str(exc.value).lower()


def test_enforce_english_raw_accepts_cyrillic_with_raw_tag():
    """Phase-1 raw:<lang> tag exception still works through the shim."""
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="", tags=["raw:ru"])
    enforce_english_raw(r)


def test_enforce_english_raw_accepts_pure_english():
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("hello world", language="")
    enforce_english_raw(r)
