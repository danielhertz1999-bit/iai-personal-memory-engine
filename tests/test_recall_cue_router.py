"""Cue-detection router tests.

Covers:
- classifier function _classify_cue (8 unit tests + parameterized).
- dispatch integration (5 tests appended after the wiring lands).

Acceptance:
- 6 verbatim-positive cues (3 EN + 3 RU) covered.
- 6 concept-negative cues covered.
- triggered_pattern label surfaced for diagnostics (logged, not in response).
- case-insensitivity for EN word-marker.
- RU patterns anchored at start-of-string.
"""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------- Task 1


def test_module_exposes_compiled_trigger_lists():
    """EN_TRIGGERS and RU_TRIGGERS must be present and contain 4 entries each."""
    from iai_mcp.cue_router import EN_TRIGGERS, RU_TRIGGERS

    assert len(EN_TRIGGERS) == 4, f"EN_TRIGGERS must have 4 entries, got {len(EN_TRIGGERS)}"
    assert len(RU_TRIGGERS) == 4, f"RU_TRIGGERS must have 4 entries, got {len(RU_TRIGGERS)}"

    # Each entry is (label, compiled_pattern)
    for label, pat in EN_TRIGGERS:
        assert isinstance(label, str) and label, "EN trigger label must be non-empty str"
        assert hasattr(pat, "search"), f"EN trigger pattern for {label!r} must be compiled regex"
    for label, pat in RU_TRIGGERS:
        assert isinstance(label, str) and label, "RU trigger label must be non-empty str"
        assert hasattr(pat, "search"), f"RU trigger pattern for {label!r} must be compiled regex"


def test_module_exposes_historical_trigger_lists():
    """EN_HISTORICAL_TRIGGERS + RU_HISTORICAL_TRIGGERS exist with 5/4 entries."""
    from iai_mcp.cue_router import EN_HISTORICAL_TRIGGERS, RU_HISTORICAL_TRIGGERS

    assert len(EN_HISTORICAL_TRIGGERS) == 5, (
        f"EN_HISTORICAL_TRIGGERS must have 5 entries, got {len(EN_HISTORICAL_TRIGGERS)}"
    )
    assert len(RU_HISTORICAL_TRIGGERS) == 4, (
        f"RU_HISTORICAL_TRIGGERS must have 4 entries, got {len(RU_HISTORICAL_TRIGGERS)}"
    )
    for label, pat in EN_HISTORICAL_TRIGGERS:
        assert isinstance(label, str) and label.startswith("historical-en-")
        assert hasattr(pat, "search")
    for label, pat in RU_HISTORICAL_TRIGGERS:
        assert isinstance(label, str) and label.startswith("historical-ru-")
        assert hasattr(pat, "search")


@pytest.mark.parametrize(
    "cue",
    [
        "find the verbatim quote about migration",
        "what did the user say on day 17?",
        "show me the exact phrase about cleanup",
    ],
)
def test_classify_cue_en_verbatim_positives(cue):
    """3 EN verbatim-positive cues each return mode=verbatim."""
    from iai_mcp.cue_router import _classify_cue

    mode, _intent, pattern = _classify_cue(cue)
    assert mode == "verbatim", f"cue {cue!r} should classify as verbatim, got {mode!r}"
    assert pattern is not None, f"cue {cue!r} should report a triggered_pattern label"


def test_classify_cue_en_quoted_phrase():
    """EN positive: cue containing a "..." quoted phrase routes to verbatim."""
    from iai_mcp.cue_router import _classify_cue

    mode, _intent, pattern = _classify_cue('recall "lancedb pre-cleanup snapshot" verbatim')
    assert mode == "verbatim"
    # quoted-phrase OR word-marker may match first; both label types are valid.
    assert pattern in ("quoted-phrase", "word-marker"), (
        f"expected quoted-phrase or word-marker label, got {pattern!r}"
    )


@pytest.mark.parametrize(
    "cue",
    [
        "найди дословно сообщение о схема-чистке",
        "точная цитата про deg_norm",
        "что я сказал в прошлой сессии о dedup",
    ],
)
def test_classify_cue_ru_verbatim_positives(cue):
    """3 RU starts-with cues each return mode=verbatim."""
    from iai_mcp.cue_router import _classify_cue

    mode, _intent, pattern = _classify_cue(cue)
    assert mode == "verbatim", f"cue {cue!r} should classify as verbatim, got {mode!r}"
    assert pattern is not None, f"cue {cue!r} should report a triggered_pattern label"
    assert pattern.startswith("ru-start-"), (
        f"expected ru-start-* label for cue {cue!r}, got {pattern!r}"
    )


def test_classify_cue_ru_european_quote_marker():
    """EN positive (european-quote): cue with «...» routes to verbatim."""
    from iai_mcp.cue_router import _classify_cue

    mode, _intent, pattern = _classify_cue('recall the «schema_reinforced event payload» definition')
    assert mode == "verbatim"
    assert pattern == "european-quote"


@pytest.mark.parametrize(
    "cue",
    [
        "tell me about schema dedup",
        "how does the rank stage work",
        "community structure of the live store",
        "каков статус Phase 6",
        "sleep daemon REM cycle behaviour",
        "что нового в проекте",
    ],
)
def test_classify_cue_concept_negatives(cue):
    """6 concept-negative cues each return mode=concept and triggered_pattern=None."""
    from iai_mcp.cue_router import _classify_cue

    mode, _intent, pattern = _classify_cue(cue)
    assert mode == "concept", f"cue {cue!r} should classify as concept, got {mode!r}"
    assert pattern is None, f"cue {cue!r} should not have a triggered_pattern, got {pattern!r}"


def test_classify_cue_triggered_pattern_label_non_none_for_verbatim():
    """Every verbatim cue carries a non-None triggered_pattern; every concept cue carries None."""
    from iai_mcp.cue_router import _classify_cue

    verbatim_cues = [
        "verbatim quote please",
        "what I said on day 7",
        '"quoted text"',
        "найди дословно вот это",
    ]
    for cue in verbatim_cues:
        mode, _intent, pattern = _classify_cue(cue)
        assert mode == "verbatim", f"{cue!r} -> mode {mode!r}"
        assert pattern is not None, f"{cue!r} -> pattern None"

    concept_cues = [
        "what is the architecture",
        "general project status",
        "опиши структуру проекта",
    ]
    for cue in concept_cues:
        mode, _intent, pattern = _classify_cue(cue)
        assert mode == "concept", f"{cue!r} -> mode {mode!r}"
        assert pattern is None, f"{cue!r} -> pattern {pattern!r}"


def test_classify_cue_case_insensitive_en():
    """EN word-marker honours re.IGNORECASE: VERBATIM, EXACT, Quote all match."""
    from iai_mcp.cue_router import _classify_cue

    for cue in ("VERBATIM what did I say", "EXACT phrase", "Quote me on this"):
        mode, _intent, _pat = _classify_cue(cue)
        assert mode == "verbatim", f"case-insensitive match failed for {cue!r}"


def test_classify_cue_ru_patterns_anchored_at_start():
    """RU triggers require the cue to START with the phrase; mid-string match returns concept."""
    from iai_mcp.cue_router import _classify_cue

    # Mid-string occurrence -> concept (RU patterns anchored ^ at start).
    mode_mid, _intent_mid, pattern_mid = _classify_cue("remind me, найди дословно not in middle")
    assert mode_mid == "concept", (
        f"RU trigger should NOT match mid-string, got mode={mode_mid!r} pattern={pattern_mid!r}"
    )

    # Start-of-string occurrence -> verbatim.
    mode_start, _intent_start, pattern_start = _classify_cue("найди дословно вот эту фразу")
    assert mode_start == "verbatim"
    assert pattern_start == "ru-start-найди-дословно"


def test_classify_cue_empty_string_returns_concept():
    """Empty / None-ish cue returns concept (defensive default)."""
    from iai_mcp.cue_router import _classify_cue

    mode, intent, pattern = _classify_cue("")
    assert mode == "concept"
    assert intent is None
    assert pattern is None


# ============================================================================
# historical_verbatim intent classification
# ============================================================================


def test_classify_cue_bench_failing_cue_routes_historical_verbatim():
    """The exact bench-failing cue 'Quote the original ETA wording.' routes
    to (verbatim, historical_verbatim) — verbatim mode via 'quote' word-marker,
    historical_verbatim intent via 'original' EN historical trigger.
    """
    from iai_mcp.cue_router import _classify_cue

    mode, intent, label = _classify_cue("Quote the original ETA wording.")
    assert mode == "verbatim", f"bench cue should be verbatim, got {mode!r}"
    assert intent == "historical_verbatim", (
        f"bench cue should be historical_verbatim, got {intent!r}"
    )
    # word-marker matches "Quote" first; label is the verbatim trigger label.
    assert label == "word-marker", f"expected word-marker label, got {label!r}"


@pytest.mark.parametrize(
    "cue",
    [
        "what was the first plan?",
        "before the change",
        "earlier statement",
        "previously mentioned",
        "the originally proposed approach",
        "the initial design document",
        "what was first about auth?",
    ],
)
def test_classify_cue_en_historical_markers(cue):
    """EN historical markers (original, before, first, earlier, previously) flag intent."""
    from iai_mcp.cue_router import _classify_cue

    _mode, intent, _label = _classify_cue(cue)
    assert intent == "historical_verbatim", (
        f"cue {cue!r} should have intent=historical_verbatim, got {intent!r}"
    )


@pytest.mark.parametrize(
    "cue",
    [
        "приведи оригинальную формулировку",
        "что было сначала?",
        "изначальный план",
        "изначально мы говорили",
        "ранее упомянутое",
        "оригинальный текст про auth",
    ],
)
def test_classify_cue_ru_historical_markers(cue):
    """RU historical markers (оригинал*, сначала, изначальн*, ранее) flag intent."""
    from iai_mcp.cue_router import _classify_cue

    _mode, intent, _label = _classify_cue(cue)
    assert intent == "historical_verbatim", (
        f"cue {cue!r} should have intent=historical_verbatim, got {intent!r}"
    )


@pytest.mark.parametrize(
    "cue",
    [
        "what about auth",
        "Quote the auth tokens.",  # verbatim but NOT historical
        "tell me about schema dedup",
        "каков статус системы",
        '"exact phrase about db migration"',  # verbatim quoted, no historical marker
    ],
)
def test_classify_cue_neutral_no_historical_intent(cue):
    """Neutral / non-historical verbatim cues do NOT trigger historical_verbatim."""
    from iai_mcp.cue_router import _classify_cue

    _mode, intent, _label = _classify_cue(cue)
    assert intent is None, (
        f"cue {cue!r} should NOT have historical_verbatim intent, got {intent!r}"
    )


def test_classify_cue_historical_intent_orthogonal_to_mode():
    """Historical intent fires independently of mode: even concept-mode cues
    carrying a historical marker get intent="historical_verbatim".
    """
    from iai_mcp.cue_router import _classify_cue

    # Concept mode + historical marker — possible mid-sentence
    mode, intent, _label = _classify_cue("tell me about the original auth approach")
    # mode may be "concept" (no verbatim trigger matched) — intent still fires
    assert intent == "historical_verbatim", (
        f"historical intent should be orthogonal to mode; got mode={mode!r} intent={intent!r}"
    )


def test_classify_cue_ru_historical_uses_word_boundary_not_anchor():
    """Open Question 2 resolution: RU historical triggers use \\b (word boundary),
    NOT ^ anchor — they fire mid-cue, not just at start.
    """
    from iai_mcp.cue_router import _classify_cue

    # Mid-cue Russian historical marker
    _mode, intent, _label = _classify_cue("напомни мне, что было изначально в плане")
    assert intent == "historical_verbatim", (
        f"mid-cue RU historical marker should fire intent; got {intent!r}"
    )


# ============================================================================
# Task 3 — dispatch integration tests
# ============================================================================

# Reuses the _ControlledEmbedder pattern + helper builders so
# dispatch end-to-end tests can pin the embedder side-effect (advisor #5).


from datetime import datetime, timezone  # noqa: E402 -- co-located fixtures
from uuid import uuid4  # noqa: E402

import numpy as np  # noqa: E402

from iai_mcp.types import EMBED_DIM, MemoryRecord  # noqa: E402


class _DispatchEmbedder:
    """Lightweight embedder for the dispatch tests — pins fixed cue vectors
    so dispatch's embedder_for_store-loaded bge does not destroy the
    hand-crafted geometry. Same trick as 's deviation #1.
    """

    DIM = EMBED_DIM

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self.fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self.fixed:
            return list(self.fixed[text])
        import hashlib
        import random
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


def _seed_populated_store(tmp_path):
    """Seed a store with one episodic record matching the test cues so
    dispatch returns a non-empty hits list under either mode.
    """
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "hippo")
    embedder = _DispatchEmbedder()

    cue_text = "verbatim quote about migration snapshot"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    # One episodic record whose embedding matches the cue (cos=1.0).
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="verbatim record about migration snapshot",
        aaak_index="",
        embedding=list(cue_vec),
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
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)

    return store, embedder, cue_text, rec


def test_dispatch_routes_verbatim_cue_to_verbatim_mode(tmp_path, monkeypatch):
    """Dispatch with a populated store + verbatim cue: response.cue_mode == 'verbatim'."""
    from iai_mcp import core
    from iai_mcp import embed as _embed_mod

    store, embedder, cue, rec = _seed_populated_store(tmp_path)
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: embedder)

    response = core.dispatch(
        store, "memory_recall",
        {"cue": cue, "session_id": "verb_cue", "cue_embedding": embedder.embed(cue)},
    )
    assert response["cue_mode"] == "verbatim", (
        f"verbatim cue should classify to verbatim mode, got {response['cue_mode']!r}"
    )
    assert "patterns_observed" in response, "patterns_observed must be in response"
    assert isinstance(response["patterns_observed"], list)


def test_dispatch_routes_concept_cue_to_concept_mode(tmp_path, monkeypatch):
    """Dispatch with a populated store + concept cue: response.cue_mode == 'concept'."""
    from iai_mcp import core
    from iai_mcp import embed as _embed_mod

    store, embedder, _cue, _rec = _seed_populated_store(tmp_path)
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: embedder)

    concept_cue = "tell me about cleanup"
    embedder.set_fixed(concept_cue, embedder.embed(concept_cue))
    response = core.dispatch(
        store, "memory_recall",
        {"cue": concept_cue, "session_id": "concept_cue",
         "cue_embedding": embedder.embed(concept_cue)},
    )
    assert response["cue_mode"] == "concept", (
        f"concept cue should classify to concept mode, got {response['cue_mode']!r}"
    )
    assert "patterns_observed" in response


def test_dispatch_empty_store_fallback_honours_classified_mode(tmp_path):
    """records_count==0 path: retrieve.recall is exercised; cue_mode reflects the
    classifier's verdict; hits[] empty (no records to return) but the field is
    still episodic-only-shaped (verbatim filter wouldn't matter)."""
    from iai_mcp import core
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "hippo")  # empty
    response = core.dispatch(
        store, "memory_recall",
        {"cue": "verbatim quote please", "session_id": "fallback",
         "cue_embedding": [0.0] * EMBED_DIM},
    )
    assert response["cue_mode"] == "verbatim", (
        f"verbatim cue should classify even on the fallback (empty-store) path, "
        f"got {response['cue_mode']!r}"
    )


def test_dispatch_passes_mode_kwarg_to_recall_for_response(tmp_path, monkeypatch):
    """Monkeypatch recall_for_response to capture kwargs; assert mode kwarg passed.

      entry-point split: core.dispatch calls recall_for_response
    (production answer-packing) instead of the deleted pipeline_recall.
    The mode-plumbing acceptance criterion is preserved verbatim — the
    cue-classifier output flows unchanged into the new entry point.
    """
    from iai_mcp import core
    from iai_mcp import pipeline as _pipeline_mod
    from iai_mcp import embed as _embed_mod
    from iai_mcp.types import RecallResponse

    store, embedder, _cue, _rec = _seed_populated_store(tmp_path)
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: embedder)

    captured: dict = {}

    def fake_recall_for_response(**kwargs):
        captured.update(kwargs)
        return RecallResponse(
            hits=[], anti_hits=[], activation_trace=[], budget_used=0,
            cue_mode=kwargs.get("mode", "concept"),
            patterns_observed=[],
        )

    monkeypatch.setattr(_pipeline_mod, "recall_for_response", fake_recall_for_response)

    verbatim_cue = "verbatim recall this exact quote"
    response = core.dispatch(
        store, "memory_recall",
        {"cue": verbatim_cue, "session_id": "kwarg_capture",
         "cue_embedding": embedder.embed(verbatim_cue)},
    )
    assert "mode" in captured, "dispatch must pass mode kwarg to recall_for_response"
    assert captured["mode"] == "verbatim", (
        f"verbatim cue should propagate as mode='verbatim' to recall_for_response, "
        f"got mode={captured.get('mode')!r}"
    )
    assert response["cue_mode"] == "verbatim"


def test_dispatch_passes_mode_kwarg_to_retrieve_recall(tmp_path, monkeypatch):
    """Empty-store fallback path: monkeypatch retrieve.recall, assert mode passed."""
    from iai_mcp import core
    from iai_mcp import retrieve as _retrieve_mod
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import RecallResponse

    store = MemoryStore(path=tmp_path / "hippo")  # empty -> fallback path

    captured: dict = {}

    def fake_recall(**kwargs):
        captured.update(kwargs)
        return RecallResponse(
            hits=[], anti_hits=[], activation_trace=[], budget_used=0,
            cue_mode=kwargs.get("mode", "verbatim"),
            patterns_observed=[],
        )

    monkeypatch.setattr(_retrieve_mod, "recall", fake_recall)

    response = core.dispatch(
        store, "memory_recall",
        {"cue": "verbatim quote about something", "session_id": "fallback_kwarg",
         "cue_embedding": [0.0] * EMBED_DIM},
    )
    assert "mode" in captured, (
        "dispatch must pass mode kwarg to retrieve.recall on empty-store fallback"
    )
    assert captured["mode"] == "verbatim", (
        f"verbatim cue should propagate as mode='verbatim' to retrieve.recall, "
        f"got mode={captured.get('mode')!r}"
    )
    assert response["cue_mode"] == "verbatim"
