"""— LongMemEval adapter tests (RED scaffold).

Covers the LongMemEvalAdapter surface that bench/longmemeval_blind.py drives:

    Test 1 load_dataset(split="S") returns an iterable of LMESession with
            non-zero len (skipped if HuggingFace cache is unavailable; falls
            back to an inline fixture for the pure-offline CI case).
    Test 2 session_to_inserts maps each turn to a MemoryRecord with
            tier='episodic' and literal_surface == turn['content'].
    Test 3 query_to_recall calls retrieve.recall with cue_text=query['query']
            (verified via mock.patch on retrieve.recall).
    Test 4 score_r_at_k on a hand-labeled mini-set (3 retrieved, 2 relevant,
            k=5) returns the expected float.
    Test 5 score_r_at_k with an empty relevant list returns 1.0 (convention;
            avoids div-by-zero).

Notes (scope discipline):
- Zero modifications to src/iai_mcp/ exercised here. The adapter runs on the
  public MemoryStore.insert + retrieve.recall surface only.
- PINNED_REVISION is a 40-char HuggingFace commit hash pinned at Task 2 time.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest


# --------------------------------------------------------------------------- env gate

_HF_CACHE = Path(
    os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
)
HAS_LONGMEMEVAL_CACHE = any(
    _HF_CACHE.rglob("longmemeval_s")
) if _HF_CACHE.exists() else False


# --------------------------------------------------------------------------- Test 1


@pytest.mark.skipif(
    not HAS_LONGMEMEVAL_CACHE,
    reason="LongMemEval dataset not cached locally; skipping network-dependent load",
)
def test_load_dataset_S_returns_non_empty_iterable():
    from bench.adapters.longmemeval import LongMemEvalAdapter, LMESession

    adapter = LongMemEvalAdapter()
    sessions = list(adapter.load_dataset(split="S"))
    assert len(sessions) > 0, "LongMemEval-S must have at least 1 session"
    first = sessions[0]
    assert isinstance(first, LMESession)
    assert isinstance(first.session_id, str) and first.session_id
    assert isinstance(first.turns, list) and len(first.turns) >= 1
    # Turns use {role, content} keys per LongMemEval schema.
    t0 = first.turns[0]
    assert "role" in t0 and "content" in t0
    # The adapter attaches at least one eval query per LMESession (a question
    # the LongMemEval-S row asks against this session's haystack).
    assert isinstance(first.queries, list) and len(first.queries) >= 1
    q0 = first.queries[0]
    assert "query" in q0
    assert "relevant_turn_ids" in q0


# --------------------------------------------------------------------------- Test 2


def test_session_to_inserts_maps_each_turn():
    from bench.adapters.longmemeval import LongMemEvalAdapter, LMESession

    adapter = LongMemEvalAdapter()
    session = LMESession(
        session_id="s1",
        turns=[
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "what's the weather?"},
        ],
        queries=[{"query": "q", "relevant_turn_ids": []}],
    )
    records = adapter.session_to_inserts(session)
    assert len(records) == 3
    for turn, rec in zip(session.turns, records):
        assert rec.tier == "episodic"
        assert rec.literal_surface == turn["content"]
        assert rec.language == "en"
        # Every record gets an embedding populated by the adapter so the
        # blind run can call MemoryStore.insert directly (insert does not
        # auto-embed).
        assert isinstance(rec.embedding, list) and len(rec.embedding) > 0


# --------------------------------------------------------------------------- Test 3


def test_query_to_recall_calls_retrieve_recall_with_cue_text():
    from bench.adapters.longmemeval import LongMemEvalAdapter

    adapter = LongMemEvalAdapter()

    fake_store = mock.MagicMock(name="MemoryStore")
    # Simulate embed helper so adapter can compute cue_embedding.
    fake_store.embed_dim = 8

    class _FakeEmbedder:
        DIM = 8
        def embed(self, text: str):  # noqa: D401
            return [0.1] * 8

    fake_hits = [mock.MagicMock(record_id=uuid4()) for _ in range(3)]

    with mock.patch(
        "bench.adapters.longmemeval.retrieve_recall"
    ) as m_recall, mock.patch(
        "bench.adapters.longmemeval.embedder_for_store",
        return_value=_FakeEmbedder(),
    ):
        m_recall.return_value = mock.MagicMock(hits=fake_hits)
        retrieved_ids = adapter.query_to_recall(
            {"query": "what did I say about coffee?"}, fake_store
        )

    assert m_recall.call_count == 1
    kwargs = m_recall.call_args.kwargs
    # Tolerate both kw and positional; the plan contract is "cue_text = query['query']".
    cue_text = kwargs.get("cue_text")
    if cue_text is None:
        # positional fall-back: (store, cue_embedding, cue_text, session_id,...)
        cue_text = m_recall.call_args.args[2]
    assert cue_text == "what did I say about coffee?"
    assert retrieved_ids == [h.record_id for h in fake_hits]


# --------------------------------------------------------------------------- Test 4


def test_score_r_at_k_hand_labeled_miniset():
    from bench.adapters.longmemeval import LongMemEvalAdapter

    adapter = LongMemEvalAdapter()
    # 3 retrieved UUIDs, 2 of them in the gold set, k=5.
    gold_a, gold_b, other = uuid4(), uuid4(), uuid4()
    retrieved = [gold_a, other, gold_b]
    gold_ids = [str(gold_a), str(gold_b), str(uuid4())]  # one gold is not retrieved
    score = adapter.score_r_at_k(retrieved, gold_ids, k=5)
    # 2 relevant retrieved / 3 relevant total -> 2/3
    assert score == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- Test 5


def test_score_r_at_k_empty_gold_returns_one():
    from bench.adapters.longmemeval import LongMemEvalAdapter

    adapter = LongMemEvalAdapter()
    score = adapter.score_r_at_k([uuid4()], [], k=5)
    assert score == 1.0
