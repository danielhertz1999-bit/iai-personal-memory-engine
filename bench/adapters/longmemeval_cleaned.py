"""Cleaned-dataset adapter for LongMemEval-S — D-02.

Mempalace's reference benchmark uses ``xiaowu0162/longmemeval-cleaned``
(commit-pinned via ``huggingface_hub.repo_info()``). This adapter mirrors
the ``LongMemEvalAdapter`` shape from ``bench/adapters/longmemeval.py`` so
the orchestrator (`bench/longmemeval_blind.py`) can swap raw vs cleaned
purely via the ``--dataset {cleaned, raw}`` CLI flag.

## boundary

This adapter is NEW (Phase 9 Task 1). The raw adapter at
``bench/adapters/longmemeval.py`` is byte-identical to its v2 state — Phase
9 does NOT modify the v1/v2 baseline path. ``--dataset raw`` continues to
load the raw revision ``2ec2a557f339...``; ``--dataset cleaned`` (the new
v3 default) routes to this module.

## Pinning discipline

Phase 9 LOCKED: pin via ``huggingface_hub.repo_info(...)``, NEVER
hardcode a magic string. The cleaned dataset's HEAD SHA is auto-discovered
on first instantiation and stored on ``self.revision`` so v3 output JSON
records exactly which dataset variant was measured. On reproducer runs,
the caller may pass ``revision=`` to pin a specific historical SHA.

## Schema

The cleaned dataset uses the same row schema as the raw dataset (cleaned
removed bad evidence; field names preserved). Each row in
``longmemeval_s_cleaned.json`` is:

    {
      "question_id":          str,
      "question_type":        str,
      "question":             str,
      "haystack_session_ids": list[str],
      "haystack_sessions":    list[list[{"role","content"}]],
      "answer_session_ids":   list[str],
    }

The adapter emits one ``LMESession`` per haystack session with the eval
query attached (matching the raw adapter's emission shape exactly), so
``main()`` in ``longmemeval_blind.py`` does NOT branch on adapter type —
it groups LMESessions by ``question_id`` either way.

## Split support

Only ``split="S"`` is supported. The cleaned dataset ships only the S split
as ``longmemeval_s_cleaned.json``; M and oracle remain in the raw dataset.
"""
from __future__ import annotations

import json
import sys
from typing import Iterable

from bench.adapters.longmemeval import LMESession


CLEANED_DATASET_ID: str = "xiaowu0162/longmemeval-cleaned"
CLEANED_FILENAME: str = "longmemeval_s_cleaned.json"


class CleanedLongMemEvalAdapter:
    """Loads ``xiaowu0162/longmemeval-cleaned`` via ``huggingface_hub``.

    Mirrors ``LongMemEvalAdapter`` so ``bench/longmemeval_blind.py`` can
    treat them interchangeably (same ``LMESession`` iterator shape).

    Pin discipline: ``revision`` defaults to the current HEAD SHA of the
    HuggingFace dataset, auto-discovered via ``repo_info()``. Pass an
    explicit revision to reproduce a historical run.
    """

    DATASET_ID: str = CLEANED_DATASET_ID

    def __init__(self, revision: str | None = None) -> None:
        if revision is not None:
            self.revision = revision
            return
        try:
            from huggingface_hub import repo_info
        except ImportError as exc:  # pragma: no cover — dev extra
            raise RuntimeError(
                "huggingface_hub not installed; run "
                "`pip install 'datasets>=2.18' huggingface_hub`"
            ) from exc
        info = repo_info(repo_id=CLEANED_DATASET_ID, repo_type="dataset")
        self.revision = info.sha

    def load_dataset(self, split: str = "S") -> Iterable[LMESession]:
        """Stream LMESessions out of ``longmemeval_s_cleaned.json``.

        Only ``split="S"`` is supported (the cleaned dataset ships the S
        split only). Raises ``ValueError`` on any other split value.
        """
        if split != "S":
            raise ValueError(
                f"unknown LongMemEval cleaned split {split!r}; "
                f"the cleaned dataset ships only the 'S' split"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover — dev extra
            raise RuntimeError(
                "huggingface_hub not installed; run "
                "`pip install 'datasets>=2.18' huggingface_hub`"
            ) from exc

        print(
            f"[LongMemEval-cleaned] resolving split={split} "
            f"revision={self.revision} filename={CLEANED_FILENAME}",
            file=sys.stderr,
            flush=True,
        )
        path = hf_hub_download(
            repo_id=CLEANED_DATASET_ID,
            filename=CLEANED_FILENAME,
            repo_type="dataset",
            revision=self.revision,
        )
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        for row in rows:
            qid = row["question_id"]
            question = row["question"]
            question_type = str(row.get("question_type", "unknown"))
            answer_session_ids = list(row.get("answer_session_ids", []))
            haystack_session_ids: list[str] = list(
                row.get("haystack_session_ids", [])
            )
            haystack_sessions: list[list[dict]] = list(
                row.get("haystack_sessions", [])
            )

            # Emit one LMESession per haystack session; attach the eval
            # query to every one so the orchestrator can run ONE recall
            # per row after inserting all haystack turns. Matches the
            # raw adapter's emission shape exactly.
            for sess_id, turns in zip(
                haystack_session_ids, haystack_sessions
            ):
                yield LMESession(
                    session_id=sess_id,
                    turns=list(turns),
                    queries=[
                        {
                            "query": question,
                            "question_id": qid,
                            "question_type": question_type,
                            "relevant_turn_ids": answer_session_ids,
                            "is_gold_session": sess_id in answer_session_ids,
                        }
                    ],
                )


__all__ = [
    "CLEANED_DATASET_ID",
    "CLEANED_FILENAME",
    "CleanedLongMemEvalAdapter",
]
