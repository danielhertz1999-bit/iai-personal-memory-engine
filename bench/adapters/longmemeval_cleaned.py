from __future__ import annotations

import json
import sys
from typing import Iterable

from bench.adapters.longmemeval import LMESession


CLEANED_DATASET_ID: str = "xiaowu0162/longmemeval-cleaned"
CLEANED_FILENAME: str = "longmemeval_s_cleaned.json"


class CleanedLongMemEvalAdapter:

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
