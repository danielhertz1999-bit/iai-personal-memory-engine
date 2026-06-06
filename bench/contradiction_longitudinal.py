#!/usr/bin/env python3
"""Contradiction-longitudinal falsifiability bench (skeleton + pre-registered criteria).

**Do not run on the construction host by default** — this module is meant for a
dedicated bench machine with an isolated ``IAI_MCP_STORE`` and optional GPU.

Pre-registered pass criteria (from CONTEXT_PEER_REVIEW v3):
- **Metric B (post-flip):** cues issued after session ``t_0`` (contradiction +
  consolidation window simulated) must rank the *current* winning fact above
  flat cosine-only retrieval on the same store slice.
- **Metric A (historical verbatim):** probes asking for superseded wording must
  still surface the archived surface (verbatim), not the post-flip fact alone.
- **Regression gate:** pipeline score on B must beat cosine baseline; A must not
  collapse below a configured verbatim hit threshold.

This file loads:file:`fixtures/contradiction_longitudinal.jsonl` (synthetic JSONL
rows: ``session``, ``text``, optional ``probe`` / ``expects``) and documents the
evaluation harness contract. A full implementation wires:

1. Fixture loader → ``MemoryStore`` inserts per session order.
2. Explicit ``memory_contradict`` (or edge-equivalent) at ``t_0``.
3. Optional sleep/consolidation tick simulation (bench-only knobs).
4. Two eval slices: ``pre_flip_cues`` vs ``post_flip_cues`` with separated metrics.

Exit code 0 only when all gates pass; non-zero on any failure. Until the harness
is completed, ``main()`` prints the criteria and exits with code 2 to avoid a
silent green run::

    python bench/contradiction_longitudinal.py --fixture bench/fixtures/contradiction_longitudinal.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures" / "contradiction_longitudinal.jsonl",
    )
    args = parser.parse_args(argv)
    rows = load_rows(args.fixture)
    print(
        json.dumps(
            {
                "loaded_rows": len(rows),
                "fixture": str(args.fixture),
                "status": "harness_stub",
                "criteria": [
                    "B: post-flip cues — pipeline beats flat cosine",
                    "A: historical verbatim probes — superseded text still retrievable",
                    "No regression: B gain without A collapse",
                ],
            },
            indent=2,
        )
    )
    # Stub: full eval is intentionally absent so CI never runs heavy retrieval.
    return 2


if __name__ == "__main__":
    sys.exit(main())
