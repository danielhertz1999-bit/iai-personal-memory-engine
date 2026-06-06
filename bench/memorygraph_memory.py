"""Measure peak RSS during MemoryGraph build at the requested size.

Run:
    python bench/memorygraph_memory.py
    python bench/memorygraph_memory.py --n 10000 --m 5000

Output:
    Single-line summary:
        "RSS delta @ N=10000 m=5000: 42.7 MB (before=18.2 MB, after=60.9 MB)"

Notes:
    - macOS reports ``ru_maxrss`` in BYTES; Linux reports KiB. The
      platform branch normalises both into megabytes so the printed
      delta is comparable across hosts.
    - The script exits 0 unconditionally; the PASS/FAIL judgment is
      performed by the caller against an external gate (no automated
      assertion inside the bench itself).
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
from pathlib import Path
from uuid import uuid4


# Resolve ``iai_mcp.*`` to this worktree's ``src/`` rather than an editable
# install elsewhere on the path. Idempotent: each ``sys.path.insert`` is
# guarded by an "if not already present" check.
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)


def rss_mb() -> float:
    """Return current peak resident-set-size in megabytes.

    macOS ``ru_maxrss`` is in bytes; Linux is in KiB (POSIX). The branch
    below normalises both forms into megabytes so the printed delta is
    cross-platform-comparable. Without the branch, macOS measurements
    would be reported 1024x too small, giving a false PASS on any
    memory gate calibrated in megabytes.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bench.memorygraph_memory",
        description="Measure peak RSS during MemoryGraph build.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of nodes to add (default: 10000).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=5000,
        help="Number of edges to add (default: 5000).",
    )
    args = parser.parse_args()

    # Defer the import so the baseline RSS reading does not include the
    # core-module load cost. The module is cached on first import, so the
    # bench measures graph-build cost only, not module-import cost.
    from iai_mcp.graph import MemoryGraph

    gc.collect()
    before = rss_mb()

    g = MemoryGraph()
    nodes = [uuid4() for _ in range(args.n)]
    for n in nodes:
        g.add_node(n, community_id=None, embedding=[0.0] * 384)

    # Sparse edge population: m edges across consecutive non-overlapping
    # pairs (step=2). Caps at len(nodes)-1 so a small N never tries to
    # add more edges than the node count permits.
    edges_added = 0
    i = 0
    while edges_added < args.m and i < len(nodes) - 1:
        g.add_edge(nodes[i], nodes[i + 1], weight=1.0)
        edges_added += 1
        i += 2

    gc.collect()
    after = rss_mb()
    delta = after - before
    print(
        f"RSS delta @ N={args.n} m={args.m}: {delta:.1f} MB "
        f"(before={before:.1f} MB, after={after:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
