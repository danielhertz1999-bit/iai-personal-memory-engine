"""One-shot helper that freezes canonical G(n, m) edge sets into
``tests/fixtures/sigma_baseline.json``.

The Rust generator at ``iai_mcp_native.graph.gnm_random_graph`` is called
for five ``(n, m, seed)`` tuples; each call's ``(u_list, v_list)`` pair
is stored under a new top-level ``gnm_baseline`` key. The fixture's
``sha256_self_check`` field is then recomputed from the canonical bytes
(sort_keys=True, indent=2, excluding the hash key) and written back.

This script is **NOT shipped** to ``src/`` and is only run once when the
canonical fixture is being extended. Re-running it re-freezes the
baseline; any non-empty diff against the prior baseline MUST be
investigated as a generator drift before the test constant is updated.

Usage::

    python tests/_helpers/freeze_gnm_baseline.py # writes the default fixture path
    python tests/_helpers/freeze_gnm_baseline.py path.json # writes to a custom path

After running this helper, update ``SIGMA_BASELINE_SHA256`` in
``tests/test_mosaicsigma_sigma_baseline.py`` to the new hash printed
on stdout and commit both files in a single atomic commit so the
SHA-lockstep gate from stays green.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

# Five (n, m, seed) tuples baked into the spec. Two graph sizes
# (n=200, m=400 and n=500, m=1000) and three / two seeds respectively.
# The σ consumer at src/iai_mcp/sigma.py calls gnm with seed = base + k
# for k in 0..2, so seed values 42, 43, 44 cover the n_random=3 path
# at the n=200 size; n=500 covers the larger size in fast_sigma() runs.
FROZEN_TUPLES = [
    (200, 400, 42),
    (200, 400, 43),
    (200, 400, 44),
    (500, 1000, 42),
    (500, 1000, 43),
]


def _key(n: int, m: int, seed: int) -> str:
    return f"n{n}_m{m}_s{seed}"


def _canonical_bytes_without_hash(doc: dict) -> bytes:
    """Serialize doc without sha256_self_check using the project canonical
    formatting (sort_keys=True, indent=2). Mirrors the formula used by
    ``test_baseline_sha256_locked`` so the recomputed hash matches.
    """
    inner = {k: v for k, v in doc.items() if k != "sha256_self_check"}
    return json.dumps(inner, sort_keys=True, indent=2).encode("utf-8")


def build_gnm_baseline() -> dict[str, dict]:
    """Call the Rust generator for each frozen tuple and collect output."""
    # Lazy import so the helper fails loudly if the wheel isn't installed.
    from iai_mcp_native import graph  # type: ignore[import-not-found]

    out: dict[str, dict] = {}
    for n, m, seed in FROZEN_TUPLES:
        u_list, v_list = graph.gnm_random_graph(n, m, seed)
        assert len(u_list) == m, f"edge count mismatch for {(n, m, seed)}"
        assert len(v_list) == m, f"edge count mismatch for {(n, m, seed)}"
        out[_key(n, m, seed)] = {
            "n": n,
            "m": m,
            "seed": seed,
            "u_list": [int(x) for x in u_list],
            "v_list": [int(x) for x in v_list],
        }
    return out


def main() -> int:
    target_str = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sigma_baseline.json"
    target = pathlib.Path(target_str)
    assert target.exists(), f"fixture file missing: {target}"

    doc = json.loads(target.read_text(encoding="utf-8"))
    gnm_baseline = build_gnm_baseline()

    # Stash under a new top-level key (sibling of `fixtures`). Stashing
    # inside `fixtures` would mix two different shapes under one key and
    # complicate the existing test_classify_regime_invariants loop.
    doc["gnm_baseline"] = gnm_baseline

    # Recompute the canonical hash from doc-minus-sha and write back.
    digest = hashlib.sha256(_canonical_bytes_without_hash(doc)).hexdigest()
    doc["sha256_self_check"] = digest

    # Write the file using the same canonical formatting the SHA was
    # computed over (sort_keys=True, indent=2 + trailing newline) so a
    # round-trip re-read produces a byte-identical hash.
    text = json.dumps(doc, sort_keys=True, indent=2)
    target.write_text(text + "\n", encoding="utf-8")

    # Print summary so the operator can paste the new SHA into the test
    # constant in the same commit.
    print("=" * 72)
    print("GNM BASELINE — frozen canonical edge sets")
    print("=" * 72)
    print(f"target           = {target}")
    print(f"sha256           = {digest}")
    print("-" * 72)
    print(f"{'key':>16s}  {'n':>5s} {'m':>6s}  {'seed':>5s}  edge[0]  edge[1]  edge[-1]")
    print("-" * 72)
    for key, entry in gnm_baseline.items():
        e0 = (entry["u_list"][0], entry["v_list"][0])
        e1 = (entry["u_list"][1], entry["v_list"][1])
        eN = (entry["u_list"][-1], entry["v_list"][-1])
        print(f"{key:>16s}  {entry['n']:>5d} {entry['m']:>6d}  {entry['seed']:>5d}  "
              f"{e0}  {e1}  {eN}")
    print("=" * 72)
    print("Next step: update SIGMA_BASELINE_SHA256 in")
    print("  tests/test_mosaicsigma_sigma_baseline.py")
    print(f"to: {digest}")
    print("Stage both files in a single atomic commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
