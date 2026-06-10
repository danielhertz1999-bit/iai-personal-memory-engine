
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

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
    inner = {k: v for k, v in doc.items() if k != "sha256_self_check"}
    return json.dumps(inner, sort_keys=True, indent=2).encode("utf-8")


def build_gnm_baseline() -> dict[str, dict]:
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

    doc["gnm_baseline"] = gnm_baseline

    digest = hashlib.sha256(_canonical_bytes_without_hash(doc)).hexdigest()
    doc["sha256_self_check"] = digest

    text = json.dumps(doc, sort_keys=True, indent=2)
    target.write_text(text + "\n", encoding="utf-8")

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
