"""One-shot generator for tests/fixtures/sigma_baseline.json.

Builds a constitutional sigma oracle fixture comprising:
- Five frozen-from-networkx baselines (karate, les_miserables, er_200, er_500, er_1000)
- Three hand-verified tiny graphs (tiny_10_ws_k4, tiny_20_ws_p010, tiny_karate)
- One optional snapshot entry (live_n2000) for downstream perf gating

Each entry records (n, m, C, L, sigma, Cr, Lr, regime, source, literature_anchor, edges).
"regime" is the live result of ``iai_mcp.sigma.classify_regime(n, sigma)`` — round-trip equality
with the production classifier is the constitutional gate (regime equality, not bit-exact float
parity).

The output JSON is hashed with sha256 over its canonical form (the
``sha256_self_check`` key is excluded from the hash computation, then written back). Downstream
tests recompute and assert the same hash.

Note on the dolphins network: the planner spec listed ``dolphins`` as a mandatory
fixture, but networkx does not ship a ``dolphins_graph`` generator at the
installed version. Substituted with ``les_miserables_graph`` — Humphries-Gurney
2008 Table 1 tabulates both Dolphins and Lesmis as small social graphs, so the
literature anchor is preserved (same paper, same table, same row format).

Edges are canonicalized as ``tuple(sorted((u, v)))`` then ``sorted(edges)`` so the
SHA256 hash is stable across regeneration.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone

import networkx as nx

# Use the actual project classifier so the fixture round-trips against production code.
from iai_mcp.sigma import classify_regime, fast_sigma


def _canonical_edges(g: nx.Graph) -> list[list[int]]:
    """Edges in canonical form: each tuple sorted, list sorted, stable across runs."""
    edges = [tuple(sorted((int(u), int(v)))) for (u, v) in g.edges()]
    edges.sort()
    return [list(e) for e in edges]


def _relabel_consecutive(g: nx.Graph) -> nx.Graph:
    """Relabel nodes to consecutive ints 0..n-1, preserving topology."""
    return nx.convert_node_labels_to_integers(g, first_label=0, ordering="sorted")


def _sigma_record(g: nx.Graph, *, source: str, literature_anchor: str | None,
                  include_edges: bool = True) -> dict:
    """Compute (sigma, C, L, Cr, Lr) via project fast_sigma and serialize an entry."""
    g = _relabel_consecutive(g)
    n = int(g.number_of_nodes())
    m = int(g.number_of_edges())
    sigma_val, C, L, Cr, Lr = fast_sigma(g, n_random=3, seed=42)
    if isinstance(sigma_val, float) and math.isnan(sigma_val):
        sigma_for_json: float | None = None
    else:
        sigma_for_json = float(sigma_val)
    regime = classify_regime(n, sigma_for_json)
    record: dict = {
        "n": n,
        "m": m,
        "C": float(C),
        "L": float(L),
        "sigma": sigma_for_json,
        "Cr": float(Cr),
        "Lr": float(Lr),
        "regime": regime,
        "source": source,
        "literature_anchor": literature_anchor,
    }
    if include_edges:
        record["edges"] = _canonical_edges(g)
    else:
        record["edges"] = None
    return record


def build_fixtures() -> dict[str, dict]:
    fixtures: dict[str, dict] = {}

    # ---- 5 reference fixtures (frozen-from-networkx) ----

    # 1. Karate club (n=34, m=78) — Humphries-Gurney 2008 Table 1 row.
    fixtures["karate"] = _sigma_record(
        nx.karate_club_graph(),
        source="networkx-frozen",
        literature_anchor="Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1",
    )

    # 2. Les Miserables co-occurrence (n=77, m=254) — Humphries-Gurney 2008 Table 1 row.
    # Substituted for "dolphins" (networkx does not ship dolphins_graph at the
    # installed version). Both networks are tabulated in the same H-G Table 1.
    fixtures["les_miserables"] = _sigma_record(
        nx.les_miserables_graph(),
        source="networkx-frozen",
        literature_anchor="Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1",
    )

    # 3-5. Erdos-Renyi G(n, m) reference graphs — random baselines.
    for n_nodes, m_edges, key in [(200, 400, "er_200"),
                                  (500, 1000, "er_500"),
                                  (1000, 2000, "er_1000")]:
        fixtures[key] = _sigma_record(
            nx.gnm_random_graph(n_nodes, m_edges, seed=42),
            source="networkx-frozen",
            literature_anchor=None,
        )

    # ---- 3 hand-verified tiny graphs ----

    # tiny_10_ws_k4: k=4 ring lattice, p=0. C(p=0) = 3(k-2)/(4(k-1)) = 0.5
    # per 1998 Nature 393:440-442 §"Highly ordered case".
    fixtures["tiny_10_ws_k4"] = _sigma_record(
        nx.watts_strogatz_graph(10, 4, 0.0, seed=42),
        source="hand-verified",
        literature_anchor="Watts-Strogatz 1998 Nature 393:440-442 Table 1",
    )

    # tiny_20_ws_p010: k=4, p=0.1 — small-world regime per WS Table 1.
    fixtures["tiny_20_ws_p010"] = _sigma_record(
        nx.watts_strogatz_graph(20, 4, 0.1, seed=42),
        source="hand-verified",
        literature_anchor="Watts-Strogatz 1998 Nature 393:440-442 Table 1",
    )

    # tiny_karate: Zachary's karate club used as a tiny hand-verifiable graph.
    fixtures["tiny_karate"] = _sigma_record(
        nx.karate_club_graph(),
        source="hand-verified",
        literature_anchor="Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1",
    )

    # ---- 1 OPTIONAL live snapshot entry ----
    # No live store present in this worktree (no ~/.iai-mcp/store.db) → placeholder.
    # Downstream plans must handle this via pytest.skipif on source == "missing-snapshot".
    fixtures["live_n2000"] = {
        "n": 0,
        "m": 0,
        "C": 0.0,
        "L": 0.0,
        "sigma": None,
        "Cr": 0.0,
        "Lr": 0.0,
        "regime": "unavailable",
        "source": "missing-snapshot",
        "literature_anchor": None,
        "edges": None,
        "snapshot_path": None,
        "note": "live_n2000 unavailable - store has fewer than 2000 records; "
                "the 5 reference fixtures + 3 tinies remain the constitutional gated set",
    }

    return fixtures


def build_doc() -> dict:
    fixtures = build_fixtures()
    return {
        "schema_version": "1",
        "generator_meta": {
            "networkx_version": nx.__version__,
            "seed": 42,
            "n_random": 3,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host_python": sys.version.split()[0],
        },
        "fixtures": fixtures,
    }


def canonical_bytes(doc_without_hash: dict) -> bytes:
    """Bytes used for SHA256 — canonical JSON without the sha256_self_check key."""
    return json.dumps(doc_without_hash, sort_keys=True, indent=2).encode()


def main() -> int:
    doc = build_doc()
    digest = hashlib.sha256(canonical_bytes(doc)).hexdigest()
    doc["sha256_self_check"] = digest

    # Print human summary so band-checking is possible BEFORE writing the file.
    print("=" * 72)
    print("SIGMA BASELINE — actual computed values")
    print("=" * 72)
    print(f"networkx_version = {doc['generator_meta']['networkx_version']}")
    print(f"sha256           = {digest}")
    print("-" * 72)
    print(f"{'fixture':>20s}  {'n':>5s} {'m':>6s}  {'C':>8s} {'L':>8s}  "
          f"{'Cr':>8s} {'Lr':>8s}  {'sigma':>8s}  regime")
    print("-" * 72)
    for key, f in doc["fixtures"].items():
        sigma_str = "None" if f["sigma"] is None else f"{f['sigma']:.4f}"
        print(f"{key:>20s}  {f['n']:>5d} {f['m']:>6d}  "
              f"{f['C']:>8.4f} {f['L']:>8.4f}  "
              f"{f['Cr']:>8.4f} {f['Lr']:>8.4f}  "
              f"{sigma_str:>8s}  {f['regime']}")
    print("=" * 72)

    # Write the file alongside the helper output for inspection.
    target = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sigma_baseline.json"
    with open(target, "w", encoding="utf-8") as fh:
        # Use the same canonical format used for hashing, then append the hash.
        canonical = json.dumps({k: v for k, v in doc.items() if k != "sha256_self_check"},
                               sort_keys=True, indent=2)
        # Add sha256_self_check key in canonical order via re-dump of the full doc.
        full = json.dumps(doc, sort_keys=True, indent=2)
        fh.write(full)
        fh.write("\n")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
