# iai — benchmarks

Every number below is measured on the current release (Hippo storage + Lilli/HD substrate + the native Rust `iai_mcp_native.embed` bge-small-en-v1.5 384-dim embedder + the MOSAIC graph engine), on an Apple M2 Max (12-core, 64 GB). Each row carries a one-line reproduce command; run them and get your own results.

**Honesty rules applied:** no tuning on the test set, no hand-picked seeds, honest-scale (multi-seed) where applicable. The only head-to-head comparison is LongMemEval vs mempalace; every other number is our own metric (no competitor column). Where a number missed its target or regressed, it says so plainly.

---

## Head-to-head — LongMemEval-S (the one competitive arena)

Validated in a single harness — both systems run on the identical dataset, metric, and gold labels, line-for-line verified:

| System | Embedder | R@5 | R@10 |
|---|---|---|---|
| **iai** (product) | bge-small-en-v1.5 (384d) | **0.962** | 0.978 |
| iai (matched embedder) | all-MiniLM-L6-v2 (384d) | 0.966 | 0.978 |
| [mempalace](https://github.com/MemPalace/mempalace) v3.3.6 | all-MiniLM-L6-v2 (384d) | 0.966 | 0.978 |

- **Config (identical both sides):** LongMemEval-S **cleaned, 500 questions**, session granularity, metric = `recall_any@k` (any gold session-id in top-k), full haystack, **raw** (no rerank). Both pure-dense, user-turns-only per session. The only difference is the embedder model.
- **On raw retrieval — the headline both projects ship — it's an exact tie on the matched embedder:** R@5 0.966 = 0.966 and R@10 0.978 = 0.978. Our product embedder (bge-small-en-v1.5) scores 0.962 R@5 — a 2-question-in-500 difference (481 vs 483 hits), within noise.
- **mempalace's published raw numbers reproduce** — we ran them in our own harness and got the same 0.966. They are honest.
- LongMemEval is a *cold, one-shot* retrieval benchmark: it inserts a fresh haystack per question and queries immediately. It does not exercise cross-session memory at all — which is where this design's edge actually is.

---

## Our own metrics (no competitor column)

| Benchmark | Result | Notes |
|---|---|---|
| **Rescue@10** (post-contradiction current fact) | **1.000** (flat cosine baseline: ~0.70) | After a fact is updated/contradicted, the *current* winning fact still ranks top-10 — where flat-vector stores collapse because the stale fact is often the more similar one. Honest-scale: 3 seeds × 1000 sessions × 2 slices. See the three-gate note below. |
| **Personal-fact drift** (recall@10) | **0.9933** | retention_loss@10 = 0.0067. Honest-scale: 3 seeds × (50 facts, 50 sessions, 30 intervening). |
| **Sleep-consolidation** (recall@10 preserved) | **1.000 → 1.000** (Δ=0) | Recall survives a full consolidation cycle. 3 seeds × 160-record corpus (20 targets + 40 confusors + 100 noise). |
| **Session-start token cost** | **1,629** (minimal) / **2,993** (standard) | Both under the ≤3,000-token session budget. tiktoken-cl100k proxy. |
| **MOSAIC community-detection parity** | **36/36** LFR-gauntlet + 10/10 | NMI vs ground-truth on karate / football / LFR n=1000 & 5000; 5× replay-deterministic; modularity-monotonic. |
| **Recall p95 latency** | 77 ms @100 · 105 ms @1k · **368 ms @10k** | Misses the internal <100 ms@10k target at scale; the rank/centrality stage dominates. |
| **Memory footprint (RSS)** | **589 MB @10,000 records** | Threshold 2,000 MB; passed. Embedder + graph runtime. |
| **Rust embedder latency** | p50 70 ms / p95 253 ms (single embed) | bge-small-en-v1.5, 384-dim. |

Reproduce (each with a temporary `IAI_MCP_STORE` and `IAI_MCP_CRYPTO_PASSPHRASE` unless noted):

```bash
python -m bench.longmemeval_blind --split S --dataset cleaned --granularity session  # LongMemEval-S raw
python -m bench.contradiction_longitudinal --scale honest --seeds 13 42 137           # Rescue@10 / longitudinal
python -m bench.personal_fact_drift --scale honest --seeds 13 42 137                  # drift / retention
python bench/sleep_ablation.py --seeds 13 42 137                                      # sleep-consolidation recall
python -m bench.tokens --wake-depth minimal                                           # session-start tokens
python -m bench.neural_map --n 100 --n 1000 --n 10000 --iterations 20                 # recall latency
python -m bench.memory_footprint --n 10000                                            # RSS footprint
```

---

## Honest caveats / not-yet-leads

- **Recall p95 at 10k = 368 ms** — above the internal <100 ms target. The rank/centrality stage dominates (betweenness recompute is ~1.7 s@10k, mitigated by a centrality cache that stays on by default). A latency-optimization candidate.
- **Historical-verbatim retrieval regressed 0.90 → 0.71 in this release** — the ability to retrieve the *superseded/archived* wording of an updated fact verbatim dropped. This is **separate from Rescue@10** (current-fact retrieval, unchanged at 1.000). Likely cause (per an external code review): the centrality-engine swap dropped Hebbian edge-weights (unweighted Brandes), shifting the seed-score landscape toward older densely-connected facts. A tracked fix for the next release.
- **The contradiction bench reports three gates, and one fails by design.** Gate A (Rescue@10 — current fact in top-10) and Gate B-contract (the system actively signals the contradiction via its dual-route / anti-hits path, which a flat cosine cannot) both pass, and the overall verdict is built from those two. Gate B-classical (rank the current fact *above* a flat cosine baseline by MRR) shows ΔMRR ≈ −0.05 and is reported as FAIL — this is expected and labeled in the bench output: the system signals contradictions rather than re-ranking, so it makes no promise to beat raw cosine on MRR. Run `python -m bench.contradiction_longitudinal --scale honest --seeds 13 42 137` and you will see all three.

---

## Verdict

**"Best-benchmarked" holds as breadth + reproducibility + honesty**, plus genuine leads on session-start budget, personal-fact retention, post-contradiction rescue, local-first operation, and deterministic community-detection parity. It does **not** hold as "tops the LongMemEval raw leaderboard" — there it is a 2-question near-tie. The honest posture: **on par with the best on LongMemEval raw, and decisively ahead on longitudinal recall.**
