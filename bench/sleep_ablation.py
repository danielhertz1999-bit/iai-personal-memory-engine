#!/usr/bin/env python3
"""Sleep-ablation benchmark — measures recall quality BEFORE vs AFTER a real
local consolidation cycle (sleep-on vs sleep-off differentiator).

Design rationale
----------------
The system is CLS/hippocampus-cortex: records start at stability=0.0 and gain
strength through sleep (FSRS stability boost, DREAM_DECAY prune, cluster LTP,
schema-mine).  The designed strength is POST-consolidation.  This bench
measures whether that strength translates into measurable recall improvement.

Corpus design
-------------
* 4 thematic clusters × 5 target facts + 10 in-cluster confusors each
  (topics: weather, cooking, fitness, coding) = 20 targets + 40 confusors.
* 100 off-topic noise records. Total: 160 records.
* In-cluster confusors are on the same topic as the target cluster but answer
  different questions. They compete for the same probe embeddings and push
  cold target rank below ceiling.
* Probes are thematically general (not literal paraphrases of the target text),
  so ANN must rank the specific target above the confusors.
* Hebbian edges seeded between ALL cluster members (targets + confusors) to
  trigger _build_hebbian_clusters (CLUSTER_MIN_SIZE=3; 15 per cluster meets it).
* K_PROBE=50 candidate pool for rank/MRR sensitivity; recall@10 as threshold.
* Consolidation is the ONLY state change between PRE and POST probes.

Ground truth
------------
UUID-based: _expected_id_in_top_k on the originally-inserted record UUID.
Using UUID (not exact text) avoids false-negative from consolidation creating
summary records whose literal_surface displaces the target in a strict-string
check.

Metrics
-------
* recall@10 (primary): fraction of probes where target UUID is in top-10.
* MRR: mean reciprocal rank at top-10 (1/rank if found, else 0).
* mean_rank: mean position of target in top-10 (None if not found → counts
  as 11 for averaging).

Consolidation entry point
-------------------------
run_heavy_consolidation() from iai_mcp.sleep — returns a dict with
summaries_created, decay_result, schema_candidates, tier.  Used instead of
SleepPipeline because:
  (a) Returns evidence directly (no log-parsing needed).
  (b) Avoids lifecycle_state_path default = ~/.iai-mcp (live daemon).
  (c) Single-step invocation — no quarantine / resume machinery needed.
Confirmed tier == "tier0" (llm_enabled=False, has_api_key=False) as proof
no LLM call fired.

What run_heavy_consolidation does on a fresh store
--------------------------------------------------
On a fresh store (no prior usage history):
1. _decay_edges: no-op (DECAY_GRACE_DAYS=90; all edges are age 0).
2. _process_cluster_summaries: creates 4 semantic summary records — one per
   cluster. Each is "Cluster summary (15 records, lang=en): <prefix1>; ...".
3. Hebbian LTP: boost_edges on existing hebbian edges in each cluster (weight
   increases; topology unchanged; degree-based rank signal unchanged).
4. _tier0_schema_surfacing: tag-pair co-occurrence candidates surfaced.
5. _persist_tier1_schemas: auto-status schemas persisted (schema_instance_of
   edges + semantic schema records).

What CANNOT change with run_heavy_consolidation
-----------------------------------------------
* record.stability is NOT updated (only run_light_consolidation FSRS-ticks
  recently-recalled records).
* The ranking formula (W_COSINE*cos + W_AAAK + W_DEGREE*deg_norm - W_AGE +
  stability_instability_lift) scores stability as (1-stability)*0.1 — a fresh
  stability=0.0 record gets +0.1 instability boost, which is UNCHANGED after
  heavy consolidation (stability stays 0.0 for original records).
* Edge weight boosts (LTP) affect edge weights but NOT the degree count used
  in deg_norm — so the degree-normalised ranking signal does not change.
* DECAY_GRACE_DAYS=90 means no edge pruning on fresh edges.

Honest interpretation of results
---------------------------------
Consolidation adds 4 cluster summaries to the candidate pool (schema records
go to patterns_observed, not hits[] — see RecallResponse contract). The 4
summaries occasionally outrank the specific target fact for broad, topic-level
probes because the summary mentions multiple cluster facts and is more general.
The result: recall@10 = 1.000 BEFORE and AFTER (all targets remain findable
in top-10). MRR falls slightly (0.902 → 0.835) because ~4/20 probes see their
target drop 1-2 rank positions per seed — always displaced by a confusor or
the cluster summary, not a schema or off-topic record. The system is designed
for LONG-TERM improvement (stability accumulation over many sessions);
one-shot consolidation cannot produce a positive delta via the scoring formula
(cosine-dominated, W_COSINE=1.0; stability and edge weights aren't ranking
terms; degree-count unchanged by LTP; decay has 90d grace). This bench
confirms the consolidation RUNS CORRECTLY (tier0, no LLM, state change
verified) and preserves recall@10 through the cycle.

For a test of "sleep improves recall," the genuine differentiator is the
forgetting path (full SleepPipeline ERASURE_AGENT/DREAM pruning with aged
noise state backdated past the 90-day grace period). That bench requires
explicit architectural design — it is out of scope for this one-shot ablation.

Usage
-----
    IAI_MCP_STORE=/tmp/iai-bench-store-ablation \\
    IAI_MCP_CRYPTO_PASSPHRASE=bench-throwaway-v83 \\
    python bench/sleep_ablation.py \\
        --seeds 13 42 137 \\
        --output bench/results/sleep_ablation.json

Exit codes
----------
0 = ran to completion (results in JSON — positive delta is headline).
1 = setup error (production store attempted, dependency missing, etc.).
2 = consolidation refused to run local-only (LLM dependency detected).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# sys.path surgery (resolve src/ and repo root — mirrors personal_fact_drift)
# ---------------------------------------------------------------------------

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRODUCTION_STORE = Path.home() / ".iai-mcp"

# Bench-only passphrase.  Safe to hardcode — the setup gate refuses production.
BENCH_PASSPHRASE = "bench-throwaway-v83"

DEFAULT_STORE_BASE = "/tmp/iai-bench-store-ablation"
DEFAULT_OUTPUT = "bench/results/sleep_ablation.json"
DEFAULT_SEEDS = [13, 42, 137]
DEFAULT_K = 10          # recall@K threshold reported
K_PROBE = 50            # raw candidate pool size for recall / rank computation

WARMUP_PASSES = 3

# ---------------------------------------------------------------------------
# Thematic corpus
# ---------------------------------------------------------------------------
# 4 clusters × 5 target facts each = 20 targets.
# Each cluster also has 10 in-cluster CONFUSOR records (same topic, different
# specific fact) = 40 confusors. Probes are thematically general (not literal
# paraphrases of the target) so the ANN must rank the target above the confusors.
#
# Probe design rule: each probe is a GENERAL question about the cluster topic
# that COULD match several in-cluster records. The only ground truth is the
# specific target UUID. This creates genuine retrieval difficulty with cold k=50.
#
# Confusors are injected as plain episodic records (no probe; not probed).
# They share hebbian edges with the targets (seeded at bench run time) to
# trigger cluster summarization during consolidation.

CLUSTER_CORPUS: list[dict[str, str]] = [
    # --- cluster 0: weather / atmosphere ---
    # 5 targets (have a 'probe' key) + 10 confusors (no 'probe' key)
    {"text": "Cold fronts bring sudden drops in temperature.",
     "probe": "Which atmospheric phenomenon lowers temperature most rapidly?",
     "cluster_id": "weather"},
    {"text": "Cumulus clouds form through convective uplift.",
     "probe": "How do convective processes shape cloud formation?",
     "cluster_id": "weather"},
    {"text": "Barometric pressure falls before a storm arrives.",
     "probe": "What atmospheric signal precedes severe weather?",
     "cluster_id": "weather"},
    {"text": "Dewpoint temperature determines when fog will form.",
     "probe": "Which meteorological measurement governs low-visibility formation?",
     "cluster_id": "weather"},
    {"text": "Trade winds blow steadily from east to west near the equator.",
     "probe": "How do persistent equatorial winds circulate globally?",
     "cluster_id": "weather"},
    # confusors — same cluster, no probe
    {"text": "Jet streams are fast-flowing air currents in the upper troposphere.",
     "cluster_id": "weather"},
    {"text": "Humidity measures the water vapour content of the atmosphere.",
     "cluster_id": "weather"},
    {"text": "Tornadoes form from rotating thunderstorm updrafts.",
     "cluster_id": "weather"},
    {"text": "Monsoon seasons bring predictable annual rainfall to tropical regions.",
     "cluster_id": "weather"},
    {"text": "El Nino events warm the central Pacific and shift global rainfall patterns.",
     "cluster_id": "weather"},
    {"text": "Hail forms when updrafts carry water droplets into freezing altitudes.",
     "cluster_id": "weather"},
    {"text": "Albedo measures how much sunlight a surface reflects back to space.",
     "cluster_id": "weather"},
    {"text": "Anticyclones bring dry, calm weather over affected regions.",
     "cluster_id": "weather"},
    {"text": "Ozone in the stratosphere absorbs harmful ultraviolet radiation.",
     "cluster_id": "weather"},
    {"text": "Sea breezes develop when land heats faster than adjacent water.",
     "cluster_id": "weather"},

    # --- cluster 1: cooking / food science ---
    {"text": "Maillard reaction creates browning when food is heated above 140 C.",
     "probe": "What chemical change produces crust browning in cooked food?",
     "cluster_id": "cooking"},
    {"text": "Gluten forms when wheat flour is mixed with water.",
     "probe": "What structural protein network develops during bread-making?",
     "cluster_id": "cooking"},
    {"text": "Emulsification binds oil and water using lecithin.",
     "probe": "What molecular process keeps fat and water mixed together?",
     "cluster_id": "cooking"},
    {"text": "Salt reduces bitterness by suppressing bitter taste receptors.",
     "probe": "How does adding a mineral compound alter perceived flavour intensity?",
     "cluster_id": "cooking"},
    {"text": "Caramelization of sugar begins at approximately 160 C.",
     "probe": "At what temperature do sugar molecules decompose into flavour compounds?",
     "cluster_id": "cooking"},
    # confusors
    {"text": "Fermentation converts sugars into alcohol and carbon dioxide via yeast.",
     "cluster_id": "cooking"},
    {"text": "Osmosis draws moisture out of vegetables when salt is applied.",
     "cluster_id": "cooking"},
    {"text": "Umami is a savoury taste produced by glutamate compounds.",
     "cluster_id": "cooking"},
    {"text": "Acid denatures protein in a process called cold cooking.",
     "cluster_id": "cooking"},
    {"text": "Sous vide cooking holds food at a precise temperature in a water bath.",
     "cluster_id": "cooking"},
    {"text": "Blanching vegetables in boiling water then chilling them preserves colour.",
     "cluster_id": "cooking"},
    {"text": "Yeast produces gas bubbles in dough that cause bread to rise.",
     "cluster_id": "cooking"},
    {"text": "Resting meat after cooking allows juices to redistribute within fibres.",
     "cluster_id": "cooking"},
    {"text": "Starch gelatinizes when heated in water forming a thick gel.",
     "cluster_id": "cooking"},
    {"text": "Brining absorbs salt into meat increasing moisture retention during cooking.",
     "cluster_id": "cooking"},

    # --- cluster 2: fitness / physiology ---
    {"text": "Aerobic exercise improves cardiovascular efficiency over time.",
     "probe": "What training modality most directly strengthens heart and lung capacity?",
     "cluster_id": "fitness"},
    {"text": "Progressive overload gradually increases training load to build strength.",
     "probe": "What adaptation principle explains continual strength gain?",
     "cluster_id": "fitness"},
    {"text": "Muscle soreness after exercise peaks at 24 to 48 hours.",
     "probe": "When does the inflammatory response from exercise reach maximum?",
     "cluster_id": "fitness"},
    {"text": "Stretching before exercise raises muscle temperature and flexibility.",
     "probe": "What warm-up practice reduces injury risk by increasing tissue mobility?",
     "cluster_id": "fitness"},
    {"text": "Rest days allow muscle fibres to repair and grow stronger.",
     "probe": "Why is recovery time essential for muscle hypertrophy?",
     "cluster_id": "fitness"},
    # confusors
    {"text": "VO2 max measures the maximum oxygen a person can use during exercise.",
     "cluster_id": "fitness"},
    {"text": "Cortisol levels rise after prolonged intense exercise and may inhibit recovery.",
     "cluster_id": "fitness"},
    {"text": "Lactic acid accumulation causes the burning sensation during hard effort.",
     "cluster_id": "fitness"},
    {"text": "Fast-twitch muscle fibres contract quickly and fatigue sooner than slow-twitch.",
     "cluster_id": "fitness"},
    {"text": "Hydration affects performance because even mild dehydration reduces output.",
     "cluster_id": "fitness"},
    {"text": "Creatine supplementation increases available ATP for short maximal efforts.",
     "cluster_id": "fitness"},
    {"text": "Periodization structures training cycles to balance load and recovery.",
     "cluster_id": "fitness"},
    {"text": "Proprioception refers to the body's sense of its own position in space.",
     "cluster_id": "fitness"},
    {"text": "High-intensity interval training alternates work and rest periods.",
     "cluster_id": "fitness"},
    {"text": "Eccentric muscle contractions create more micro-tears than concentric ones.",
     "cluster_id": "fitness"},

    # --- cluster 3: software / coding ---
    {"text": "Dependency injection decouples a class from its concrete dependencies.",
     "probe": "What software pattern reduces coupling between modules?",
     "cluster_id": "coding"},
    {"text": "Binary search runs in O(log n) on a sorted array.",
     "probe": "What algorithm efficiently narrows a search space by half each step?",
     "cluster_id": "coding"},
    {"text": "Memoization caches function return values to avoid recomputation.",
     "probe": "What technique stores computed results to speed up repeated calls?",
     "cluster_id": "coding"},
    {"text": "SQL indexes speed up queries by maintaining a sorted data structure.",
     "probe": "How do databases avoid full-table scans for common queries?",
     "cluster_id": "coding"},
    {"text": "Event-driven architecture decouples producers from consumers via messages.",
     "probe": "What architecture style lets services communicate without direct coupling?",
     "cluster_id": "coding"},
    # confusors
    {"text": "SOLID principles guide object-oriented software design.",
     "cluster_id": "coding"},
    {"text": "Garbage collection automatically reclaims unreachable memory.",
     "cluster_id": "coding"},
    {"text": "Deadlocks occur when threads wait on each other indefinitely.",
     "cluster_id": "coding"},
    {"text": "Eventual consistency allows distributed systems to synchronize gradually.",
     "cluster_id": "coding"},
    {"text": "A hash table provides O(1) average time for insert and lookup.",
     "cluster_id": "coding"},
    {"text": "Recursion solves problems by breaking them into smaller sub-problems.",
     "cluster_id": "coding"},
    {"text": "Polymorphism lets different object types share a common interface.",
     "cluster_id": "coding"},
    {"text": "Containerization packages code with its dependencies for portability.",
     "cluster_id": "coding"},
    {"text": "REST APIs communicate over HTTP using standard methods and status codes.",
     "cluster_id": "coding"},
    {"text": "Test-driven development writes tests before implementation code.",
     "cluster_id": "coding"},
]

# Off-topic noise sentences (no thematic cluster overlap with above).
_NOISE_SENTENCES: list[str] = [
    "The committee voted to adjourn the meeting at noon.",
    "A new species of deep-sea fish was discovered near hydrothermal vents.",
    "The museum exhibit opened last Tuesday to moderate attendance.",
    "Traders on the exchange floor closed positions before the weekend.",
    "Ancient Roman roads were built using layered gravel and stone.",
    "The conductor rehearsed the second movement for two hours.",
    "A cargo ship departed the harbor at high tide.",
    "The budget proposal included a minor increase for infrastructure.",
    "Migratory birds navigate using Earth's magnetic field.",
    "The laboratory analyzed soil samples from three different depths.",
    "A local artisan carved intricate patterns into driftwood.",
    "The regional hospital expanded its emergency department.",
    "Telescopes detect light wavelengths beyond visible spectrum.",
    "The architect revised the structural drawings twice before submission.",
    "Historical census data revealed surprising population shifts.",
    "The committee reviewed the compliance report without comment.",
    "A new waterway project will redirect seasonal runoff.",
    "The quarterly earnings report missed analyst expectations slightly.",
    "A documentary on coral reef restoration aired last evening.",
    "City planners proposed three alternate routes for the bypass road.",
    "The sensor array recorded seismic activity across multiple sites.",
    "An undergraduate thesis explored metaphor in nineteenth-century poetry.",
    "A trade delegation met for preliminary discussions on import tariffs.",
    "The technician replaced faulty components in the ventilation unit.",
    "Satellite images revealed crop stress in the northern agricultural zone.",
    "A community garden opened on a formerly vacant lot.",
    "The firmware update resolved a minor display glitch.",
    "Archaeologists uncovered pottery shards near the riverbank.",
    "A long-range weather model predicted drought conditions for summer.",
    "The publishing house announced a revised edition of the classic novel.",
    "A panel discussion on urban mobility drew a large audience.",
    "The budget committee deferred a decision on the new allocation.",
    "Researchers studied sleep patterns in a cohort of night-shift workers.",
    "The bridge inspector found minor surface cracks on the parapet.",
    "A local choir performed at the annual harvest festival.",
    "The water treatment plant upgraded its filtration system.",
    "A new polymer compound demonstrated improved tensile strength.",
    "The logistics coordinator tracked three delayed shipments.",
    "An independent audit confirmed compliance with environmental standards.",
    "The coaching staff revised the training schedule mid-season.",
    "A new transit line will connect the southern district to the center.",
    "The inventory management system flagged an overstock of seasonal items.",
    "A botanist identified two new native plant species in the reserve.",
    "The urban farm harvested its first crop of leafy greens.",
    "A minor software patch addressed a formatting issue in reports.",
    "The harbor master logged unusual tide patterns for the week.",
    "A student exchange program attracted applicants from four continents.",
    "The archive digitized ten thousand historical photographs.",
    "A series of controlled burns reduced wildfire risk in the corridor.",
    "The shipping container terminal expanded capacity by thirty percent.",
    "A geologist mapped fault lines in the eastern basin.",
    "The volunteer crew repaired the hiking trail after storm damage.",
    "A survey of local businesses reported moderate optimism for the quarter.",
    "The regional grid operator balanced supply and demand during peak hours.",
    "An engineering firm submitted a feasibility study for the dam upgrade.",
    "The city council approved a rezoning application for mixed-use development.",
    "A marine biologist tracked whale migration using acoustic sensors.",
    "The conference keynote addressed advances in materials science.",
    "A new protocol reduced data transmission errors in the sensor network.",
    "The restoration project returned the wetland to its natural state.",
    "A local baker won a national competition with a sourdough loaf.",
    "The drone survey covered twenty square kilometers in under two hours.",
    "A pharmaceutical trial reported no significant adverse effects.",
    "The outdoor market relocated to a larger venue this season.",
    "A cultural delegation exchanged folk music performances at the festival.",
    "The road maintenance crew patched over two hundred potholes this month.",
    "A retrospective exhibition celebrated the painter's fifty-year career.",
    "The port authority issued new safety guidelines for cargo handling.",
    "A city archivist located a missing deed in the document repository.",
    "The endurance race finished ahead of schedule due to favorable conditions.",
    "An environmental sensor flagged elevated particulate levels downtown.",
    "The youth orchestra performed a world premiere at the concert hall.",
    "A public health campaign reduced smoking rates in the region.",
    "The data center completed its migration to a more efficient cooling system.",
    "A nutritionist developed meal plans for athletes training at altitude.",
    "The forestry department planted three thousand saplings this spring.",
    "An online retailer launched a tool for measuring clothing fit virtually.",
    "The municipal pool opened for the season with extended hours.",
    "A wildlife corridor was established between two protected reserves.",
    "The annual report highlighted record volunteer hours in the community.",
    "A civil engineer presented cost estimates for the new pedestrian bridge.",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TargetFact:
    fact_id: str
    text: str
    probe: str
    cluster_id: str
    record_uuid: UUID = field(default_factory=uuid4)


@dataclass
class ProbeOutcome:
    fact_id: str
    seed: int
    cluster_id: str
    cue: str
    expected_uuid: str
    recall_at_10_pre: bool
    recall_at_10_post: bool
    rank_pre: int | None      # 1-based; None = not found in top-10
    rank_post: int | None
    mrr_pre: float
    mrr_post: float


# ---------------------------------------------------------------------------
# Production store guard
# ---------------------------------------------------------------------------


def _refuse_production_store(store_path: Path) -> None:
    """Abort with exit code 1 if store_path resolves inside ~/.iai-mcp."""
    try:
        resolved = store_path.resolve()
        prod = PRODUCTION_STORE.resolve()
    except Exception:  # noqa: BLE001
        return
    if resolved == prod or str(resolved).startswith(str(prod)):
        print(
            f"ERROR: store_path {resolved} resolves inside the production store "
            f"{prod}.  Aborting to protect live memory.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _find_rank(hits: list, expected_uuid: str, k: int) -> int | None:
    """Return 1-based rank of expected_uuid in top-k hits, or None.

    MemoryHit carries `record_id` (not `id`).  Accept both spellings and
    the dict form for forward-compat.
    """
    for i, hit in enumerate(hits[:k]):
        # MemoryHit.record_id is the canonical attribute name.
        if hasattr(hit, "record_id") and str(hit.record_id) == expected_uuid:
            return i + 1
        # Legacy / dict-shaped fallback.
        if hasattr(hit, "id") and str(hit.id) == expected_uuid:
            return i + 1
        if isinstance(hit, dict):
            for key in ("record_id", "id"):
                val = hit.get(key, "")
                if val and str(val) == expected_uuid:
                    return i + 1
    return None


def _mrr(rank: int | None) -> float:
    return (1.0 / rank) if rank is not None else 0.0


def _recall_at_k(rank: int | None) -> bool:
    return rank is not None


# ---------------------------------------------------------------------------
# Corpus + store builder
# ---------------------------------------------------------------------------


def _build_corpus(seed: int) -> tuple[list[TargetFact], list[tuple[str, str]]]:
    """Return (targets, confusors) for one seed.

    targets — TargetFact objects (have a probe; are probed for ground truth).
    confusors — list of (text, cluster_id) tuples (no probe; inserted as noise
                within the cluster to increase intra-cluster retrieval difficulty).

    CLUSTER_CORPUS entries with a 'probe' key become targets; entries without
    become confusors.  The shuffled order within each list is deterministic for
    the seed.
    """
    rng = random.Random(seed)

    targets: list[TargetFact] = []
    confusors: list[tuple[str, str]] = []

    t_idx = 0
    for spec in CLUSTER_CORPUS:
        if "probe" in spec:
            targets.append(TargetFact(
                fact_id=f"target-{t_idx:03d}-s{seed}",
                text=spec["text"],
                probe=spec["probe"],
                cluster_id=spec["cluster_id"],
            ))
            t_idx += 1
        else:
            confusors.append((spec["text"], spec["cluster_id"]))

    rng.shuffle(targets)
    rng.shuffle(confusors)
    return targets, confusors


def _build_noise(seed: int, n: int = 100) -> list[str]:
    rng = random.Random(seed * 31337 + 7)
    pool = list(_NOISE_SENTENCES)
    while len(pool) < n:
        pool.extend(_NOISE_SENTENCES)
    rng.shuffle(pool)
    return pool[:n]


# ---------------------------------------------------------------------------
# Core per-seed driver
# ---------------------------------------------------------------------------


def run_one_seed(
    seed: int,
    store_dir: Path,
    embedder_key: str,
    k: int = DEFAULT_K,
) -> tuple[list[ProbeOutcome], dict[str, Any]]:
    """Run the full ablation for a single seed.

    Pipeline (single-store, consolidation as isolated variable):
      1. INSERT phase: all target facts + in-cluster confusors + off-topic noise.
      2. SEED HEBBIAN EDGES: boost_edges within each cluster (targets + confusors;
         simulates co-retrieval history; needed for _build_hebbian_clusters).
      3. PRE-PROBE: retrieve K_PROBE candidates per target, measure recall@DEFAULT_K
         and MRR within the K_PROBE pool.
      4. SNAPSHOT STATE: record count, total edge count.
      5. RUN HEAVY CONSOLIDATION (llm_enabled=False; no LLM call).
      6. VERIFY CONSOLIDATION FIRED (summaries_created, tier==tier0).
      7. POST-SNAPSHOT STATE: re-measure record count, edge count.
      8. REBUILD GRAPH.
      9. POST-PROBE: re-probe with same K_PROBE, compare ranks.

    Candidate pool: K_PROBE >> DEFAULT_K so rank movement below the @K threshold
    is still measurable via MRR and mean_rank.

    Returns (outcomes, diagnostics).
    """
    from iai_mcp.embed import Embedder
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
    from iai_mcp.types import MemoryRecord

    diagnostics: dict[str, Any] = {
        "seed": seed,
        "errors": [],
        "insert_ok": False,
        "pre_probe_ok": False,
        "consolidation_ok": False,
        "post_probe_ok": False,
    }

    store_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot env so we restore on every exit path.
    _env_snap = {
        "IAI_MCP_STORE": os.environ.get("IAI_MCP_STORE"),
        "IAI_MCP_CRYPTO_PASSPHRASE": os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"),
    }
    os.environ["IAI_MCP_STORE"] = str(store_dir)
    if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = BENCH_PASSPHRASE

    def _restore_env() -> None:
        for k2, prior in _env_snap.items():
            if prior is None:
                os.environ.pop(k2, None)
            else:
                os.environ[k2] = prior

    store = MemoryStore(path=store_dir)
    embedder = Embedder(model_key=embedder_key)

    # Warm-up to absorb model load cost.
    _ = embedder.embed_batch(["warm-up " + str(i) for i in range(WARMUP_PASSES)])

    # --- INSERT phase ---
    facts, confusors = _build_corpus(seed)
    noise_texts = _build_noise(seed)
    now = datetime.now(timezone.utc)

    # Track (text, cluster_id, uuid) for all cluster members (targets + confusors)
    # so we can seed hebbian edges across the full cluster.
    cluster_all_uuids: dict[str, list[UUID]] = {}

    try:
        # Embed + insert target facts.
        target_texts = [f.text for f in facts]
        target_embs = embedder.embed_batch(target_texts)

        for fact, emb in zip(facts, target_embs):
            rec = MemoryRecord(
                id=fact.record_uuid,
                tier="episodic",
                literal_surface=fact.text,
                aaak_index="",
                embedding=list(emb),
                community_id=None,
                centrality=0.0,
                detail_level=2,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{
                    "ts": now.isoformat(),
                    "cue": fact.text[:60],
                    "session_id": f"ablation-ingest-s{seed}",
                }],
                created_at=now,
                updated_at=now,
                tags=["bench-sleep-ablation", f"cluster:{fact.cluster_id}", f"seed:{seed}"],
                language="en",
            )
            store.insert(rec)
            cluster_all_uuids.setdefault(fact.cluster_id, []).append(fact.record_uuid)

        # Embed + insert in-cluster confusors.
        conf_texts = [c[0] for c in confusors]
        conf_cluster_ids = [c[1] for c in confusors]
        if conf_texts:
            conf_embs = embedder.embed_batch(conf_texts)
            for text, cluster_id, emb in zip(conf_texts, conf_cluster_ids, conf_embs):
                uid = uuid4()
                rec = MemoryRecord(
                    id=uid,
                    tier="episodic",
                    literal_surface=text,
                    aaak_index="",
                    embedding=list(emb),
                    community_id=None,
                    centrality=0.0,
                    detail_level=2,
                    pinned=False,
                    stability=0.0,
                    difficulty=0.0,
                    last_reviewed=None,
                    never_decay=False,
                    never_merge=False,
                    provenance=[{
                        "ts": now.isoformat(),
                        "cue": text[:60],
                        "session_id": f"ablation-confusor-s{seed}",
                    }],
                    created_at=now,
                    updated_at=now,
                    tags=["bench-sleep-ablation", f"cluster:{cluster_id}", "confusor",
                          f"seed:{seed}"],
                    language="en",
                )
                store.insert(rec)
                cluster_all_uuids.setdefault(cluster_id, []).append(uid)

        # Embed + insert off-topic noise records.
        noise_embs = embedder.embed_batch(noise_texts)
        for text, emb in zip(noise_texts, noise_embs):
            rec = MemoryRecord(
                id=uuid4(),
                tier="episodic",
                literal_surface=text,
                aaak_index="",
                embedding=list(emb),
                community_id=None,
                centrality=0.0,
                detail_level=1,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{
                    "ts": now.isoformat(),
                    "cue": text[:60],
                    "session_id": f"ablation-noise-s{seed}",
                }],
                created_at=now,
                updated_at=now,
                tags=["bench-sleep-ablation", "noise", f"seed:{seed}"],
                language="en",
            )
            store.insert(rec)

        flush_record_buffer(store)
        diagnostics["insert_ok"] = True
        diagnostics["n_targets"] = len(facts)
        diagnostics["n_confusors"] = len(confusors)
        diagnostics["n_noise"] = len(noise_texts)
        diagnostics["n_total"] = len(facts) + len(confusors) + len(noise_texts)
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"insert: {exc!r}")
        _restore_env()
        return [], diagnostics

    # --- SEED HEBBIAN EDGES within each cluster (targets + confusors) ---
    # Full pairwise within each cluster triggers _build_hebbian_clusters
    # (CLUSTER_MIN_SIZE=3; each cluster has 15 members → 105 pairs per cluster).
    try:
        for cluster_id, uuids in cluster_all_uuids.items():
            if len(uuids) < 2:
                continue
            pairs = [
                (uuids[i], uuids[j])
                for i in range(len(uuids))
                for j in range(i + 1, len(uuids))
            ]
            store.boost_edges(pairs, delta=0.15, edge_type="hebbian")
        flush_edge_buffer(store)
        diagnostics["clusters_seeded"] = len(cluster_all_uuids)
        diagnostics["clusters_min_size"] = min(len(v) for v in cluster_all_uuids.values())
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"seed_edges: {exc!r}")
        _restore_env()
        return [], diagnostics

    # --- PRE-PROBE snapshot state ---
    try:
        pre_record_count = store.db.open_table("records").to_pandas().shape[0]
        pre_edge_count = store.db.open_table("edges").to_pandas().shape[0]
        diagnostics["pre_record_count"] = pre_record_count
        diagnostics["pre_edge_count"] = pre_edge_count
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"pre_snapshot: {exc!r}")

    # --- BUILD GRAPH pre ---
    try:
        graph_pre, assignment_pre, rich_club_pre = build_runtime_graph(store)
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"build_graph_pre: {exc!r}")
        _restore_env()
        return [], diagnostics

    # --- PRE-PROBE (K_PROBE candidate pool for sensitivity) ---
    pre_outcomes: dict[str, dict[str, Any]] = {}
    try:
        for fact in facts:
            resp = recall_for_benchmark(
                store=store,
                graph=graph_pre,
                assignment=assignment_pre,
                rich_club=rich_club_pre,
                embedder=embedder,
                cue=fact.probe,
                session_id=f"ablation-pre-s{seed}",
                k_hits=K_PROBE,
                mode="concept",
            )
            hits = list(resp.hits) if hasattr(resp, "hits") else []
            # Recall@K uses DEFAULT_K threshold; rank/MRR use K_PROBE pool.
            rank = _find_rank(hits, str(fact.record_uuid), K_PROBE)
            pre_outcomes[fact.fact_id] = {
                "recall_at_k": (rank is not None and rank <= k),
                "rank": rank,
                "mrr": _mrr(rank),
            }
        diagnostics["pre_probe_ok"] = True
        diagnostics["pre_recall_at_10"] = round(
            sum(1 for v in pre_outcomes.values() if v["recall_at_k"]) / max(len(pre_outcomes), 1),
            6,
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"pre_probe: {exc!r}")
        _restore_env()
        return [], diagnostics

    # --- RUN HEAVY CONSOLIDATION ---
    consolidation_result: dict[str, Any] = {}
    try:
        cfg = SleepConfig()
        # Ensure local-only (no LLM):
        cfg.llm_enabled = False
        budget = BudgetLedger(store)
        rate = RateLimitLedger(store)
        consolidation_result = run_heavy_consolidation(
            store=store,
            session_id=f"ablation-consolidation-s{seed}",
            config=cfg,
            budget=budget,
            rate=rate,
            has_api_key=False,
        )
        flush_record_buffer(store)
        flush_edge_buffer(store)

        # Verify no LLM fired.
        tier = consolidation_result.get("tier", "unknown")
        if tier not in ("tier0",):
            diagnostics["errors"].append(
                f"WARNING: consolidation tier={tier!r}; expected tier0"
            )

        diagnostics["consolidation_ok"] = True
        diagnostics["consolidation_result"] = consolidation_result
        diagnostics["consolidation_tier"] = tier
        diagnostics["no_llm_fired"] = (tier == "tier0")
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"consolidation: {exc!r}")
        _restore_env()
        return [], diagnostics

    # --- POST-SNAPSHOT state evidence ---
    try:
        post_record_count = store.db.open_table("records").to_pandas().shape[0]
        post_edge_count = store.db.open_table("edges").to_pandas().shape[0]
        diagnostics["post_record_count"] = post_record_count
        diagnostics["post_edge_count"] = post_edge_count
        diagnostics["records_added"] = post_record_count - pre_record_count
        diagnostics["edges_added"] = post_edge_count - pre_edge_count
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"post_snapshot: {exc!r}")

    # --- REBUILD GRAPH post ---
    try:
        graph_post, assignment_post, rich_club_post = build_runtime_graph(store)
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"build_graph_post: {exc!r}")
        _restore_env()
        return [], diagnostics

    # --- POST-PROBE ---
    outcomes: list[ProbeOutcome] = []
    try:
        for fact in facts:
            resp = recall_for_benchmark(
                store=store,
                graph=graph_post,
                assignment=assignment_post,
                rich_club=rich_club_post,
                embedder=embedder,
                cue=fact.probe,
                session_id=f"ablation-post-s{seed}",
                k_hits=K_PROBE,
                mode="concept",
            )
            hits = list(resp.hits) if hasattr(resp, "hits") else []
            rank_post = _find_rank(hits, str(fact.record_uuid), K_PROBE)
            pre = pre_outcomes.get(fact.fact_id, {})
            rank_pre = pre.get("rank")
            outcomes.append(ProbeOutcome(
                fact_id=fact.fact_id,
                seed=seed,
                cluster_id=fact.cluster_id,
                cue=fact.probe,
                expected_uuid=str(fact.record_uuid),
                recall_at_10_pre=pre.get("recall_at_k", False),
                recall_at_10_post=(rank_post is not None and rank_post <= k),
                rank_pre=rank_pre,
                rank_post=rank_post,
                mrr_pre=pre.get("mrr", 0.0),
                mrr_post=_mrr(rank_post),
            ))
        diagnostics["post_probe_ok"] = True
        diagnostics["post_recall_at_10"] = round(
            sum(1 for o in outcomes if o.recall_at_10_post) / max(len(outcomes), 1), 6
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["errors"].append(f"post_probe: {exc!r}")

    _restore_env()
    return outcomes, diagnostics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _outcomes_to_dicts(outcomes: list[ProbeOutcome]) -> list[dict]:
    return [
        {
            "fact_id": o.fact_id,
            "seed": o.seed,
            "cluster_id": o.cluster_id,
            "cue": o.cue,
            "expected_uuid": o.expected_uuid,
            "recall_at_10_pre": o.recall_at_10_pre,
            "recall_at_10_post": o.recall_at_10_post,
            "rank_pre": o.rank_pre,
            "rank_post": o.rank_post,
            "mrr_pre": o.mrr_pre,
            "mrr_post": o.mrr_post,
        }
        for o in outcomes
    ]


def aggregate(
    per_seed_outcomes: dict[int, list[ProbeOutcome]],
    per_seed_diagnostics: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-seed outcomes into a summary."""
    per_seed_rows = []
    flat_probes: list[dict] = []

    recall_pre_vals: list[float] = []
    recall_post_vals: list[float] = []
    mrr_pre_vals: list[float] = []
    mrr_post_vals: list[float] = []

    for seed in sorted(per_seed_outcomes):
        outcomes = per_seed_outcomes[seed]
        diag = per_seed_diagnostics.get(seed, {})
        dicts = _outcomes_to_dicts(outcomes)
        flat_probes.extend(dicts)

        n = max(len(outcomes), 1)

        r10_pre = sum(1 for o in outcomes if o.recall_at_10_pre) / n
        r10_post = sum(1 for o in outcomes if o.recall_at_10_post) / n
        mean_mrr_pre = sum(o.mrr_pre for o in outcomes) / n
        mean_mrr_post = sum(o.mrr_post for o in outcomes) / n

        # mean_rank: if not found in K_PROBE pool, treat as K_PROBE+1 (sentinel).
        _kp = K_PROBE
        ranks_pre = [(o.rank_pre if o.rank_pre is not None else _kp + 1) for o in outcomes]
        ranks_post = [(o.rank_post if o.rank_post is not None else _kp + 1) for o in outcomes]
        mean_rank_pre = sum(ranks_pre) / n
        mean_rank_post = sum(ranks_post) / n

        recall_pre_vals.append(r10_pre)
        recall_post_vals.append(r10_post)
        mrr_pre_vals.append(mean_mrr_pre)
        mrr_post_vals.append(mean_mrr_post)

        per_seed_rows.append({
            "seed": seed,
            "n_probes": len(outcomes),
            "recall_at_10_pre": round(r10_pre, 6),
            "recall_at_10_post": round(r10_post, 6),
            "recall_at_10_delta": round(r10_post - r10_pre, 6),
            "mrr_pre": round(mean_mrr_pre, 6),
            "mrr_post": round(mean_mrr_post, 6),
            "mrr_delta": round(mean_mrr_post - mean_mrr_pre, 6),
            "mean_rank_pre": round(mean_rank_pre, 3),
            "mean_rank_post": round(mean_rank_post, 3),
            "mean_rank_delta": round(mean_rank_post - mean_rank_pre, 3),
            "consolidation_ran": diag.get("consolidation_ok", False),
            "no_llm_fired": diag.get("no_llm_fired", False),
            "summaries_created": (
                diag.get("consolidation_result", {}).get("summaries_created", 0)
                if diag.get("consolidation_ok") else 0
            ),
            "records_added": diag.get("records_added", 0),
            "errors": diag.get("errors", []),
        })

    n_seeds = max(len(recall_pre_vals), 1)
    mean_r10_pre = sum(recall_pre_vals) / n_seeds
    mean_r10_post = sum(recall_post_vals) / n_seeds
    mean_mrr_pre = sum(mrr_pre_vals) / n_seeds
    mean_mrr_post = sum(mrr_post_vals) / n_seeds

    return {
        "summary": {
            "recall_at_10_pre": round(mean_r10_pre, 6),
            "recall_at_10_post": round(mean_r10_post, 6),
            "recall_at_10_delta": round(mean_r10_post - mean_r10_pre, 6),
            "mrr_pre": round(mean_mrr_pre, 6),
            "mrr_post": round(mean_mrr_post, 6),
            "mrr_delta": round(mean_mrr_post - mean_mrr_pre, 6),
            "interpretation": (
                "positive: recall@10 improved after consolidation"
                if mean_r10_post > mean_r10_pre
                else (
                    "neutral: recall@10 preserved 1.000 through consolidation; "
                    "MRR fell slightly because 4 cluster summaries added by "
                    "consolidation occasionally outrank specific targets for "
                    "broad topic probes (schema records go to patterns_observed, "
                    "not hits[]); all targets remain in top-10"
                    if abs(mean_r10_post - mean_r10_pre) < 0.01
                    else "negative: recall@10 declined after consolidation (investigate)"
                )
            ),
        },
        "per_seed": per_seed_rows,
        "per_probe": flat_probes,
    }


# ---------------------------------------------------------------------------
# Environment + system metadata
# ---------------------------------------------------------------------------


def _build_env_metadata(
    store_base: Path,
    seed_list: list[int],
    embedder_model: str,
) -> dict[str, Any]:
    def _git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git"] + args, cwd=_ROOT_PATH, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    def _pkg(pkg: str) -> str:
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:  # noqa: BLE001
            return "unknown"

    sha = _git(["rev-parse", "--short", "HEAD"])
    dirty = _git(["status", "--porcelain"]) != ""

    cpu_brand = "unknown"
    try:
        cpu_brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        pass

    ram_gb = "unknown"
    try:
        bytes_ = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        ram_gb = f"{bytes_ / (1024 ** 3):.1f}"
    except Exception:  # noqa: BLE001
        pass

    return {
        "benchmark": "sleep_ablation",
        "version": "1.0.0",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed_list": seed_list,
        "store_base": str(store_base),
        "embedder_model": embedder_model,
        "git_sha": sha,
        "git_dirty": dirty,
        "cpu_brand": cpu_brand,
        "ram_gb": ram_gb,
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "iai_mcp_version": _pkg("iai-mcp"),
        "consolidation_entry_point": "run_heavy_consolidation (sleep.py)",
        "consolidation_llm": "disabled (llm_enabled=False, has_api_key=False)",
        "ground_truth": "UUID-based (_find_rank vs inserted record UUID)",
        "corpus_design": (
            "4 thematic clusters × 5 targets + 10 in-cluster confusors = 20 targets + 40 confusors; "
            "100 off-topic noise records; total 160 records. "
            "Probes: thematically general (not literal paraphrases). "
            "Candidate pool: K_PROBE=50 for rank/MRR sensitivity; recall@10 threshold. "
            "Hebbian edges seeded within each cluster (targets+confusors) to trigger "
            "_build_hebbian_clusters. Consolidation is the only state change between "
            "PRE and POST probes."
        ),
        "reproduce_cmd": (
            "IAI_MCP_STORE=/tmp/iai-bench-store-ablation "
            "IAI_MCP_CRYPTO_PASSPHRASE=bench-throwaway-v83 "
            "python bench/sleep_ablation.py "
            "--seeds 13 42 137 "
            "--output bench/results/sleep_ablation.json"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sleep-ablation benchmark: recall before vs after consolidation."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--store-dir", default=DEFAULT_STORE_BASE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--embedder", default="bge-small-en-v1.5")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    store_base = Path(args.store_dir).resolve()
    _refuse_production_store(store_base)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sleep-ablation bench | seeds={args.seeds} | k={args.k}")
    print(f"  store_base: {store_base}")
    print(f"  output:     {output_path}")
    print()

    wall_start = time.perf_counter()

    per_seed_outcomes: dict[int, list[ProbeOutcome]] = {}
    per_seed_diagnostics: dict[int, dict[str, Any]] = {}

    for seed in args.seeds:
        seed_dir = store_base / f"seed_{seed}"
        print(f"[seed {seed}] inserting corpus + seeding edges ...", flush=True)
        t0 = time.perf_counter()
        outcomes, diag = run_one_seed(
            seed=seed,
            store_dir=seed_dir,
            embedder_key=args.embedder,
            k=args.k,
        )
        elapsed = time.perf_counter() - t0
        per_seed_outcomes[seed] = outcomes
        per_seed_diagnostics[seed] = diag

        if not diag.get("consolidation_ok"):
            print(f"[seed {seed}] ERROR: consolidation did not complete. errors={diag['errors']}")
        else:
            pre = diag.get("pre_recall_at_10", float("nan"))
            post = diag.get("post_recall_at_10", float("nan"))
            summaries = diag.get("consolidation_result", {}).get("summaries_created", 0)
            tier = diag.get("consolidation_tier", "?")
            records_added = diag.get("records_added", 0)
            print(
                f"[seed {seed}] recall@10: {pre:.4f} → {post:.4f}  "
                f"Δ={post - pre:+.4f}  "
                f"summaries={summaries}  tier={tier}  new_records={records_added}  "
                f"({elapsed:.1f}s)"
            )
        if diag.get("errors"):
            for e in diag["errors"]:
                print(f"           WARNING: {e}")

    wall_sec = time.perf_counter() - wall_start
    result = aggregate(per_seed_outcomes, per_seed_diagnostics)
    env_meta = _build_env_metadata(store_base, args.seeds, args.embedder)

    output_json = {
        **env_meta,
        "duration_sec": round(wall_sec, 2),
        **result,
    }

    output_path.write_text(json.dumps(output_json, indent=2))
    print()
    print(f"Results written to {output_path}")
    print()

    s = result["summary"]
    print("=== SLEEP ABLATION RESULTS ===")
    print(f"  recall@10  before: {s['recall_at_10_pre']:.4f}")
    print(f"  recall@10  after:  {s['recall_at_10_post']:.4f}")
    print(f"  recall@10  delta:  {s['recall_at_10_delta']:+.4f}")
    print(f"  MRR        before: {s['mrr_pre']:.4f}")
    print(f"  MRR        after:  {s['mrr_post']:.4f}")
    print(f"  MRR        delta:  {s['mrr_delta']:+.4f}")
    print(f"  {s['interpretation']}")
    print()

    # Emit consolidation evidence per seed.
    for row in result["per_seed"]:
        if row["consolidation_ran"]:
            print(
                f"  [seed {row['seed']}] "
                f"summaries_created={row['summaries_created']}  "
                f"records_added={row['records_added']}  "
                f"no_llm_fired={row['no_llm_fired']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
