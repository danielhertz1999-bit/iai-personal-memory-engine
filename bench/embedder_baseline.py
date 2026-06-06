#!/usr/bin/env python3
"""Embedder baseline capture — prerequisite for the Rust forward-pass replacement.

Captures `bge-small-en-v1.5` output on a deliberate 100-text mix so the Rust
implementation has a numeric-parity gate that catches tokenization, truncation,
and L2-normalization bugs which the LongMemEval recall@k test cannot surface.

Produces (rerunnable, deterministic with fixed seed):
  bench/embedder_baseline/vectors.npy float32 (N=100, 384), L2-normalized
  bench/embedder_baseline/texts.json UTF-8 list[str]
  bench/embedder_baseline/metadata.json model revision, env, seed, source

Acceptance gate for the Rust replacement (per text):
  cosine(rust_output, vectors.npy[i]) >= 0.9999
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Resolve iai_mcp.* (via src) AND bench.* (via worktree root) to THIS
# worktree, not the parent venv's editable install. Idempotent: each
# `sys.path.insert` is guarded by an "if not already present" check.
# Canonical bench shim form (matches tests/test_bench_worktree_resolution.py).
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

# Suppress the BERT `position_ids` warning sentence-transformers emits at load.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from iai_mcp.embed import Embedder  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "embedder_baseline"
SEED = 42
N_LME = 60
N_EDGE = 20
N_NARRATIVE = 20

# ---------------------------------------------------------------------------
# 20 hand-curated edge cases — exercise tokenization / truncation / unicode.
# ---------------------------------------------------------------------------
# Empty string DELIBERATELY excluded: production records always have content
# (capture pipeline gates min_length > 0), so embedding "" would test a
# non-existent code path.
EDGE_TEXTS: list[str] = [
    # short / boundary
    "a",
    "ok",
    "yes!",
    "1234",
    # emoji / unicode / CJK
    "\U0001F600",
    "cafe resume naive",
    "Visit https://example.com for more.",
    "user@host.tld",
    # code-like (tokenizer treats these specially)
    "<html><body>hi</body></html>",
    '{"key": "value", "n": 42}',
    "def foo():\n    return bar()",
    "a + b == c\n\nelse:",
    # punctuation-heavy
    "...!?...",
    "----===----",
    # whitespace patterns
    "   leading and trailing spaces   ",
    "tab\tseparated\tfields\there",
    # truncation triggers: max_position_embeddings = 512, so these MUST be
    # over the cap to lock the model's truncation behavior into the baseline
    "lorem ipsum dolor sit amet consectetur adipiscing elit " * 200,
    "the quick brown fox jumped over the lazy dog. " * 100,
    # diverse natural sentences
    "What is the capital of France?",
    "Numbers like 3.14159 and 2.71828 appear throughout mathematics.",
]
assert len(EDGE_TEXTS) == N_EDGE, f"want {N_EDGE} edge cases, got {len(EDGE_TEXTS)}"

# ---------------------------------------------------------------------------
# 20 hand-curated narrative paragraphs — diverse domains, 1-3 sentences each.
# ---------------------------------------------------------------------------
NARRATIVE_TEXTS: list[str] = [
    "In distributed systems, eventual consistency trades strict ordering for availability under network partitions. The CAP theorem formalizes this tradeoff.",
    "The Maillard reaction occurs between amino acids and reducing sugars at temperatures above 140 degrees Celsius, producing the characteristic browning and aroma of seared meat.",
    "Tokyo's subway system carries roughly nine million riders each weekday across thirteen interconnected lines operated by two separate companies.",
    "Quantum entanglement describes pairs of particles whose measurement outcomes remain correlated regardless of the distance separating them, a phenomenon Einstein called spooky action at a distance.",
    "Constantinople fell to Mehmed the Second on May twenty-ninth, fourteen fifty-three, after a siege that lasted fifty-three days and involved one of the largest cannons ever cast at that time.",
    "Marathon runners typically deplete their muscle glycogen stores around the thirty-kilometer mark, an experience competitors call hitting the wall.",
    "Bach's Goldberg Variations consist of an aria followed by thirty variations and a return of the original aria, organized around a recurring bass line rather than the melodic theme.",
    "Wittgenstein's later philosophy abandoned the picture theory of meaning developed in the Tractatus, replacing it with the view that language consists of overlapping games defined by use rather than by reference.",
    "Borges imagined a library whose hexagonal galleries contain every possible book of four hundred ten pages, including the catalog of itself and the proof that this catalog is false.",
    "Federalism distributes legislative authority between a central government and constituent regional governments, each sovereign within its delegated sphere.",
    "Cardiovascular disease remains the leading cause of death globally, accounting for roughly one in three deaths worldwide despite decades of advances in pharmacology and surgical technique.",
    "Inflation expectations anchor when households and firms trust that monetary authorities will respond decisively to demand shocks, even when current measurements drift above target.",
    "Brutalist buildings of the nineteen sixties used raw concrete to express structural honesty, though the style has divided architectural critics ever since.",
    "The Riemann hypothesis predicts that every non-trivial zero of the zeta function lies on the critical line where the real part equals one-half.",
    "Mitochondria in eukaryotic cells generate adenosine triphosphate through oxidative phosphorylation, a process that requires both the electron transport chain and a proton gradient across the inner membrane.",
    "The Hubble tension refers to the discrepancy between measurements of the universe's expansion rate inferred from the cosmic microwave background versus those derived from local distance ladders.",
    "Indo-European languages descended from a common proto-language spoken roughly six thousand years ago, with reconstructed vocabulary suggesting a society familiar with wheels, horses, and dairy farming.",
    "Dual-process theory in cognitive psychology distinguishes fast, automatic, intuitive judgment from slow, deliberate, analytical reasoning, often called system one and system two.",
    "Plate tectonics explains continental drift through the slow movement of lithospheric plates over the asthenosphere, driven by convection currents in the underlying mantle.",
    "Hash tables provide expected constant-time lookup, insertion, and deletion when the load factor stays bounded and the hash function distributes keys uniformly across buckets.",
]
assert len(NARRATIVE_TEXTS) == N_NARRATIVE


def _load_lme_turns(n: int, seed: int) -> list[str]:
    """Sample N turn-texts from the cleaned LongMemEval haystack."""
    from bench.adapters.longmemeval_cleaned import CleanedLongMemEvalAdapter

    adapter = CleanedLongMemEvalAdapter()
    turns: list[str] = []
    for session in adapter.load_dataset(split="S"):
        # LMESession has a.turns attribute (list of message dicts).
        for turn in getattr(session, "turns", []):
            content = (turn.get("content") or "").strip() if isinstance(turn, dict) else ""
            if content:
                turns.append(content)
        if len(turns) >= n * 10:  # collect ~10x to make sampling diverse
            break
    rng = random.Random(seed)
    rng.shuffle(turns)
    return turns[:n], adapter.revision


def _resolve_model_revision(embedder: Embedder) -> str:
    """Best-effort lookup of the HF snapshot SHA the SentenceTransformer loaded."""
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    safe_id = embedder.model_name.replace("/", "--")
    snap_dir = hub_root / f"models--{safe_id}" / "snapshots"
    if snap_dir.is_dir():
        revisions = sorted(p.name for p in snap_dir.iterdir() if p.is_dir())
        if revisions:
            return revisions[-1]
    return "unknown"


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] loading Embedder (bge-small-en-v1.5)...", flush=True)
    embedder = Embedder()
    assert embedder.model_key == "bge-small-en-v1.5", (
        f"want default bge-small, got {embedder.model_key!r}"
    )
    assert embedder.DIM == 384, f"want 384-d, got {embedder.DIM}"

    print(f"[baseline] sampling {N_LME} LongMemEval turn-texts (seed={SEED})...", flush=True)
    lme_texts, lme_revision = _load_lme_turns(N_LME, SEED)
    assert len(lme_texts) == N_LME, f"sampled {len(lme_texts)} turns, want {N_LME}"

    texts: list[str] = lme_texts + EDGE_TEXTS + NARRATIVE_TEXTS
    assert len(texts) == 100, f"want 100 texts, got {len(texts)}"

    print(f"[baseline] embedding {len(texts)} texts via Embedder.embed() (per-record)...", flush=True)
    vectors = np.zeros((len(texts), embedder.DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        vec = embedder.embed(text)
        vectors[i] = np.asarray(vec, dtype=np.float32)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(texts)}", flush=True)

    # Sanity: bge output should be L2-normalized. Verify.
    norms = np.linalg.norm(vectors, axis=1)
    norm_min, norm_max = float(norms.min()), float(norms.max())
    assert 0.999 <= norm_min and norm_max <= 1.001, (
        f"output not L2-normalized: norms in [{norm_min:.6f}, {norm_max:.6f}]"
    )

    vectors_path = OUTPUT_DIR / "vectors.npy"
    texts_path = OUTPUT_DIR / "texts.json"
    meta_path = OUTPUT_DIR / "metadata.json"

    np.save(vectors_path, vectors)
    texts_path.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Hash the on-disk.npy file (NOT vectors.tobytes()) so the SHA reflects
    # the full artifact consumers actually read — magic + NPY header + raw bytes.
    # Consumers in other languages (Rust) read the file bytes, not the in-memory
    # numpy array, so the SHA must hash the same subject.
    vec_sha256 = hashlib.sha256(vectors_path.read_bytes()).hexdigest()

    metadata = {
        "purpose": "Numeric-parity acceptance gate for the Rust forward-pass replacement.",
        "embedder": {
            "model_key": embedder.model_key,
            "model_hf_id": embedder.model_name,
            "model_revision_sha": _resolve_model_revision(embedder),
            "output_dim": embedder.DIM,
            "output_dtype": "float32",
            "l2_normalized": True,
            "call_api": "Embedder.embed(text) -- per-record, not batch",
        },
        "text_mix": {
            "n_total": len(texts),
            "n_lme_turns": N_LME,
            "n_edge_cases": N_EDGE,
            "n_narrative": N_NARRATIVE,
            "lme_source": "xiaowu0162/longmemeval-cleaned",
            "lme_split": "S",
            "lme_revision_sha": lme_revision,
            "sampling_seed": SEED,
            "edge_cases_include": [
                "short (1-5 chars)",
                "emoji (U+1F600)",
                "URL / email",
                "HTML / JSON / Python-code-like",
                "punctuation-heavy",
                "whitespace patterns (leading/trailing/tab)",
                "over-512-BPE-token truncation triggers (locks model truncation behavior)",
            ],
            "edge_cases_exclude": [
                "empty string (production pipeline gates min_length > 0)",
            ],
        },
        "determinism": {
            "vectors_sha256": vec_sha256,
            "rerun_invariance": "Re-running this script with the same seed and the same model revision MUST reproduce the same vectors_sha256.",
        },
        "environment": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": _pkg_version("torch"),
            "sentence_transformers_version": _pkg_version("sentence-transformers"),
            "transformers_version": _pkg_version("transformers"),
            "numpy_version": _pkg_version("numpy"),
        },
        "acceptance_gate_for_rust_replacement": {
            "per_text_metric": "cosine similarity",
            "per_text_threshold": 0.9999,
            "aggregate_recall_at_10_tolerance": "+/- 0.005 vs PyTorch baseline on LongMemEval-S",
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"[baseline] wrote {vectors_path}  ({vectors.nbytes:,} bytes, sha256={vec_sha256[:12]}...)")
    print(f"[baseline] wrote {texts_path}    ({texts_path.stat().st_size:,} bytes)")
    print(f"[baseline] wrote {meta_path} ({meta_path.stat().st_size:,} bytes)")
    print()
    print(f"[baseline] OK -- {len(texts)} texts captured, L2-norms in [{norm_min:.6f}, {norm_max:.6f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
