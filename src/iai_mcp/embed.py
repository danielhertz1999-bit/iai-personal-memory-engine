"""Embedding layer -- configurable embedder with a 3-model registry.

Plan 05-08 (2026-04-20): the DEFAULT is now ``bge-small-en-v1.5`` (384d
English-only), reverting the Phase-2 deviation. PROJECT.md line
125 always specified bge-small-en-v1.5 as the intended default; Phase-2
swapped in bge-m3 (1024d multilingual) as D-08a. User directive
2026-04-19: the brain stores English, surface translation is Claude's
job. bge-m3 stays selectable via env var / kwarg for anyone who needs
multilingual semantic match at the 5x RAM cost.

Configurable 4-model registry:
- "bge-m3"                 -> BAAI/bge-m3               -> 1024d (opt-in, multilingual)
- "multilingual-e5-small"  -> intfloat/multilingual-e5-small -> 384d (compromise)
- "bge-small-en-v1.5"      -> BAAI/bge-small-en-v1.5    -> 384d (DEFAULT, English)
- "all-MiniLM-L6-v2"       -> sentence-transformers/all-MiniLM-L6-v2 -> 384d (English alternative embedder option; included for compatibility testing)

Selection priority at Embedder() instantiation:
1. Explicit `model_key` constructor arg
2. IAI_MCP_EMBED_MODEL environment variable
3. MODEL_REGISTRY default ("bge-small-en-v1.5")

The model is loaded once per process and cached in a module-level dict so
multiple Embedder() instances share the underlying SentenceTransformer.

Deterministic: `normalize_embeddings=True` is always passed,
`show_progress_bar=False`. Same input text always produces the same output
vector across calls within a process.
"""
from __future__ import annotations

import os
import threading

from sentence_transformers import SentenceTransformer


# 4-model registry. Name convention: short logical key -> HF repo id + dim.
# (2026-04-29): all-MiniLM-L6-v2 added as additive ablation entry;
# DEFAULT_MODEL_KEY unchanged (English-Only Brain lock from / Plan 05-08).
MODEL_REGISTRY: dict[str, dict] = {
    "bge-m3": {"hf": "BAAI/bge-m3", "dim": 1024},
    "multilingual-e5-small": {"hf": "intfloat/multilingual-e5-small", "dim": 384},
    "bge-small-en-v1.5": {"hf": "BAAI/bge-small-en-v1.5", "dim": 384},
    "all-MiniLM-L6-v2": {"hf": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384},
}
DEFAULT_MODEL_KEY = "bge-small-en-v1.5"


def _resolve_model_key(model_key: str | None = None) -> str:
    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown embed model key {model_key!r}; valid: {sorted(MODEL_REGISTRY)}"
            )
        return model_key
    env_key = os.environ.get("IAI_MCP_EMBED_MODEL")
    if env_key:
        if env_key not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown embed model key {env_key!r} from IAI_MCP_EMBED_MODEL; "
                f"valid: {sorted(MODEL_REGISTRY)}"
            )
        return env_key
    return DEFAULT_MODEL_KEY


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(hf_id: str) -> SentenceTransformer:
    """Process-local lazy-load + cache. Thread-safe via lock around cache mutation."""
    with _MODEL_LOCK:
        if hf_id not in _MODEL_CACHE:
            _MODEL_CACHE[hf_id] = SentenceTransformer(hf_id)
        return _MODEL_CACHE[hf_id]


class Embedder:
    """English-Only Brain embedder with a configurable model registry.

    Default model is `bge-small-en-v1.5` (384d, English) per Plan 05-08.
    Used by the retrieval pipeline (stage 1, cue embedding) and by session-start
    assembler. `.DIM` is per-instance (varies by model). `.DEFAULT_DIM` is a
    class-level default pointing at the registry's default model dimension.

    The opt-in `bge-m3` (1024d multilingual) path stays in the registry for
    users who explicitly need multilingual semantic match at the 5x RAM cost,
    but it is opt-in via `IAI_MCP_EMBED_MODEL=bge-m3` — not the product.

    Backward compatibility:
    - `Embedder.DIM` is kept as a class attribute aliased to the default model
      dimension so tests that reference `Embedder.DIM` still work; new
      code should prefer `Embedder().DIM` (instance attr) for correctness.
    - `Embedder.DEFAULT_MODEL` is the HF id of the default model (bge-small-en-v1.5).
    """

    DEFAULT_MODEL_KEY: str = DEFAULT_MODEL_KEY
    DEFAULT_DIM: int = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["dim"]
    # Legacy class-level attributes (Phase 1 test compatibility).
    # New code should construct Embedder() and read .DIM from the instance.
    DEFAULT_MODEL: str = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["hf"]
    DIM: int = DEFAULT_DIM

    def __init__(
        self,
        model_key: str | None = None,
        *,
        model_name: str | None = None,
    ) -> None:
        """Initialise an Embedder.

        Parameters
        ----------
        model_key:
            Logical key from MODEL_REGISTRY ("bge-m3" | "multilingual-e5-small" |
            "bge-small-en-v1.5"). If None, uses IAI_MCP_EMBED_MODEL env var or
            the registry default.
        model_name:
            Legacy parameter: full HuggingFace repo id (e.g. "BAAI/bge-small-en-v1.5").
            Prefer model_key for new code. If both are provided, model_key wins.
        """
        if model_key is None and model_name is not None:
            # Reverse-lookup: find the key whose hf matches this name.
            match = next(
                (k for k, v in MODEL_REGISTRY.items() if v["hf"] == model_name),
                None,
            )
            if match is None:
                raise ValueError(
                    f"model_name {model_name!r} is not in MODEL_REGISTRY; "
                    f"valid hf ids: {[v['hf'] for v in MODEL_REGISTRY.values()]}"
                )
            key = match
        else:
            key = _resolve_model_key(model_key)
        self.model_key: str = key
        spec = MODEL_REGISTRY[key]
        self.model_name: str = spec["hf"]
        self.DIM: int = int(spec["dim"])  # instance attr overrides class attr
        self._model = _get_model(self.model_name)

    def embed(self, text: str) -> list[float]:
        """Encode a single string to a DIM-length list[float]. Normalised, deterministic."""
        vec = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode preserving input order. Returns N vectors for N inputs."""
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [v.tolist() for v in vecs]


def embedder_for_store(store) -> "Embedder":
    """Store-aware Embedder factory. Picks the model whose output dim matches
    the existing LanceDB records schema, so a legacy 1024d store from the
    pre-Plan-05-08 bge-m3 era stays queryable until it is re-embedded down to
    the 384d English-Only-Brain default.

    Resolution order:
    1. If store.embed_dim has an exact match in MODEL_REGISTRY, prefer the
       model whose logical key name indicates the canonical model at that dim
       (bge-small-en-v1.5 for 384d default; bge-m3 for legacy/opt-in 1024d).
    2. Otherwise fall through to the env/registry default via Embedder().

    This decouples runtime model selection from a global env var so a single
    process can operate multiple stores at different dims while the migration
    from a legacy 1024d store down to 384d completes.
    """
    target_dim = getattr(store, "embed_dim", None)
    if target_dim is None:
        return Embedder()
    preferred = {384: "bge-small-en-v1.5", 1024: "bge-m3"}
    key = preferred.get(int(target_dim))
    # Tests and migrations may monkey-patch `Embedder` with a stub that takes no
    # kwargs. Fall back to the zero-arg form in that case so the fake surface
    # stays compatible; real production code still respects store.embed_dim.
    try:
        if key is not None and key in MODEL_REGISTRY:
            return Embedder(model_key=key)
        for reg_key, spec in MODEL_REGISTRY.items():
            if int(spec["dim"]) == int(target_dim):
                return Embedder(model_key=reg_key)
    except TypeError:
        pass
    return Embedder()
