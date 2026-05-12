"""Embedding layer -- configurable embedder with a 3-model registry.

Plan 05-08 (2026-04-20): the DEFAULT is now ``bge-small-en-v1.5`` (384d
English-only), reverting the Phase-2 deviation. PROJECT.md line
125 always specified bge-small-en-v1.5 as the intended default; Phase-2
swapped in bge-m3 (1024d multilingual). User directive
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
from dataclasses import dataclass

import numpy as np
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

# Opt-in WRITE-side quantization knob. Default (env unset) keeps the fp32
# path byte-identical; int8 is exposed via the additive
# Embedder.embed_quantized() surface only. Extensible: future modes (e.g.
# "fp16") can be added to this set.
VALID_QUANTIZE_MODES: set[str] = {"int8"}


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


def _resolve_quantize_mode() -> str | None:
    """Read ``IAI_MCP_EMBED_QUANTIZE`` env var; return mode or None.

    Empty string or unset → ``None`` (fp32 default — unchanged behavior).
    ``"int8"`` (case-sensitive, lower-case only) → ``"int8"``.
    Any other non-empty value → ``ValueError``. NO silent fallback.

    Case-sensitivity choice: lower-case only. ``"INT8"`` is rejected so the
    knob value matches the canonical mode token used in storage metadata
    when a future task wires int8 into a parallel Lance column path.

    Default remains fp32 until a manual LongMemEval-S A/B subset validates
    <1% recall loss on the int8 path.
    """
    raw = os.environ.get("IAI_MCP_EMBED_QUANTIZE", "")
    if not raw:
        return None
    if raw not in VALID_QUANTIZE_MODES:
        raise ValueError(
            f"IAI_MCP_EMBED_QUANTIZE={raw!r} is not a valid quantization mode; "
            f"valid: {sorted(VALID_QUANTIZE_MODES)} or unset for fp32 default"
        )
    return raw


@dataclass(frozen=True)
class QuantizedVector:
    """int8-quantized embedding with per-vector min/max calibration metadata.

    Reconstruct the fp32 approximation via:

        fp32[i] ≈ (values[i] - zero_point) * scale

    Per-vector calibration is used because BGE per-dim values cluster
    narrowly around 0 but are NOT confined to [-1, 1] despite L2
    normalization. A global codebook would waste resolution; per-vector
    min/max maps each vector's full dynamic range onto the [-128, 127]
    int8 codebook, preserving cos >= 0.99 round-trip on real probes.
    """

    values: list[int]   # length == dim; each in [-128, 127] (signed int8)
    scale: float        # per-vector scale = (vmax - vmin) / 255
    zero_point: int     # per-vector zero-point in the int8 codebook
    dim: int            # convenience; equals len(values)


def _quantize_int8(vec: list[float]) -> QuantizedVector:
    """Per-vector min/max int8 quantization of a fp32 embedding vector.

    Inverse: ``fp32[i] ≈ (values[i] - zero_point) * scale``. Empirically
    preserves cos >= 0.99 on real bge-small-en-v1.5 outputs (test 4 in
    tests/test_embed_quantize.py).
    """
    arr = np.asarray(vec, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    # Degenerate case: all-equal vector → scale=1.0, zero_point=0, values=zeros.
    # In practice BGE never produces this, but guard anyway so the helper is
    # total over the input space.
    if vmax == vmin:
        return QuantizedVector(
            values=[0] * len(vec), scale=1.0, zero_point=0, dim=len(vec)
        )
    scale = (vmax - vmin) / 255.0
    # Map fp32 vmin → -128, fp32 vmax → 127, define zero_point so the
    # inverse (values[i] - zero_point) * scale recovers fp32.
    zero_point = int(round(-vmin / scale)) - 128
    quantized = np.round(arr / scale).astype(np.int32) + zero_point
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return QuantizedVector(
        values=[int(x) for x in quantized.tolist()],
        scale=float(scale),
        zero_point=int(zero_point),
        dim=len(vec),
    )


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
        # Read the WRITE-side quantization knob once at construction so an
        # invalid value fails loud at startup rather than later at first
        # .embed_quantized() call. Does NOT change .embed() / .embed_batch()
        # behavior — the int8 surface is exposed exclusively via
        # .embed_quantized().
        self._quantize_mode: str | None = _resolve_quantize_mode()

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

    def embed_quantized(self, text: str) -> QuantizedVector:
        """Encode ``text`` and return an int8-quantized vector + metadata.

        Always available regardless of env var — the env var gates init-time
        validation (so an invalid value fails loud at startup), not method
        availability. For ambient ergonomics, callers that opt in via
        ``IAI_MCP_EMBED_QUANTIZE=int8`` should branch on ``self._quantize_mode``.

        Storage integration: a separate future task wires int8 into the Lance
        store via a parallel column path (gated on a manual LongMemEval-S A/B
        confirming <1% recall loss). This task is the embedder surface only;
        the Lance schema is intentionally unchanged here.
        """
        fp32 = self.embed(text)
        return _quantize_int8(fp32)


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
