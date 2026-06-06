"""Tests for iai_mcp.embed -- bge-small-en-v1.5 path (legacy model).

  made bge-m3 the default. The 3-model registry still exposes
bge-small-en-v1.5 (384d, English-only) for English-only deployments. These
tests exercise the model explicitly via `Embedder(model_key=...)` so
they remain valid regression gates.

Multilingual behaviour is covered by tests/test_embed_multilingual.py.
"""
from __future__ import annotations

import threading

import pytest

from iai_mcp.embed import Embedder


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


def test_embed_returns_384_dim_vector() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    v = emb.embed("hello world")
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)


def test_embed_is_deterministic() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    a = emb.embed("exact same text")
    b = emb.embed("exact same text")
    assert a == b


def test_embed_batch_preserves_order_and_dim() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    texts = ["one", "two", "three"]
    vecs = emb.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)
    # Batch must equal sequential calls (determinism across batching path too).
    assert vecs[0] == emb.embed("one")


def test_embed_empty_string_still_returns_384d() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    v = emb.embed("")
    assert len(v) == 384


def test_embedder_dim_matches_output() -> None:
    emb = Embedder(model_key="bge-small-en-v1.5")
    assert emb.DIM == 384
    v = emb.embed("anything")
    assert len(v) == emb.DIM


def test_bge_small_en_still_registered_for_legacy() -> None:
    """keeps the model in the registry for English-only deployments."""
    from iai_mcp.embed import MODEL_REGISTRY

    assert "bge-small-en-v1.5" in MODEL_REGISTRY
    assert MODEL_REGISTRY["bge-small-en-v1.5"]["dim"] == 384


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_concurrent_encode_byte_identical_to_serial() -> None:
    """Two threads encoding on ONE shared Embedder produce byte-identical
    output to serial encoding.

    The held singleton serves concurrent in-process recalls that fan out via
    ``asyncio.to_thread``; this empirically closes the shared-singleton
    concurrency claim (encode releases the GIL and takes ``&self`` with no
    interior mutability, so concurrent encodes are sound). Native-skip ONLY
    if the Rust extension is genuinely absent.
    """
    emb = Embedder(model_key="bge-small-en-v1.5")
    cues = [
        "the hippocampus stays awake",
        "consolidation runs at night",
        "embedding is a pure deterministic function",
        "two threads share one warm instance",
        "byte-identical output across re-encode",
    ]

    # Serial baseline.
    serial = {cue: emb.embed(cue) for cue in cues}

    # Concurrent: two worker threads share the ONE instance, each encoding the
    # full cue set repeatedly to maximize overlap on the shared encode path.
    concurrent: dict[str, list[float]] = {}
    lock = threading.Lock()
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            for _ in range(3):
                for cue in cues:
                    v = emb.embed(cue)
                    with lock:
                        concurrent[cue] = v
        except BaseException as exc:  # noqa: BLE001 -- surface to the assertion
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent encode raised: {errors}"
    for cue in cues:
        assert concurrent[cue] == serial[cue], (
            f"concurrent encode diverged from serial for cue: {cue!r}"
        )
