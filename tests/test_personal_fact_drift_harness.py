"""Contract for `bench/personal_fact_drift.py`.

The single-user harness surface: first-person phrasing, Recall@10 metric,
retention_loss_at_10 ceiling.

Honesty constraint: smoke scale runs in seconds on a real store. Tests stay
isolated from any on-disk store. The setup-gate test pins that contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _cleanup_deferred_provenance(tmp_path: Path):
    """Remove leftover deferred-provenance JSONL files written under tmp_path.

    The recall pipeline writes a .deferred-provenance.jsonl next to each
    store it touches (drained on daemon WAKE in production). In unit-test
    scope, pytest's tmp_path cleanup already removes these files, but this
    fixture documents the invariant and guards future tests that might use
    a fixed (non-tmp) store path.
    """
    yield
    for jsonl in tmp_path.rglob(".deferred-provenance.jsonl"):
        jsonl.unlink(missing_ok=True)


def test_harness_runs_end_to_end_on_smoke(tmp_path: Path) -> None:
    """Smoke run completes, writes a JSON with the expected shape."""
    from bench.personal_fact_drift import main

    store_dir = tmp_path / "bench-store"
    results_dir = tmp_path / "results"

    # main() may sys.exit() or return; catch both.
    exit_code: int | None = None
    try:
        exit_code = main([
            "--scale=smoke",
            f"--store-dir={store_dir}",
            "--seeds", "13", "42", "137",
            f"--output-dir={results_dir}",
        ])
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0

    # Smoke scale is shape-only — accept either pass (0) or data-miss (1).
    # Setup error (2) is a real bug. Other codes are unexpected.
    assert exit_code in (0, 1), f"smoke exit_code={exit_code!r}; expected 0 or 1"

    json_files = list(results_dir.glob("personal_fact_drift_*.json"))
    assert json_files, f"no JSON output written under {results_dir}"

    with json_files[0].open() as fh:
        data = json.load(fh)

    assert "env" in data, "missing top-level 'env'"
    assert "summary" in data, "missing top-level 'summary'"
    summary = data["summary"]
    assert "recall_at_10" in summary, "summary missing recall_at_10"
    assert "retention_loss_at_10" in summary, "summary missing retention_loss_at_10"
    assert "per_probe" in summary, "summary missing per_probe"

    r10 = summary["recall_at_10"]
    loss = summary["retention_loss_at_10"]
    assert isinstance(r10, (int, float)), f"recall_at_10 type={type(r10).__name__}"
    assert isinstance(loss, (int, float)), f"retention_loss_at_10 type={type(loss).__name__}"
    assert 0.0 <= float(r10) <= 1.0, f"recall_at_10={r10} out of [0,1]"
    # retention_loss can be negative if post-recall > pre-recall (improbable but
    # legal — it would only happen if intervention helps retrieval).
    assert -1.0 <= float(loss) <= 1.0, f"retention_loss_at_10={loss} out of [-1,1]"


def test_harness_refuses_production_store(tmp_path: Path) -> None:
    """Setup-gate: --store-dir resolving to the production store must exit 2."""
    import bench.personal_fact_drift as _bench

    # Pass the harness's OWN production-store constant so the gate compares it
    # against itself -> always matches -> exit 2, independent of HOME-redirect
    # timing (the module-import-cached constant otherwise diverges from a
    # per-test Path.home() recomputation). The gate exits before any store open,
    # so this never touches the real store.
    exit_code: int | None = None
    try:
        exit_code = _bench.main([
            "--scale=smoke",
            f"--store-dir={_bench.PRODUCTION_STORE}",
            "--seeds", "13", "42", "137",
            f"--output-dir={tmp_path / 'results'}",
        ])
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0

    assert exit_code == 2, f"production-store gate did not fire; exit={exit_code!r}"


def test_corpus_generation_is_deterministic() -> None:
    """generate_fact_corpus(seed=13) returns the same tuple of lists on two calls."""
    from bench.personal_fact_drift import generate_fact_corpus

    facts_a, probes_a = generate_fact_corpus(seed=13, n_facts=20, n_probes=10)
    facts_b, probes_b = generate_fact_corpus(seed=13, n_facts=20, n_probes=10)

    assert facts_a == facts_b, "fact list differs between runs at seed=13"
    assert probes_a == probes_b, "probe list differs between runs at seed=13"
    assert len(facts_a) == 20, f"expected 20 facts, got {len(facts_a)}"
    assert len(probes_a) == 10, f"expected 10 probes, got {len(probes_a)}"


def test_harness_smoke_scale_uses_tiny_corpus() -> None:
    """Smoke preset MUST be tiny so the smoke test runs in seconds."""
    from bench.personal_fact_drift import SCALE_PRESETS

    assert "smoke" in SCALE_PRESETS, "SCALE_PRESETS missing 'smoke' key"
    preset = SCALE_PRESETS["smoke"]
    # preset = (n_facts, n_probes, n_intervening_sessions, n_chatter_turns)
    assert len(preset) == 4, f"smoke preset shape={preset}; expected 4-tuple"
    n_facts, n_probes, n_inter, n_chatter = preset
    assert n_facts <= 10, f"smoke n_facts={n_facts}; must be <= 10"
    assert n_inter <= 2, f"smoke n_intervening_sessions={n_inter}; must be <= 2"


def test_recall_at_10_metric_math() -> None:
    """8/10 probes hitting top-10 → Recall@10 = 0.8 exactly."""
    from bench.personal_fact_drift import _compute_recall_at_10

    probe_results = [
        {"recall_at_10_post": True, "recall_at_10_pre": True, "probe_id": f"p{i}"}
        for i in range(8)
    ] + [
        {"recall_at_10_post": False, "recall_at_10_pre": True, "probe_id": f"p{i}"}
        for i in range(8, 10)
    ]
    r10 = _compute_recall_at_10(probe_results)
    assert r10 == 0.8, f"expected 0.8, got {r10!r}"


def test_retention_loss_at_10_metric_math() -> None:
    """retention_loss = pre_recall - post_recall.

    Setup: 10/10 pre-hit, 7/10 post-hit ⇒ retention_loss = 1.0 - 0.7 = 0.3.
    """
    from bench.personal_fact_drift import _compute_retention_loss_at_10

    probe_results = [
        {"recall_at_10_post": True, "recall_at_10_pre": True, "probe_id": f"p{i}"}
        for i in range(7)
    ] + [
        {"recall_at_10_post": False, "recall_at_10_pre": True, "probe_id": f"p{i}"}
        for i in range(7, 10)
    ]
    loss = _compute_retention_loss_at_10(probe_results)
    assert loss == pytest.approx(0.3), f"expected 0.3, got {loss!r}"


# ---------------------------------------------------------------------------
# single-user contract
# ---------------------------------------------------------------------------


def test_corpus_has_no_user_id_in_text() -> None:
    """Single-user reality: corpus text must NOT contain `User-` identifiers."""
    from bench.personal_fact_drift import generate_fact_corpus

    facts, probes = generate_fact_corpus(seed=13, n_facts=50, n_probes=20)

    for fact in facts:
        assert "User-" not in fact.text, (
            f"fact.text leaks multi-user identifier: {fact.text!r}"
        )
        # PersonalFact dataclass exposes its declared fields. Confirm there
        # is no `user_id` field on the structure.
        assert not hasattr(fact, "user_id"), (
            "PersonalFact still carries a user_id field — single-user reality "
            "forbids it"
        )

    for probe in probes:
        assert "User-" not in probe.text, (
            f"probe leaks multi-user identifier in fact.text: {probe.text!r}"
        )
        assert "User-" not in probe.probe, (
            f"probe phrasing leaks multi-user identifier: {probe.probe!r}"
        )


def test_probe_phrasing_is_first_person() -> None:
    """Every probe must be a first-person operator question.

    Starts with a Wh-question word; contains one of " I ", " my ", " me "
    (whitespace-bounded so we don't get fooled by "Italy", "myth", etc.).
    """
    from bench.personal_fact_drift import generate_fact_corpus

    facts, probes = generate_fact_corpus(seed=13, n_facts=50, n_probes=20)
    wh_words = ("what", "where", "when", "which", "how", "who")
    first_person_markers = (" i ", " my ", " me ")

    for probe in probes:
        text = probe.probe
        lower = text.lower()
        first_word = lower.split()[0] if lower.split() else ""
        # Strip a trailing comma/colon/etc from the leading wh-word.
        first_word_clean = first_word.rstrip("'?.,:;!")
        assert first_word_clean in wh_words, (
            f"probe does not start with a Wh-word: {text!r}"
        )
        # Pad to allow boundary checks at start/end.
        padded = f" {lower} "
        assert any(marker in padded for marker in first_person_markers), (
            f"probe lacks first-person marker (I/my/me): {text!r}"
        )


def test_metric_name_is_recall_at_10() -> None:
    """Summary uses recall_at_10 + retention_loss_at_10 keys (not the old names)."""
    from bench.personal_fact_drift import aggregate, ProbeOutcome

    # Build a minimal outcomes dict.
    sample = ProbeOutcome(
        probe_id="p0",
        seed=13,
        cue="What color do I prefer?",
        expects="My favorite color is teal.",
        category="preference",
        attribute="preference",
        recall_at_10_pre=True,
        recall_at_10_post=True,
        top1_pre="My favorite color is teal.",
        top1_post="My favorite color is teal.",
        top1_changed=False,
    )
    summary = aggregate({13: [sample]})

    assert "recall_at_10" in summary, "summary missing recall_at_10"
    assert "retention_loss_at_10" in summary, "summary missing retention_loss_at_10"
    # Old keys must be gone — leaving them in confuses downstream CI parsers.
    assert "precision_at_10" not in summary, (
        "summary still carries deprecated precision_at_10 key"
    )
    assert "drift" not in summary, "summary still carries deprecated drift key"

    gate = summary.get("ship_gate", {})
    assert "recall_at_10_threshold" in gate, (
        "ship_gate missing recall_at_10_threshold"
    )
    assert "retention_loss_ceiling" in gate, (
        "ship_gate missing retention_loss_ceiling"
    )


# ---------------------------------------------------------------------------
# FACT_SPECS unique-probe contract
# ---------------------------------------------------------------------------


def test_each_fact_has_unique_probe_one_to_one() -> None:
    """Every generated fact's probe is unique across the corpus.

    With a shared-template design, 500 facts would share 15 distinct probe
    strings → recall@10 bounded by ~10/33 = 0.30. The corrected design makes
    each fact a distinct semantic row with its own probe.
    """
    from bench.personal_fact_drift import generate_fact_corpus

    facts, _ = generate_fact_corpus(seed=13, n_facts=50, n_probes=50)
    probes = [f.probe for f in facts]
    assert len(set(probes)) == len(probes), (
        f"probe collisions detected: {len(probes)} probes, "
        f"{len(set(probes))} unique. Sample dupes: "
        f"{[p for p in probes if probes.count(p) > 1][:5]}"
    )


def test_no_template_placeholders_in_facts() -> None:
    """Every fact text must be a concrete sentence with no `{v}/{u}/{p}` placeholders.

    Catches future regressions where template-instantiation leaks unformatted
    placeholders into the corpus.
    """
    from bench.personal_fact_drift import generate_fact_corpus

    facts, _ = generate_fact_corpus(seed=13, n_facts=50, n_probes=50)
    for fact in facts:
        for placeholder in ("{v}", "{u}", "{p}"):
            assert placeholder not in fact.text, (
                f"fact text contains placeholder {placeholder!r}: {fact.text!r}"
            )
            assert placeholder not in fact.probe, (
                f"probe contains placeholder {placeholder!r}: {fact.probe!r}"
            )


def test_honest_scale_n_facts_is_50() -> None:
    """SCALE_PRESETS['honest'] must declare 50 facts (was 500 in 19-03)."""
    from bench.personal_fact_drift import SCALE_PRESETS

    assert "honest" in SCALE_PRESETS, "SCALE_PRESETS missing 'honest' key"
    preset = SCALE_PRESETS["honest"]
    n_facts = preset[0]
    assert n_facts == 50, (
        f"honest scale n_facts={n_facts}; expected 50 (1:1 with FACT_SPECS)"
    )


def test_fact_specs_list_has_50_distinct_facts() -> None:
    """FACT_SPECS must be a list of 50 dicts with distinct fact texts."""
    from bench.personal_fact_drift import FACT_SPECS

    assert isinstance(FACT_SPECS, list), "FACT_SPECS must be a list"
    assert len(FACT_SPECS) == 50, (
        f"FACT_SPECS has {len(FACT_SPECS)} entries; expected 50"
    )
    texts = [spec["text"] for spec in FACT_SPECS]
    assert len(set(texts)) == 50, (
        f"FACT_SPECS has duplicate fact texts: "
        f"{len(set(texts))} unique of {len(texts)} total"
    )
    probes = [spec["probe"] for spec in FACT_SPECS]
    assert len(set(probes)) == 50, (
        f"FACT_SPECS has duplicate probes: "
        f"{len(set(probes))} unique of {len(probes)} total"
    )


def test_fact_specs_cover_three_categories() -> None:
    """FACT_SPECS spans preference / project / constraint with roughly even counts."""
    from bench.personal_fact_drift import FACT_SPECS

    categories = [spec["category"] for spec in FACT_SPECS]
    expected_cats = {"preference", "project", "constraint"}
    assert set(categories) == expected_cats, (
        f"FACT_SPECS categories={set(categories)}; expected {expected_cats}"
    )
    # Roughly balanced: ≥14 per category (out of 50 with 3 categories).
    from collections import Counter

    counts = Counter(categories)
    for cat in expected_cats:
        assert counts[cat] >= 14, (
            f"category {cat!r} has only {counts[cat]} facts; expected >=14"
        )
