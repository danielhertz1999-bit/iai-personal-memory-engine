"""Grep-based static guards for constitutional invariants.

Verifies C1..C6 hold across the daemon-side module set.

Catalog:
- C3: no ANTHROPIC_API_KEY anywhere in daemon-side code.
- Pitfall 2: no fcntl.lockf (close-fd trap) anywhere in src/iai_mcp/.
- C5: no assignment to `.literal_surface` in daemon-side modules.
- no hardcoded Western clock-time in quiet_window.py.
- seal: PROFILE_KNOBS still has exactly 14 entries (daemon does NOT
  add knobs).
- C6: identity_audit.py does NOT import ProcessLock / concurrency module.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"

# Daemon-side modules. Some (bedtime, host_cli) may not exist yet (future
# plans). We scan whichever ones exist today.
DAEMON_MODULES: tuple[str, ...] = (
    "daemon.py",
    "dream.py",
    "identity_audit.py",
    "bedtime.py",
    "host_cli.py",
    "insight.py",
    "quiet_window.py",
    "daemon_state.py",
    "concurrency.py",
    "hippea_cascade.py", # / D5-05
)


def _existing_daemon_files() -> list[Path]:
    return [SRC / n for n in DAEMON_MODULES if (SRC / n).exists()]


# ---------------------------------------------------------------------------
# C3: ANTHROPIC_API_KEY must never appear in daemon-side code
# ---------------------------------------------------------------------------

def test_no_api_key_in_daemon():
    """C3 ( / ): zero paid-API cost. ANTHROPIC_API_KEY must not
    appear in ANY daemon-side module. Insight module uses `claude -p`
    subprocess with the user's subscription instead."""
    offenders: list[str] = []
    for f in _existing_daemon_files():
        text = f.read_text()
        if "ANTHROPIC_API_KEY" in text:
            offenders.append(f.name)
    assert not offenders, f"C3 violation: ANTHROPIC_API_KEY found in {offenders}"


# ---------------------------------------------------------------------------
# Pitfall 2: fcntl.lockf must never be used (POSIX close-fd trap)
# ---------------------------------------------------------------------------

def test_no_lockf_anywhere():
    """Pitfall 2 (apenwarr 2010): POSIX fcntl.lockf is released when ANY fd
    referring to the same file is closed. We must use BSD fcntl.flock which
    is bound to the open file description. Scan ALL iai_mcp/*.py, not just
    daemon modules -- mixing the two is also a bug."""
    offenders: list[str] = []
    for f in SRC.glob("*.py"):
        text = f.read_text()
        if "fcntl.lockf" in text:
            offenders.append(f.name)
    assert not offenders, f"Pitfall 2 violation: fcntl.lockf in {offenders}"


# ---------------------------------------------------------------------------
# C5: daemon must NEVER assign to record.literal_surface
# ---------------------------------------------------------------------------

def test_no_literal_surface_mutation_in_daemon():
    """C5 literal preservation. Daemon-side modules must not contain
    `.literal_surface =` assignment syntax. Reading `.literal_surface` is
    allowed; writing is forbidden."""
    pattern = re.compile(r"\.literal_surface\s*=")
    offenders: list[tuple[str, list[str]]] = []
    for f in _existing_daemon_files():
        text = f.read_text()
        matches = pattern.findall(text)
        if matches:
            offenders.append((f.name, matches))
    assert not offenders, f"C5 violation: {offenders}"


# ---------------------------------------------------------------------------
# no hardcoded Western 9-5 / clock-time in quiet_window.py
# ---------------------------------------------------------------------------

def test_no_hardcoded_clock_time_in_quiet_window():
    """D-05 global-product mandate: quiet window must be LEARNED from event
    history, never hardcoded. Flag obvious clock-time literals."""
    f = SRC / "quiet_window.py"
    if not f.exists():
        return  # module not yet created
    text = f.read_text()
    # Look for common patterns that would indicate clock-based decisions.
    forbidden = [
        r"\b22:00\b",
        r"\b02:00\b",
        r"hour\s*==\s*22\b",
        r"hour\s*==\s*2\b",
    ]
    offenders: list[str] = []
    for pat in forbidden:
        if re.search(pat, text):
            offenders.append(pat)
    assert not offenders, (
        f"D-05 violation: hardcoded clock-time patterns in quiet_window.py: {offenders}"
    )


# ---------------------------------------------------------------------------
# seal: PROFILE_KNOBS has exactly 11 entries
# (10 autistic-kernel + 1 operator wake_depth MCP-12; AUTIST-02/08/11/12 removed)
# ---------------------------------------------------------------------------

def test_profile_knobs_still_sealed():
    """11-knob registry is sealed (-02 post AUTIST-02/08/11/12 removal).
    Daemon must not add new knobs. Transient state (hebbian-rate boost during
    developmental sigma, etc.) belongs in events or .daemon-state.json,
    never in PROFILE_KNOBS."""
    from iai_mcp import profile
    assert len(profile.PROFILE_KNOBS) == 11, (
        f"PROFILE_KNOBS unseal: expected 11, got {len(profile.PROFILE_KNOBS)}"
    )


# ---------------------------------------------------------------------------
# / D5-04: profile knob names must NEVER appear in the
# session-start payload at any wake_depth. Knobs are applied server-side via
# response_decorator.apply_profile; their names must not cross the MCP wire.
# ---------------------------------------------------------------------------


def test_no_profile_knob_in_session_start_payload(tmp_path):
    """: knob names must not leak into the NEW pointer fields at
    wake_depth=minimal (<=30 raw tok design budget).

    The legacy L0 identity kernel (`_seed_l0_identity`) historically recites
    a handful of autistic-kernel defaults inline in the literal_surface
    ('literal_preservation=strong, masking_off=true, ...'). That predates
     and lives inside the user's identity record itself, not a
    decorator output — so it's scoped into the standard/deep l0 segment and
    explicitly exempt from this grep guard.

    The invariant this guard DEFENDS is: the lazy minimal payload
    (identity_pointer / brain_handle / topic_cluster_hint) MUST NOT contain
    knob names. Knobs are applied server-side by response_decorator
    (D5-04); knob names must never reach the MCP wire.
    """
    from iai_mcp import profile
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.core import _seed_l0_identity
    from iai_mcp.session import assemble_session_start
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    assignment = CommunityAssignment()

    for mode in ("minimal", "standard", "deep"):
        state = profile.default_state()
        state["wake_depth"] = mode
        payload = assemble_session_start(
            store, assignment, [], profile_state=state,
        )
        # Only scan the NEW lazy fields. Legacy l0 / l1 / l2 / rich_club
        # carry user-authored identity content and remain exempt per design.
        lazy_text = " ".join(
            [
                payload.identity_pointer,
                payload.brain_handle,
                payload.topic_cluster_hint,
            ],
        )
        for knob_name in profile.PROFILE_KNOBS:
            # wake_depth is the operator-facing knob; its echo in the
            # payload field `wake_depth` is a meta-attribute, not inline
            # knob text in the lazy pointers.
            assert knob_name not in lazy_text, (
                f" violation: knob name '{knob_name}' found in "
                f"lazy session-start payload at wake_depth={mode} "
                f"(identity_pointer/brain_handle/topic_cluster_hint)"
            )


# ---------------------------------------------------------------------------
# Pitfall 1: wake_depth=minimal payload (<=30 raw tok) is below the
# Anthropic Sonnet 4.6 cache minimum (2048 tok). Adding cache_control in
# session.py would be silently ignored — wastes a breakpoint slot. Guard
# against accidental regression.
# ---------------------------------------------------------------------------


def test_no_cache_control_in_session_assembler():
    """Pitfall 1: session.py must not set cache_control (minimal prefix
    cannot be cached on Sonnet 4.6 / Opus 4.7; standard+deep caching lives
    in the TS wrapper, not the Python assembler).
    """
    f = SRC / "session.py"
    assert f.exists(), "session.py missing"
    text = f.read_text()
    # Comments that mention "cache_control" are fine (they document the
    # pitfall). We only guard against actual code references like setattr/
    # cache_control=... — scan for the pattern with an equals sign.
    pattern = re.compile(r"cache_control\s*[:=]")
    offenders = pattern.findall(text)
    assert not offenders, (
        f"Pitfall 1 violation: cache_control assignment/kwarg in session.py: "
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# C3 + : response_decorator must be pure-local. No Anthropic
# SDK import, no ANTHROPIC_API_KEY read, no paid-API coupling.
# ---------------------------------------------------------------------------


def test_no_api_key_in_response_decorator():
    """C3 + : response_decorator.py stays local-only."""
    f = SRC / "response_decorator.py"
    assert f.exists, "response_decorator.py missing after "
    text = f.read_text()
    lower = text.lower()
    assert "anthropic" not in lower, (
        "C3 violation: response_decorator references 'anthropic'"
    )
    assert "ANTHROPIC_API_KEY" not in text, (
        "C3 violation: response_decorator references ANTHROPIC_API_KEY"
    )
    assert "import anthropic" not in lower, (
        "C3 violation: response_decorator imports anthropic SDK"
    )


# ---------------------------------------------------------------------------
# C6: identity_audit.py must not import ProcessLock
# ---------------------------------------------------------------------------

def test_identity_audit_has_no_lock_import():
    """C6: continuous audit runs even when daemon is paused. To make that
    invariant mechanical, identity_audit.py must NOT import the concurrency
    module -- the only way to accidentally take a lock is to import it."""
    f = SRC / "identity_audit.py"
    if not f.exists():
        return
    text = f.read_text()
    # No import of iai_mcp.concurrency, no `ProcessLock` symbol reference.
    assert "iai_mcp.concurrency" not in text, (
        "C6 violation: identity_audit.py imports iai_mcp.concurrency"
    )
    assert "ProcessLock" not in text, (
        "C6 violation: identity_audit.py references ProcessLock"
    )
    # Also: no `fcntl.` calls (belt-and-braces).
    assert "fcntl." not in text, (
        "C6 violation: identity_audit.py uses fcntl directly"
    )


# ---------------------------------------------------------------------------
# : HIPPEA cascade module guards
# ---------------------------------------------------------------------------

def test_no_api_key_in_hippea_cascade():
    """C3 (D5-05): HIPPEA cascade is pure-local. ANTHROPIC_API_KEY and
    `anthropic` SDK imports are forbidden in hippea_cascade.py."""
    f = SRC / "hippea_cascade.py"
    if not f.exists():
        return  # module not yet created
    text = f.read_text()
    assert "ANTHROPIC_API_KEY" not in text, (
        "C3 violation: ANTHROPIC_API_KEY in hippea_cascade.py"
    )
    assert "import anthropic" not in text, (
        "C3 violation: `import anthropic` in hippea_cascade.py"
    )
    assert "from anthropic" not in text, (
        "C3 violation: `from anthropic` in hippea_cascade.py"
    )


def test_hippea_cascade_is_read_only_against_store():
    """C6 (D5-05): cascade prefetch never mutates the store.

    Grep for store-mutating call patterns (with trailing open-paren so the
    module's own enumerated-forbidden list in the docstring does not trip
    this guard).
    """
    f = SRC / "hippea_cascade.py"
    if not f.exists():
        return
    text = f.read_text()
    forbidden_calls = [
        "store.insert(",
        "store.append_provenance(",
        "store.append_provenance_batch(",
        "store.update(",
        "store.boost_edges(",
        "store.add_contradicts_edge(",
    ]
    offenders = [p for p in forbidden_calls if p in text]
    assert not offenders, (
        f"C6 violation: hippea_cascade.py contains store mutators: {offenders}"
    )
