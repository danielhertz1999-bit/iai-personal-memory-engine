# Lilli Laws -- structural slot (no runtime hooks)

This directory is a structural placeholder for the embedded-laws module. No
runtime checks are active in the current release.

## Why this exists today

Lilli's eventual integration target is robotic embodiment. Embedded laws are
most relevant when Lilli is plugged into a real action loop -- they validate
proposed actions before motor commands leave the cognitive layer. In the
current release, Lilli does not ACT (just remembers + retrieves), so the laws
have no triggering surface yet. Reserving the slot now means the activation
drop happens at a single integration point in the future.

## The four laws (planned content, not active code)

| Tier | Identifier | Concern | Activation |
|---|---|---|---|
| L0 | hard-stop | safety-critical refusals (no shorthand for harm) | future |
| L1 | informed-consent | user authority over irreversible operations | future |
| L2 | transparency | exposing what the system is about to do | future |
| L3 | proportionality | cost vs benefit of an action | future |

## Activation plan (future robotics integration)

1. Add a `LawCheckResult` dataclass and a `evaluate(action) -> LawCheckResult`
   function under this directory.
2. Wire `evaluate` into every action-emitting code path in the (future)
   robotics action loop -- never call from cognition-only paths.
3. Set `LAWS_ACTIVE = True` once `evaluate` is implemented and the action loop
   routes through it.
4. Defense-in-depth around the activation: legal license layer, provenance
   watermarking, cultural shaming for bad uses, technical obfuscation of the
   law-check logic. No single layer is airtight -- the alignment problem is
   unsolved -- so the structural goal is to raise attacker effort, not promise
   safety.

## Until then

Treat this README as the authority for what should land here. Do NOT add
runtime hooks pre-emptively. Empty slot is the correct state.
