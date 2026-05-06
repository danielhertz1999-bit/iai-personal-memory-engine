"""D-STORAGE events table interface.

Single source of runtime state. Every kind of event — S4 contradictions,
trajectory metrics, LLM health probes, schema induction runs, CLS consolidation
runs, migration traces, alerts — goes through write_event.

No .jsonl files. No .json files scattered under internal storage or
internal storage. Everything persists in the LanceDB `events` table.

CLI queries (iai-mcp health, iai-mcp trajectory) read via query_events.

events.data_json is AES-256-GCM encrypted at rest (some event
payloads carry user quotes / cues -- safest default). The event UUID is the
associated data binding. kind / severity / domain / ts / session_id stay
plaintext so audit queries (`iai-mcp health`, `iai-mcp trajectory`) can filter
on them without decrypting.

Phase 3 additions (new event kinds — free-form strings, no taxonomy enum):
- CONN-05 TEM factorization: `migration_v3_to_v4`.
- CONN-07 small-world sigma: `sigma_observation`, `sigma_drift`
  (sigma-curve diagnostic per Ashby ultrastability).
- M2/M4/M6 live wiring: `retrieval_used`, `profile_updated`,
  `session_started` (existing emit sites extended; not all new — verify via
  ctx_search before emitting duplicates).
- Chapman ecological self-regulation:
    * `formality_score_weekly` — per-turn aggregate of user SURFACE formality.
    * `camouflaging_detected` — over-formal trajectory detected over 5-point weekly window.
    * `register_relaxed` — OUR `camouflaging_relaxation` knob bumped; the system
      relaxes its OWN register (never the user's; masking modeling is out-of-scope).

Phase 6 additions (Plan 06-01 schema dedup):
- `schema_reinforced` — emitted when `persist_schema` finds an existing
  schema for the candidate pattern and reinforces incoming
  `schema_instance_of` edges from new evidence onto the existing keeper
  instead of inserting a duplicate row. Payload:
    {schema_id: str, pattern: str, evidence_added: int, total_evidence: int}
  Source IDs: [keeper_schema_id, *new_evidence_ids[:5]] mirroring the
  existing `schema_induction_run` shape.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from iai_mcp.crypto import (
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.store import EVENTS_TABLE, MemoryStore


def write_event(
    store: MemoryStore,
    kind: str,
    data: dict[str, Any],
    *,
    severity: str | None = None,
    domain: str | None = None,
    session_id: str = "-",
    source_ids: list[UUID] | None = None,
) -> UUID:
    """Persist a single event to the LanceDB events table.

    Parameters
    ----------
    store:
        Open MemoryStore instance.
    kind:
        Logical event kind (e.g. "s4_contradiction", "trajectory_metric",
        "llm_health", "migration_v1_to_v2"). Free-form string; downstream
        consumers filter on it.
    data:
        JSON-serialisable kind-specific payload. Encoded to data_json.
    severity:
        Optional alert severity ("info" | "warning" | "critical"). Stored
        as empty string for non-alert events.
    domain:
        Optional monotropic-domain tag. Stored as empty string when absent.
    session_id:
        Session identifier; defaults to "-" when no session is active.
    source_ids:
        Optional list of MemoryRecord UUIDs that triggered this event.

    Returns the newly-minted event UUID.
    """
    event_id = uuid4()
    # encrypt data_json with AD = event UUID bytes. kind / severity /
    # domain / ts / session_id stay plaintext for filter queries.
    data_plain = json.dumps(data)
    ad = str(event_id).encode("ascii")
    data_ct = encrypt_field(data_plain, store._key(), associated_data=ad)
    row = {
        "id": str(event_id),
        "kind": kind,
        "severity": severity or "",
        "domain": domain or "",
        "ts": datetime.now(timezone.utc),
        "data_json": data_ct,
        "session_id": session_id,
        "source_ids_json": json.dumps([str(x) for x in (source_ids or [])]),
    }
    store.db.open_table(EVENTS_TABLE).add([row])
    return event_id


def query_events(
    store: MemoryStore,
    kind: str | None = None,
    since: datetime | None = None,
    severity: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query events matching the given filters, newest first.

    Parameters
    ----------
    store:
        Open MemoryStore instance.
    kind:
        Filter by event kind. None returns all kinds.
    since:
        Only return events with ts >= since. Naive datetimes are treated as UTC.
    severity:
        Exact-match filter on severity field.
    limit:
        Maximum rows returned (default 100). Caller can pass e.g. 1 to get
        only the most recent event of a given kind (iai-mcp health).

    Returns a list of dicts with keys: id, kind, severity, domain, ts, data,
    session_id, source_ids. data and source_ids are decoded from JSON.
    """
    tbl = store.db.open_table(EVENTS_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return []
    if kind is not None:
        df = df[df["kind"] == kind]
    if severity is not None:
        df = df[df["severity"] == severity]
    if since is not None:
        # Ensure tz-aware comparison
        since_cmp = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        # Pandas Timestamp compares naturally with tz-aware datetimes
        df = df[df["ts"] >= since_cmp]
    if df.empty:
        return []
    df = df.sort_values("ts", ascending=False).head(limit)
    out: list[dict] = []
    for _, row in df.iterrows():
        # decrypt data_json when it carries the iai:enc:v1: prefix.
        # Pre-02-08 rows stay plaintext; migration rewrites them lazily.
        raw_data = row["data_json"] or "{}"
        if is_encrypted(raw_data):
            ad = str(row["id"]).encode("ascii")
            try:
                raw_data = decrypt_field(raw_data, store._key(), associated_data=ad)
            except Exception:
                # Rule 1 diagnostic semantics: a corrupt event row should not
                # fail the entire query. Return empty payload + mark in meta.
                raw_data = "{}"
        try:
            data = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            data = {}
        try:
            source_ids = json.loads(row["source_ids_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            source_ids = []
        out.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "severity": row["severity"] or None,
                "domain": row["domain"] or None,
                "ts": row["ts"],
                "data": data,
                "session_id": row["session_id"],
                "source_ids": source_ids,
            }
        )
    return out
