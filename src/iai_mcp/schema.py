"""Back-compat shim -- schema induction has moved to iai_mcp.lilli.cycle.schema.

Re-exports the public API so callers that have not yet migrated keep working.
"""
from __future__ import annotations

from iai_mcp.lilli.cycle.schema import (
    AUTO_INDUCT_COOCCURRENCE,
    AUTO_INDUCT_CONFIDENCE,
    MAX_EVIDENCE_PER_SCHEMA,
    PROVISIONAL_ENTROPY_MIN,
    USER_APPROVAL_COOCCURRENCE,
    USER_APPROVAL_CONFIDENCE,
    SchemaCandidate,
    induce_schemas_tier0,
    induce_schemas_tier1,
    persist_schema,
    provisional_schemas_for_recall,
)

__all__ = [
    "induce_schemas_tier0", "induce_schemas_tier1", "persist_schema",
    "provisional_schemas_for_recall", "SchemaCandidate",
    "AUTO_INDUCT_COOCCURRENCE", "AUTO_INDUCT_CONFIDENCE",
    "USER_APPROVAL_COOCCURRENCE", "USER_APPROVAL_CONFIDENCE",
    "MAX_EVIDENCE_PER_SCHEMA", "PROVISIONAL_ENTROPY_MIN",
]
