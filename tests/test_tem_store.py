from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

def _kwargs(**override):
    from iai_mcp.types import EMBED_DIM

    base = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface="hello",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    base.update(override)
    return base

def test_structure_hv_defaults_to_empty_bytes() -> None:
    from iai_mcp.types import MemoryRecord

    rec = MemoryRecord(**_kwargs())
    assert rec.structure_hv == b""
    assert isinstance(rec.structure_hv, bytes)

def test_structure_hv_accepts_exact_length() -> None:
    from iai_mcp.types import MemoryRecord, STRUCTURE_HV_BYTES

    payload = bytes(STRUCTURE_HV_BYTES)
    rec = MemoryRecord(**_kwargs(structure_hv=payload))
    assert rec.structure_hv == payload
    assert len(rec.structure_hv) == STRUCTURE_HV_BYTES

def test_structure_hv_rejects_wrong_length() -> None:
    from iai_mcp.types import MemoryRecord

    with pytest.raises(ValueError, match=r"structure_hv must be empty"):
        MemoryRecord(**_kwargs(structure_hv=b"too short"))

    with pytest.raises(ValueError, match=r"structure_hv must be empty"):
        MemoryRecord(**_kwargs(structure_hv=b"x" * 999))

def test_structure_hv_rejects_non_bytes() -> None:
    from iai_mcp.types import MemoryRecord

    with pytest.raises(ValueError, match=r"structure_hv must be bytes"):
        MemoryRecord(**_kwargs(structure_hv=[1, 0, 1]))

    with pytest.raises(ValueError, match=r"structure_hv must be bytes"):
        MemoryRecord(**_kwargs(structure_hv="not bytes"))

def test_module_constants_match_canonical_dims() -> None:
    from iai_mcp.types import STRUCTURE_HV_BYTES, STRUCTURE_HV_DIM

    assert STRUCTURE_HV_DIM == 10000
    assert STRUCTURE_HV_BYTES == 1250
    assert STRUCTURE_HV_DIM // 8 == STRUCTURE_HV_BYTES

def test_schema_version_v4_accepted() -> None:
    from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_V4

    rec = MemoryRecord(**_kwargs(schema_version=SCHEMA_VERSION_V4))
    assert rec.schema_version == 4
