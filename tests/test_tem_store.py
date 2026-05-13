""" RED: structure_hv field schema validation on MemoryRecord.

MemoryRecord.structure_hv: bytes is the renamed Phase-2 hd_vector slot. It
must accept empty bytes (pre-migration sentinel) OR exactly STRUCTURE_HV_BYTES
(1250 bytes, D=10000 BSC packed) -- anything else is a constitutional schema
violation that __post_init__ rejects.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


def _kwargs(**override):
    """Build a minimal valid MemoryRecord kwargs dict; tests override fields."""
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


# ---------------------------------------------------------------- field default


def test_structure_hv_defaults_to_empty_bytes() -> None:
    """Default value is b"" (pre-migration sentinel)."""
    from iai_mcp.types import MemoryRecord

    rec = MemoryRecord(**_kwargs())
    assert rec.structure_hv == b""
    assert isinstance(rec.structure_hv, bytes)


def test_structure_hv_accepts_exact_length() -> None:
    """Exactly STRUCTURE_HV_BYTES (1250) bytes must be accepted."""
    from iai_mcp.types import MemoryRecord, STRUCTURE_HV_BYTES

    payload = bytes(STRUCTURE_HV_BYTES)  # all-zero sentinel; right shape
    rec = MemoryRecord(**_kwargs(structure_hv=payload))
    assert rec.structure_hv == payload
    assert len(rec.structure_hv) == STRUCTURE_HV_BYTES


def test_structure_hv_rejects_wrong_length() -> None:
    """Anything that is not empty AND not STRUCTURE_HV_BYTES bytes raises."""
    from iai_mcp.types import MemoryRecord

    with pytest.raises(ValueError, match=r"structure_hv must be empty"):
        MemoryRecord(**_kwargs(structure_hv=b"too short"))

    with pytest.raises(ValueError, match=r"structure_hv must be empty"):
        MemoryRecord(**_kwargs(structure_hv=b"x" * 999))


def test_structure_hv_rejects_non_bytes() -> None:
    """Non-bytes input (list/str/None) is rejected at the type boundary."""
    from iai_mcp.types import MemoryRecord

    with pytest.raises(ValueError, match=r"structure_hv must be bytes"):
        MemoryRecord(**_kwargs(structure_hv=[1, 0, 1]))

    with pytest.raises(ValueError, match=r"structure_hv must be bytes"):
        MemoryRecord(**_kwargs(structure_hv="not bytes"))


def test_module_constants_match_canonical_dims() -> None:
    """STRUCTURE_HV_DIM=10000 (D-TEM-01); STRUCTURE_HV_BYTES=1250 (D/8)."""
    from iai_mcp.types import STRUCTURE_HV_BYTES, STRUCTURE_HV_DIM

    assert STRUCTURE_HV_DIM == 10000
    assert STRUCTURE_HV_BYTES == 1250
    assert STRUCTURE_HV_DIM // 8 == STRUCTURE_HV_BYTES


def test_schema_version_v4_accepted() -> None:
    """schema_version=4 (marker) must be accepted alongside 1/2/3."""
    from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_V4

    rec = MemoryRecord(**_kwargs(schema_version=SCHEMA_VERSION_V4))
    assert rec.schema_version == 4
