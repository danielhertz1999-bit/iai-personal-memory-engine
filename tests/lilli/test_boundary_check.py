"""Tests for the forbidden-import boundary enforcement hook in tests/lilli/conftest.py.

Tests exercise scan_for_forbidden_imports() directly so that the hook
itself is not triggered by the test's own imports (the test does not
import any forbidden modules; it writes temp files instead).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.lilli.conftest import ForbiddenImport, scan_for_forbidden_imports


# ---------------------------------------------------------------------------
# Positive tests — clean files should produce zero hits
# ---------------------------------------------------------------------------

class TestScanCleanFiles:
    """Files with only permitted imports produce no violations."""

    def test_lilli_only_imports(self, tmp_path: Path) -> None:
        """Imports from iai_mcp.lilli.* are permitted."""
        src = tmp_path / "clean.py"
        src.write_text(
            "from iai_mcp.lilli.brain import Brain\n"
            "from iai_mcp.lilli.tiers.bsc import BSCTier\n"
            "import iai_mcp.lilli.core.seed\n",
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

    def test_whitelisted_cross_tier_imports(self, tmp_path: Path) -> None:
        """Permitted cross-tier modules (events, store, types, hippo, tem) are allowed."""
        src = tmp_path / "cross_tier.py"
        src.write_text(
            "from iai_mcp.events import write_event\n"
            "from iai_mcp.store import MemoryStore\n"
            "from iai_mcp.types import MemoryRecord\n"
            "from iai_mcp.hippo import HippoDB\n"
            "from iai_mcp.tem import TEM\n"
            "from iai_mcp.migrate import run_migration\n",
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

    def test_stdlib_and_external_only(self, tmp_path: Path) -> None:
        """Pure stdlib + external imports produce no violations."""
        src = tmp_path / "stdlib.py"
        src.write_text(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import torch\n",
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file produces no violations."""
        src = tmp_path / "empty.py"
        src.write_text("", encoding="utf-8")
        assert scan_for_forbidden_imports(src) == []

    def test_comment_containing_forbidden_name(self, tmp_path: Path) -> None:
        """A comment mentioning a forbidden module name is not flagged."""
        src = tmp_path / "with_comment.py"
        src.write_text(
            "# This module never imports iai_mcp.daemon\n"
            "# iai_mcp.lifecycle is forbidden in lilli tests\n"
            "from iai_mcp.lilli.brain import Brain\n",
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

    def test_string_literal_with_forbidden_name(self, tmp_path: Path) -> None:
        """A string literal containing a forbidden module name is not flagged."""
        src = tmp_path / "strings.py"
        src.write_text(
            'FORBIDDEN = "iai_mcp.daemon"\n'
            'DOC = "Do not import iai_mcp.lifecycle."\n',
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []


# ---------------------------------------------------------------------------
# Negative tests — forbidden imports must be detected
# ---------------------------------------------------------------------------

class TestScanForbiddenImports:
    """Files importing forbidden infrastructure modules must be flagged."""

    def test_direct_daemon_import(self, tmp_path: Path) -> None:
        """'import iai_mcp.daemon' is forbidden."""
        src = tmp_path / "bad.py"
        src.write_text("import iai_mcp.daemon\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.daemon"
        assert hits[0].line_number == 1

    def test_from_daemon_import(self, tmp_path: Path) -> None:
        """'from iai_mcp.daemon import...' is forbidden."""
        src = tmp_path / "bad.py"
        src.write_text("from iai_mcp.daemon import Daemon\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.daemon"

    def test_lifecycle_module(self, tmp_path: Path) -> None:
        """'from iai_mcp.lifecycle import...' is forbidden."""
        src = tmp_path / "bad.py"
        src.write_text(
            "from iai_mcp.lilli.brain import Brain\n"
            "from iai_mcp.lifecycle import LifecycleState\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.lifecycle"
        assert hits[0].line_number == 2

    def test_lifecycle_submodule(self, tmp_path: Path) -> None:
        """'from iai_mcp.lifecycle_state import...' is forbidden."""
        src = tmp_path / "bad.py"
        src.write_text(
            "from iai_mcp.lifecycle_state import load_state\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.lifecycle_state"

    def test_s2_coordinator(self, tmp_path: Path) -> None:
        """'from iai_mcp.s2_coordinator import...' is forbidden."""
        src = tmp_path / "bad.py"
        src.write_text(
            "from iai_mcp.s2_coordinator import S2Coordinator\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.s2_coordinator"

    def test_socket_server(self, tmp_path: Path) -> None:
        """'import iai_mcp.socket_server' is forbidden."""
        src = tmp_path / "bad.py"
        src.write_text("import iai_mcp.socket_server\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.socket_server"

    def test_returns_first_and_only_violation_in_one_line_file(
        self, tmp_path: Path
    ) -> None:
        """Multiple forbidden imports in one file returns ALL violations."""
        src = tmp_path / "multi.py"
        src.write_text(
            "from iai_mcp.daemon import Daemon\n"
            "from iai_mcp.s2_coordinator import S2Coordinator\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 2
        modules = {h.module for h in hits}
        assert "iai_mcp.daemon" in modules
        assert "iai_mcp.s2_coordinator" in modules

    def test_forbidden_import_returns_forbidden_import_namedtuple(
        self, tmp_path: Path
    ) -> None:
        """Return type is ForbiddenImport with expected fields."""
        src = tmp_path / "typed.py"
        src.write_text("from iai_mcp.daemon_config import DaemonConfig\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, ForbiddenImport)
        assert hit.path == src
        assert hit.line_number == 1
        assert hit.module == "iai_mcp.daemon_config"
        assert "daemon_config" in hit.line_text

    def test_indented_import_is_detected(self, tmp_path: Path) -> None:
        """Indented imports (e.g. inside if block) are also detected."""
        src = tmp_path / "indented.py"
        src.write_text(
            "if True:\n"
            "    from iai_mcp.daemon import Daemon\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.daemon"
        assert hits[0].line_number == 2


# ---------------------------------------------------------------------------
# Hook integration test — proves pytest_collection_modifyitems fires
# ---------------------------------------------------------------------------

class TestHookIntegration:
    """Prove that pytest_collection_modifyitems raises UsageError when a file
    inside tests/lilli/ contains a forbidden import.

    We call the hook directly with a synthetic pytest.Item whose
    module.__file__ resolves to a real path inside tests/lilli/ (a temporary
    file written there for the duration of the test).
    """

    def _make_fake_item(self, source_path: Path) -> pytest.Item:
        """Return a minimal object that satisfies the hook's attribute reads."""

        class FakeModule:
            __file__ = str(source_path)

        class FakeItem:
            module = FakeModule()

        return FakeItem()  # type: ignore[return-value]

    def test_hook_raises_usage_error_on_forbidden_import(
        self, tmp_path: Path
    ) -> None:
        """Hook fires UsageError when a lilli-local file imports daemon."""
        from tests.lilli.conftest import pytest_collection_modifyitems

        # Write a file with a forbidden import INSIDE tests/lilli/ so the
        # hook's relative_to() filter accepts it.
        lilli_dir = Path(__file__).resolve().parent
        bad_module_path = lilli_dir / "_test_hook_temp_violation.py"
        try:
            bad_module_path.write_text(
                "from iai_mcp.daemon import Daemon\n", encoding="utf-8"
            )
            fake_item = self._make_fake_item(bad_module_path)

            with pytest.raises(pytest.UsageError, match="forbidden import"):
                pytest_collection_modifyitems(config=None, items=[fake_item])
        finally:
            if bad_module_path.exists():
                bad_module_path.unlink()

    def test_hook_is_silent_for_clean_lilli_file(self, tmp_path: Path) -> None:
        """Hook does NOT raise when a lilli-local file has only clean imports."""
        from tests.lilli.conftest import pytest_collection_modifyitems

        lilli_dir = Path(__file__).resolve().parent
        clean_module_path = lilli_dir / "_test_hook_temp_clean.py"
        try:
            clean_module_path.write_text(
                "from iai_mcp.lilli.brain import Brain\n"
                "from iai_mcp.events import write_event\n",
                encoding="utf-8",
            )
            fake_item = self._make_fake_item(clean_module_path)
            # Must not raise
            pytest_collection_modifyitems(config=None, items=[fake_item])
        finally:
            if clean_module_path.exists():
                clean_module_path.unlink()

    def test_hook_ignores_files_outside_lilli_dir(self, tmp_path: Path) -> None:
        """Hook must NOT raise for files outside tests/lilli/ even with forbidden imports."""
        from tests.lilli.conftest import pytest_collection_modifyitems

        # File is in tmp_path, which is outside tests/lilli/
        outside_file = tmp_path / "other_test.py"
        outside_file.write_text(
            "from iai_mcp.daemon import Daemon\n", encoding="utf-8"
        )
        fake_item = self._make_fake_item(outside_file)
        # Must not raise — hook only applies to tests/lilli/ files
        pytest_collection_modifyitems(config=None, items=[fake_item])
