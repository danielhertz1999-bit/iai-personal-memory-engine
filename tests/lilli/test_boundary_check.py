from __future__ import annotations

from pathlib import Path

import pytest

from tests.lilli.conftest import ForbiddenImport, scan_for_forbidden_imports

class TestScanCleanFiles:

    def test_lilli_only_imports(self, tmp_path: Path) -> None:
        src = tmp_path / "clean.py"
        src.write_text(
            "from iai_mcp.lilli.brain import Brain\n"
            "from iai_mcp.lilli.tiers.bsc import BSCTier\n"
            "import iai_mcp.lilli.core.seed\n",
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

    def test_whitelisted_cross_tier_imports(self, tmp_path: Path) -> None:
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
        src = tmp_path / "empty.py"
        src.write_text("", encoding="utf-8")
        assert scan_for_forbidden_imports(src) == []

    def test_comment_containing_forbidden_name(self, tmp_path: Path) -> None:
        src = tmp_path / "with_comment.py"
        src.write_text(
            "# This module never imports iai_mcp.daemon\n"
            "# iai_mcp.lifecycle is forbidden in lilli tests\n"
            "from iai_mcp.lilli.brain import Brain\n",
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

    def test_string_literal_with_forbidden_name(self, tmp_path: Path) -> None:
        src = tmp_path / "strings.py"
        src.write_text(
            'FORBIDDEN = "iai_mcp.daemon"\n'
            'DOC = "Do not import iai_mcp.lifecycle."\n',
            encoding="utf-8",
        )
        assert scan_for_forbidden_imports(src) == []

class TestScanForbiddenImports:

    def test_direct_daemon_import(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.py"
        src.write_text("import iai_mcp.daemon\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.daemon"
        assert hits[0].line_number == 1

    def test_from_daemon_import(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.py"
        src.write_text("from iai_mcp.daemon import Daemon\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.daemon"

    def test_lifecycle_module(self, tmp_path: Path) -> None:
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
        src = tmp_path / "bad.py"
        src.write_text(
            "from iai_mcp.lifecycle_state import load_state\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.lifecycle_state"

    def test_s2_coordinator(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.py"
        src.write_text(
            "from iai_mcp.s2_coordinator import S2Coordinator\n",
            encoding="utf-8",
        )
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.s2_coordinator"

    def test_socket_server(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.py"
        src.write_text("import iai_mcp.socket_server\n", encoding="utf-8")
        hits = scan_for_forbidden_imports(src)
        assert len(hits) == 1
        assert hits[0].module == "iai_mcp.socket_server"

    def test_returns_first_and_only_violation_in_one_line_file(
        self, tmp_path: Path
    ) -> None:
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

class TestHookIntegration:

    def _make_fake_item(self, source_path: Path) -> pytest.Item:

        class FakeModule:
            __file__ = str(source_path)

        class FakeItem:
            module = FakeModule()

        return FakeItem()  # type: ignore[return-value]

    def test_hook_raises_usage_error_on_forbidden_import(
        self, tmp_path: Path
    ) -> None:
        from tests.lilli.conftest import pytest_collection_modifyitems

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
            pytest_collection_modifyitems(config=None, items=[fake_item])
        finally:
            if clean_module_path.exists():
                clean_module_path.unlink()

    def test_hook_ignores_files_outside_lilli_dir(self, tmp_path: Path) -> None:
        from tests.lilli.conftest import pytest_collection_modifyitems

        outside_file = tmp_path / "other_test.py"
        outside_file.write_text(
            "from iai_mcp.daemon import Daemon\n", encoding="utf-8"
        )
        fake_item = self._make_fake_item(outside_file)
        pytest_collection_modifyitems(config=None, items=[fake_item])
