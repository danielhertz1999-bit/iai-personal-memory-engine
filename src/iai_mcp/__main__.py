"""Package entry point so ``python -m iai_mcp`` works.

The Windows PowerShell hooks (``_deploy/hooks/*.ps1``) invoke the CLI as
``python -m iai_mcp <subcommand>`` rather than via the ``iai-mcp`` console
script (which may not be on PATH inside a hook subprocess). That form requires
this module; without it Python raises "No module named iai_mcp.__main__" and
every hook silently no-ops. Delegates to the same entry point as the
``iai-mcp`` console script and ``python -m iai_mcp.cli``.
"""

from __future__ import annotations

import sys

from iai_mcp.cli import main

if __name__ == "__main__":
    sys.exit(main())
