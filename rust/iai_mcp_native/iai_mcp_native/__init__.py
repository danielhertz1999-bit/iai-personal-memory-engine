"""Native Python wheel for the iai_mcp Rust workspace.

Re-exports the embed and graph sub-modules from the compiled extension so
that both ``from iai_mcp_native import embed, graph`` and
``import iai_mcp_native.embed`` resolve. The extension's own
``#[pymodule]`` body also writes ``iai_mcp_native.embed`` and
``iai_mcp_native.graph`` into ``sys.modules`` to support the dotted-import
form.
"""

from .iai_mcp_native import *  # noqa: F401,F403 — re-export native API surface

# Mirror the Maturin auto-generated wrapper: expose the native module's
# docstring + __all__ at the package level when available.
from . import iai_mcp_native as _native

__doc__ = _native.__doc__ if _native.__doc__ else __doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
