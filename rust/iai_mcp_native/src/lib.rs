//! iai_mcp_native — single-cdylib wheel exposing two Python sub-modules
//! (`embed` + `graph`).
//!
//! The two core crates (`iai_mcp_embed_core` and `iai_mcp_graph_core`) are
//! plain `rlib`s with no `#[pymodule]` entry of their own — instead they
//! each expose a `register(py, m)` helper that this wrapper calls from
//! inside its `#[pymodule]` body. The result is one `.so` file with two
//! logical Python sub-modules:
//!
//! ```python
//! from iai_mcp_native import embed, graph
//! e = embed.Embedder()
//! v = graph.answer()
//! ```
//!
//! The wrapper also registers the dotted sub-module names into
//! `sys.modules` so `import iai_mcp_native.embed` works as a stand-alone
//! statement, not just `from iai_mcp_native import embed`. This is the
//! workaround documented in the Maturin Book for PyO3 sub-modules; without
//! it the dotted-import path raises `ModuleNotFoundError`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::define_stub_info_gatherer;

#[pymodule]
fn iai_mcp_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Embedder sub-module — Bert / bge-small-en-v1.5 forward pass.
    let embed = PyModule::new_bound(py, "embed")?;
    iai_mcp_embed_core::register(py, &embed)?;
    m.add_submodule(&embed)?;

    // Graph sub-module — pure-Rust algorithms layer (currently a wiring
    // probe; real algorithm work begins in later plans).
    let graph = PyModule::new_bound(py, "graph")?;
    iai_mcp_graph_core::register(py, &graph)?;
    m.add_submodule(&graph)?;

    // Register the dotted sub-module names in `sys.modules` so a separate
    // `import iai_mcp_native.embed` statement also resolves. Without this
    // step, only `from iai_mcp_native import embed` works.
    let sys_modules: Bound<'_, PyDict> = py
        .import_bound("sys")?
        .getattr("modules")?
        .downcast_into()?;
    sys_modules.set_item("iai_mcp_native.embed", &embed)?;
    sys_modules.set_item("iai_mcp_native.graph", &graph)?;

    Ok(())
}

// Stub-metadata gatherer for the `stub_gen` binary. The macro walks the
// `#[gen_stub_*]` attributes declared in the consumed core crates because
// each crate runs its own `define_stub_info_gatherer!(stub_info)` and the
// wrapper aggregates them at build time.
define_stub_info_gatherer!(stub_info);
