# NOTICE

This project is MIT-licensed (see `LICENSE`). The load-bearing components are
**our own original code**; this file attributes only the genuine **third-party**
software the project depends on. Every third-party runtime dependency is MIT,
BSD, Apache-2.0, or PSF — all compatible with this project's MIT license.

## Our own components (not third-party)

These are original code written for this project, under its MIT `LICENSE` — they
are **not** dependencies and are listed here only to make the boundary explicit:

- **Hippo** — the storage engine (encrypted records + ANN vector index + graph +
  event ledger in one local store). Our code, built on the standard-library
  `sqlite3`, the `hnswlib` index, and the audited `cryptography` AES-256-GCM
  primitive.
- **MOSAIC** — our community-detection algorithm, written from scratch for the
  memory-graph workload (`src/iai_mcp/mosaic*.py`). Pure Python + Numba.
- **Lilli** — our hyperdimensional cognitive/memory substrate (BSC / FHRR /
  sparse VSA tiers) and the recall, capture, and sleep-consolidation pipelines
  (`src/iai_mcp/lilli/` and the daemon).
- **Native engine** (`iai_mcp_native`) — our Rust embedder + graph kernels.

For Python deps, version pins live in `pyproject.toml`; for the Rust engine, in
`rust/*/Cargo.toml`; for npm, in `mcp-wrapper/package.json`.

## Python runtime dependencies

| Package      | License             | Author / maintainer                           | Upstream URL                            |
| ------------ | ------------------- | --------------------------------------------- | --------------------------------------- |
| cachetools   | MIT                 | Thomas Kemmer                                 | https://github.com/tkem/cachetools/     |
| cryptography | Apache-2.0 or BSD-3 | Python Cryptographic Authority + contributors | https://github.com/pyca/cryptography    |
| hnswlib      | Apache-2.0          | Yury Malkov et al.                            | https://github.com/nmslib/hnswlib       |
| keyring      | MIT                 | Jason R. Coombs (maintainer)                  | https://github.com/jaraco/keyring       |
| numba        | BSD-2-Clause        | Numba project (Anaconda Inc.)                 | https://numba.pydata.org                |
| numpy        | BSD-3-Clause        | NumPy developers                              | https://numpy.org                       |
| pandas       | BSD-3-Clause        | pandas development team (PyData)              | https://pandas.pydata.org               |
| psutil       | BSD-3-Clause        | Giampaolo Rodola                              | https://github.com/giampaolo/psutil     |
| pyarrow      | Apache-2.0          | Apache Arrow project                          | https://arrow.apache.org/               |
| scipy        | BSD-3-Clause        | SciPy developers                              | https://scipy.org                       |
| tiktoken     | MIT                 | OpenAI                                        | https://github.com/openai/tiktoken      |

The standard-library `sqlite3` binds SQLite, which is public domain.

## Rust native engine dependencies

The package builds a native extension (`iai_mcp_native`) from the `rust/`
workspace via `setuptools-rust` during `pip install`. The compiled extension
links the following crates — all permissively licensed (MIT, Apache-2.0, or
BSD). (Exact per-crate SPDX is regenerated with `cargo license` before release;
the values below are the upstream norms.)

| Crate          | License            | Upstream                                            |
| -------------- | ------------------ | --------------------------------------------------- |
| pyo3           | Apache-2.0         | https://github.com/PyO3/pyo3                        |
| pyo3-stub-gen  | MIT or Apache-2.0  | https://github.com/Jij-Inc/pyo3-stub-gen            |
| candle-core    | MIT or Apache-2.0  | https://github.com/huggingface/candle               |
| candle-nn      | MIT or Apache-2.0  | https://github.com/huggingface/candle               |
| safetensors    | Apache-2.0         | https://github.com/huggingface/safetensors          |
| tokenizers     | Apache-2.0         | https://github.com/huggingface/tokenizers           |
| hf-hub         | Apache-2.0         | https://github.com/huggingface/hf-hub               |
| serde          | MIT or Apache-2.0  | https://github.com/serde-rs/serde                   |
| serde_json     | MIT or Apache-2.0  | https://github.com/serde-rs/json                    |
| thiserror      | MIT or Apache-2.0  | https://github.com/dtolnay/thiserror                |
| petgraph       | MIT or Apache-2.0  | https://github.com/petgraph/petgraph                |
| rustworkx-core | Apache-2.0         | https://github.com/Qiskit/rustworkx                 |
| rayon          | MIT or Apache-2.0  | https://github.com/rayon-rs/rayon                   |
| fixedbitset    | MIT or Apache-2.0  | https://github.com/petgraph/fixedbitset             |
| rand           | MIT or Apache-2.0  | https://github.com/rust-random/rand                 |
| rand_pcg       | MIT or Apache-2.0  | https://github.com/rust-random/rand                 |
| numpy (rust)   | BSD-2-Clause       | https://github.com/PyO3/rust-numpy                  |
| uuid           | MIT or Apache-2.0  | https://github.com/uuid-rs/uuid                     |
| accelerate-src | MIT or Apache-2.0  | https://github.com/blas-lapack-rs/accelerate-src    |

`accelerate-src` binds Apple's Accelerate framework on macOS; it is build/link
glue only and ships no Apple code.

## Python build dependency

| Package         | License | Upstream                                |
| --------------- | ------- | --------------------------------------- |
| setuptools-rust | MIT     | https://github.com/PyO3/setuptools-rust |

Build-time only (compiles the Rust extension during `pip install`); not imported
at runtime.

## Python optional dependencies

Installable via extras but NOT pulled in by a default install.

### `compress` extra (opt-in; large model weights)

| Package    | License    | Upstream                                  |
| ---------- | ---------- | ----------------------------------------- |
| llmlingua  | MIT        | https://github.com/microsoft/LLMLingua    |
| accelerate | Apache-2.0 | https://github.com/huggingface/accelerate |

### `migration` extra (one-time legacy-store import only)

| Package | License    | Upstream                           |
| ------- | ---------- | ---------------------------------- |
| lancedb | Apache-2.0 | https://github.com/lancedb/lancedb |

Installed only to import data from a pre-Hippo store; not used by a fresh install
or at runtime.

### `dev` extra (test-only, not shipped at runtime)

| Package             | License      | Upstream                                       |
| ------------------- | ------------ | ---------------------------------------------- |
| pytest              | MIT          | https://github.com/pytest-dev/pytest           |
| pytest-cov          | MIT          | https://github.com/pytest-dev/pytest-cov       |
| ruff                | MIT          | https://github.com/astral-sh/ruff              |
| networkx            | BSD-3-Clause | https://networkx.org/                          |
| hypothesis-networkx | MIT          | https://github.com/pckroon/hypothesis-networkx |
| scikit-learn        | BSD-3-Clause | https://scikit-learn.org                       |

`networkx` is a test-only oracle for the graph-algorithm parity checks; it is not
imported on any runtime path.

## TypeScript wrapper runtime dependencies

The `mcp-wrapper/` subdirectory contains the TypeScript MCP wrapper. Its runtime
dependencies (NOT devDependencies, which are build-only) are:

| Package                   | Version pin | License | Upstream                                               |
| ------------------------- | ----------- | ------- | ------------------------------------------------------ |
| @modelcontextprotocol/sdk | ^1.0.0      | MIT     | https://github.com/modelcontextprotocol/typescript-sdk |
| zod                       | ^3.23.0     | MIT     | https://github.com/colinhacks/zod                      |

The wrapper's `devDependencies` (`@types/node`, `typescript`, `tsx`) are
build-time only and not bundled.

## License compatibility summary

Every third-party runtime dependency above — Python, Rust, and TypeScript — is
licensed under one of **MIT**, **BSD-2/BSD-3**, **Apache-2.0**, or **PSF**, or a
permissive dual-license that includes one of these. SQLite is public domain. All
are permissive and compatible with this project's MIT `LICENSE`.

## Updating this file

Regenerate when `pyproject.toml` dependencies, `rust/*/Cargo.toml`, or
`mcp-wrapper/package.json` change:

```
pip install pip-licenses && pip-licenses --format=markdown --with-urls --with-authors
cargo install cargo-license && cargo license --manifest-path rust/iai_mcp_native/Cargo.toml
```
