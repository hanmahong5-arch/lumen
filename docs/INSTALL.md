# Installing Lumen from source

Lumen is **not published to any package registry yet**. There is no `lumen-ai`
package on PyPI, and the `lumen-cli` crate on crates.io is an **unrelated
third-party project** (an AI programming language, not this tool). The only
supported install path today is building from source.

## Prerequisites

- **Rust** — a stable toolchain with Edition 2024 support, installed via
  [rustup](https://rustup.rs). Build verified with `rustc 1.96.0` / `cargo 1.96.0`.
- **Python** (SDK only) — Python 3.10+ with `pip` and `venv`. Verified with
  Python 3.12.3.
- **Git**.
- Verified on Linux (WSL2 Ubuntu). On Windows, build inside WSL.

## 1. Get the source

```bash
git clone https://github.com/hanmahong5-arch/lumen.git
cd lumen
```

## 2. Build the CLI

```bash
cargo build --release
```

The binary lands at `target/release/lumen`. A cold build (fresh clone, deps
not yet cached) takes a few minutes; incremental rebuilds finish in well
under a minute.

## 3. Verify

```bash
./target/release/lumen --version
# lumen 0.1.0

./target/release/lumen --help
# lists: replay, cost, traces, dashboard, metrics, pull, export, kova, demo, tour
```

`cargo build` does not put `lumen` on your PATH — either call it by path as
above, or copy `target/release/lumen` into a directory on your PATH.

## 4. Python SDK (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./lumen-sdk                 # SDK core
# or: pip install -e "./lumen-sdk[langgraph]"   / "./lumen-sdk[all]"

python -c "import lumen; print(lumen.__version__)"
# 0.2.1
```

## 5. Run the test suite (optional)

```bash
cargo test -p lumen-core -p lumen-cli
# all suites pass; e.g. lumen-core: "test result: ok. 14 passed; 0 failed"
```

## Troubleshooting

**`pip install lumen-ai` fails, or `cargo install lumen-cli` installs the
wrong software.** Expected: Lumen is not on PyPI (the request 404s), and the
`lumen-cli` name on crates.io belongs to an unrelated project. Build from
source as described above.

**`python3 -m venv` fails with "ensurepip is not available" (Debian/Ubuntu).**
The venv module ships separately there:

```bash
sudo apt install python3.12-venv   # match your Python minor version
```

then recreate the virtual environment.

**`lumen: command not found` after a successful build.** The release binary is
at `target/release/lumen`; `cargo build` never installs to PATH. Invoke it by
path or copy it somewhere on your PATH.
