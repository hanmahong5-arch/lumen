# Installing Lumen

> **Do not** `cargo install lumen-cli` or assume `pip install lumen-ai` works yet.
> The `lumen-cli` name on crates.io is an **unrelated** third-party project.
> The Python package is not on PyPI yet — install the SDK from this repo.

## Fast path — CLI binary (recommended)

No Rust toolchain required. Downloads the GitHub Release for your machine,
verifies `SHA256SUMS`, and installs to `~/.local/bin` (override with `PREFIX=`).

```bash
curl -fsSL https://raw.githubusercontent.com/hanmahong5-arch/lumen/main/docs/install-cli.sh | bash
# pin a release:  LUMEN_VERSION=lumen-cli-v0.3.0 bash …
export PATH="$HOME/.local/bin:$PATH"
lumen --version   # lumen 0.3.0
lumen demo        # opens a sample lifecycle HTML — zero Kova / LLM key needed
```

Windows (PowerShell): download
`lumen-x86_64-pc-windows-msvc.exe` from the
[latest CLI release](https://github.com/hanmahong5-arch/lumen/releases/tag/lumen-cli-v0.3.0),
rename to `lumen.exe`, put it on your `PATH`, then run `lumen demo`.

Script source: [docs/install-cli.sh](install-cli.sh).

## Python SDK (from source)

```bash
git clone https://github.com/hanmahong5-arch/lumen.git
cd lumen
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ./lumen-sdk                 # SDK core
# or: pip install -e "./lumen-sdk[langgraph]"   / "./lumen-sdk[all]"

python -c "import lumen; print(lumen.__version__)"
# 0.2.1
```

## Build the CLI from source (optional)

Needs a stable Rust toolchain with Edition 2024 support (verified with
`rustc 1.96.0`). On Windows, prefer WSL.

```bash
git clone https://github.com/hanmahong5-arch/lumen.git
cd lumen
cargo build --release
./target/release/lumen --version   # lumen 0.3.0
```

`cargo build` does not put `lumen` on your `PATH` — invoke by path or copy the
binary into a directory that is.

## Tests (optional)

```bash
cargo test -p lumen-core -p lumen-cli
cd lumen-sdk && python -m pytest
```

CI runs both on every push/PR to `main` (`.github/workflows/ci.yaml`).

## Troubleshooting

**`cargo install lumen-cli` installs the wrong software.** Expected — that
crates.io crate is unrelated. Use [install-cli.sh](install-cli.sh) or build
from this repo.

**`pip install lumen-ai` 404s.** Expected until a PyPI publish; use
`pip install -e ./lumen-sdk` from a clone.

**`python3 -m venv` fails with "ensurepip is not available" (Debian/Ubuntu).**

```bash
sudo apt install python3.12-venv   # match your Python minor version
```

**`lumen: command not found` after install-cli.sh.** Add
`export PATH="$HOME/.local/bin:$PATH"` (or your `PREFIX`) to your shell profile.
