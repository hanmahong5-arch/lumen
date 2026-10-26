# Installing Lumen

> **Do not** `cargo install lumen-cli` or assume `pip install lumen-ai` works yet.
> The `lumen-cli` name on crates.io is an **unrelated** third-party project.
> The Python package is not on PyPI yet — install the SDK from this repo
> (first publish: [RELEASE.md](RELEASE.md)).

## Fast path — CLI binary (recommended)

No Rust toolchain required. Downloads the **latest** `lumen-cli-v*` GitHub
Release for your machine, verifies `SHA256SUMS`, and installs it.

```bash
curl -fsSL https://raw.githubusercontent.com/hanmahong5-arch/lumen/main/docs/install-cli.sh | bash
# pin a release:  LUMEN_VERSION=lumen-cli-v0.3.1 bash …
export PATH="$HOME/.local/bin:$PATH"
lumen --version
lumen demo        # opens a sample lifecycle HTML — zero Kova / LLM key needed
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/hanmahong5-arch/lumen/main/docs/install-cli.ps1 | iex
# pin:  $env:LUMEN_VERSION = 'lumen-cli-v0.3.1'; .\docs\install-cli.ps1
```

Default install dir: `%LOCALAPPDATA%\lumen\bin` (override with `$env:PREFIX`).
Add that directory to your user `PATH`, then run `lumen demo`.

Scripts: [install-cli.sh](install-cli.sh) · [install-cli.ps1](install-cli.ps1).
Release process: [RELEASE.md](RELEASE.md).

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

Needs a stable Rust toolchain with Edition 2024 support. On Windows, prefer WSL.

```bash
git clone https://github.com/hanmahong5-arch/lumen.git
cd lumen
cargo build --release
./target/release/lumen --version
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
crates.io crate is unrelated. Use [install-cli.sh](install-cli.sh) /
[install-cli.ps1](install-cli.ps1) or build from this repo.

**`pip install lumen-ai` 404s.** Expected until a PyPI publish; use
`pip install -e ./lumen-sdk` from a clone. See [RELEASE.md](RELEASE.md).

**`python3 -m venv` fails with "ensurepip is not available" (Debian/Ubuntu).**

```bash
sudo apt install python3.12-venv   # match your Python minor version
```

**`lumen: command not found` after install-cli.sh.** Add
`export PATH="$HOME/.local/bin:$PATH"` (or your `PREFIX`) to your shell profile.

**Windows: `lumen` not found after install-cli.ps1.** Add
`%LOCALAPPDATA%\lumen\bin` to your user PATH (the script prints the exact
commands).
