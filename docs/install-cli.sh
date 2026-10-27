#!/usr/bin/env bash
# Install the Lumen CLI binary from GitHub Releases (no Rust toolchain required).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/hanmahong5-arch/lumen/main/docs/install-cli.sh | bash
#   LUMEN_VERSION=lumen-cli-v0.3.2 bash docs/install-cli.sh
#   PREFIX=~/.local/bin bash docs/install-cli.sh
#
# Do NOT `cargo install lumen-cli` — that crates.io name is an unrelated project.

set -euo pipefail

REPO="${LUMEN_REPO:-hanmahong5-arch/lumen}"
PREFIX="${PREFIX:-$HOME/.local/bin}"

resolve_version() {
  if [ -n "${LUMEN_VERSION:-}" ]; then
    printf '%s\n' "$LUMEN_VERSION"
    return
  fi

  # Prefer GitHub "latest" when it is a CLI tag; else first lumen-cli-v* in the list.
  local api="https://api.github.com/repos/${REPO}/releases"
  local tag=""
  local body=""

  if body="$(curl -fsSL "${api}/latest" 2>/dev/null || true)" && [ -n "$body" ]; then
    tag="$(printf '%s' "$body" | tr ',' '\n' | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
    case "$tag" in
      lumen-cli-v*) printf '%s\n' "$tag"; return ;;
    esac
  fi

  if body="$(curl -fsSL "${api}?per_page=30" 2>/dev/null || true)" && [ -n "$body" ]; then
    tag="$(printf '%s' "$body" | tr ',' '\n' | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\(lumen-cli-v[^"]*\)".*/\1/p' | head -1)"
    if [ -n "$tag" ]; then
      printf '%s\n' "$tag"
      return
    fi
  fi

  # Last-known good pin if the API is unreachable (offline / rate-limited).
  printf '%s\n' "lumen-cli-v0.3.0"
}

VERSION="$(resolve_version)"
BASE="https://github.com/${REPO}/releases/download/${VERSION}"

uname_s="$(uname -s | tr '[:upper:]' '[:lower:]')"
uname_m="$(uname -m)"

case "${uname_s}-${uname_m}" in
  linux-x86_64|linux-amd64)  asset="lumen-x86_64-unknown-linux-gnu" ;;
  darwin-arm64|darwin-aarch64) asset="lumen-aarch64-apple-darwin" ;;
  darwin-x86_64)             asset="lumen-x86_64-apple-darwin" ;;
  mingw*|msys*|cygwin*)
    echo "error: on Windows use docs/install-cli.ps1 (PowerShell), or WSL." >&2
    echo "  Asset: ${BASE}/lumen-x86_64-pc-windows-msvc.exe" >&2
    exit 1
    ;;
  *)
    echo "error: unsupported platform ${uname_s}/${uname_m}" >&2
    echo "  download a matching asset from ${BASE}/" >&2
    exit 1
    ;;
esac

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "→ downloading ${asset} (${VERSION})"
curl -fsSL "${BASE}/${asset}" -o "${tmpdir}/${asset}"
curl -fsSL "${BASE}/SHA256SUMS" -o "${tmpdir}/SHA256SUMS"

(
  cd "$tmpdir"
  if command -v sha256sum >/dev/null 2>&1; then
    grep " ${asset}\$" SHA256SUMS | sha256sum -c -
  elif command -v shasum >/dev/null 2>&1; then
    expected="$(grep " ${asset}\$" SHA256SUMS | awk '{print $1}')"
    actual="$(shasum -a 256 "$asset" | awk '{print $1}')"
    if [ "$expected" != "$actual" ]; then
      echo "error: checksum mismatch for ${asset}" >&2
      exit 1
    fi
    echo "${asset}: OK"
  else
    echo "warning: no sha256sum/shasum; skipping checksum verify" >&2
  fi
)

mkdir -p "$PREFIX"
install -m 755 "${tmpdir}/${asset}" "${PREFIX}/lumen"

echo "✓ installed ${PREFIX}/lumen"
"${PREFIX}/lumen" --version || true
echo
echo "Next:  lumen demo          # offline sample lifecycle HTML (no Kova required)"
echo "       lumen --help"
# Require PATH to resolve to *this* binary — a stale `cargo install lumen-cli`
# (unrelated crates.io crate) would otherwise silence the hint.
ours="${PREFIX}/lumen"
on_path="$(command -v lumen 2>/dev/null || true)"
same=0
if [ -n "$on_path" ]; then
  if [ "$on_path" = "$ours" ] || { [ -e "$on_path" ] && [ -e "$ours" ] && [ "$on_path" -ef "$ours" ]; }; then
    same=1
  fi
fi
if [ "$same" -eq 0 ]; then
  echo
  if [ -z "$on_path" ]; then
    echo "Add to PATH if needed:  export PATH=\"${PREFIX}:\$PATH\""
  else
    echo "warning: \`lumen\` on PATH is ${on_path}, not ${ours}"
    echo "  Prefer this install:  export PATH=\"${PREFIX}:\$PATH\""
    echo "  Do NOT \`cargo install lumen-cli\` — that crates.io name is unrelated."
  fi
fi
