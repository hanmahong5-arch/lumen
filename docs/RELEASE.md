# Cutting a Lumen release

Two artifacts, two tag prefixes (or one shared tag when both ship together):

| Artifact | Tag | Workflow |
|---|---|---|
| CLI binaries | `lumen-cli-vX.Y.Z` | `.github/workflows/release-lumen-cli.yaml` |
| Python SDK (`lumen-ai`) | `lumen-sdk-vX.Y.Z` | `.github/workflows/publish-lumen-sdk.yaml` |
| Both | `vX.Y.Z` | both of the above |

Do **not** `cargo publish` — `lumen-cli` on crates.io is an unrelated package.

## CLI binary release

1. Bump `[workspace.package] version` in root `Cargo.toml` (and confirm
   `lumen --version` would print the new number).
2. Merge to `main`, CI green.
3. Tag and push:

```bash
git tag -a lumen-cli-v0.3.2 -m "lumen-cli 0.3.1"
git push origin lumen-cli-v0.3.2
```

4. Wait for the Release workflow matrix (linux / windows / darwin). Confirm
   assets + `SHA256SUMS` on the GitHub Release page.
5. Smoke:

```bash
curl -fsSL https://raw.githubusercontent.com/hanmahong5-arch/lumen/main/docs/install-cli.sh | bash
lumen --version
lumen demo
```

Windows: `irm …/docs/install-cli.ps1 | iex`.

## Python SDK → PyPI (one-time + each cut)

### One-time: Trusted Publisher

1. GitHub Environment **`pypi`** already exists on `hanmahong5-arch/lumen`
   (created 2026-11-06). Optional: add a required reviewer under
   Settings → Environments → pypi.
2. On [pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
   create a **Pending publisher** for project `lumen-ai` (creates the project
   on first successful upload). Match the OIDC claims from the failed
   `lumen-sdk-v0.2.1` run exactly:
   - Owner: `hanmahong5-arch`
   - Repository: `lumen`
   - Workflow name: `publish-lumen-sdk.yaml`
   - Environment name: `pypi`

3. Re-run publish: `gh workflow run publish-lumen-sdk.yaml` will not upload
   (tag-only). Either delete+retag `lumen-sdk-v0.2.1` after the publisher is
   live, or bump to `0.2.2` / tag `lumen-sdk-v0.2.2` and push.

No long-lived `PYPI_API_TOKEN` is required (OIDC). Until step 2 is done,
`pip install lumen-ai` stays 404 — keep INSTALL.md honest. First tag attempt
(`lumen-sdk-v0.2.1`, run 42147653742) failed with `invalid-publisher` as
expected before the pending publisher exists.

### Each SDK cut

1. Bump `version` in `lumen-sdk/pyproject.toml` **and**
   `lumen-sdk/lumen/__init__.py` (`__version__`) to the same value.
2. Merge to `main`, CI green.
3. Tag:

```bash
git tag -a lumen-sdk-v0.2.2 -m "lumen-ai 0.2.2"
git push origin lumen-sdk-v0.2.2
```

4. Confirm the Publish workflow uploaded; then:

```bash
pip install lumen-ai==0.2.2
python -c "import lumen; print(lumen.__version__)"
```

Only cut a shared `vX.Y.Z` when **both** CLI and SDK versions are intentionally
aligned and the Trusted Publisher is already green once.

## Install tax checklist

After any CLI release: install scripts default to GitHub **latest**
`lumen-cli-v*` (override with `LUMEN_VERSION=`). Keep
[INSTALL.md](INSTALL.md) examples honest about PyPI status until the first
`pip install lumen-ai` succeeds.
