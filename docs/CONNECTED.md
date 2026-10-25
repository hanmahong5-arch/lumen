# Connected path — Kova / Forge in ~10 minutes

Lumen **Core** works offline. This runbook unlocks the **Connected** layer
(Flow, Terminal, Metrics, advisory) against a live Kova (and optionally Netdata).

## Prerequisites

- A running `kova-rest` (local or STAGE) with an API key
- A Lumen CLI binary on `PATH` ([INSTALL.md](INSTALL.md) / `docs/install-cli.sh`)
- Optional: Netdata scraping kova Prometheus `/metrics` (no service code changes)

## Golden path

```bash
# 1. Point Lumen at Kova (API key stays server-side; never sent to the browser)
export LUMEN_KOVA_URL=http://localhost:3010
export LUMEN_KOVA_API_KEY=sk-kova-…   # or KOVA_API_KEY

# 2. Pull a recent run (add --deep for full lifecycle sidecars)
lumen pull --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY" --deep

# 3. Shareable incident artifact
lumen export <run-id>                  # opens *-lifecycle.html

# 4. Live UI — Flow / Terminal / advisory unlock when Kova is configured
lumen dashboard --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY"

# 5. Optional Metrics tab
lumen dashboard --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY" \
  --netdata-url http://localhost:19999
```

## Forge embed

Enterprise demos should set `FORGE_LUMEN_BIN` to the Lumen binary so lifecycle
export is a **non-skippable** demo screen (same HTML as `lumen export`).
If the binary is missing, fall back to Core sample: `lumen demo` (offline).

## Contract hygiene

- Trace JSON shape is vendored in `lumen-core` from `kova-types` (schema version
  in `trace_types.rs`). After a kova schema bump, run the sibling
  `lumen-trace-schema-scan` (when the kova tree is checked out) and refresh
  `lumen-core/tests/fixtures` + the Rust/Python drift guards.
- Do not re-aggregate Flow / Governor stats in Lumen — proxy Kova as the
  single source of truth.

## What not to promise

- Connected features without a reachable Kova/Netdata
- That Lumen itself runs durable WAL agents (that is Kova)
- OTLP export (not implemented; enabling it only warns)
