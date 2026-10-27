# Connected path — Kova / Forge in ~10 minutes

Lumen **Core** works offline. This runbook unlocks the **Connected** layer
(Flow, Terminal, Metrics, advisory) against a live Kova (and optionally Netdata).

**三仓唯一操作面**（静态 / 运行 / 部署）：[`2b-svc-kova/docs/kova/TRIPLE.md`](../../2b-svc-kova/docs/kova/TRIPLE.md) — 在 Kova 仓跑 `bash scripts/triple.sh help`（含 viewer key 配方与 Demo Day joint）。

## Prerequisites

- A running `kova-rest` (local or STAGE) with an API key
- A Lumen CLI binary on `PATH` ([INSTALL.md](INSTALL.md) / `docs/install-cli.sh`)
- Optional: Netdata scraping kova Prometheus `/metrics` (no service code changes)

## Golden path

```bash
# 1. Point Lumen at Kova (API key stays server-side; never sent to the browser)
#    Prefer 127.0.0.1 on Windows (localhost can hit ::1 first) — see TRIPLE.md
export LUMEN_KOVA_URL=http://127.0.0.1:3010
export LUMEN_KOVA_API_KEY=sk-kova-…   # or KOVA_API_KEY; triple.sh env --howto-viewer

# 2. Live UI — Flow / Terminal / advisory unlock; traces auto-sync from Kova
#    every 5s into ./traces (UI refresh is also 4s). No separate pull needed.
lumen dashboard --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY"
#    tune: --sync-interval 2   ·  --sync-deep   ·  --no-sync

# 3. Headless live sync (no UI) — same pull loop
lumen pull --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY" --watch
#    one-shot + sidecars for offline export:
lumen pull --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY" --deep

# 4. Shareable incident artifact
lumen export <run-id>                  # opens *-lifecycle.html

# 5. Optional Metrics tab
lumen dashboard --kova-url "$LUMEN_KOVA_URL" --api-key "$LUMEN_KOVA_API_KEY" \
  --netdata-url http://localhost:19999
```

## What is “live” today

| Surface | Mechanism | Latency |
|---|---|---|
| Traces timeline | Background pull → disk → `/api/traces` (4s UI poll) | ~sync interval (default 5s) |
| Terminal / Flow / advisory | Direct REST proxy to kova-rest | request RTT |
| Metrics / ML anomaly | Netdata scrape of kova `/metrics` | Netdata + 4s UI poll |

Kova does not expose a trace SSE/WebSocket today; Lumen’s Connected path is
**pull-based**. Tightening `--sync-interval` (e.g. 2) is the supported way to
feel more real-time without changing Kova.

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

## OTLP export (optional)

Set `LUMEN_OTLP_ENABLED=1` and `LUMEN_OTLP_ENDPOINT=http://localhost:4318`
(or `export.otlp` in `lumen.toml`). Each completed trace is POSTed as
OTLP/HTTP JSON to `{endpoint}/v1/traces` in addition to the local
`trace_dir`. Stdlib only — no OpenTelemetry SDK dependency.
