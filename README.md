# Lumen

> Illuminate your AI agents. Never lose a run. Never burn tokens blindly.

**Developer tool for AI agent reliability and observability.** For anyone building
agents with LangGraph / CrewAI / AutoGen / custom Python who needs: visibility
(where money goes), debugging (what happened at step 7/12), reliability (resume
after crash without re-running), budget control (stop before credits burn).

## Install

```bash
pip install lumen-ai              # Python SDK (primary interface)
pip install "lumen-ai[all]"       # + all integrations
pip install "lumen-ai[langgraph]" # + LangGraph integration
cargo install lumen-cli           # CLI (Rust, optional)
```

## Quick Start — one-line auto-instrumentation

```python
from lumen import instrument, trace, traced_fn

instrument()  # auto-detect OpenAI / Anthropic / LangGraph; all later calls recorded

# Group multiple calls into one trace:
with trace("research-task") as t:
    r1 = client.chat.completions.create(model="gpt-4o", messages=[...])
    r2 = client.chat.completions.create(model="gpt-4o", messages=[...])
    print(f"Trace: {t.trace_id}, cost: ${t.total_cost_usd:.4f}")

# Decorator form (also: async with atrace(...)):
@traced_fn("summarizer")
def summarize(text):
    return client.chat.completions.create(...)
```

## Quick Start — LangGraph + Lumen

```python
from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
from langgraph.prebuilt import create_react_agent

checkpointer = LumenCheckpointer(storage_dir="./checkpoints")  # crash-safe, zero ext services
tracer = LumenTracer(trace_dir="./traces")                     # replay + cost tracking

graph = create_react_agent(model, tools, checkpointer=checkpointer)
result = graph.invoke(
    {"messages": [HumanMessage("Research quantum computing")]},
    config={"callbacks": [tracer]},
)
# then: lumen traces / cost / replay --trace-dir ./traces
```

## What it does

- **Replay** (`lumen replay <id>`) — deterministically replay any agent run from trace JSON, zero LLM cost. Supports `--from-step N`.
- **Cost tracking** (`lumen cost`) — per-agent, per-model dollar breakdown; flags per-run **cost outliers** (> 2× run mean). For metric-level **ML anomaly** detection, see Metrics & ML anomaly below.
- **Crash recovery** — process dies, agent resumes from last checkpoint. `LumenCheckpointer` does file-backed 3μs checkpoint writes (vs SQLite ~100μs, PostgreSQL ~1ms), zero external services.

Python API:

```python
from lumen import CostTracker, ReplayEngine

report = CostTracker(trace_dir="./traces").report(last="24h")
print(f"Total: ${report.total_usd:.2f} across {report.total_runs} runs")

trace = ReplayEngine(trace_dir="./traces").replay("abc123")
for step in trace.steps:
    print(f"Step {step.step}: {step.tool_name or step.content}")
```

## CLI

```bash
lumen replay <trace-id>              # Replay an agent run (zero cost)
lumen replay <id> --from-step 7      # Replay from a specific step
lumen cost --last 24h                # Cost report (flags per-run cost outliers)
lumen cost --last 7d --format json   # JSON output
lumen traces                         # List all agent runs
lumen traces --trace-dir ./my-traces # Custom trace directory
lumen dashboard                      # Web UI: trace timeline + Metrics + Terminal tabs
lumen dashboard --netdata-url http://localhost:19999   # + live metrics/ML anomaly
lumen dashboard --kova-url http://localhost:3010 --api-key sk-kova-…  # + Terminal tab
lumen metrics --last 10m             # Headless kova metrics + ML anomaly snapshot
lumen kova "agents"                  # Headless one-shot of a kova control verb
lumen kova "agent foo delete" --yes  # …destructive verbs need --yes
```

## Metrics & ML anomaly (Netdata)

Lumen is the **single observability surface** for Kova: traces + cost (read from trace JSON) **and**
live metrics + per-metric **ML anomaly detection** (consumed from [Netdata](https://netdata.cloud)).
The dashboard's **Metrics tab** and `lumen metrics` render kova's Prometheus metrics — agent-dispatch
queue depth, WAL bytes, throughput, completion tokens, cost — each with an **ML anomaly-rate ribbon**,
plus a fleet "ML anomaly rate" KPI and active Netdata health alarms.

```bash
export LUMEN_NETDATA_URL=http://localhost:19999     # or --netdata-url; over Tailscale: http://100.122.83.20:19999
lumen dashboard                                     # Metrics tab now live
lumen metrics --last 1h --format json               # headless snapshot for stress campaigns
```

**Single-origin proxy (no CORS).** The browser only talks to Lumen's loopback `:9700`; Lumen makes the
Netdata calls server-side. It works whether Lumen is co-located with Netdata or one Tailscale hop away.
Consuming Netdata's HTTP API is a REST call — **no GPLv3 netdata code is linked into Lumen**, and
`lumen-core` stays HTTP-free.

**SSRF discipline** (the proxy is browser-reachable): one fixed base URL; only `/api/v1/data` +
`/api/v1/alarms`; a chart whitelist (`prometheus.kova_*` / `anomaly_detection.*`); relative-only
`after`/`before` windows; fixed `group=average`; `points ≤ 1000`; no redirects; a ~2 MiB response cap;
and no header passthrough in either direction.

**Two distinct anomaly signals (by design).** Netdata ML anomaly is *per-metric, per-timestep* (a
metric sample deviates from its learned history) — the headline signal on the Metrics tab. It has no
`trace_id` dimension, so it can't be pinned to one run. Lumen's `cost` command/Overview keeps a
separate *per-run, offline* **cost outlier** flag (a run costing > 2× the mean of runs on disk) — cheap
and works with no Netdata, but no longer the headline anomaly story. A per-run **deep-link** bridges
them: from a trace, "view metrics for this run's window" opens the Metrics tab scoped to that run's
`[started_at_ms, completed_at_ms]`, so you see kova's metrics + anomaly ribbon *as they were while the
run executed*.

> Netdata scrapes kova-rest's Prometheus `/metrics` into `prometheus.kova_*` charts (one dimension per
> tester job `kova-rest-<tester>`) — **your services are unchanged**. See the obs-pack contract
> (`2b-svc-kova/deploy/obs-pack/`, Netdata on `:19999`) and `2b-svc-kova/doc/coord/contracts.md`.
>
> Known gap: the inline dashboard has no CSP (loopback-only dev tool; the page is all-inline, so a CSP
> would need `'unsafe-inline'` and add little). All Netdata-derived strings are HTML/JS-escaped.

## Terminal — kova control console

The dashboard's **Terminal tab** (and the headless `lumen kova "<command>"`) drive a running Kova
without SSH-ing to it: inspect agents / workflows / traces and run common control verbs from a
Ghostty-styled console in the browser.

```bash
export LUMEN_KOVA_URL=http://localhost:3010                 # or --kova-url; over Tailscale: http://100.122.83.20:3010
export LUMEN_KOVA_API_KEY=sk-kova-…                         # or --api-key; held server-side, never sent to the browser
lumen dashboard                                             # Terminal tab now live
lumen kova "agents"                                         # headless one-shot (no browser)
```

**Honest scope — it is a kova-REST command console, not Ghostty/a PTY.** Kova is API-first: every
control verb is a discrete REST call, so the "terminal" is a **REST interpreter**, not a shell. A typed
line is parsed into one **whitelisted** command that maps to exactly one `(method, /api/v1/… path,
body)`; there is no raw shell, no `std::process` exec, and no arbitrary path/method/host. (A true PTY
shell into Kova would be arbitrary RCE — out of scope; Forge is the home for that if ever wanted.)

**Verbs:**
- **Read:** `status` · `agents` · `agent <id> [traces|history|status]` · `workflows` · `workflow <id>` ·
  `awaiting` (workflows awaiting input) · `approvals` (kova's pending-HITL-approval projection) ·
  `schedules` · `schedule <id>` · `tools` · `queues` · `traces` · `trace <id>` · `llm` · `help`/`clear` (local).
- **Safe mutations:** `agent <id> run <msg…>` · `agent <id> stop|pause|resume|restart` ·
  `agent <id> approve|deny [comment…]` · `workflow <id> cancel [reason…]` · `workflow <id> resume <input…>` ·
  `schedule <id> pause|resume`.
- **Destructive (confirm required):** `agent <id> reset [reason…]` · `agent <id> terminate [reason…]` ·
  `agent <id> delete` · `schedule <id> delete` · `trace <id> delete`.

**Safety model:**
- The dashboard stays **loopback-only** (`127.0.0.1`); the kova API key is held **server-side** (reuses
  `resolve_api_key`) and never reaches the browser — the page only ever POSTs command *text* to Lumen.
- **Whitelist chokepoint:** the HTTP seam is only ever called with an interpreter-built `(method, path)`;
  ids are percent-encoded so no extra path segment / query / `..` traversal is reachable. One fixed
  `base_url`, `redirects(0)`, ~2 MiB bounded read.
- **Destructive confirm** is enforced **server-side** (not bypassable from the UI): a destructive command
  is never sent until a second confirmed request arrives (`⏎`/`y` in the UI, `--yes` for `lumen kova`).
- **CSRF-guarded:** the server rejects any state-changing request (`POST`) whose `Origin` isn't the
  dashboard's own loopback origin, so a page the operator happens to visit can't drive kova through the
  console (the key is server-side, and a cross-site `POST` would otherwise borrow it). Non-browser
  clients (the `lumen` CLI, curl) send no `Origin` and are unaffected.
- Kova independently enforces **RBAC + audit** server-side, so the key's role bounds reach; all
  kova-derived output is inert in the DOM (`textContent`). Read verbs need no LLM; agent *runs* need a
  working LLM key on Kova.
- Known gap (same as the Metrics tab): no CSP — loopback-only, all-inline page. A `--console-token` gate
  on the dashboard itself (for shared / multi-user hosts) is noted future work.

See `2b-svc-kova/doc/coord/contracts.md` (Lumen → kova-rest control-plane) for the consumed endpoint set.

## Pricing estimation

Estimates costs from token usage even when the provider doesn't report cost directly.
Supports 30+ models incl. Claude, GPT-4o, Gemini, Llama, Mistral, DeepSeek.

## v0.2 Status

Shipped ✅: `instrument()` one-line auto-detection · OpenAI + Anthropic SDK auto-instrumentation ·
LangGraph CheckpointSaver + Tracer · `with trace(...)` / `async with atrace(...)` / `@traced_fn` ·
cost tracking (Python + CLI) · replay (Python + CLI) · pricing estimation (30+ models) ·
budget tracker with kill-on-exceeded · cost anomaly detection · trace redaction (PII / custom
patterns) · trace diff & compare · TraceQuery (filter / aggregate) · runtime metrics
(Prometheus-style) · hierarchical config (`lumen.toml` / env / builder) · web dashboard
(trace timeline + Netdata Metrics tab + kova-control Terminal tab).

⏳ v0.3: `lumen.Agent.run()` direct execution · web dashboard (`lumen up`) · `pytest-lumen` plugin.
⏳ Planned: CrewAI / AutoGen integration.

## Architecture

Your Python code → **Lumen SDK** (`lumen-ai`: cost tracking, anomaly detection, replay engine;
LangGraph CheckpointSaver + Tracer callback) → **Lumen Core** (Rust, embedded: replay engine,
cost aggregation, pricing tables) → your LLM provider (OpenAI, Anthropic, etc.).

## License

MIT
