# Lumen

> Illuminate your AI agents. Never lose a run. Never burn tokens blindly.

Lumen is a **developer tool for AI agent reliability and observability**.

```bash
pip install lumen-ai
```

## Quick Start: One-line auto-instrumentation

```python
pip install "lumen-ai[all]"
```

```python
from lumen import instrument, trace

instrument()  # auto-detect OpenAI / Anthropic / LangGraph

# All subsequent LLM calls are recorded automatically.
# Group multiple calls into one trace:
with trace("research-task") as t:
    r1 = client.chat.completions.create(model="gpt-4o", messages=[...])
    r2 = client.chat.completions.create(model="gpt-4o", messages=[...])
    print(f"Trace: {t.trace_id}, cost: ${t.total_cost_usd:.4f}")

# Decorator form:
from lumen import traced_fn

@traced_fn("summarizer")
def summarize(text):
    return client.chat.completions.create(...)
```

## Quick Start: LangGraph + Lumen

```python
pip install "lumen-ai[langgraph]"
```

```python
from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
from langgraph.prebuilt import create_react_agent

# Crash-safe checkpointing (zero external services)
checkpointer = LumenCheckpointer(storage_dir="./checkpoints")

# Execution tracing for replay + cost tracking
tracer = LumenTracer(trace_dir="./traces")

graph = create_react_agent(model, tools, checkpointer=checkpointer)
result = graph.invoke(
    {"messages": [HumanMessage("Research quantum computing")]},
    config={"callbacks": [tracer]},
)

# Now you can:
# lumen traces --trace-dir ./traces
# lumen cost --last 24h --trace-dir ./traces
# lumen replay <trace-id> --trace-dir ./traces
```

## Who is Lumen for?

Lumen is for **AI application developers** — anyone building agents with LangGraph, CrewAI, AutoGen, or custom Python code who needs:

- **Visibility**: Where is my money going? Which agent costs $50/day?
- **Debugging**: Agent failed at step 7 of 12 — what happened?
- **Reliability**: Process crashed — can I resume without re-running everything?
- **Budget control**: Stop the agent before it burns my API credits

## What it does

### Replay — Deterministically replay any agent run. Zero LLM cost.

```bash
$ lumen replay abc123
🔄 Replaying trace abc123 (12 iterations, $2.34 original cost)

  Step 1: ✓ search_web(call_01) → 3 results
  Step 2: ✓ read_url(call_02) → 2,847 chars
  ...
  Step 11: ✓ summarize(call_11) → 450 chars
  Step 12: 💡 Final answer: "Quantum computing in 2026..."

✅ Run completed successfully
```

### Cost tracking — Know exactly where every dollar goes.

```bash
$ lumen cost --last 24h
📊 Cost Report (last 24h)
  Total: $47.23 across 156 runs

  By Agent:
    research-agent           $31.20 (66%)
    summary-agent            $12.03 (25%)
    routing-agent            $4.00  (8%)

  ⚠️  Anomalies:
    run #42 (research-agent): $18.50 (3x avg) — 47 iterations
```

### Cost tracking in Python

```python
from lumen import CostTracker, ReplayEngine

tracker = CostTracker(trace_dir="./traces")
report = tracker.report(last="24h")
print(f"Total: ${report.total_usd:.2f} across {report.total_runs} runs")

engine = ReplayEngine(trace_dir="./traces")
trace = engine.replay("abc123")
for step in trace.steps:
    print(f"Step {step.step}: {step.tool_name or step.content}")
```

### Crash recovery — Process dies? Agent resumes from last checkpoint.

```python
from lumen.integrations.langgraph import LumenCheckpointer

checkpointer = LumenCheckpointer(storage_dir="./checkpoints")
graph = create_react_agent(model, tools, checkpointer=checkpointer)

# 3μs checkpoint writes (vs SQLite ~100μs, PostgreSQL ~1ms)
# Zero external services required
```

## Why Lumen?

| Pain Point | Before | With Lumen |
|-----------|--------|------------|
| Agent crashes | Lost. Re-run and hope. | Auto-recovered from checkpoint. |
| Debug failures | Add print statements | `lumen replay <trace-id>` |
| Cost tracking | `print(response.usage)` | Per-agent, per-model dashboard |
| Agent loops | Burn $50 before you notice | Auto-terminated at limit |
| LangGraph checkpoints | SQLite (slow) or PG (heavy) | File-backed, 3μs, zero deps |

## Architecture

```
Your Python Code
  │
  ▼
Lumen SDK (pip install lumen-ai)
  │  Cost tracking · Anomaly detection · Replay engine
  │  LangGraph: CheckpointSaver + Tracer callback
  ▼
Lumen Core (Rust, embedded)
  │  Replay engine · Cost aggregation · Pricing tables
  ▼
Your LLM Provider (OpenAI, Anthropic, etc.)
```

## CLI

```bash
lumen replay <trace-id>                  # Replay an agent run (zero cost)
lumen replay <id> --from-step 7          # Replay from a specific step
lumen cost --last 24h                    # Cost report
lumen cost --last 7d --format json       # JSON output
lumen traces                             # List all agent runs
lumen traces --trace-dir ./my-traces     # Custom trace directory
lumen dashboard                          # Web UI (coming soon)
```

## Pricing estimation

Lumen estimates costs from token usage even when your LLM provider doesn't report costs directly. Supports 30+ models including Claude, GPT-4o, Gemini, Llama, Mistral, and DeepSeek.

## v0.2 Status

| Feature | Status |
|---------|--------|
| `instrument()` one-line auto-detection | ✅ |
| OpenAI SDK auto-instrumentation | ✅ |
| Anthropic SDK auto-instrumentation | ✅ |
| LangGraph CheckpointSaver + Tracer | ✅ |
| `with trace(...)` / `async with atrace(...)` / `@traced_fn` | ✅ |
| Cost tracking (Python + CLI) | ✅ |
| Replay (Python + CLI) | ✅ |
| Pricing estimation (30+ models) | ✅ |
| Budget tracker with kill-on-exceeded | ✅ |
| Cost anomaly detection | ✅ |
| Trace redaction (PII / custom patterns) | ✅ |
| Trace diff & compare | ✅ |
| TraceQuery (filter / aggregate) | ✅ |
| Runtime metrics (Prometheus-style) | ✅ |
| Hierarchical config (`lumen.toml` / env / builder) | ✅ |
| `lumen.Agent.run()` direct execution | ⏳ v0.3 |
| Web dashboard (`lumen up`) | ⏳ v0.3 |
| `pytest-lumen` plugin | ⏳ v0.3 |
| CrewAI / AutoGen integration | ⏳ Planned |

## Install

```bash
# Python SDK (primary interface)
pip install lumen-ai

# With LangGraph integration
pip install "lumen-ai[langgraph]"

# CLI (Rust, optional)
cargo install lumen-cli
```

## License

MIT
