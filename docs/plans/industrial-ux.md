# Industrial UX roadmap — status

Service plan for the Trust → First Kiss → Habit → Power-Up → deep-bet sequence.
Strategic narrative lives in the Cursor plan; this file is the **shipped checklist**.

**Product veto** (existence / scope): [necessity.md](./necessity.md) — ship only what
serves the black-box formula (money × honest narrative × local-complete + optional Kova).
If a candidate fails that page, do not pick it here.

Audience: independent LangGraph first, Kova Connected later. Repo stays a **sibling**
of `2b-svc-kova` (vendored AgentTrace, not a path dependency).

## Shipped

| Phase | Outcome | Evidence |
|---|---|---|
| Trust | PR CI (Rust + pytest), honest public API (no fake Agent/OTLP), install script + INSTALL.md, version alignment | `.github/workflows/ci.yaml`, `docs/INSTALL.md`, `lumen.agent` stubs not exported |
| First Kiss | Offline `lumen demo`, README demo-first, CLI next-step copy | `lumen-cli` demo path, root README |
| Habit | Cost narrative (`--format md`), resume ritual, shareable lifecycle HTML | `docs/RESUME.md`, `lumen cost --format md`, `lumen export` |
| Power-Up | Core vs Connected docs, Forge `FORGE_LUMEN_BIN` note | `docs/CONNECTED.md` |
| Forge embed | `/demo` auto-open lifecycle + gate Closed Beta when `lifecycle_ready` | `2b-bs-forge` `demo-lifecycle-gate.ts` + `/demo` page |
| Deep bet | LangGraph `resume_hint`, Rust↔Python cost/replay parity fixtures, real OTLP/HTTP JSON export | `behavior_expectations.json`, `lumen/_otlp.py`, PR #10–#13 |

## Deliberately not shipped

- Nesting this repo under Kova / merging crates
- Direct `lumen.Agent.run()` as a second runtime (Kova owns WAL agents)
- CrewAI / AutoGen / `pytest-lumen` / PyPI publish (until release gate exists)
- OTLP protobuf / gRPC (HTTP JSON is enough for collectors today)

## Install tax (in progress)

- CLI: `docs/install-cli.sh` / `install-cli.ps1` resolve latest `lumen-cli-v*`;
  cut tags per [RELEASE.md](../RELEASE.md).
- PyPI: Trusted Publishing workflow ready; **blocked on one-time pypi.org
  pending publisher** + first `lumen-sdk-v*` tag (SDK still install-from-git).

## Connected live sync

Dashboard with `--kova-url` background-pulls traces every 5s (override
`--sync-interval` / `--no-sync`). Headless: `lumen pull --watch`. Kova has no
trace SSE yet — pull-based is the honest real-time path. Shipped in
`lumen-cli-v0.3.2` binaries.

## Next candidates (pick one per cycle)

1. **First PyPI cut** — GitHub env `pypi` exists; finish one-time pypi.org Pending publisher, then tag `lumen-sdk-v*`
2. ~~**Step-shape replay alignment**~~ — done (shared `replay_decisions` in behavior_expectations)
3. **Kova-side trace push/SSE** — only if pull interval is proven insufficient
4. **STAGE true-LLM Demo Day** — after backend image with baked `FORGE_LUMEN_BIN` deploys (see kova runs ledger 2026-11-06)

Rule: measure progress by install → 5-minute incident HTML → shareable artifact, not feature-list length.
