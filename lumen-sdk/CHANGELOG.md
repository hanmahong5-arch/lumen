# Changelog

All notable changes to the `lumen-ai` Python SDK are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before 0.2.0 predate this repository's public git history and are not
itemized here.

## [0.2.1] - Unreleased

### Added
- `AgentTrace.parent_trace_id` (`Optional[str]`), mirroring the kova-types
  trace schema, so child/parent trace relationships survive the SDK parser.
- `AgentTrace.schema_version` (`int`, defaults to `0` for legacy JSON without
  the key), mirroring kova-types `TRACE_SCHEMA_VERSION`.
- Shared Rust↔Python drift-guard fixtures (`trace_examples.json`) read by both
  the vendored Rust and Python trace parsers, including a `failed_trace`
  example with `"status": {"Failed": "<reason>"}` so the
  reason-carrying status variant cannot silently break the parser.

### Fixed
- Step `metadata` was declared on the dataclass but never extracted during
  parsing, dropping crash-recovery markers; `_parse_trace` now reads it.
- `duration_ms` is clamped to ≥1 so traces that complete within the same
  millisecond no longer surface as a misleading `0ms`.

## [0.2.0] - 2026-05-07

### Added
- Initial public release: tracing instrumentation (`instrument()`, `trace`,
  `atrace`, `traced_fn`), trace replay, per-call cost tracking and pricing
  tables, budget guard, trace query API (`TraceQuery`), redaction, anomaly
  detection, and integrations for OpenAI, Anthropic, and LangGraph
  (checkpoint-based crash recovery).
