"""Protocol interfaces for pluggable Lumen backends.

Defines structural typing contracts that allow alternative implementations
without requiring inheritance. Any object satisfying the protocol's shape
can be used — duck typing with static verification.

Protocols defined:
  - TraceWriter: Write trace data to storage
  - TraceReader: Read/query trace data from storage
  - PricingProvider: Estimate LLM costs
  - RedactionProvider: Sanitize sensitive data
  - MetricsCollector: Record and query observability metrics

Design rationale:
  - Protocol over ABC: No runtime overhead, supports duck typing, works
    with existing classes without modification
  - runtime_checkable: Enables isinstance() checks for error messages
  - Minimal surface: Each protocol has 1-3 methods. Compose, don't inherit.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class TraceWriter(Protocol):
    """Writes trace data to a storage backend.

    Default implementation: JSON file writer in _context.py.
    Alternatives: SQLite, PostgreSQL, S3, OTLP exporter.
    """

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        """Persist a complete trace.

        Args:
            trace_data: Serializable dict in Lumen trace format.
        """
        ...


@runtime_checkable
class TraceReader(Protocol):
    """Reads trace data from a storage backend.

    Default implementation: JSON file reader in trace_reader.py.
    Alternatives: SQLite reader, API client, cached reader.
    """

    def load_traces(self) -> list[Any]:
        """Load all traces, newest first."""
        ...

    def get_trace(self, trace_id: str) -> Any | None:
        """Load a single trace by ID. Returns None if not found."""
        ...


@runtime_checkable
class PricingProvider(Protocol):
    """Estimates cost for an LLM call.

    Default implementation: Trie-based lookup in pricing.py.
    Alternatives: API-based pricing, custom rate cards.
    """

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Return estimated cost in USD."""
        ...


@runtime_checkable
class RedactionProvider(Protocol):
    """Sanitizes sensitive data in text.

    Default implementation: RedactionEngine in _redaction.py.
    Alternatives: External DLP service, Presidio.
    """

    def redact_text(self, text: str) -> str:
        """Return text with sensitive data replaced."""
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Collects runtime metrics for observability.

    Default implementation: MetricsWindow in _ring_buffer.py.
    Alternatives: Prometheus client, StatsD, OpenTelemetry metrics.
    """

    def record_latency(self, duration_ms: float) -> None:
        """Record an LLM call latency observation."""
        ...

    def record_cost(self, cost_usd: float) -> None:
        """Record a cost observation."""
        ...

    def record_tokens(self, prompt: int, completion: int) -> None:
        """Record token usage."""
        ...
