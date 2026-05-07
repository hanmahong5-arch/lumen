"""Composable trace processing middleware.

Transforms trace data through a chain of processors before writing.
Each middleware can modify, enrich, or drop traces.

Built-in middleware:
  - RedactionMiddleware: Apply PII redaction patterns
  - ContentStripMiddleware: Strip verbose content from metadata
  - SizeLimitMiddleware: Truncate oversized trace data
  - TimestampMiddleware: Ensure ISO 8601 timestamps

Users can add custom middleware via instrument() or config.

Example::

    from lumen._middleware import MiddlewareChain, RedactionMiddleware

    chain = MiddlewareChain([
        RedactionMiddleware(redaction_config),
        custom_enricher,
    ])
    result = chain.process(trace_data)
    if result is not None:
        writer.write_trace(result)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("lumen")


@runtime_checkable
class TraceMiddleware(Protocol):
    """Protocol for trace processing middleware.

    Return the (possibly modified) trace data, or None to drop the trace.
    """

    def process(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        ...


class MiddlewareChain:
    """Executes a chain of middleware in order.

    If any middleware returns None, the trace is dropped and
    subsequent middleware are not called.
    """

    __slots__ = ("_middlewares",)

    def __init__(self, middlewares: list[Any] | None = None) -> None:
        self._middlewares: list[Any] = list(middlewares) if middlewares else []

    def add(self, middleware: Any) -> None:
        """Append a middleware to the chain."""
        self._middlewares.append(middleware)

    def process(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        """Run trace data through all middleware. Returns None if dropped."""
        data: dict[str, Any] | None = trace_data
        for mw in self._middlewares:
            if data is None:
                return None
            try:
                data = mw.process(data)
            except Exception:
                logger.debug(
                    "Middleware %r failed, passing trace through",
                    type(mw).__name__, exc_info=True,
                )
        return data

    def __len__(self) -> int:
        return len(self._middlewares)


class RedactionMiddleware:
    """Apply PII redaction patterns to trace steps.

    Redacts error messages and string values in metadata.
    """

    __slots__ = ("_redaction_config",)

    def __init__(self, redaction_config: Any) -> None:
        self._redaction_config = redaction_config

    def process(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        if not self._redaction_config.enabled:
            return trace_data

        try:
            steps = json.loads(json.dumps(trace_data.get("steps", []), default=str))
        except (TypeError, ValueError):
            return trace_data

        for step in steps:
            step_type = step.get("step_type", {})
            if isinstance(step_type, dict) and "Error" in step_type:
                msg = step_type["Error"].get("message", "")
                step_type["Error"]["message"] = self._redaction_config.redact(msg)

            metadata = step.get("metadata", [])
            if metadata:
                step["metadata"] = [
                    [k, self._redaction_config.redact(v) if isinstance(v, str) else v]
                    for k, v in metadata
                ]

        trace_data = dict(trace_data)
        trace_data["steps"] = steps
        return trace_data


class ContentStripMiddleware:
    """Strip verbose content from metadata, keeping only structure."""

    __slots__ = ()

    def process(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        try:
            steps = json.loads(json.dumps(trace_data.get("steps", []), default=str))
        except (TypeError, ValueError):
            return trace_data

        for step in steps:
            meta = step.get("metadata", [])
            step["metadata"] = [(k, "[stripped]") for k, _v in meta] if meta else []

        trace_data = dict(trace_data)
        trace_data["steps"] = steps
        return trace_data


class SizeLimitMiddleware:
    """Truncate trace data that exceeds a size limit."""

    __slots__ = ("_max_bytes",)

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes

    def process(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        try:
            size = len(json.dumps(trace_data, default=str))
        except (TypeError, ValueError):
            return trace_data

        if size <= self._max_bytes:
            return trace_data

        # Progressively strip content to reduce size
        trace_data = dict(trace_data)
        steps = trace_data.get("steps", [])
        for step in steps:
            meta = step.get("metadata", [])
            step["metadata"] = [(k, "[truncated:size_limit]") for k, _v in meta] if meta else []

        return trace_data


class TimestampMiddleware:
    """Add ISO 8601 timestamps to trace data for human readability."""

    __slots__ = ()

    def process(self, trace_data: dict[str, Any]) -> dict[str, Any] | None:
        import datetime

        trace_data = dict(trace_data)
        started = trace_data.get("started_at_ms")
        if isinstance(started, (int, float)) and started > 0:
            trace_data["started_at"] = datetime.datetime.fromtimestamp(
                started / 1000, tz=datetime.timezone.utc,
            ).isoformat()

        completed = trace_data.get("completed_at_ms")
        if isinstance(completed, (int, float)) and completed > 0:
            trace_data["completed_at"] = datetime.datetime.fromtimestamp(
                completed / 1000, tz=datetime.timezone.utc,
            ).isoformat()

        return trace_data
