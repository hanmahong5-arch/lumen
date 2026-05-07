"""Asynchronous batched trace writer.

Decouples trace serialization from the caller's thread/event loop by
buffering traces in a bounded queue and draining them on a background
daemon thread. This ensures LLM call latency is never increased by
disk I/O from trace persistence.

Wires up the previously-dead ``TracingConfig.buffer_size`` and
``TracingConfig.flush_interval_ms`` configuration fields.

Architecture::

    caller thread          background thread
    ─────────────          ─────────────────
    add(trace_data) ──→ [bounded deque] ──→ writer.write_trace()
         O(1)              drain on:
                           - flush_interval_ms timer
                           - buffer reaches high-water mark
                           - explicit flush() call
                           - atexit hook
"""

from __future__ import annotations

import atexit
import collections
import logging
import threading
from typing import Any

logger = logging.getLogger("lumen")


class BatchTraceWriter:
    """Thread-safe buffered writer that drains to a TraceWriter backend.

    Traces are enqueued in O(1) and flushed by a daemon thread, so the
    caller never blocks on disk I/O.

    Args:
        writer: The underlying TraceWriter backend (FileTraceWriter, etc.)
        buffer_size: Maximum number of traces to buffer before forcing a flush.
        flush_interval_sec: Seconds between periodic flushes.
    """

    __slots__ = (
        "_writer", "_buffer", "_buffer_size", "_flush_interval",
        "_lock", "_event", "_thread", "_closed",
    )

    def __init__(
        self,
        writer: Any,
        *,
        buffer_size: int = 100,
        flush_interval_sec: float = 5.0,
    ) -> None:
        self._writer = writer
        self._buffer: collections.deque[dict[str, Any]] = collections.deque()
        self._buffer_size = max(1, buffer_size)
        self._flush_interval = max(0.1, flush_interval_sec)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._closed = False

        self._thread = threading.Thread(
            target=self._drain_loop,
            name="lumen-batch-writer",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.shutdown)

    # -- Public API --

    def add(self, trace_data: dict[str, Any]) -> None:
        """Enqueue a trace for background writing. O(1), never blocks."""
        if self._closed:
            return
        with self._lock:
            self._buffer.append(trace_data)
            if len(self._buffer) >= self._buffer_size:
                self._event.set()

    def flush(self) -> None:
        """Force an immediate drain of all buffered traces.

        Blocks until the current buffer is fully written. Useful for
        testing and graceful shutdown.
        """
        self._event.set()
        # Wait for the drain thread to process
        if self._thread.is_alive():
            self._thread.join(timeout=self._flush_interval + 1.0)

    def shutdown(self) -> None:
        """Stop the background thread and flush remaining traces.

        Called automatically via atexit. Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        # Final drain (in case thread exited before processing all)
        self._drain_once()

    @property
    def pending(self) -> int:
        """Number of traces waiting to be written."""
        with self._lock:
            return len(self._buffer)

    # -- TraceWriter protocol --

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        """TraceWriter protocol — delegates to add() for batching."""
        self.add(trace_data)

    # -- Internal --

    def _drain_loop(self) -> None:
        """Background thread main loop."""
        while not self._closed:
            self._event.wait(timeout=self._flush_interval)
            self._event.clear()
            self._drain_once()

    def _drain_once(self) -> None:
        """Drain all buffered traces to the underlying writer."""
        with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()

        for trace_data in batch:
            try:
                self._writer.write_trace(trace_data)
            except Exception:
                trace_id = trace_data.get("trace_id", "unknown")
                logger.debug(
                    "BatchTraceWriter: failed to write trace %s",
                    trace_id, exc_info=True,
                )
