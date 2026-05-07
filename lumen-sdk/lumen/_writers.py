"""Pluggable trace writer backends.

Implements the TraceWriter protocol with concrete backends:
  - FileTraceWriter: JSON-per-file on local disk (current default behavior)
  - NullWriter: Discards traces (for sampling-dropped or testing)
  - CompositeWriter: Fan-out to multiple writers

The FileTraceWriter extracts the atomic-write logic that was previously
inlined in _context.py, making it reusable and testable in isolation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("lumen")


class FileTraceWriter:
    """Write each trace as a JSON file with atomic rename.

    File layout: ``{trace_dir}/{trace_id}.json``

    Implements the TraceWriter protocol. Never raises — logs warnings
    on failure so tracing issues never crash user code.
    """

    __slots__ = ("_trace_dir",)

    def __init__(self, trace_dir: str = "./traces") -> None:
        self._trace_dir = trace_dir

    @property
    def trace_dir(self) -> str:
        return self._trace_dir

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        """Write a single trace to disk atomically."""
        from lumen._errors import warn_trace_write

        trace_id = trace_data.get("trace_id", "unknown")

        try:
            trace_dir = Path(self._trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            warn_trace_write(
                f"Cannot create trace directory '{self._trace_dir}': {exc}. "
                f"Check permissions or set a writable path via LUMEN_TRACE_DIR.",
                trace_id=trace_id,
            )
            return

        path = trace_dir / f"{trace_id}.json"
        tmp_path = trace_dir / f".{trace_id}.json.tmp"

        try:
            json_bytes = json.dumps(trace_data, indent=2, default=str)
        except (TypeError, ValueError, OverflowError) as exc:
            warn_trace_write(
                f"Trace data contains non-serializable values: {exc}. "
                f"Retrying with lossy serialization.",
                trace_id=trace_id,
            )
            try:
                json_bytes = json.dumps(trace_data, indent=2, default=_safe_default)
            except Exception as exc2:
                warn_trace_write(
                    f"Trace serialization failed completely: {exc2}. Trace lost.",
                    trace_id=trace_id,
                )
                return

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json_bytes)
        except OSError as exc:
            warn_trace_write(
                f"Cannot write trace file '{tmp_path}': {exc}. "
                f"Check disk space and permissions.",
                trace_id=trace_id,
            )
            return

        try:
            os.replace(tmp_path, path)
        except OSError as exc:
            warn_trace_write(
                f"Atomic rename failed for trace '{trace_id}': {exc}. "
                f"Temp file may remain at '{tmp_path}'.",
                trace_id=trace_id,
            )


class NullWriter:
    """Discards all traces. Used for sampled-out requests and testing.

    Implements the TraceWriter protocol.
    """

    __slots__ = ()

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        pass


class CompositeWriter:
    """Fan-out writer that delegates to multiple backends.

    Each backend is called independently — one failure does not
    block the others. Implements the TraceWriter protocol.

    Example::

        writer = CompositeWriter([
            FileTraceWriter("./traces"),
            custom_s3_writer,
        ])
    """

    __slots__ = ("_writers",)

    def __init__(self, writers: list[Any]) -> None:
        self._writers = list(writers)

    @property
    def writers(self) -> list[Any]:
        return list(self._writers)

    def write_trace(self, trace_data: dict[str, Any]) -> None:
        for w in self._writers:
            try:
                w.write_trace(trace_data)
            except Exception:
                trace_id = trace_data.get("trace_id", "unknown")
                logger.debug(
                    "CompositeWriter: backend %r failed for trace %s",
                    type(w).__name__, trace_id, exc_info=True,
                )


def _safe_default(obj: Any) -> str:
    """Fallback serializer for non-JSON-serializable objects."""
    try:
        return repr(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"
