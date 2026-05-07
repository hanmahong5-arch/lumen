"""Context-aware trace session management for grouping LLM calls.

Uses ``contextvars.ContextVar`` instead of ``threading.local`` for correct
behavior with both threads and async (coroutines on the same thread get
independent context via ``copy_context()``).

Used by openai_tracer, anthropic_tracer, and instrument.py to coordinate
trace recording across multiple SDK calls within a single logical operation.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lumen._budget import BudgetTracker
    from lumen.config import LumenConfig

# ContextVar: works correctly with both threads and asyncio tasks
_active_session: contextvars.ContextVar[TraceSession | None] = contextvars.ContextVar(
    "lumen_active_session", default=None,
)


class TraceSession:
    """An active trace being recorded (one or more LLM/tool steps).

    Created either automatically (one per LLM call) or explicitly via
    the ``trace()`` context manager to group multiple calls.
    """

    def __init__(
        self,
        name: str,
        trace_dir: str,
        config: LumenConfig | None = None,
        *,
        user_id: str = "",
        session_id: str = "",
        budget_tracker: BudgetTracker | None = None,
        writer: Any = None,
        parent_trace_id: str = "",
    ) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.name = name
        self.trace_dir = trace_dir
        self.config = config
        self.user_id = user_id
        self.session_id = session_id
        self.parent_trace_id = parent_trace_id
        self.started_at_ms = _now_ms()
        self.steps: list[dict[str, Any]] = []
        self.step_num = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.status = "Running"
        self.failure_reason: str | None = None
        self._budget_tracker = budget_tracker
        self._writer = writer
        self._estimated_bytes = 0

    def add_llm_step(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
        duration_ms: int,
        started_at_ms: int,
        cost_usd: float = 0.0,
        input_messages: list[dict[str, Any]] | None = None,
        output_content: str | None = None,
    ) -> None:
        """Record an LLM call step.

        Args:
            model: Model name (e.g., "gpt-4o").
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            finish_reason: Why the LLM stopped (stop/tool_use/length).
            duration_ms: Call duration in milliseconds.
            started_at_ms: When the call started (epoch ms).
            cost_usd: Estimated cost in USD.
            input_messages: Raw input messages (captured in full mode).
            output_content: Raw output text (captured in full mode).
        """
        tokens_dict = None
        if prompt_tokens > 0 or completion_tokens > 0:
            tokens_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        # Budget enforcement — check per-run limit before recording
        if self._budget_tracker and cost_usd > 0:
            self._budget_tracker.check_per_run(cost_usd, self.trace_id)
            self._budget_tracker.record(cost_usd, self.trace_id)

        metadata: list[Any] = []
        if input_messages is not None:
            # Size limiting: skip content if trace is already large
            if not self._size_limit_reached():
                try:
                    serialized = json.dumps(input_messages, ensure_ascii=False, default=str)
                    self._estimated_bytes += len(serialized)
                    metadata.append(("input_messages", serialized))
                except (TypeError, ValueError):
                    metadata.append(("input_messages", "[serialization_error]"))
            else:
                metadata.append(("input_messages", "[truncated:size_limit]"))
        if output_content is not None:
            if not self._size_limit_reached():
                self._estimated_bytes += len(output_content)
                metadata.append(("output_content", output_content))
            else:
                metadata.append(("output_content", "[truncated:size_limit]"))

        span_id = uuid.uuid4().hex[:8]
        self.steps.append({
            "step_num": self.step_num,
            "span_id": span_id,
            "step_type": {"LlmCall": {"model": model, "finish_reason": finish_reason}},
            "started_at_ms": started_at_ms,
            "duration_ms": duration_ms,
            "tokens": tokens_dict,
            "metadata": metadata,
        })
        self.step_num += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost_usd

        # Record to runtime metrics (if initialized)
        _record_metrics(
            model=model,
            agent_name=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )

    def add_tool_step(
        self,
        *,
        tool_name: str,
        success: bool,
        duration_ms: int,
        started_at_ms: int,
    ) -> None:
        """Record a tool execution step."""
        span_id = uuid.uuid4().hex[:8]
        self.steps.append({
            "step_num": self.step_num,
            "span_id": span_id,
            "step_type": {"ToolCall": {"tool_name": tool_name, "success": success}},
            "started_at_ms": started_at_ms,
            "duration_ms": duration_ms,
            "tokens": None,
            "metadata": [],
        })
        self.step_num += 1

    def add_error_step(self, message: str) -> None:
        """Record an error step."""
        span_id = uuid.uuid4().hex[:8]
        self.steps.append({
            "step_num": self.step_num,
            "span_id": span_id,
            "step_type": {"Error": {"message": message}},
            "started_at_ms": _now_ms(),
            "duration_ms": 0,
            "tokens": None,
            "metadata": [],
        })
        self.step_num += 1
        self.status = "Failed"
        self.failure_reason = message

    def _size_limit_reached(self) -> bool:
        """Check if estimated trace size exceeds max_trace_size_mb."""
        if self.config is None:
            return False
        max_bytes = self.config.tracing.max_trace_size_mb * 1024 * 1024
        return self._estimated_bytes >= max_bytes

    def finalize(self) -> None:
        """Write the trace to disk and reset.

        Applies redaction, content stripping, and budget finalization
        before writing the trace JSON atomically.
        """
        if not self.steps:
            # Finish budget run even for empty sessions
            if self._budget_tracker:
                self._budget_tracker.finish_run(self.trace_id)
            return

        if self.status == "Running":
            self.status = "Completed"

        status: str | dict[str, str]
        if self.status == "Failed" and self.failure_reason:
            status = {"Failed": self.failure_reason}
        else:
            status = self.status

        # Apply redaction — use RedactionEngine if presets configured,
        # otherwise fall back to simple regex redaction
        steps = self.steps
        if self.config and self.config.tracing.redaction.enabled:
            steps = _redact_steps(steps, self.config.tracing.redaction)

        # Strip content if metadata_only or off mode
        content_mode = "full"
        if self.config:
            content_mode = self.config.tracing.content_capture
        if content_mode == "metadata_only":
            steps = _strip_content(steps)
        elif content_mode == "off":
            steps = _strip_content(steps)

        trace_data: dict[str, Any] = {
            "trace_id": self.trace_id,
            "agent_id": self.name,
            "task_id": 0,
            "steps": steps,
            "status": status,
            "total_tokens": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
            },
            "total_cost_usd": self.total_cost_usd,
            "started_at_ms": self.started_at_ms,
            "completed_at_ms": _now_ms(),
        }

        # Nested trace support — record parent for span tree reconstruction
        if self.parent_trace_id:
            trace_data["parent_trace_id"] = self.parent_trace_id

        # Optional session/user tracking
        if self.user_id:
            trace_data["user_id"] = self.user_id
        if self.session_id:
            trace_data["session_id"] = self.session_id
        if self.config:
            if self.config.project_name:
                trace_data["project"] = self.config.project_name
            if self.config.environment:
                trace_data["environment"] = self.config.environment

        # Finalize budget tracking for this run
        if self._budget_tracker:
            self._budget_tracker.finish_run(self.trace_id)

        # Write trace — use injected writer if available, else fall back
        if self._writer is not None:
            try:
                self._writer.write_trace(trace_data)
            except Exception:
                logger.debug("Writer failed for trace %s, falling back", self.trace_id, exc_info=True)
                _write_trace_safe(trace_data, self.trace_dir, self.trace_id)
        else:
            _write_trace_safe(trace_data, self.trace_dir, self.trace_id)

        # Emit TraceCompleted event and check for cost anomalies
        _emit_trace_completed(self)
        _check_anomaly(self)


def get_active_session() -> TraceSession | None:
    """Return the active trace session for this context.

    Works correctly with both threads and async tasks — each gets
    independent context via ``contextvars``.
    """
    return _active_session.get()


def set_active_session(session: TraceSession | None) -> None:
    """Set the active trace session for this context."""
    _active_session.set(session)


def should_sample(config: LumenConfig | None) -> bool:
    """Decide whether to trace this request based on sampling rate."""
    if config is None:
        return True
    if not config.tracing.enabled:
        return False
    rate = config.tracing.sampling_rate
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def _now_ms() -> int:
    return int(time.time() * 1000)


def _redact_steps(
    steps: list[dict[str, Any]],
    redaction: Any,
) -> list[dict[str, Any]]:
    """Return a deep copy of steps with sensitive data redacted.

    Redacts:
      - Error messages
      - All string values in metadata pairs
    """
    try:
        redacted = json.loads(json.dumps(steps, default=str))
    except (TypeError, ValueError):
        logger.debug("Cannot deep-copy steps for redaction, operating on originals")
        redacted = steps
    for step in redacted:
        step_type = step.get("step_type", {})
        if isinstance(step_type, dict):
            # Redact error messages
            if "Error" in step_type:
                msg = step_type["Error"].get("message", "")
                step_type["Error"]["message"] = redaction.redact(msg)

        # Redact metadata values (list of [key, value] pairs)
        metadata = step.get("metadata", [])
        if metadata:
            step["metadata"] = [
                [k, redaction.redact(v) if isinstance(v, str) else v]
                for k, v in metadata
            ]
    return redacted


def _strip_content(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip detailed content, keeping only structural metadata."""
    try:
        stripped = json.loads(json.dumps(steps, default=str))
    except (TypeError, ValueError):
        logger.debug("Cannot deep-copy steps for stripping, operating on originals")
        stripped = steps
    for step in stripped:
        # Keep step_type structure but remove verbose content
        meta = step.get("metadata", [])
        step["metadata"] = [
            (k, "[stripped]") for k, _v in meta
        ] if meta else []
    return stripped


logger = logging.getLogger("lumen")


def _write_trace_safe(
    trace_data: dict[str, Any],
    trace_dir_str: str,
    trace_id: str,
) -> None:
    """Write trace JSON atomically with comprehensive error handling.

    Handles: mkdir failure, JSON serialization errors (circular refs,
    non-serializable objects), disk full, permission denied, and
    atomic rename failure. Never raises — logs a warning on failure.
    """
    from lumen._errors import warn_trace_write

    try:
        trace_dir = Path(trace_dir_str)
        trace_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warn_trace_write(
            f"Cannot create trace directory '{trace_dir_str}': {exc}. "
            f"Check permissions or set a writable path via LUMEN_TRACE_DIR.",
            trace_id=trace_id,
        )
        return

    path = trace_dir / f"{trace_id}.json"
    tmp_path = trace_dir / f".{trace_id}.json.tmp"

    # Serialize — catch non-serializable objects gracefully
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

    # Write to temp file, then atomic rename
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


def _safe_default(obj: Any) -> str:
    """Fallback serializer for non-JSON-serializable objects."""
    try:
        return repr(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def _emit_trace_completed(session: TraceSession) -> None:
    """Emit a TraceCompleted event to the default event bus.

    Best-effort: never raises — event emission failure must not
    break trace persistence.
    """
    try:
        from lumen._event_bus import TraceCompleted, get_default_bus

        completed_at = _now_ms()
        event = TraceCompleted(
            trace_id=session.trace_id,
            agent_id=session.name,
            status=session.status,
            total_cost_usd=session.total_cost_usd,
            step_count=len(session.steps),
            duration_ms=completed_at - session.started_at_ms,
        )
        get_default_bus().emit(event)
    except Exception:
        logger.debug("Event emission failed for trace %s", session.trace_id, exc_info=True)


# Module-level anomaly detector — lazy initialized, thread-safe
_anomaly_detector = None
_anomaly_lock = threading.Lock()


def _check_anomaly(session: TraceSession) -> None:
    """Check if the trace cost is anomalously high.

    Uses a module-level AnomalyDetector with configurable thresholds
    from the session's config. Thread-safe initialization.
    Best-effort — never raises.
    """
    global _anomaly_detector  # noqa: PLW0603

    if session.total_cost_usd <= 0:
        return

    try:
        from lumen._anomaly import AnomalyDetector

        if _anomaly_detector is None:
            with _anomaly_lock:
                if _anomaly_detector is None:
                    multiplier = 2.0
                    min_cost = 0.01
                    if session.config:
                        multiplier = session.config.cost.anomaly_multiplier
                        min_cost = session.config.cost.anomaly_min_usd
                    _anomaly_detector = AnomalyDetector(
                        multiplier=multiplier,
                        min_cost_usd=min_cost,
                    )

        _anomaly_detector.check(
            trace_id=session.trace_id,
            agent_id=session.name,
            cost_usd=session.total_cost_usd,
        )
    except Exception:
        logger.debug("Anomaly detection failed for trace %s", session.trace_id, exc_info=True)


# ── Runtime metrics bridge ────────────────────────────────────────────────

_runtime_metrics: Any = None


def set_runtime_metrics(metrics: Any) -> None:
    """Set the global runtime metrics collector (called from instrument())."""
    global _runtime_metrics  # noqa: PLW0603
    _runtime_metrics = metrics


def get_runtime_metrics() -> Any:
    """Return the global runtime metrics collector, or None."""
    return _runtime_metrics


def _record_metrics(
    *,
    model: str,
    agent_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    duration_ms: int,
) -> None:
    """Feed an LLM call into the runtime metrics pipeline.

    Best-effort — never raises. Does nothing if metrics not initialized.
    """
    if _runtime_metrics is None:
        return
    try:
        _runtime_metrics.record(
            model=model,
            agent_name=agent_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.debug("Metrics recording failed", exc_info=True)
