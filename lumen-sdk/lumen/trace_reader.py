"""Load AgentTrace JSON files from a trace directory.

Mirrors the Rust trace_reader in lumen-core. Reads {trace_id}.json files,
skips non-trace and temp files, returns traces sorted newest-first.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TokenUsage:
    """Token usage for an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TraceStep:
    """A single step within an agent execution trace."""

    step_num: int = 0
    step_type: str = ""  # "LlmCall", "ToolCall", "Checkpoint", "Error"
    started_at_ms: int = 0
    duration_ms: int = 0
    tokens: TokenUsage | None = None
    metadata: list[tuple[str, str]] = field(default_factory=list)

    # LlmCall fields
    model: str | None = None
    finish_reason: str | None = None

    # ToolCall fields
    tool_name: str | None = None
    tool_success: bool | None = None

    # Error fields
    error_message: str | None = None

    # Checkpoint fields
    checkpoint_iteration: int | None = None


@dataclass
class AgentTrace:
    """Complete trace of a single agent execution."""

    trace_id: str = ""
    agent_id: str = ""
    task_id: int = 0
    steps: list[TraceStep] = field(default_factory=list)
    status: str = "Running"  # "Running", "Completed", "Failed"
    failure_reason: str | None = None
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost_usd: float = 0.0
    started_at_ms: int = 0
    completed_at_ms: int | None = None

    # v0.2 optional fields
    user_id: str = ""
    session_id: str = ""
    project: str = ""
    environment: str = ""


def _parse_step(raw: dict) -> TraceStep:
    """Parse a single trace step from JSON."""
    step = TraceStep(
        step_num=raw.get("step_num", 0),
        started_at_ms=raw.get("started_at_ms", 0),
        duration_ms=raw.get("duration_ms", 0),
    )

    tokens_raw = raw.get("tokens")
    if tokens_raw is not None:
        step.tokens = TokenUsage(
            prompt_tokens=tokens_raw.get("prompt_tokens", 0),
            completion_tokens=tokens_raw.get("completion_tokens", 0),
        )

    step_type = raw.get("step_type", {})
    if isinstance(step_type, dict):
        if "LlmCall" in step_type:
            step.step_type = "LlmCall"
            step.model = step_type["LlmCall"].get("model", "")
            step.finish_reason = step_type["LlmCall"].get("finish_reason", "")
        elif "ToolCall" in step_type:
            step.step_type = "ToolCall"
            step.tool_name = step_type["ToolCall"].get("tool_name", "")
            step.tool_success = step_type["ToolCall"].get("success", True)
        elif "Checkpoint" in step_type:
            step.step_type = "Checkpoint"
            step.checkpoint_iteration = step_type["Checkpoint"].get("iteration", 0)
        elif "Error" in step_type:
            step.step_type = "Error"
            step.error_message = step_type["Error"].get("message", "")

    return step


def _parse_trace(raw: dict) -> AgentTrace:
    """Parse an AgentTrace from JSON dict."""
    trace = AgentTrace(
        trace_id=raw.get("trace_id", ""),
        agent_id=raw.get("agent_id", ""),
        task_id=raw.get("task_id", 0),
        total_cost_usd=raw.get("total_cost_usd", 0.0),
        started_at_ms=raw.get("started_at_ms", 0),
        completed_at_ms=raw.get("completed_at_ms"),
    )

    # v0.2 optional fields
    trace.user_id = raw.get("user_id", "")
    trace.session_id = raw.get("session_id", "")
    trace.project = raw.get("project", "")
    trace.environment = raw.get("environment", "")

    # Parse status
    status = raw.get("status", "Running")
    if isinstance(status, str):
        trace.status = status
    elif isinstance(status, dict) and "Failed" in status:
        trace.status = "Failed"
        trace.failure_reason = status["Failed"]

    # Parse total_tokens
    total_tok = raw.get("total_tokens", {})
    if total_tok:
        trace.total_tokens = TokenUsage(
            prompt_tokens=total_tok.get("prompt_tokens", 0),
            completion_tokens=total_tok.get("completion_tokens", 0),
        )

    # Parse steps
    for step_raw in raw.get("steps", []):
        trace.steps.append(_parse_step(step_raw))

    return trace


def load_traces(trace_dir: str | Path) -> list[AgentTrace]:
    """Load all agent traces from a directory of {trace_id}.json files.

    Returns traces sorted newest-first by started_at_ms.
    Skips dotfiles, .tmp files, and non-parseable JSON.
    Returns empty list if directory doesn't exist.
    """
    trace_dir = Path(trace_dir)
    if not trace_dir.exists():
        return []

    traces: list[AgentTrace] = []

    try:
        entries = os.listdir(trace_dir)
    except OSError:
        return []

    for name in entries:
        # Skip dotfiles (temp files from atomic writes)
        if name.startswith("."):
            continue
        # Only read .json files
        if not name.endswith(".json"):
            continue
        # Skip .tmp files
        if name.endswith(".tmp"):
            continue

        path = trace_dir / name
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        # Must have trace_id to be a valid trace
        if "trace_id" not in raw:
            continue

        traces.append(_parse_trace(raw))

    # Sort newest first
    traces.sort(key=lambda t: t.started_at_ms, reverse=True)
    return traces
