"""Tests for trace_reader module."""

from __future__ import annotations

import json
from pathlib import Path

from lumen.trace_reader import load_traces


def test_load_two_traces(trace_dir: Path) -> None:
    traces = load_traces(trace_dir)
    assert len(traces) == 2
    # Newest first
    assert traces[0].trace_id == "trace_abc123"
    assert traces[1].trace_id == "trace_def456"


def test_trace_fields(trace_dir: Path) -> None:
    traces = load_traces(trace_dir)
    t = traces[0]
    assert t.agent_id == "research-agent"
    assert t.task_id == 42
    assert t.status == "Completed"
    assert t.started_at_ms == 1710000000000
    assert t.completed_at_ms == 1710000004900
    assert t.total_tokens.prompt_tokens == 2500
    assert t.total_tokens.completion_tokens == 550


def test_steps_parsed(trace_dir: Path) -> None:
    traces = load_traces(trace_dir)
    steps = traces[0].steps
    assert len(steps) == 5

    # First step is LlmCall
    assert steps[0].step_type == "LlmCall"
    assert steps[0].model == "claude-sonnet-4-6"
    assert steps[0].finish_reason == "tool_use"
    assert steps[0].tokens is not None
    assert steps[0].tokens.prompt_tokens == 500

    # Second step is ToolCall
    assert steps[1].step_type == "ToolCall"
    assert steps[1].tool_name == "search_web"
    assert steps[1].tool_success is True


def test_failed_trace_status(trace_dir: Path) -> None:
    traces = load_traces(trace_dir)
    t = traces[1]
    assert t.status == "Failed"
    assert t.failure_reason == "tool timeout"


def test_error_step(trace_dir: Path) -> None:
    traces = load_traces(trace_dir)
    steps = traces[1].steps
    error_step = steps[2]
    assert error_step.step_type == "Error"
    assert error_step.error_message == "tool timeout after 5000ms"


def test_nonexistent_dir() -> None:
    traces = load_traces("/tmp/lumen_nonexistent_xyz_99999")
    assert traces == []


def test_empty_dir(tmp_path: Path) -> None:
    traces = load_traces(tmp_path)
    assert traces == []


def test_skips_non_trace_json(trace_dir: Path) -> None:
    # config.json doesn't have trace_id, should be skipped
    traces = load_traces(trace_dir)
    ids = [t.trace_id for t in traces]
    assert "config" not in str(ids)


def test_skips_dotfiles(trace_dir: Path) -> None:
    # Write a dotfile that looks like a trace
    (trace_dir / ".temp_trace.json").write_text(json.dumps({
        "trace_id": "hidden",
        "agent_id": "x",
        "task_id": 0,
        "steps": [],
        "status": "Completed",
        "total_tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        "total_cost_usd": 0.0,
        "started_at_ms": 0,
    }))
    traces = load_traces(trace_dir)
    ids = [t.trace_id for t in traces]
    assert "hidden" not in ids
