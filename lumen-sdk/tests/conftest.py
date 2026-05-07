"""Shared test fixtures for Lumen SDK tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample trace JSON files."""
    # Successful trace (newer)
    trace1 = {
        "trace_id": "trace_abc123",
        "agent_id": "research-agent",
        "task_id": 42,
        "steps": [
            {
                "step_num": 0,
                "step_type": {"LlmCall": {"model": "claude-sonnet-4-6", "finish_reason": "tool_use"}},
                "started_at_ms": 1710000000000,
                "duration_ms": 1200,
                "tokens": {"prompt_tokens": 500, "completion_tokens": 100},
                "metadata": [],
            },
            {
                "step_num": 1,
                "step_type": {"ToolCall": {"tool_name": "search_web", "success": True}},
                "started_at_ms": 1710000001200,
                "duration_ms": 800,
                "tokens": None,
                "metadata": [],
            },
            {
                "step_num": 2,
                "step_type": {"LlmCall": {"model": "claude-sonnet-4-6", "finish_reason": "tool_use"}},
                "started_at_ms": 1710000002000,
                "duration_ms": 900,
                "tokens": {"prompt_tokens": 800, "completion_tokens": 150},
                "metadata": [],
            },
            {
                "step_num": 3,
                "step_type": {"ToolCall": {"tool_name": "read_url", "success": True}},
                "started_at_ms": 1710000002900,
                "duration_ms": 500,
                "tokens": None,
                "metadata": [],
            },
            {
                "step_num": 4,
                "step_type": {"LlmCall": {"model": "claude-sonnet-4-6", "finish_reason": "end_turn"}},
                "started_at_ms": 1710000003400,
                "duration_ms": 1500,
                "tokens": {"prompt_tokens": 1200, "completion_tokens": 300},
                "metadata": [],
            },
        ],
        "status": "Completed",
        "total_tokens": {"prompt_tokens": 2500, "completion_tokens": 550},
        "total_cost_usd": 0.0,  # Intentionally zero to test estimation
        "started_at_ms": 1710000000000,
        "completed_at_ms": 1710000004900,
    }

    # Failed trace (older, with gpt-4o model)
    trace2 = {
        "trace_id": "trace_def456",
        "agent_id": "summary-agent",
        "task_id": 99,
        "steps": [
            {
                "step_num": 0,
                "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "tool_use"}},
                "started_at_ms": 1709990000000,
                "duration_ms": 2000,
                "tokens": {"prompt_tokens": 1000, "completion_tokens": 200},
                "metadata": [],
            },
            {
                "step_num": 1,
                "step_type": {"ToolCall": {"tool_name": "run_code", "success": False}},
                "started_at_ms": 1709990002000,
                "duration_ms": 5000,
                "tokens": None,
                "metadata": [],
            },
            {
                "step_num": 2,
                "step_type": {"Error": {"message": "tool timeout after 5000ms"}},
                "started_at_ms": 1709990007000,
                "duration_ms": 0,
                "tokens": None,
                "metadata": [],
            },
        ],
        "status": {"Failed": "tool timeout"},
        "total_tokens": {"prompt_tokens": 1000, "completion_tokens": 200},
        "total_cost_usd": 0.0,
        "started_at_ms": 1709990000000,
        "completed_at_ms": 1709990007000,
    }

    (tmp_path / "trace_abc123.json").write_text(json.dumps(trace1))
    (tmp_path / "trace_def456.json").write_text(json.dumps(trace2))

    # Non-trace JSON (should be skipped)
    (tmp_path / "config.json").write_text('{"not": "a trace"}')
    # Non-JSON file (should be skipped)
    (tmp_path / "notes.txt").write_text("some notes")

    return tmp_path
