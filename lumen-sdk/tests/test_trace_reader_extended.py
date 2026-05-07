"""Extended tests for the trace reader — edge cases and v0.2 fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.trace_reader import (
    AgentTrace,
    TokenUsage,
    TraceStep,
    _parse_step,
    _parse_trace,
    load_traces,
)


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_defaults(self) -> None:
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total() == 0

    def test_total(self) -> None:
        t = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert t.total() == 150

    def test_large_values(self) -> None:
        t = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        assert t.total() == 1_500_000


class TestParseStep:
    """Test _parse_step with various step types."""

    def test_llm_call(self) -> None:
        raw = {
            "step_num": 0,
            "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "stop"}},
            "started_at_ms": 1000,
            "duration_ms": 500,
            "tokens": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        step = _parse_step(raw)
        assert step.step_type == "LlmCall"
        assert step.model == "gpt-4o"
        assert step.finish_reason == "stop"
        assert step.tokens is not None
        assert step.tokens.total() == 150

    def test_tool_call(self) -> None:
        raw = {
            "step_num": 1,
            "step_type": {"ToolCall": {"tool_name": "search", "success": True}},
            "started_at_ms": 2000,
            "duration_ms": 300,
        }
        step = _parse_step(raw)
        assert step.step_type == "ToolCall"
        assert step.tool_name == "search"
        assert step.tool_success is True
        assert step.tokens is None

    def test_error(self) -> None:
        raw = {
            "step_num": 2,
            "step_type": {"Error": {"message": "timeout"}},
            "started_at_ms": 3000,
            "duration_ms": 0,
        }
        step = _parse_step(raw)
        assert step.step_type == "Error"
        assert step.error_message == "timeout"

    def test_checkpoint(self) -> None:
        raw = {
            "step_num": 3,
            "step_type": {"Checkpoint": {"iteration": 5}},
            "started_at_ms": 4000,
            "duration_ms": 10,
        }
        step = _parse_step(raw)
        assert step.step_type == "Checkpoint"
        assert step.checkpoint_iteration == 5

    def test_missing_fields(self) -> None:
        raw = {}
        step = _parse_step(raw)
        assert step.step_num == 0
        assert step.step_type == ""
        assert step.started_at_ms == 0

    def test_unknown_step_type(self) -> None:
        raw = {"step_type": {"CustomType": {"data": "foo"}}}
        step = _parse_step(raw)
        assert step.step_type == ""

    def test_no_tokens(self) -> None:
        raw = {
            "step_type": {"LlmCall": {"model": "gpt-4o"}},
            "tokens": None,
        }
        step = _parse_step(raw)
        assert step.tokens is None


class TestParseTrace:
    """Test _parse_trace with various trace structures."""

    def test_completed_trace(self) -> None:
        raw = {
            "trace_id": "abc",
            "agent_id": "agent-1",
            "task_id": 42,
            "status": "Completed",
            "total_tokens": {"prompt_tokens": 200, "completion_tokens": 100},
            "total_cost_usd": 0.05,
            "started_at_ms": 1000,
            "completed_at_ms": 5000,
            "steps": [],
        }
        trace = _parse_trace(raw)
        assert trace.trace_id == "abc"
        assert trace.agent_id == "agent-1"
        assert trace.task_id == 42
        assert trace.status == "Completed"
        assert trace.total_cost_usd == 0.05

    def test_failed_trace(self) -> None:
        raw = {
            "trace_id": "def",
            "agent_id": "a",
            "status": {"Failed": "connection lost"},
            "total_tokens": {},
            "steps": [],
        }
        trace = _parse_trace(raw)
        assert trace.status == "Failed"
        assert trace.failure_reason == "connection lost"

    def test_v02_fields(self) -> None:
        raw = {
            "trace_id": "v2",
            "agent_id": "a",
            "status": "Completed",
            "user_id": "u-1",
            "session_id": "s-2",
            "project": "proj",
            "environment": "prod",
            "total_tokens": {},
            "steps": [],
        }
        trace = _parse_trace(raw)
        assert trace.user_id == "u-1"
        assert trace.session_id == "s-2"
        assert trace.project == "proj"
        assert trace.environment == "prod"

    def test_missing_v02_fields(self) -> None:
        raw = {
            "trace_id": "old",
            "agent_id": "a",
            "status": "Completed",
            "total_tokens": {},
            "steps": [],
        }
        trace = _parse_trace(raw)
        assert trace.user_id == ""
        assert trace.session_id == ""
        assert trace.project == ""
        assert trace.environment == ""

    def test_steps_parsed(self) -> None:
        raw = {
            "trace_id": "s1",
            "agent_id": "a",
            "status": "Completed",
            "total_tokens": {},
            "steps": [
                {"step_num": 0, "step_type": {"LlmCall": {"model": "gpt-4o"}}},
                {"step_num": 1, "step_type": {"ToolCall": {"tool_name": "search"}}},
            ],
        }
        trace = _parse_trace(raw)
        assert len(trace.steps) == 2
        assert trace.steps[0].step_type == "LlmCall"
        assert trace.steps[1].step_type == "ToolCall"


class TestLoadTraces:
    """Test trace loading from directory."""

    def test_sorted_newest_first(self, trace_dir: Path) -> None:
        for i, ts in enumerate([1000, 3000, 2000]):
            data = {
                "trace_id": f"t{i}",
                "agent_id": "a",
                "status": "Completed",
                "total_tokens": {},
                "started_at_ms": ts,
                "steps": [],
            }
            with open(trace_dir / f"t{i}.json", "w") as f:
                json.dump(data, f)

        traces = load_traces(str(trace_dir))
        assert [t.started_at_ms for t in traces] == [3000, 2000, 1000]

    def test_skips_invalid_json(self, trace_dir: Path) -> None:
        (trace_dir / "good.json").write_text(json.dumps({
            "trace_id": "g1", "agent_id": "a", "status": "Completed",
            "total_tokens": {}, "steps": [],
        }))
        (trace_dir / "bad.json").write_text("not valid json{{{")
        traces = load_traces(str(trace_dir))
        assert len(traces) == 1

    def test_skips_non_json(self, trace_dir: Path) -> None:
        (trace_dir / "good.json").write_text(json.dumps({
            "trace_id": "g1", "agent_id": "a", "status": "Completed",
            "total_tokens": {}, "steps": [],
        }))
        (trace_dir / "notes.txt").write_text("some notes")
        (trace_dir / "data.csv").write_text("a,b,c")
        traces = load_traces(str(trace_dir))
        assert len(traces) == 1

    def test_skips_json_without_trace_id(self, trace_dir: Path) -> None:
        (trace_dir / "good.json").write_text(json.dumps({
            "trace_id": "g1", "agent_id": "a", "status": "Completed",
            "total_tokens": {}, "steps": [],
        }))
        (trace_dir / "config.json").write_text(json.dumps({"key": "value"}))
        traces = load_traces(str(trace_dir))
        assert len(traces) == 1

    def test_skips_dotfiles(self, trace_dir: Path) -> None:
        (trace_dir / ".tmp.json").write_text(json.dumps({
            "trace_id": "tmp", "agent_id": "a", "status": "Completed",
            "total_tokens": {}, "steps": [],
        }))
        traces = load_traces(str(trace_dir))
        assert len(traces) == 0

    def test_empty_directory(self, trace_dir: Path) -> None:
        traces = load_traces(str(trace_dir))
        assert traces == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        traces = load_traces(str(tmp_path / "nope"))
        assert traces == []

    def test_many_traces(self, trace_dir: Path) -> None:
        for i in range(50):
            data = {
                "trace_id": f"trace-{i:04d}",
                "agent_id": f"agent-{i % 5}",
                "status": "Completed" if i % 3 else "Failed",
                "total_tokens": {"prompt_tokens": i * 10, "completion_tokens": i * 5},
                "total_cost_usd": i * 0.001,
                "started_at_ms": 1000 + i,
                "steps": [{"step_num": 0, "step_type": {"LlmCall": {"model": "gpt-4o"}}}],
            }
            with open(trace_dir / f"trace-{i:04d}.json", "w") as f:
                json.dump(data, f)

        traces = load_traces(str(trace_dir))
        assert len(traces) == 50
        # Sorted newest first
        assert traces[0].started_at_ms > traces[-1].started_at_ms
