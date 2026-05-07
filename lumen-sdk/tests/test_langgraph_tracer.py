"""Tests for LumenTracer callback handler.

Uses mock callback events — no real LLM or langchain-core required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from lumen.integrations.langgraph_tracer import LumenTracer


def _make_llm_response(
    model: str = "claude-sonnet-4-6",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    finish_reason: str = "stop",
    has_tool_calls: bool = False,
) -> MagicMock:
    """Create a mock LLM response."""
    response = MagicMock()
    response.llm_output = {
        "model_name": model,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    gen = MagicMock()
    gen.generation_info = {"finish_reason": finish_reason}

    msg = MagicMock()
    msg.usage_metadata = None
    msg.response_metadata = {}
    if has_tool_calls:
        msg.tool_calls = [{"name": "search", "args": {}}]
    else:
        msg.tool_calls = []
    gen.message = msg

    response.generations = [[gen]]
    return response


class TestLumenTracerBasic:
    """Tests that don't require langchain-core."""

    def test_trace_written_on_chain_end(self, tmp_path: Path) -> None:
        tracer = LumenTracer(trace_dir=str(tmp_path), agent_name="test-agent")

        # Simulate: chain_start → llm → tool → llm → chain_end
        tracer.on_chain_start({}, {})
        tracer.on_llm_start({}, ["hello"])
        tracer.on_llm_end(_make_llm_response(
            model="claude-sonnet-4-6",
            prompt_tokens=500,
            completion_tokens=100,
            has_tool_calls=True,
        ))
        tracer.on_tool_start({}, "search query")
        tracer.on_tool_end("3 results found", name="search_web")
        tracer.on_llm_start({}, ["process results"])
        tracer.on_llm_end(_make_llm_response(
            model="claude-sonnet-4-6",
            prompt_tokens=800,
            completion_tokens=200,
            finish_reason="stop",
        ))
        tracer.on_chain_end({"output": "done"})

        # Verify trace file written
        traces = list(tmp_path.glob("*.json"))
        assert len(traces) == 1

        with open(traces[0]) as f:
            data = json.load(f)

        assert data["agent_id"] == "test-agent"
        assert data["status"] == "Completed"
        assert len(data["steps"]) == 3  # 2 LLM + 1 tool
        assert data["total_tokens"]["prompt_tokens"] == 1300
        assert data["total_tokens"]["completion_tokens"] == 300

    def test_cost_estimated(self, tmp_path: Path) -> None:
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})
        tracer.on_llm_start({}, ["hello"])
        tracer.on_llm_end(_make_llm_response(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        ))
        tracer.on_chain_end({})

        traces = list(tmp_path.glob("*.json"))
        with open(traces[0]) as f:
            data = json.load(f)

        # gpt-4o: $2.50/$10 per 1K → (1000*2.5 + 500*10)/1000 = $7.50
        assert abs(data["total_cost_usd"] - 7.50) < 0.01

    def test_tool_success_and_failure(self, tmp_path: Path) -> None:
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})
        tracer.on_tool_start({}, "query1")
        tracer.on_tool_end("ok", name="search")
        tracer.on_tool_start({}, "query2")
        tracer.on_tool_error(RuntimeError("timeout"), name="fetch")
        tracer.on_chain_end({})

        traces = list(tmp_path.glob("*.json"))
        with open(traces[0]) as f:
            data = json.load(f)

        assert data["steps"][0]["step_type"]["ToolCall"]["success"] is True
        assert data["steps"][0]["step_type"]["ToolCall"]["tool_name"] == "search"
        assert data["steps"][1]["step_type"]["ToolCall"]["success"] is False
        assert data["steps"][1]["step_type"]["ToolCall"]["tool_name"] == "fetch"

    def test_chain_error_writes_trace(self, tmp_path: Path) -> None:
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})
        tracer.on_llm_start({}, ["hello"])
        tracer.on_llm_error(RuntimeError("API key invalid"))
        tracer.on_chain_error(RuntimeError("API key invalid"))

        traces = list(tmp_path.glob("*.json"))
        with open(traces[0]) as f:
            data = json.load(f)

        assert data["status"] == {"Failed": "API key invalid"}

    def test_nested_chains(self, tmp_path: Path) -> None:
        """Only outermost chain end triggers trace write."""
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})  # outer
        tracer.on_chain_start({}, {})  # inner
        tracer.on_llm_start({}, ["hello"])
        tracer.on_llm_end(_make_llm_response())
        tracer.on_chain_end({})  # inner end — should NOT write

        assert len(list(tmp_path.glob("*.json"))) == 0

        tracer.on_chain_end({})  # outer end — SHOULD write

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_atomic_write(self, tmp_path: Path) -> None:
        """No .tmp files left after write."""
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})
        tracer.on_chain_end({})

        tmp_files = list(tmp_path.glob(".*.tmp"))
        assert len(tmp_files) == 0

    def test_trace_id_in_filename(self, tmp_path: Path) -> None:
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})
        tracer.on_chain_end({})

        traces = list(tmp_path.glob("*.json"))
        assert len(traces) == 1

        with open(traces[0]) as f:
            data = json.load(f)

        assert traces[0].stem == data["trace_id"]

    def test_flush(self, tmp_path: Path) -> None:
        tracer = LumenTracer(trace_dir=str(tmp_path))

        tracer.on_chain_start({}, {})
        tracer.on_llm_start({}, ["hello"])
        tracer.on_llm_end(_make_llm_response())

        # Force flush without chain_end
        tracer.flush()

        traces = list(tmp_path.glob("*.json"))
        assert len(traces) == 1

    def test_compatible_with_lumen_reader(self, tmp_path: Path) -> None:
        """Verify trace files can be loaded by lumen.trace_reader."""
        from lumen.trace_reader import load_traces

        tracer = LumenTracer(trace_dir=str(tmp_path), agent_name="compat-test")

        tracer.on_chain_start({}, {})
        tracer.on_llm_start({}, ["hello"])
        tracer.on_llm_end(_make_llm_response(
            model="gpt-4o",
            prompt_tokens=500,
            completion_tokens=100,
        ))
        tracer.on_chain_end({})

        # Load with Lumen trace reader
        traces = load_traces(tmp_path)
        assert len(traces) == 1
        assert traces[0].agent_id == "compat-test"
        assert traces[0].status == "Completed"
        assert len(traces[0].steps) == 1
        assert traces[0].steps[0].step_type == "LlmCall"
        assert traces[0].steps[0].model == "gpt-4o"
