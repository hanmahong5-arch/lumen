"""Tests for the instrument() one-liner and trace() context manager."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from lumen._context import TraceSession, get_active_session, set_active_session
from lumen.config import LumenConfig
from lumen.instrument import trace, uninstrument


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


class TestTraceContextManager:
    """Test the trace() context manager for grouping calls."""

    def test_creates_trace_file(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        with trace("test-agent", config=config) as session:
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                finish_reason="stop",
                duration_ms=500,
                started_at_ms=1000000,
                cost_usd=0.001,
            )

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["agent_id"] == "test-agent"
        assert data["status"] == "Completed"
        assert len(data["steps"]) == 1
        assert data["total_tokens"]["prompt_tokens"] == 100

    def test_session_is_active_within_context(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        assert get_active_session() is None

        with trace("test", config=config) as session:
            assert get_active_session() is session
            assert session.name == "test"

        assert get_active_session() is None

    def test_user_id_and_session_id(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        with trace("agent", user_id="u-1", session_id="s-2", config=config) as session:
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                duration_ms=100,
                started_at_ms=1000000,
            )

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["user_id"] == "u-1"
        assert data["session_id"] == "s-2"

    def test_error_in_context_creates_failed_trace(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        with pytest.raises(ValueError, match="boom"):
            with trace("agent", config=config) as session:
                session.add_llm_step(
                    model="gpt-4o",
                    prompt_tokens=50,
                    completion_tokens=25,
                    finish_reason="stop",
                    duration_ms=200,
                    started_at_ms=1000000,
                )
                raise ValueError("boom")

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["status"] == {"Failed": "boom"}

    def test_empty_context_no_file(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        with trace("agent", config=config):
            pass  # no steps

        # No trace should be written for empty sessions
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 0

    def test_sampling_zero_skips(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "sampling_rate": 0.0,
            },
        })

        with trace("agent", config=config) as session:
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                finish_reason="stop",
                duration_ms=500,
                started_at_ms=1000000,
            )

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 0

    def test_nested_sessions_restore(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        with trace("outer", config=config) as outer:
            outer.add_llm_step(
                model="gpt-4o",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                duration_ms=100,
                started_at_ms=1000000,
            )
            assert get_active_session() is outer

            with trace("inner", config=config) as inner:
                inner.add_llm_step(
                    model="gpt-4o",
                    prompt_tokens=20,
                    completion_tokens=10,
                    finish_reason="stop",
                    duration_ms=200,
                    started_at_ms=2000000,
                )
                assert get_active_session() is inner

            assert get_active_session() is outer

        assert get_active_session() is None

        # Should produce 2 trace files
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 2

    def test_project_and_env_in_trace(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "project": {"name": "my-proj", "environment": "staging"},
            "tracing": {"trace_dir": str(trace_dir)},
        })

        with trace("agent", config=config) as session:
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                duration_ms=100,
                started_at_ms=1000000,
            )

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["project"] == "my-proj"
        assert data["environment"] == "staging"


class TestTraceSession:
    """Test TraceSession directly."""

    def test_add_multiple_steps(self, trace_dir: Path) -> None:
        session = TraceSession(name="test", trace_dir=str(trace_dir))

        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="tool_use",
            duration_ms=500,
            started_at_ms=1000000,
            cost_usd=0.001,
        )
        session.add_tool_step(
            tool_name="search",
            success=True,
            duration_ms=200,
            started_at_ms=1000500,
        )
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=200,
            completion_tokens=100,
            finish_reason="stop",
            duration_ms=800,
            started_at_ms=1000700,
            cost_usd=0.002,
        )
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)

        assert len(data["steps"]) == 3
        assert data["steps"][0]["step_type"]["LlmCall"]["finish_reason"] == "tool_use"
        assert data["steps"][1]["step_type"]["ToolCall"]["tool_name"] == "search"
        assert data["steps"][2]["step_type"]["LlmCall"]["finish_reason"] == "stop"
        assert data["total_tokens"]["prompt_tokens"] == 300
        assert data["total_tokens"]["completion_tokens"] == 150
        assert data["total_cost_usd"] == pytest.approx(0.003)

    def test_error_step_marks_failed(self, trace_dir: Path) -> None:
        session = TraceSession(name="test", trace_dir=str(trace_dir))
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            duration_ms=100,
            started_at_ms=1000000,
        )
        session.add_error_step("connection timeout")
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["status"] == {"Failed": "connection timeout"}

    def test_finalize_empty_no_file(self, trace_dir: Path) -> None:
        session = TraceSession(name="test", trace_dir=str(trace_dir))
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 0


class TestRedactionInTrace:
    """Test that redaction is applied when writing traces."""

    def test_error_message_redacted(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "redaction": {
                    "enabled": True,
                    "patterns": [r"sk-[a-zA-Z0-9]+"],
                },
            },
        })

        session = TraceSession(name="test", trace_dir=str(trace_dir), config=config)
        session.add_error_step("Invalid API key: sk-abc123def456")
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        error_step = data["steps"][0]
        msg = error_step["step_type"]["Error"]["message"]
        assert "sk-abc123def456" not in msg
        assert "[REDACTED]" in msg
