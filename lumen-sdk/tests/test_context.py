"""Comprehensive tests for the TraceSession and thread-local context."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from lumen._budget import BudgetExceededError, BudgetTracker
from lumen._context import (
    TraceSession,
    _now_ms,
    _redact_steps,
    _strip_content,
    get_active_session,
    set_active_session,
    should_sample,
)
from lumen.config import LumenConfig


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


class TestTraceSessionBasic:
    """Test basic TraceSession creation and attributes."""

    def test_trace_id_generated(self, trace_dir: Path) -> None:
        s = TraceSession(name="test", trace_dir=str(trace_dir))
        assert len(s.trace_id) == 12
        assert s.trace_id.isalnum()

    def test_unique_trace_ids(self, trace_dir: Path) -> None:
        ids = {TraceSession(name="t", trace_dir=str(trace_dir)).trace_id for _ in range(100)}
        assert len(ids) == 100

    def test_initial_state(self, trace_dir: Path) -> None:
        s = TraceSession(name="agent", trace_dir=str(trace_dir))
        assert s.name == "agent"
        assert s.status == "Running"
        assert s.steps == []
        assert s.step_num == 0
        assert s.total_prompt_tokens == 0
        assert s.total_completion_tokens == 0
        assert s.total_cost_usd == 0.0
        assert s.failure_reason is None

    def test_started_at_is_now(self, trace_dir: Path) -> None:
        before = _now_ms()
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        after = _now_ms()
        assert before <= s.started_at_ms <= after

    def test_user_and_session_ids(self, trace_dir: Path) -> None:
        s = TraceSession(
            name="t", trace_dir=str(trace_dir),
            user_id="u-1", session_id="s-2",
        )
        assert s.user_id == "u-1"
        assert s.session_id == "s-2"


class TestTraceSessionLlmStep:
    """Test LLM step recording."""

    def test_add_llm_step(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000,
        )
        assert len(s.steps) == 1
        assert s.step_num == 1
        assert s.total_prompt_tokens == 100
        assert s.total_completion_tokens == 50

    def test_cost_accumulates(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=100, completion_tokens=50,
            finish_reason="stop", duration_ms=500, started_at_ms=1000,
            cost_usd=0.01,
        )
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=200, completion_tokens=100,
            finish_reason="stop", duration_ms=800, started_at_ms=2000,
            cost_usd=0.02,
        )
        assert s.total_cost_usd == pytest.approx(0.03)

    def test_llm_step_with_content_capture(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
            input_messages=[{"role": "user", "content": "hello"}],
            output_content="world",
        )
        assert len(s.steps[0]["metadata"]) == 2
        assert s.steps[0]["metadata"][0][0] == "input_messages"
        assert s.steps[0]["metadata"][1] == ("output_content", "world")

    def test_llm_step_no_content(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        assert s.steps[0]["metadata"] == []

    def test_zero_tokens_no_tokens_dict(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=0, completion_tokens=0,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        assert s.steps[0]["tokens"] is None


class TestTraceSessionToolStep:
    """Test tool step recording."""

    def test_add_tool_step(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_tool_step(
            tool_name="search", success=True,
            duration_ms=200, started_at_ms=1000,
        )
        assert len(s.steps) == 1
        step = s.steps[0]
        assert step["step_type"]["ToolCall"]["tool_name"] == "search"
        assert step["step_type"]["ToolCall"]["success"] is True

    def test_failed_tool_step(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_tool_step(
            tool_name="execute", success=False,
            duration_ms=100, started_at_ms=1000,
        )
        assert s.steps[0]["step_type"]["ToolCall"]["success"] is False


class TestTraceSessionErrorStep:
    """Test error step recording."""

    def test_add_error_step(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_error_step("connection timeout")
        assert len(s.steps) == 1
        assert s.status == "Failed"
        assert s.failure_reason == "connection timeout"

    def test_error_step_structure(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_error_step("boom")
        step = s.steps[0]
        assert step["step_type"]["Error"]["message"] == "boom"
        assert step["duration_ms"] == 0


class TestTraceSessionFinalize:
    """Test trace finalization and file writing."""

    def test_empty_session_no_file(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.finalize()
        assert list(trace_dir.glob("*.json")) == []

    def test_completed_status(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        s.finalize()
        files = list(trace_dir.glob("*.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            data = json.load(f)
        assert data["status"] == "Completed"

    def test_failed_status_in_json(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_error_step("bad")
        s.finalize()
        files = list(trace_dir.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)
        assert data["status"] == {"Failed": "bad"}

    def test_creates_dir_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "deep" / "nested" / "traces"
        s = TraceSession(name="t", trace_dir=str(new_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        s.finalize()
        assert new_dir.exists()
        assert len(list(new_dir.glob("*.json"))) == 1

    def test_atomic_write_no_temp_files(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        s.finalize()
        # No .tmp files should remain
        assert list(trace_dir.glob(".*")) == []

    def test_trace_data_structure(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
            "project": {"name": "proj", "environment": "staging"},
        })
        s = TraceSession(
            name="agent", trace_dir=str(trace_dir), config=config,
            user_id="u1", session_id="s1",
        )
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=100, completion_tokens=50,
            finish_reason="stop", duration_ms=500, started_at_ms=1000,
            cost_usd=0.01,
        )
        s.finalize()

        files = list(trace_dir.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)

        assert data["trace_id"] == s.trace_id
        assert data["agent_id"] == "agent"
        assert data["user_id"] == "u1"
        assert data["session_id"] == "s1"
        assert data["project"] == "proj"
        assert data["environment"] == "staging"
        assert data["total_tokens"]["prompt_tokens"] == 100
        assert data["total_tokens"]["completion_tokens"] == 50
        assert data["total_cost_usd"] == pytest.approx(0.01)
        assert data["started_at_ms"] > 0
        assert data["completed_at_ms"] >= data["started_at_ms"]


class TestTraceSessionBudget:
    """Test budget integration in TraceSession."""

    def test_budget_recorded_on_llm_step(self, trace_dir: Path) -> None:
        bt = BudgetTracker(budget=100.0)
        s = TraceSession(name="t", trace_dir=str(trace_dir), budget_tracker=bt)
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=100, completion_tokens=50,
            finish_reason="stop", duration_ms=500, started_at_ms=1000,
            cost_usd=0.05,
        )
        assert bt.get_run_cost(s.trace_id) == pytest.approx(0.05)

    def test_budget_finished_on_finalize(self, trace_dir: Path) -> None:
        bt = BudgetTracker()
        s = TraceSession(name="t", trace_dir=str(trace_dir), budget_tracker=bt)
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
            cost_usd=0.01,
        )
        s.finalize()
        assert bt.run_count == 1
        assert bt.get_run_cost(s.trace_id) == 0.0  # cleaned up

    def test_budget_finished_on_empty_session(self, trace_dir: Path) -> None:
        bt = BudgetTracker()
        s = TraceSession(name="t", trace_dir=str(trace_dir), budget_tracker=bt)
        s.finalize()
        # finish_run is called even for empty sessions for cleanup
        assert bt.run_count == 1
        assert bt.get_run_cost(s.trace_id) == 0.0

    def test_per_run_exceeded(self, trace_dir: Path) -> None:
        bt = BudgetTracker(max_per_run=0.05)
        s = TraceSession(name="t", trace_dir=str(trace_dir), budget_tracker=bt)
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=100, completion_tokens=50,
            finish_reason="stop", duration_ms=500, started_at_ms=1000,
            cost_usd=0.04,
        )
        with pytest.raises(BudgetExceededError):
            s.add_llm_step(
                model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                finish_reason="stop", duration_ms=500, started_at_ms=2000,
                cost_usd=0.03,  # 0.04 + 0.03 > 0.05
            )

    def test_no_budget_tracker_ok(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        s.add_llm_step(
            model="gpt-4o", prompt_tokens=100, completion_tokens=50,
            finish_reason="stop", duration_ms=500, started_at_ms=1000,
            cost_usd=999.0,
        )
        s.finalize()  # no exception


class TestThreadLocalContext:
    """Test thread-local session management."""

    def test_no_active_session_initially(self) -> None:
        assert get_active_session() is None

    def test_set_and_get(self, trace_dir: Path) -> None:
        s = TraceSession(name="t", trace_dir=str(trace_dir))
        set_active_session(s)
        assert get_active_session() is s
        set_active_session(None)
        assert get_active_session() is None

    def test_thread_isolation(self, trace_dir: Path) -> None:
        s1 = TraceSession(name="main", trace_dir=str(trace_dir))
        set_active_session(s1)

        seen_in_thread: list[Any] = []

        def worker() -> None:
            seen_in_thread.append(get_active_session())
            s2 = TraceSession(name="worker", trace_dir=str(trace_dir))
            set_active_session(s2)
            seen_in_thread.append(get_active_session())

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Worker sees None initially, then its own session
        assert seen_in_thread[0] is None
        assert seen_in_thread[1] is not None
        assert seen_in_thread[1].name == "worker"

        # Main thread still has its own session
        assert get_active_session() is s1
        set_active_session(None)

    def test_multiple_threads_isolated(self, trace_dir: Path) -> None:
        results: dict[int, str] = {}
        barrier = threading.Barrier(5)

        def worker(idx: int) -> None:
            s = TraceSession(name=f"t-{idx}", trace_dir=str(trace_dir))
            set_active_session(s)
            barrier.wait()  # all threads set session simultaneously
            active = get_active_session()
            results[idx] = active.name if active else "none"
            set_active_session(None)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should see only its own session
        for i in range(5):
            assert results[i] == f"t-{i}"


class TestShouldSample:
    """Test sampling decision logic."""

    def test_none_config_always_samples(self) -> None:
        assert should_sample(None) is True

    def test_disabled_never_samples(self) -> None:
        config = LumenConfig.from_dict({"tracing": {"enabled": False}})
        assert should_sample(config) is False

    def test_rate_1_always_samples(self) -> None:
        config = LumenConfig.from_dict({"tracing": {"sampling_rate": 1.0}})
        for _ in range(100):
            assert should_sample(config) is True

    def test_rate_0_never_samples(self) -> None:
        config = LumenConfig.from_dict({"tracing": {"sampling_rate": 0.0}})
        for _ in range(100):
            assert should_sample(config) is False

    def test_rate_half_statistical(self) -> None:
        config = LumenConfig.from_dict({"tracing": {"sampling_rate": 0.5}})
        sampled = sum(1 for _ in range(1000) if should_sample(config))
        # Should be roughly 500, allow wide margin
        assert 300 < sampled < 700


class TestRedactSteps:
    """Test the _redact_steps helper."""

    def test_redacts_error_message(self) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "redaction": {"enabled": True, "patterns": [r"secret-\d+"]},
            },
        })
        steps = [
            {"step_type": {"Error": {"message": "key: secret-123"}}, "metadata": []},
        ]
        result = _redact_steps(steps, config.tracing.redaction)
        assert "secret-123" not in result[0]["step_type"]["Error"]["message"]

    def test_redacts_metadata_values(self) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "redaction": {"enabled": True, "patterns": [r"password=\w+"]},
            },
        })
        steps = [
            {
                "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "stop"}},
                "metadata": [["input", "password=hunter2"]],
            },
        ]
        result = _redact_steps(steps, config.tracing.redaction)
        assert "hunter2" not in result[0]["metadata"][0][1]

    def test_preserves_non_matching(self) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "redaction": {"enabled": True, "patterns": [r"zzzzz"]},
            },
        })
        steps = [
            {"step_type": {"Error": {"message": "normal error"}}, "metadata": []},
        ]
        result = _redact_steps(steps, config.tracing.redaction)
        assert result[0]["step_type"]["Error"]["message"] == "normal error"


class TestStripContent:
    """Test the _strip_content helper."""

    def test_strips_metadata(self) -> None:
        steps = [
            {
                "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "stop"}},
                "metadata": [["input", "secret data"], ["output", "more secrets"]],
            },
        ]
        result = _strip_content(steps)
        for key, val in result[0]["metadata"]:
            assert val == "[stripped]"

    def test_preserves_step_type(self) -> None:
        steps = [
            {
                "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "stop"}},
                "metadata": [],
            },
        ]
        result = _strip_content(steps)
        assert result[0]["step_type"]["LlmCall"]["model"] == "gpt-4o"

    def test_empty_metadata_preserved(self) -> None:
        steps = [
            {"step_type": {"Error": {"message": "err"}}, "metadata": []},
        ]
        result = _strip_content(steps)
        assert result[0]["metadata"] == []
