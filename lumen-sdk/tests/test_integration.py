"""End-to-end integration tests exercising the full Lumen pipeline.

Tests the full flow: instrument() → LLM calls → trace files → load_traces
→ query → cost report → replay, ensuring all components work together.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lumen._budget import BudgetExceededError, BudgetTracker
from lumen._context import TraceSession, get_active_session
from lumen._redaction import RedactionEngine
from lumen.config import LumenConfig
from lumen.cost import CostTracker
from lumen.instrument import atrace, trace, traced_fn
from lumen.query import TraceQuery
from lumen.trace_reader import load_traces


# ── Mock OpenAI SDK ──────────────────────────────────────────────────────────

@dataclass
class MockUsage:
    prompt_tokens: int = 150
    completion_tokens: int = 60


@dataclass
class MockMessage:
    role: str = "assistant"
    content: str = "Hello from the model!"
    tool_calls: list[Any] | None = None


@dataclass
class MockChoice:
    index: int = 0
    message: MockMessage = field(default_factory=MockMessage)
    finish_reason: str = "stop"


@dataclass
class MockChatCompletion:
    id: str = "chatcmpl-abc"
    model: str = "gpt-4o"
    choices: list[MockChoice] = field(default_factory=lambda: [MockChoice()])
    usage: MockUsage = field(default_factory=MockUsage)


def _make_mock_openai() -> types.ModuleType:
    openai_mod = types.ModuleType("openai")
    resources_mod = types.ModuleType("openai.resources")
    chat_mod = types.ModuleType("openai.resources.chat")
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, *args: Any, **kwargs: Any) -> MockChatCompletion:
            return MockChatCompletion(model=kwargs.get("model", "gpt-4o"))

    class AsyncCompletions:
        async def create(self, *args: Any, **kwargs: Any) -> MockChatCompletion:
            return MockChatCompletion(model=kwargs.get("model", "gpt-4o"))

    completions_mod.Completions = Completions  # type: ignore[attr-defined]
    completions_mod.AsyncCompletions = AsyncCompletions  # type: ignore[attr-defined]
    chat_mod.completions = completions_mod  # type: ignore[attr-defined]
    resources_mod.chat = chat_mod  # type: ignore[attr-defined]
    openai_mod.resources = resources_mod  # type: ignore[attr-defined]
    return openai_mod


@pytest.fixture()
def mock_openai() -> types.ModuleType:
    mod = _make_mock_openai()
    completions_mod = mod.resources.chat.completions  # type: ignore[attr-defined]
    sys.modules["openai"] = mod
    sys.modules["openai.resources"] = mod.resources  # type: ignore[attr-defined]
    sys.modules["openai.resources.chat"] = mod.resources.chat  # type: ignore[attr-defined]
    sys.modules["openai.resources.chat.completions"] = completions_mod
    yield mod
    for key in list(sys.modules):
        if key.startswith("openai"):
            del sys.modules[key]
    if "lumen.integrations.openai_tracer" in sys.modules:
        tracer_mod = sys.modules["lumen.integrations.openai_tracer"]
        tracer_mod._patched = False  # type: ignore[attr-defined]
        tracer_mod._HAS_OPENAI = False  # type: ignore[attr-defined]


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


# ── Full pipeline tests ─────────────────────────────────────────────────────

class TestFullPipeline:
    """Test the complete instrument → call → query → cost flow."""

    def test_instrument_call_query(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
            "project": {"name": "test-proj", "environment": "test"},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        patch_openai(config)

        # Make 3 calls
        client = completions_mod.Completions()
        for _ in range(3):
            client.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        # Verify trace files
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 3

        # Query traces
        result = TraceQuery(str(trace_dir)).execute()
        assert result.total_scanned == 3
        assert result.total_matched == 3

        # Filter by model
        gpt_traces = TraceQuery(str(trace_dir)).model("gpt-4o").execute()
        assert gpt_traces.total_matched == 3

        # Cost report
        tracker = CostTracker(config=config)
        report = tracker.report("1h")
        assert report.total_runs == 3
        assert report.total_usd > 0

        unpatch_openai()

    def test_grouped_trace_with_query(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        patch_openai(config)

        # Group 3 calls into one trace
        with trace("research-agent", user_id="u-test", session_id="s-1", config=config):
            client = completions_mod.Completions()
            client.create(model="gpt-4o", messages=[])
            client.create(model="gpt-4o", messages=[])
            client.create(model="gpt-4o", messages=[])

        # Should be 1 trace with 3 steps
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        result = TraceQuery(str(trace_dir)).execute()
        assert result.total_matched == 1
        assert len(result.traces[0].steps) == 3

        # Verify user/session
        by_user = result.group_by_user()
        assert "u-test" in by_user
        by_session = result.group_by_session()
        assert "s-1" in by_session

        unpatch_openai()


class TestContentCapture:
    """Test content capture modes."""

    def test_full_capture(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "content_capture": "full",
            },
        })

        session = TraceSession(name="test", trace_dir=str(trace_dir), config=config)
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000000,
            input_messages=[{"role": "user", "content": "What is AI?"}],
            output_content="AI is artificial intelligence.",
        )
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        step = data["steps"][0]
        assert len(step["metadata"]) == 2
        assert step["metadata"][0][0] == "input_messages"
        assert "What is AI?" in step["metadata"][0][1]
        assert step["metadata"][1][0] == "output_content"
        assert step["metadata"][1][1] == "AI is artificial intelligence."

    def test_metadata_only_strips_content(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "content_capture": "metadata_only",
            },
        })

        session = TraceSession(name="test", trace_dir=str(trace_dir), config=config)
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000000,
            input_messages=[{"role": "user", "content": "Secret info"}],
            output_content="Sensitive output",
        )
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        step = data["steps"][0]
        # Content should be stripped
        for key, val in step["metadata"]:
            assert val == "[stripped]"

    def test_off_mode_strips_everything(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "content_capture": "off",
            },
        })

        session = TraceSession(name="test", trace_dir=str(trace_dir), config=config)
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000000,
            input_messages=[{"role": "user", "content": "Secret"}],
            output_content="Also secret",
        )
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        # Verify no actual content
        raw = json.dumps(data)
        assert "Secret" not in raw
        assert "Also secret" not in raw


class TestBudgetIntegration:
    """Test budget enforcement in trace sessions."""

    def test_budget_tracking_in_session(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
            "agent": {"max_cost_per_run_usd": 10.0},
            "cost": {"budget_usd": 100.0},
        })
        bt = BudgetTracker.from_config(config)

        session = TraceSession(
            name="test",
            trace_dir=str(trace_dir),
            config=config,
            budget_tracker=bt,
        )
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000000,
            cost_usd=0.05,
        )
        session.finalize()

        assert bt.total_spent == pytest.approx(0.05)
        assert bt.run_count == 1

    def test_per_run_budget_exceeded(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
            "agent": {"max_cost_per_run_usd": 0.10},
        })
        bt = BudgetTracker.from_config(config)

        session = TraceSession(
            name="test",
            trace_dir=str(trace_dir),
            config=config,
            budget_tracker=bt,
        )
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000000,
            cost_usd=0.08,
        )

        with pytest.raises(BudgetExceededError, match="per-run"):
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                finish_reason="stop",
                duration_ms=500,
                started_at_ms=1000500,
                cost_usd=0.05,  # 0.08 + 0.05 = 0.13 > 0.10
            )


class TestRedactionIntegration:
    """Test redaction in trace sessions."""

    def test_error_message_redaction(self, trace_dir: Path) -> None:
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
        session.add_error_step("Bad key: sk-abc123def456")
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        msg = data["steps"][0]["step_type"]["Error"]["message"]
        assert "sk-abc123def456" not in msg
        assert "[REDACTED]" in msg

    def test_content_redaction_in_metadata(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "content_capture": "full",
                "redaction": {
                    "enabled": True,
                    "patterns": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                },
            },
        })

        session = TraceSession(name="test", trace_dir=str(trace_dir), config=config)
        session.add_llm_step(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
            duration_ms=500,
            started_at_ms=1000000,
            input_messages=[{"role": "user", "content": "Email me at test@example.com"}],
        )
        session.finalize()

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            raw = f.read()

        # The email in the metadata should be redacted
        assert "test@example.com" not in raw


class TestTracedFnDecorator:
    """Test the @traced_fn decorator."""

    def test_sync_decorator(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        @traced_fn("decorated-agent", config=config)
        def my_func(x: int) -> int:
            session = get_active_session()
            assert session is not None
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                duration_ms=100,
                started_at_ms=1000000,
            )
            return x * 2

        result = my_func(21)
        assert result == 42

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)
        assert data["agent_id"] == "decorated-agent"

    def test_async_decorator(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        @traced_fn("async-agent", config=config)
        async def my_async_func(x: int) -> int:
            session = get_active_session()
            assert session is not None
            session.add_llm_step(
                model="gpt-4o",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                duration_ms=100,
                started_at_ms=1000000,
            )
            return x + 1

        result = asyncio.get_event_loop().run_until_complete(my_async_func(41))
        assert result == 42

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1


class TestAsyncTrace:
    """Test the async trace context manager."""

    def test_atrace_creates_trace(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        async def run() -> None:
            async with atrace("async-agent", config=config) as session:
                session.add_llm_step(
                    model="gpt-4o",
                    prompt_tokens=100,
                    completion_tokens=50,
                    finish_reason="stop",
                    duration_ms=500,
                    started_at_ms=1000000,
                )

        asyncio.get_event_loop().run_until_complete(run())

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)
        assert data["agent_id"] == "async-agent"

    def test_atrace_error_handling(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        async def run() -> None:
            async with atrace("async-agent", config=config) as session:
                session.add_llm_step(
                    model="gpt-4o",
                    prompt_tokens=10,
                    completion_tokens=5,
                    finish_reason="stop",
                    duration_ms=100,
                    started_at_ms=1000000,
                )
                raise ValueError("async boom")

        with pytest.raises(ValueError, match="async boom"):
            asyncio.get_event_loop().run_until_complete(run())

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)
        assert data["status"] == {"Failed": "async boom"}


class TestQueryWithRedaction:
    """Test querying traces that were written with redaction."""

    def test_query_redacted_traces(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "redaction": {
                    "enabled": True,
                    "patterns": [r"sk-[a-zA-Z0-9]+"],
                },
            },
        })

        # Write traces with sensitive data
        for i in range(3):
            session = TraceSession(
                name=f"agent-{i}",
                trace_dir=str(trace_dir),
                config=config,
            )
            session.add_error_step(f"Key error: sk-secret{i}abc")
            session.finalize()

        # Query should work on redacted traces
        result = TraceQuery(str(trace_dir)).execute()
        assert result.total_matched == 3

        # Verify redaction persisted in loaded traces
        traces = load_traces(str(trace_dir))
        for t in traces:
            assert t.steps[0].step_type == "Error"
            assert "sk-secret" not in (t.steps[0].error_message or "")


class TestMultiAgentScenario:
    """Test realistic multi-agent scenario with all features."""

    def test_multi_agent_with_budget(self, trace_dir: Path) -> None:
        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
            "project": {"name": "multi-agent", "environment": "test"},
        })
        bt = BudgetTracker(budget=10.0)

        # Agent 1: research
        s1 = TraceSession(
            name="research-agent",
            trace_dir=str(trace_dir),
            config=config,
            user_id="alice",
            session_id="sess-1",
            budget_tracker=bt,
        )
        s1.add_llm_step(
            model="gpt-4o", prompt_tokens=500, completion_tokens=200,
            finish_reason="stop", duration_ms=1000, started_at_ms=1000000,
            cost_usd=0.10,
        )
        s1.add_tool_step(
            tool_name="web_search", success=True, duration_ms=500, started_at_ms=1001000,
        )
        s1.add_llm_step(
            model="gpt-4o", prompt_tokens=800, completion_tokens=300,
            finish_reason="stop", duration_ms=1500, started_at_ms=1001500,
            cost_usd=0.15,
        )
        s1.finalize()

        # Agent 2: coding
        s2 = TraceSession(
            name="code-agent",
            trace_dir=str(trace_dir),
            config=config,
            user_id="bob",
            session_id="sess-2",
            budget_tracker=bt,
        )
        s2.add_llm_step(
            model="claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=500,
            finish_reason="stop", duration_ms=2000, started_at_ms=2000000,
            cost_usd=0.20,
        )
        s2.finalize()

        # Verify budget
        assert bt.total_spent == pytest.approx(0.45)
        assert bt.run_count == 2

        # Query by agent
        research = TraceQuery(str(trace_dir)).agent("research-agent").execute()
        assert research.total_matched == 1
        assert len(research.traces[0].steps) == 3

        # Query by project
        proj = TraceQuery(str(trace_dir)).project("multi-agent").execute()
        assert proj.total_matched == 2

        # Query summary
        all_result = TraceQuery(str(trace_dir)).execute()
        summary = all_result.summary()
        assert summary["total_traces"] == 2
        assert "research-agent" in summary["agents"]
        assert "code-agent" in summary["agents"]

        # Cost report
        tracker = CostTracker(config=config)
        report = tracker.report("1h")
        assert report.total_runs == 2
        assert "research-agent" in report.by_agent
        assert "code-agent" in report.by_agent


class TestEventBusIntegration:
    """Verify TraceCompleted events fire on session finalize."""

    def test_trace_completed_event_emitted(self, trace_dir: Path) -> None:
        from lumen._event_bus import TraceCompleted, get_default_bus

        bus = get_default_bus()
        received: list[TraceCompleted] = []

        @bus.on(TraceCompleted)
        def capture(event: TraceCompleted):
            received.append(event)

        try:
            session = TraceSession(
                name="event-test",
                trace_dir=str(trace_dir),
            )
            session.add_llm_step(
                model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                finish_reason="stop", duration_ms=500, started_at_ms=1000000,
                cost_usd=0.05,
            )
            session.finalize()

            assert len(received) == 1
            e = received[0]
            assert e.agent_id == "event-test"
            assert e.status == "Completed"
            assert e.total_cost_usd == pytest.approx(0.05)
            assert e.step_count == 1
            assert e.duration_ms > 0
        finally:
            bus.off(TraceCompleted, capture)

    def test_failed_trace_emits_failed_status(self, trace_dir: Path) -> None:
        from lumen._event_bus import TraceCompleted, get_default_bus

        bus = get_default_bus()
        received: list[TraceCompleted] = []
        handler = lambda e: received.append(e)
        bus.on(TraceCompleted, handler)

        try:
            session = TraceSession(
                name="fail-agent",
                trace_dir=str(trace_dir),
            )
            session.add_error_step("something broke")
            session.finalize()

            assert len(received) == 1
            assert received[0].status == "Failed"
        finally:
            bus.off(TraceCompleted, handler)

    def test_empty_session_no_event(self, trace_dir: Path) -> None:
        from lumen._event_bus import TraceCompleted, get_default_bus

        bus = get_default_bus()
        received: list[TraceCompleted] = []
        handler = lambda e: received.append(e)
        bus.on(TraceCompleted, handler)

        try:
            session = TraceSession(
                name="empty",
                trace_dir=str(trace_dir),
            )
            session.finalize()  # no steps → early return, no event
            assert len(received) == 0
        finally:
            bus.off(TraceCompleted, handler)


class TestContextVarsIsolation:
    """Verify contextvars-based session isolation for async tasks."""

    def test_async_tasks_get_independent_sessions(self, trace_dir: Path) -> None:
        """Two concurrent async tasks should have independent trace sessions."""
        from lumen._context import get_active_session, set_active_session

        results: dict[str, str | None] = {}

        async def task_a():
            session = TraceSession(name="task-a", trace_dir=str(trace_dir))
            set_active_session(session)
            await asyncio.sleep(0.01)  # yield to event loop
            active = get_active_session()
            results["a"] = active.name if active else None

        async def task_b():
            session = TraceSession(name="task-b", trace_dir=str(trace_dir))
            set_active_session(session)
            await asyncio.sleep(0.01)
            active = get_active_session()
            results["b"] = active.name if active else None

        async def main():
            # asyncio.create_task copies the current context
            t1 = asyncio.create_task(task_a())
            t2 = asyncio.create_task(task_b())
            await t1
            await t2

        asyncio.run(main())
        assert results["a"] == "task-a"
        assert results["b"] == "task-b"

    def test_trace_index_with_real_traces(self, trace_dir: Path) -> None:
        """TraceIndex integrates with real trace files from TraceSession."""
        from lumen._trace_index import TraceIndex

        # Write some traces
        for i in range(5):
            session = TraceSession(
                name=f"agent-{i % 2}",
                trace_dir=str(trace_dir),
            )
            session.add_llm_step(
                model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                finish_reason="stop", duration_ms=500,
                started_at_ms=1000000 + i * 1000,
                cost_usd=0.01 * (i + 1),
            )
            session.finalize()

        # Build index and query
        idx = TraceIndex(str(trace_dir))
        count = idx.build()
        assert count == 5

        a0 = idx.by_agent("agent-0")
        assert len(a0) == 3  # indices 0, 2, 4

        a1 = idx.by_agent("agent-1")
        assert len(a1) == 2  # indices 1, 3

        top = idx.most_expensive(2)
        assert top[0].total_cost_usd >= top[1].total_cost_usd

    def test_metrics_window_tracking(self) -> None:
        """MetricsWindow tracks latency statistics correctly."""
        from lumen._ring_buffer import MetricsWindow

        w = MetricsWindow(100)
        latencies = [120.0, 150.0, 200.0, 95.0, 310.0, 180.0]
        for lat in latencies:
            w.record(lat)

        assert w.count == 6
        assert w.mean == pytest.approx(sum(latencies) / 6)
        assert w.min == 95.0
        assert w.max == 310.0
        assert w.percentile(50) > 0
