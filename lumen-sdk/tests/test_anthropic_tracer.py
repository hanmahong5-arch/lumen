"""Tests for the Anthropic SDK integration.

Uses mock objects to simulate Anthropic SDK responses without requiring
the actual anthropic package or API keys.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ── Mock Anthropic SDK structures ─────────────────────────────────────────

@dataclass
class MockAnthropicUsage:
    input_tokens: int = 200
    output_tokens: int = 80


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = "Hello from Claude!"


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_abc"
    name: str = "search"
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockAnthropicMessage:
    id: str = "msg_abc"
    type: str = "message"
    role: str = "assistant"
    model: str = "claude-sonnet-4-6"
    content: list[Any] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: MockAnthropicUsage | None = None

    def __post_init__(self) -> None:
        if not self.content:
            self.content = [MockTextBlock()]
        if self.usage is None:
            self.usage = MockAnthropicUsage()


def _make_mock_anthropic_module() -> types.ModuleType:
    """Create a mock anthropic module hierarchy for testing."""
    anthropic_mod = types.ModuleType("anthropic")
    resources_mod = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")

    class Messages:
        def create(self, *args: Any, **kwargs: Any) -> MockAnthropicMessage:
            return MockAnthropicMessage(
                model=kwargs.get("model", "claude-sonnet-4-6"),
            )

    class AsyncMessages:
        async def create(self, *args: Any, **kwargs: Any) -> MockAnthropicMessage:
            return MockAnthropicMessage(
                model=kwargs.get("model", "claude-sonnet-4-6"),
            )

    messages_mod.Messages = Messages  # type: ignore[attr-defined]
    messages_mod.AsyncMessages = AsyncMessages  # type: ignore[attr-defined]
    resources_mod.messages = messages_mod  # type: ignore[attr-defined]
    anthropic_mod.resources = resources_mod  # type: ignore[attr-defined]

    return anthropic_mod


@pytest.fixture()
def mock_anthropic() -> types.ModuleType:
    """Install mock anthropic module and clean up tracer state."""
    mod = _make_mock_anthropic_module()
    messages_mod = mod.resources.messages  # type: ignore[attr-defined]

    sys.modules["anthropic"] = mod
    sys.modules["anthropic.resources"] = mod.resources  # type: ignore[attr-defined]
    sys.modules["anthropic.resources.messages"] = messages_mod

    yield mod

    for key in list(sys.modules):
        if key.startswith("anthropic"):
            del sys.modules[key]

    if "lumen.integrations.anthropic_tracer" in sys.modules:
        tracer_mod = sys.modules["lumen.integrations.anthropic_tracer"]
        tracer_mod._patched = False  # type: ignore[attr-defined]
        tracer_mod._HAS_ANTHROPIC = False  # type: ignore[attr-defined]


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


class TestAnthropicTracer:
    """Test Anthropic tracing without active trace session."""

    def test_sync_call_creates_trace(
        self,
        mock_anthropic: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.anthropic_tracer" in sys.modules:
            del sys.modules["lumen.integrations.anthropic_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.anthropic_tracer import patch_anthropic, unpatch_anthropic

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        messages_mod = mock_anthropic.resources.messages  # type: ignore[attr-defined]
        assert patch_anthropic(config) is True

        client = messages_mod.Messages()
        response = client.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.model == "claude-sonnet-4-6"

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["agent_id"] == "anthropic"
        assert data["status"] == "Completed"
        assert len(data["steps"]) == 1

        step = data["steps"][0]
        assert step["step_type"]["LlmCall"]["model"] == "claude-sonnet-4-6"
        assert step["step_type"]["LlmCall"]["finish_reason"] == "stop"  # end_turn → stop
        assert step["tokens"]["prompt_tokens"] == 200
        assert step["tokens"]["completion_tokens"] == 80
        assert data["total_cost_usd"] > 0

        unpatch_anthropic()

    def test_tool_use_detected(
        self,
        mock_anthropic: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.anthropic_tracer" in sys.modules:
            del sys.modules["lumen.integrations.anthropic_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.anthropic_tracer import patch_anthropic, unpatch_anthropic

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        messages_mod = mock_anthropic.resources.messages  # type: ignore[attr-defined]

        original = messages_mod.Messages.create
        def tool_create(self: Any, *args: Any, **kwargs: Any) -> MockAnthropicMessage:
            return MockAnthropicMessage(
                model="claude-sonnet-4-6",
                content=[MockToolUseBlock(name="search")],
                stop_reason="tool_use",
            )

        messages_mod.Messages.create = tool_create
        patch_anthropic(config)

        client = messages_mod.Messages()
        client.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[])

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        step = data["steps"][0]
        assert step["step_type"]["LlmCall"]["finish_reason"] == "tool_use"

        messages_mod.Messages.create = original
        unpatch_anthropic()

    def test_sampling_rate_zero(
        self,
        mock_anthropic: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.anthropic_tracer" in sys.modules:
            del sys.modules["lumen.integrations.anthropic_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.anthropic_tracer import patch_anthropic, unpatch_anthropic

        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "sampling_rate": 0.0,
            },
        })

        messages_mod = mock_anthropic.resources.messages  # type: ignore[attr-defined]
        patch_anthropic(config)

        client = messages_mod.Messages()
        client.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[])

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 0

        unpatch_anthropic()

    def test_trace_readable_by_load_traces(
        self,
        mock_anthropic: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.anthropic_tracer" in sys.modules:
            del sys.modules["lumen.integrations.anthropic_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.anthropic_tracer import patch_anthropic, unpatch_anthropic
        from lumen.trace_reader import load_traces

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        messages_mod = mock_anthropic.resources.messages  # type: ignore[attr-defined]
        patch_anthropic(config)

        client = messages_mod.Messages()
        client.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[])

        traces = load_traces(str(trace_dir))
        assert len(traces) == 1
        assert traces[0].status == "Completed"
        assert traces[0].steps[0].step_type == "LlmCall"
        assert traces[0].steps[0].model == "claude-sonnet-4-6"

        unpatch_anthropic()

    def test_error_creates_failed_trace(
        self,
        mock_anthropic: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.anthropic_tracer" in sys.modules:
            del sys.modules["lumen.integrations.anthropic_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.anthropic_tracer import patch_anthropic, unpatch_anthropic

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        messages_mod = mock_anthropic.resources.messages  # type: ignore[attr-defined]

        original = messages_mod.Messages.create
        def failing_create(self: Any, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Rate limited")
        messages_mod.Messages.create = failing_create

        patch_anthropic(config)

        client = messages_mod.Messages()
        with pytest.raises(RuntimeError, match="Rate limited"):
            client.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[])

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["status"] == {"Failed": "Rate limited"}

        messages_mod.Messages.create = original
        unpatch_anthropic()
