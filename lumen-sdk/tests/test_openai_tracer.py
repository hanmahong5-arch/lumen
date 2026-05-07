"""Tests for the OpenAI SDK integration.

Uses mock objects to simulate OpenAI SDK responses without requiring
the actual openai package or API keys.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Mock OpenAI SDK structures ────────────────────────────────────────────

@dataclass
class MockUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


@dataclass
class MockToolCall:
    id: str = "call_abc"
    type: str = "function"
    function: Any = None


@dataclass
class MockMessage:
    role: str = "assistant"
    content: str = "Hello!"
    tool_calls: list[MockToolCall] | None = None


@dataclass
class MockChoice:
    index: int = 0
    message: MockMessage | None = None
    finish_reason: str = "stop"


@dataclass
class MockChatCompletion:
    id: str = "chatcmpl-abc"
    model: str = "gpt-4o"
    choices: list[MockChoice] | None = None
    usage: MockUsage | None = None

    def __post_init__(self) -> None:
        if self.choices is None:
            self.choices = [MockChoice(message=MockMessage())]
        if self.usage is None:
            self.usage = MockUsage()


def _make_mock_openai_module() -> types.ModuleType:
    """Create a mock openai module hierarchy for testing."""
    # Create module structure
    openai_mod = types.ModuleType("openai")
    resources_mod = types.ModuleType("openai.resources")
    chat_mod = types.ModuleType("openai.resources.chat")
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, *args: Any, **kwargs: Any) -> MockChatCompletion:
            return MockChatCompletion(
                model=kwargs.get("model", "gpt-4o"),
            )

    class AsyncCompletions:
        async def create(self, *args: Any, **kwargs: Any) -> MockChatCompletion:
            return MockChatCompletion(
                model=kwargs.get("model", "gpt-4o"),
            )

    completions_mod.Completions = Completions  # type: ignore[attr-defined]
    completions_mod.AsyncCompletions = AsyncCompletions  # type: ignore[attr-defined]
    chat_mod.completions = completions_mod  # type: ignore[attr-defined]
    resources_mod.chat = chat_mod  # type: ignore[attr-defined]
    openai_mod.resources = resources_mod  # type: ignore[attr-defined]

    return openai_mod


@pytest.fixture()
def mock_openai() -> types.ModuleType:
    """Install mock openai module and clean up tracer state."""
    mod = _make_mock_openai_module()
    completions_mod = mod.resources.chat.completions  # type: ignore[attr-defined]

    # Install into sys.modules
    sys.modules["openai"] = mod
    sys.modules["openai.resources"] = mod.resources  # type: ignore[attr-defined]
    sys.modules["openai.resources.chat"] = mod.resources.chat  # type: ignore[attr-defined]
    sys.modules["openai.resources.chat.completions"] = completions_mod

    yield mod

    # Cleanup
    for key in list(sys.modules):
        if key.startswith("openai"):
            del sys.modules[key]

    # Reset tracer state
    if "lumen.integrations.openai_tracer" in sys.modules:
        tracer_mod = sys.modules["lumen.integrations.openai_tracer"]
        tracer_mod._patched = False  # type: ignore[attr-defined]
        tracer_mod._HAS_OPENAI = False  # type: ignore[attr-defined]


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    """Provide a temp directory for trace output."""
    d = tmp_path / "traces"
    d.mkdir()
    return d


class TestOpenAITracerStandalone:
    """Test OpenAI tracing without active trace session."""

    def test_sync_call_creates_trace(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force reimport with mock installed
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        assert patch_openai(config) is True

        # Make a call
        client = completions_mod.Completions()
        response = client.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert response.model == "gpt-4o"

        # Verify trace was written
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            trace_data = json.load(f)

        assert trace_data["agent_id"] == "openai"
        assert trace_data["status"] == "Completed"
        assert len(trace_data["steps"]) == 1

        step = trace_data["steps"][0]
        assert step["step_type"]["LlmCall"]["model"] == "gpt-4o"
        assert step["step_type"]["LlmCall"]["finish_reason"] == "stop"
        assert step["tokens"]["prompt_tokens"] == 100
        assert step["tokens"]["completion_tokens"] == 50
        assert trace_data["total_cost_usd"] > 0

        unpatch_openai()

    def test_tool_use_detected(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]

        # Override create to return tool_calls
        original_create = completions_mod.Completions.create

        def tool_create(self: Any, *args: Any, **kwargs: Any) -> MockChatCompletion:
            return MockChatCompletion(
                model="gpt-4o",
                choices=[MockChoice(
                    message=MockMessage(
                        content="",
                        tool_calls=[MockToolCall(id="call_1")],
                    ),
                    finish_reason="tool_calls",
                )],
            )

        completions_mod.Completions.create = tool_create
        patch_openai(config)

        client = completions_mod.Completions()
        response = client.create(model="gpt-4o", messages=[])

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            trace_data = json.load(f)

        step = trace_data["steps"][0]
        assert step["step_type"]["LlmCall"]["finish_reason"] == "tool_use"

        # Restore
        completions_mod.Completions.create = original_create
        unpatch_openai()

    def test_sampling_rate_zero(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {
                "trace_dir": str(trace_dir),
                "sampling_rate": 0.0,
            },
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        patch_openai(config)

        client = completions_mod.Completions()
        client.create(model="gpt-4o", messages=[])

        # No trace should be written
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 0

        unpatch_openai()

    def test_trace_readable_by_load_traces(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai
        from lumen.trace_reader import load_traces

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        patch_openai(config)

        client = completions_mod.Completions()
        client.create(model="gpt-4o", messages=[])

        traces = load_traces(str(trace_dir))
        assert len(traces) == 1
        assert traces[0].status == "Completed"
        assert traces[0].steps[0].step_type == "LlmCall"
        assert traces[0].steps[0].model == "gpt-4o"

        unpatch_openai()


class TestOpenAITracerWithSession:
    """Test OpenAI tracing with trace() context manager."""

    def test_grouped_calls(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.instrument import trace
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        patch_openai(config)

        with trace("research", config=config) as session:
            client = completions_mod.Completions()
            client.create(model="gpt-4o", messages=[])
            client.create(model="gpt-4o", messages=[])

        # Should produce ONE trace with TWO steps
        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            trace_data = json.load(f)

        assert trace_data["agent_id"] == "research"
        assert len(trace_data["steps"]) == 2
        assert trace_data["total_tokens"]["prompt_tokens"] == 200
        assert trace_data["total_tokens"]["completion_tokens"] == 100

        unpatch_openai()

    def test_session_with_user_id(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.instrument import trace
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]
        patch_openai(config)

        with trace("agent", user_id="u-123", session_id="s-456", config=config):
            client = completions_mod.Completions()
            client.create(model="gpt-4o", messages=[])

        trace_files = list(trace_dir.glob("*.json"))
        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["user_id"] == "u-123"
        assert data["session_id"] == "s-456"

        unpatch_openai()

    def test_session_error_handling(
        self,
        mock_openai: types.ModuleType,
        trace_dir: Path,
    ) -> None:
        if "lumen.integrations.openai_tracer" in sys.modules:
            del sys.modules["lumen.integrations.openai_tracer"]

        from lumen.config import LumenConfig
        from lumen.instrument import trace
        from lumen.integrations.openai_tracer import patch_openai, unpatch_openai

        config = LumenConfig.from_dict({
            "tracing": {"trace_dir": str(trace_dir)},
        })

        completions_mod = mock_openai.resources.chat.completions  # type: ignore[attr-defined]

        # Make create raise an error
        original = completions_mod.Completions.create
        def failing_create(self: Any, *args: Any, **kwargs: Any) -> None:
            raise ValueError("API key invalid")
        completions_mod.Completions.create = failing_create

        patch_openai(config)

        with pytest.raises(ValueError, match="API key invalid"):
            with trace("agent", config=config):
                client = completions_mod.Completions()
                client.create(model="gpt-4o", messages=[])

        trace_files = list(trace_dir.glob("*.json"))
        assert len(trace_files) == 1

        with open(trace_files[0]) as f:
            data = json.load(f)

        assert data["status"] == {"Failed": "API key invalid"}

        completions_mod.Completions.create = original
        unpatch_openai()
