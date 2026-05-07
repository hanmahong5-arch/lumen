"""LangGraph callback handler that writes AgentTrace JSON files.

Bridges LangGraph's callback interface to Lumen's trace format,
enabling replay, cost tracking, and observability for LangGraph agents.

Example::

    from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer

    checkpointer = LumenCheckpointer(storage_dir="./checkpoints")
    tracer = LumenTracer(trace_dir="./traces")

    graph = create_react_agent(model, tools, checkpointer=checkpointer)
    result = graph.invoke(
        {"messages": [HumanMessage("Research quantum computing")]},
        config={"callbacks": [tracer]},
    )

    # Traces automatically written to ./traces/{trace_id}.json
    # lumen cost --last 24h --trace-dir ./traces
    # lumen replay <trace-id> --trace-dir ./traces

Install with: ``pip install "lumen-ai[langgraph]"``
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from lumen.pricing import estimate_cost

try:
    from langchain_core.callbacks import BaseCallbackHandler

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

# Dynamic base class
_Base = BaseCallbackHandler if _HAS_LANGCHAIN else object


class LumenTracer(_Base):  # type: ignore[misc]
    """LangChain callback handler that writes Lumen trace JSON files.

    Captures LLM calls, tool executions, and errors during a LangGraph
    agent run and writes the trace to a JSON file compatible with
    ``lumen replay`` and ``lumen cost``.

    Args:
        trace_dir: Directory to write trace JSON files.
        agent_name: Name to identify this agent in traces.

    Requires ``langchain-core>=0.3``. Install with::

        pip install "lumen-ai[langgraph]"
    """

    def __init__(
        self,
        trace_dir: str = "./traces",
        agent_name: str = "langgraph-agent",
    ) -> None:
        if _HAS_LANGCHAIN:
            super().__init__()
        self.trace_dir = Path(trace_dir)
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # deferred to _write_trace where we warn properly
        self.agent_name = agent_name

        # State for current trace
        self._trace_id = ""
        self._started_at_ms = 0
        self._steps: list[dict[str, Any]] = []
        self._step_num = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._status = "Running"
        self._failure_reason: str | None = None
        self._chain_depth = 0

        # Per-call timing
        self._llm_start_ms = 0
        self._tool_start_ms = 0

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _ensure_trace(self) -> None:
        """Initialize trace state if not already started."""
        if not self._trace_id:
            self._trace_id = uuid.uuid4().hex[:12]
            self._started_at_ms = self._now_ms()
            self._steps = []
            self._step_num = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_cost_usd = 0.0
            self._status = "Running"
            self._failure_reason = None

    # ── LLM callbacks ────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        self._llm_start_ms = self._now_ms()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        self._llm_start_ms = self._now_ms()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        duration_ms = self._now_ms() - self._llm_start_ms

        # Extract token usage and model from response
        model = ""
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = response.llm_output
            model = llm_output.get("model_name", "") or llm_output.get("model", "")
            usage = llm_output.get("token_usage", {}) or llm_output.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0

        # Try to get model/tokens from generations
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        info = gen.generation_info
                        if not finish_reason or finish_reason == "stop":
                            finish_reason = info.get("finish_reason", "stop") or "stop"
                    if hasattr(gen, "message"):
                        msg = gen.message
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            um = msg.usage_metadata
                            if hasattr(um, "input_tokens"):
                                prompt_tokens = prompt_tokens or um.input_tokens
                            if hasattr(um, "output_tokens"):
                                completion_tokens = completion_tokens or um.output_tokens
                        if hasattr(msg, "response_metadata") and msg.response_metadata:
                            rm = msg.response_metadata
                            model = model or rm.get("model_name", "") or rm.get("model", "")
                        # Check if LLM decided to use tools
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            finish_reason = "tool_use"

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Estimate cost for this call
        if model and (prompt_tokens > 0 or completion_tokens > 0):
            self._total_cost_usd += estimate_cost(model, prompt_tokens, completion_tokens)

        tokens_dict = None
        if prompt_tokens > 0 or completion_tokens > 0:
            tokens_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        self._steps.append({
            "step_num": self._step_num,
            "step_type": {"LlmCall": {"model": model, "finish_reason": finish_reason}},
            "started_at_ms": self._llm_start_ms,
            "duration_ms": duration_ms,
            "tokens": tokens_dict,
            "metadata": [],
        })
        self._step_num += 1

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        self._steps.append({
            "step_num": self._step_num,
            "step_type": {"Error": {"message": str(error)}},
            "started_at_ms": self._now_ms(),
            "duration_ms": 0,
            "tokens": None,
            "metadata": [],
        })
        self._step_num += 1
        self._status = "Failed"
        self._failure_reason = str(error)

    # ── Tool callbacks ───────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        self._tool_start_ms = self._now_ms()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        duration_ms = self._now_ms() - self._tool_start_ms
        tool_name = kwargs.get("name", "unknown_tool")

        self._steps.append({
            "step_num": self._step_num,
            "step_type": {"ToolCall": {"tool_name": tool_name, "success": True}},
            "started_at_ms": self._tool_start_ms,
            "duration_ms": duration_ms,
            "tokens": None,
            "metadata": [],
        })
        self._step_num += 1

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        duration_ms = self._now_ms() - self._tool_start_ms
        tool_name = kwargs.get("name", "unknown_tool")

        self._steps.append({
            "step_num": self._step_num,
            "step_type": {"ToolCall": {"tool_name": tool_name, "success": False}},
            "started_at_ms": self._tool_start_ms,
            "duration_ms": duration_ms,
            "tokens": None,
            "metadata": [],
        })
        self._step_num += 1

    # ── Chain callbacks (for detecting run completion) ────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._ensure_trace()
        self._chain_depth += 1

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._chain_depth -= 1
        # Only write trace when the outermost chain ends
        if self._chain_depth <= 0 and self._trace_id:
            if self._status == "Running":
                self._status = "Completed"
            self._write_trace()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._chain_depth -= 1
        if self._chain_depth <= 0 and self._trace_id:
            self._status = "Failed"
            self._failure_reason = str(error)
            # Record error step
            self._steps.append({
                "step_num": self._step_num,
                "step_type": {"Error": {"message": str(error)}},
                "started_at_ms": self._now_ms(),
                "duration_ms": 0,
                "tokens": None,
                "metadata": [],
            })
            self._step_num += 1
            self._write_trace()

    # ── Trace persistence ────────────────────────────────────────────────

    def _write_trace(self) -> None:
        """Write the complete trace to a JSON file atomically.

        Never raises — trace write failures are logged as warnings.
        """
        if not self._trace_id:
            return

        status: str | dict[str, str]
        if self._status == "Failed" and self._failure_reason:
            status = {"Failed": self._failure_reason}
        else:
            status = self._status

        trace_data = {
            "trace_id": self._trace_id,
            "agent_id": self.agent_name,
            "task_id": 0,
            "steps": self._steps,
            "status": status,
            "total_tokens": {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
            },
            "total_cost_usd": self._total_cost_usd,
            "started_at_ms": self._started_at_ms,
            "completed_at_ms": self._now_ms(),
        }

        from lumen._context import _write_trace_safe
        _write_trace_safe(trace_data, str(self.trace_dir), self._trace_id)

        # Reset for next trace
        self._trace_id = ""

    @property
    def last_trace_id(self) -> str:
        """The trace ID of the most recently completed trace."""
        return self._trace_id

    def flush(self) -> None:
        """Force write the current trace (useful for testing or cleanup)."""
        if self._trace_id:
            if self._status == "Running":
                self._status = "Completed"
            self._write_trace()
