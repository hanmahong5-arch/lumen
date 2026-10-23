"""Internal stubs — NOT part of the public ``lumen`` API.

``Agent.run()`` and ``@traced`` are unimplemented. Do not import these from
``lumen``; use the LangGraph integration instead::

    from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
    from langgraph.prebuilt import create_react_agent

    checkpointer = LumenCheckpointer(storage_dir="./checkpoints")
    tracer = LumenTracer(trace_dir="./traces")
    graph = create_react_agent(model, tools, checkpointer=checkpointer)
    result = graph.invoke({"messages": [...]}, config={"callbacks": [tracer]})
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class AgentConfig:
    """Configuration for a Lumen agent (stub; unused)."""

    model: str = "claude-sonnet-4-6"
    max_iterations: int = 20
    max_cost_usd: float | None = None
    trace_dir: str = "./traces"
    tools: list[Any] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of an agent run (stub; unused)."""

    output: str
    trace_id: str
    cost_usd: float
    iterations: int
    duration_s: float
    success: bool


class Agent:
    """Unimplemented high-level agent. Prefer LangGraph + LumenTracer."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        tools: Sequence[Any] | None = None,
        max_iterations: int = 20,
        max_cost_usd: float | None = None,
        trace_dir: str = "./traces",
    ):
        self.config = AgentConfig(
            model=model,
            max_iterations=max_iterations,
            max_cost_usd=max_cost_usd,
            trace_dir=trace_dir,
            tools=list(tools or []),
        )
        self._last_trace_id: str | None = None

    async def run(self, task: str) -> AgentResult:
        """Not implemented — raises ``NotImplementedError``."""
        raise NotImplementedError(
            "Agent.run() is not implemented. "
            "Use the LangGraph integration instead:\n\n"
            "  from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer\n"
            "  graph = create_react_agent(model, tools, checkpointer=LumenCheckpointer())\n"
            "  result = graph.invoke(input, config={'callbacks': [LumenTracer()]})"
        )

    @property
    def last_trace_id(self) -> str | None:
        """Trace ID of the most recent run."""
        return self._last_trace_id


def traced(fn: Callable | None = None, *, trace_dir: str = "./traces"):
    """Unimplemented decorator. Prefer ``traced_fn`` or ``LumenTracer``."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
