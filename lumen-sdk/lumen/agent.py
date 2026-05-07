"""High-level Agent API — coming in Lumen v0.2.

For v0.1, use the LangGraph integration for agent execution with tracing:

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
    """Configuration for a Lumen agent."""

    model: str = "claude-sonnet-4-6"
    max_iterations: int = 20
    max_cost_usd: float | None = None
    trace_dir: str = "./traces"
    tools: list[Any] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of an agent run."""

    output: str
    trace_id: str
    cost_usd: float
    iterations: int
    duration_s: float
    success: bool


class Agent:
    """A durable, observable AI agent.

    .. note::

        ``Agent.run()`` is not yet implemented in Lumen v0.1.
        Use the LangGraph integration instead::

            from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
            graph = create_react_agent(model, tools, checkpointer=LumenCheckpointer())
            result = graph.invoke(input, config={"callbacks": [LumenTracer()]})

        Direct ``Agent.run()`` with WAL-backed execution is planned for v0.2.
    """

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
        """Run the agent on a task.

        .. note::

            Not yet implemented in v0.1. Use ``LumenTracer`` with LangGraph instead.

        Raises:
            NotImplementedError: Always. Direct agent execution coming in v0.2.
        """
        raise NotImplementedError(
            "Agent.run() is not yet implemented in Lumen v0.1. "
            "Use the LangGraph integration instead:\n\n"
            "  from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer\n"
            "  graph = create_react_agent(model, tools, checkpointer=LumenCheckpointer())\n"
            "  result = graph.invoke(input, config={'callbacks': [LumenTracer()]})\n\n"
            "Direct Agent.run() with WAL-backed execution is planned for v0.2."
        )

    @property
    def last_trace_id(self) -> str | None:
        """Trace ID of the most recent run."""
        return self._last_trace_id


def traced(fn: Callable | None = None, *, trace_dir: str = "./traces"):
    """Decorator that automatically traces an async function.

    .. note::

        Not yet implemented in v0.1. Use ``LumenTracer`` callback instead.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # TODO v0.2: Start trace session, record inputs/outputs
            result = await func(*args, **kwargs)
            return result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
