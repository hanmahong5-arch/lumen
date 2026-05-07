"""Lumen — Illuminate your AI agents.

Replay any run. Track every dollar. Never lose agent state.

Quick start::

    from lumen import instrument, trace

    instrument()  # auto-detect OpenAI / Anthropic / LangGraph

    # All LLM calls are now traced automatically.
    # Group calls into one trace:
    with trace("research-task"):
        response = client.chat.completions.create(model="gpt-4o", ...)

    # Async context manager:
    async with atrace("research-task") as t:
        ...

    # Decorator form:
    @traced_fn("my-agent")
    def my_func():
        ...

    # Query traces:
    from lumen import TraceQuery
    results = TraceQuery("./traces").agent("research").since("7d").execute()
"""

__version__ = "0.2.0"

from lumen._anomaly import AnomalyDetector
from lumen._batch_writer import BatchTraceWriter
from lumen._budget import BudgetExceededError, BudgetTracker
from lumen._diff import TraceDiff, compare_traces
from lumen._errors import (
    ConfigWarning,
    IntegrationWarning,
    LumenWarning,
    RedactionWarning,
    TraceWriteWarning,
)
from lumen._event_bus import EventBus, TraceCompleted, BudgetAlert, CostAnomaly, get_default_bus
from lumen._lru_cache import LRUCache
from lumen._metrics import RuntimeMetrics
from lumen._middleware import MiddlewareChain, TraceMiddleware
from lumen._protocols import (
    MetricsCollector,
    PricingProvider,
    RedactionProvider,
    TraceReader,
    TraceWriter,
)
from lumen._redaction import RedactionEngine, RedactionPattern
from lumen._ring_buffer import MetricsWindow, RingBuffer
from lumen._trace_index import TraceIndex, TraceMeta
from lumen._trie import PrefixTrie
from lumen._writers import CompositeWriter, FileTraceWriter, NullWriter
from lumen.agent import Agent, AgentResult, traced
from lumen.config import LumenConfig
from lumen.cost import CostReport, CostTracker
from lumen.instrument import atrace, get_metrics, instrument, trace, traced_fn, uninstrument
from lumen.pricing import estimate_cost
from lumen.query import TraceQuery, TraceQueryResult
from lumen.replay import ReplayEngine, ReplayTrace
from lumen.trace_reader import AgentTrace, load_traces

__all__ = [
    "Agent",
    "AgentResult",
    "AgentTrace",
    "AnomalyDetector",
    "BatchTraceWriter",
    "BudgetAlert",
    "BudgetExceededError",
    "BudgetTracker",
    "compare_traces",
    "CompositeWriter",
    "ConfigWarning",
    "CostAnomaly",
    "CostReport",
    "CostTracker",
    "EventBus",
    "FileTraceWriter",
    "get_default_bus",
    "get_metrics",
    "IntegrationWarning",
    "LRUCache",
    "LumenConfig",
    "LumenWarning",
    "MetricsCollector",
    "MetricsWindow",
    "MiddlewareChain",
    "NullWriter",
    "PrefixTrie",
    "PricingProvider",
    "RedactionEngine",
    "RedactionPattern",
    "RedactionProvider",
    "RedactionWarning",
    "ReplayEngine",
    "ReplayTrace",
    "RingBuffer",
    "RuntimeMetrics",
    "TraceCompleted",
    "TraceDiff",
    "TraceIndex",
    "TraceMeta",
    "TraceMiddleware",
    "TraceQuery",
    "TraceQueryResult",
    "TraceReader",
    "TraceWriter",
    "TraceWriteWarning",
    "atrace",
    "estimate_cost",
    "instrument",
    "load_traces",
    "trace",
    "traced",
    "traced_fn",
    "uninstrument",
]
