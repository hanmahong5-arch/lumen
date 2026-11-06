"""Lumen — Illuminate your AI agents.

Replay any run. Track every dollar. Resume after a crash (LangGraph checkpointer).

Quick start::

    from lumen import instrument, trace

    instrument()  # patches OpenAI / Anthropic when installed
    # LangGraph: pass LumenTracer via callbacks (see integrations.langgraph)

    with trace("research-task"):
        response = client.chat.completions.create(model="gpt-4o", ...)

    from lumen import TraceQuery
    results = TraceQuery("./traces").agent("research").since("7d").execute()
"""

__version__ = "0.2.1"

from lumen._anomaly import AnomalyDetector
from lumen._batch_writer import BatchTraceWriter
from lumen._budget import BudgetExceededError, BudgetTracker
from lumen._errors import (
    ConfigWarning,
    IntegrationWarning,
    LumenWarning,
    RedactionWarning,
    TraceWriteWarning,
)
from lumen._otlp import OtlpHttpTraceWriter
from lumen._redaction import RedactionEngine, RedactionPattern
from lumen._writers import CompositeWriter, FileTraceWriter, NullWriter
from lumen.config import LumenConfig
from lumen.cost import CostReport, CostTracker
from lumen.instrument import atrace, get_metrics, instrument, trace, traced_fn, uninstrument
from lumen.pricing import estimate_cost, is_fallback_pricing
from lumen.query import TraceQuery, TraceQueryResult
from lumen.replay import ReplayEngine, ReplayTrace, ReplayStep, ReplayToolCall
from lumen.trace_reader import AgentTrace, load_traces

# Internal helpers (trie, ring buffer, event bus, middleware, agent stubs) stay
# importable from their modules but are not part of the public black-box surface.

__all__ = [
    "AgentTrace",
    "AnomalyDetector",
    "BatchTraceWriter",
    "BudgetExceededError",
    "BudgetTracker",
    "CompositeWriter",
    "ConfigWarning",
    "CostReport",
    "CostTracker",
    "FileTraceWriter",
    "get_metrics",
    "IntegrationWarning",
    "is_fallback_pricing",
    "LumenConfig",
    "LumenWarning",
    "NullWriter",
    "OtlpHttpTraceWriter",
    "RedactionEngine",
    "RedactionPattern",
    "RedactionWarning",
    "ReplayEngine",
    "ReplayStep",
    "ReplayToolCall",
    "ReplayTrace",
    "TraceQuery",
    "TraceQueryResult",
    "TraceWriteWarning",
    "atrace",
    "estimate_cost",
    "instrument",
    "load_traces",
    "trace",
    "traced_fn",
    "uninstrument",
]
