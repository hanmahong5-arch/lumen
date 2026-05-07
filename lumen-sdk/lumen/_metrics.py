"""Runtime metrics collection for live observability.

Wires up the MetricsCollector protocol and MetricsWindow data structure
that were previously defined but never connected to the trace pipeline.

Every LLM call records latency, cost, and token usage into sliding
windows, enabling real-time p50/p99/mean queries without loading
trace files from disk.

Example::

    from lumen._metrics import RuntimeMetrics

    metrics = RuntimeMetrics()
    metrics.record(model="gpt-4o", agent_name="research",
                   prompt_tokens=500, completion_tokens=200,
                   cost_usd=0.01, duration_ms=1200)

    print(metrics.summary())
    # {'calls': 1, 'total_cost_usd': 0.01, 'latency_p50_ms': 1200.0, ...}

    print(metrics.by_model("gpt-4o").latency.mean)
    # 1200.0
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from lumen._ring_buffer import MetricsWindow


@dataclass
class ModelMetrics:
    """Per-model metrics windows."""

    latency: MetricsWindow = field(default_factory=lambda: MetricsWindow(1000))
    cost: MetricsWindow = field(default_factory=lambda: MetricsWindow(1000))
    prompt_tokens: MetricsWindow = field(default_factory=lambda: MetricsWindow(1000))
    completion_tokens: MetricsWindow = field(default_factory=lambda: MetricsWindow(1000))
    calls: int = 0
    errors: int = 0


class RuntimeMetrics:
    """Collects real-time metrics from LLM calls across the SDK.

    Thread-safe — uses a lock for shared mutable state. Each model
    and agent gets its own set of MetricsWindow instances.

    Implements the MetricsCollector protocol from _protocols.py.
    """

    __slots__ = ("_lock", "_global", "_by_model", "_by_agent", "_total_calls", "_total_errors")

    def __init__(self, window_size: int = 1000) -> None:
        self._lock = threading.Lock()
        self._global = ModelMetrics(
            latency=MetricsWindow(window_size),
            cost=MetricsWindow(window_size),
            prompt_tokens=MetricsWindow(window_size),
            completion_tokens=MetricsWindow(window_size),
        )
        self._by_model: dict[str, ModelMetrics] = {}
        self._by_agent: dict[str, ModelMetrics] = {}
        self._total_calls = 0
        self._total_errors = 0

    def record(
        self,
        *,
        model: str,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        duration_ms: int,
    ) -> None:
        """Record a completed LLM call into all relevant windows."""
        with self._lock:
            self._total_calls += 1

            # Global
            self._global.latency.record(float(duration_ms))
            self._global.cost.record(cost_usd)
            self._global.prompt_tokens.record(float(prompt_tokens))
            self._global.completion_tokens.record(float(completion_tokens))
            self._global.calls += 1

            # Per-model
            mm = self._by_model.get(model)
            if mm is None:
                mm = ModelMetrics()
                self._by_model[model] = mm
            mm.latency.record(float(duration_ms))
            mm.cost.record(cost_usd)
            mm.prompt_tokens.record(float(prompt_tokens))
            mm.completion_tokens.record(float(completion_tokens))
            mm.calls += 1

            # Per-agent
            am = self._by_agent.get(agent_name)
            if am is None:
                am = ModelMetrics()
                self._by_agent[agent_name] = am
            am.latency.record(float(duration_ms))
            am.cost.record(cost_usd)
            am.prompt_tokens.record(float(prompt_tokens))
            am.completion_tokens.record(float(completion_tokens))
            am.calls += 1

    def record_error(self) -> None:
        """Increment the global error counter."""
        with self._lock:
            self._total_errors += 1
            self._global.errors += 1

    # -- MetricsCollector protocol methods --

    def record_latency(self, duration_ms: float) -> None:
        with self._lock:
            self._global.latency.record(duration_ms)

    def record_cost(self, cost_usd: float) -> None:
        with self._lock:
            self._global.cost.record(cost_usd)

    def record_tokens(self, prompt: int, completion: int) -> None:
        with self._lock:
            self._global.prompt_tokens.record(float(prompt))
            self._global.completion_tokens.record(float(completion))

    # -- Query API --

    def model_names(self) -> list[str]:
        """Return all model names that have been recorded."""
        with self._lock:
            return list(self._by_model.keys())

    def agent_names(self) -> list[str]:
        """Return all agent names that have been recorded."""
        with self._lock:
            return list(self._by_agent.keys())

    def by_model(self, model: str) -> ModelMetrics | None:
        """Return per-model metrics, or None if not seen."""
        with self._lock:
            return self._by_model.get(model)

    def by_agent(self, agent_name: str) -> ModelMetrics | None:
        """Return per-agent metrics, or None if not seen."""
        with self._lock:
            return self._by_agent.get(agent_name)

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_errors(self) -> int:
        return self._total_errors

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of global metrics as a plain dict.

        Useful for dashboards, logging, and health checks.
        """
        with self._lock:
            g = self._global
            return {
                "calls": self._total_calls,
                "errors": self._total_errors,
                "total_cost_usd": g.cost.total,
                "latency_p50_ms": g.latency.percentile(50),
                "latency_p99_ms": g.latency.percentile(99),
                "latency_mean_ms": g.latency.mean,
                "cost_mean_usd": g.cost.mean,
                "avg_prompt_tokens": g.prompt_tokens.mean,
                "avg_completion_tokens": g.completion_tokens.mean,
                "models": list(self._by_model.keys()),
                "agents": list(self._by_agent.keys()),
            }

    def clear(self) -> None:
        """Reset all metrics. Useful for testing."""
        with self._lock:
            self._global = ModelMetrics()
            self._by_model.clear()
            self._by_agent.clear()
            self._total_calls = 0
            self._total_errors = 0
