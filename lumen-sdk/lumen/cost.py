"""LLM cost tracking and reporting.

Loads trace JSON files, estimates costs from token usage when
total_cost_usd is zero, and aggregates by agent/model with anomaly detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lumen.pricing import estimate_cost
from lumen.trace_reader import AgentTrace, load_traces


@dataclass
class CostAnomaly:
    """A run flagged for anomalous cost."""

    trace_id: str
    agent_name: str
    cost_usd: float
    multiplier: float
    reason: str


@dataclass
class CostReport:
    """Aggregated cost report."""

    total_usd: float = 0.0
    total_runs: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_agent: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)
    anomalies: list[CostAnomaly] = field(default_factory=list)


def _parse_duration_ms(last: str) -> int:
    """Parse a duration string like '24h' or '7d' into milliseconds."""
    s = last.strip()
    try:
        if s.endswith("h"):
            return int(s[:-1]) * 3_600_000
        if s.endswith("d"):
            return int(s[:-1]) * 86_400_000
        if s.endswith("m"):
            return int(s[:-1]) * 60_000
    except ValueError:
        pass
    # Default 24h
    return 86_400_000


def _trace_cost(trace: AgentTrace) -> float:
    """Get or estimate cost for a trace.

    If total_cost_usd > 0, use it directly.
    Otherwise estimate from per-step token usage and model pricing.
    """
    if trace.total_cost_usd > 0.0:
        return trace.total_cost_usd

    # Estimate from individual LLM call steps
    total = 0.0
    for step in trace.steps:
        if step.step_type == "LlmCall" and step.tokens is not None and step.model:
            total += estimate_cost(
                step.model,
                step.tokens.prompt_tokens,
                step.tokens.completion_tokens,
            )

    # Fallback: estimate from aggregate tokens if no per-step data
    if total == 0.0 and trace.total_tokens.total() > 0:
        # Use a generic rate since we don't know the model
        total = estimate_cost(
            "unknown",
            trace.total_tokens.prompt_tokens,
            trace.total_tokens.completion_tokens,
        )

    return total


class CostTracker:
    """Track and aggregate LLM costs across agent runs.

    Reads trace JSON files from a directory and estimates costs from
    token usage when total_cost_usd is not recorded.

    Example::

        tracker = CostTracker(trace_dir="./traces")
        report = tracker.report(last="24h")
        print(f"Total: ${report.total_usd:.2f}")
        for agent, cost in report.by_agent.items():
            print(f"  {agent}: ${cost:.2f}")

        # Or with config:
        from lumen import LumenConfig
        tracker = CostTracker(config=LumenConfig.auto())
    """

    def __init__(
        self,
        trace_dir: str = "./traces",
        *,
        config: Any = None,
    ):
        if config is not None:
            self.trace_dir = config.tracing.trace_dir
            self._anomaly_multiplier = config.cost.anomaly_multiplier
            self._anomaly_min_usd = config.cost.anomaly_min_usd
        else:
            self.trace_dir = trace_dir
            self._anomaly_multiplier = 2.0
            self._anomaly_min_usd = 0.01

    def report(self, last: str = "24h") -> CostReport:
        """Generate a cost report for the given time window.

        Args:
            last: Time window string like "24h", "7d", "30d".

        Returns:
            CostReport with totals, breakdowns, and anomaly flags.
        """
        traces = load_traces(self.trace_dir)

        # Filter by time window
        duration_ms = _parse_duration_ms(last)
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - duration_ms

        filtered = [t for t in traces if t.started_at_ms >= since_ms]

        if not filtered:
            return CostReport()

        # Calculate per-trace costs
        costs: list[tuple[AgentTrace, float]] = [(t, _trace_cost(t)) for t in filtered]

        total_usd = sum(c for _, c in costs)
        total_runs = len(filtered)
        total_prompt = sum(t.total_tokens.prompt_tokens for t in filtered)
        total_completion = sum(t.total_tokens.completion_tokens for t in filtered)

        # Group by agent
        by_agent: dict[str, float] = {}
        for trace, cost in costs:
            by_agent[trace.agent_id] = by_agent.get(trace.agent_id, 0.0) + cost

        # Group by model (extract from LlmCall steps)
        by_model: dict[str, float] = {}
        for trace, cost in costs:
            llm_models = [
                step.model
                for step in trace.steps
                if step.step_type == "LlmCall" and step.model
            ]
            if llm_models:
                per_call = cost / len(llm_models)
                for model in llm_models:
                    by_model[model] = by_model.get(model, 0.0) + per_call

        # Detect anomalies: runs > Nx average cost
        avg_cost = total_usd / total_runs if total_runs > 0 else 0.0
        anomalies: list[CostAnomaly] = []
        for trace, cost in costs:
            if cost > avg_cost * self._anomaly_multiplier and cost > self._anomaly_min_usd:
                multiplier = cost / avg_cost if avg_cost > 0 else 0.0
                anomalies.append(CostAnomaly(
                    trace_id=trace.trace_id,
                    agent_name=trace.agent_id,
                    cost_usd=cost,
                    multiplier=multiplier,
                    reason=f"{len(trace.steps)} steps, {multiplier:.0f}x avg",
                ))

        return CostReport(
            total_usd=total_usd,
            total_runs=total_runs,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            by_agent=by_agent,
            by_model=by_model,
            anomalies=anomalies,
        )
