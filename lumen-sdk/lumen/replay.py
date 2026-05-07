"""Deterministic replay of agent runs from trace JSON files.

Loads trace data and reconstructs the step-by-step execution sequence,
grouping LLM decisions with their tool call results. Zero LLM cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.pricing import estimate_cost
from lumen.trace_reader import AgentTrace, load_traces


@dataclass
class ReplayStep:
    """A single step in a replayed agent run."""

    step: int
    decision_type: str  # "tool_call" or "final_answer"
    tool_name: str | None = None
    tool_output: str | None = None
    content: str | None = None
    success: bool = True
    timestamp_ms: int = 0


@dataclass
class ReplayTrace:
    """Full replay of an agent run."""

    trace_id: str = ""
    steps: list[ReplayStep] = field(default_factory=list)
    original_cost_usd: float = 0.0
    total_iterations: int = 0
    success: bool = False


def _estimate_trace_cost(trace: AgentTrace) -> float:
    """Estimate cost from per-step tokens if total_cost_usd is zero."""
    if trace.total_cost_usd > 0.0:
        return trace.total_cost_usd

    total = 0.0
    for step in trace.steps:
        if step.step_type == "LlmCall" and step.tokens is not None and step.model:
            total += estimate_cost(
                step.model,
                step.tokens.prompt_tokens,
                step.tokens.completion_tokens,
            )
    return total


def _build_replay_steps(trace: AgentTrace) -> list[ReplayStep]:
    """Group trace steps into replay steps.

    Algorithm (mirrors Rust lumen-core replay::build_replay_steps):
    - Each LlmCall with finish_reason="tool_use" groups with subsequent ToolCall steps
    - Each LlmCall with terminal finish_reason ("stop","end_turn") becomes a FinalAnswer
    - Error steps become FinalAnswer with error content
    """
    steps_raw = trace.steps
    replay_steps: list[ReplayStep] = []
    step_num = 0
    i = 0

    while i < len(steps_raw):
        s = steps_raw[i]

        if s.step_type == "LlmCall":
            step_num += 1
            finish = s.finish_reason or ""

            if finish in ("stop", "end_turn"):
                # Terminal LLM call → FinalAnswer
                replay_steps.append(ReplayStep(
                    step=step_num,
                    decision_type="final_answer",
                    content=f"Final answer (finish_reason={finish})",
                    timestamp_ms=s.started_at_ms,
                ))
                i += 1
            else:
                # tool_use → collect subsequent ToolCall steps
                i += 1
                while i < len(steps_raw) and steps_raw[i].step_type == "ToolCall":
                    tc = steps_raw[i]
                    replay_steps.append(ReplayStep(
                        step=step_num,
                        decision_type="tool_call",
                        tool_name=tc.tool_name,
                        success=tc.tool_success if tc.tool_success is not None else True,
                        timestamp_ms=tc.started_at_ms,
                    ))
                    i += 1

                # If no tool calls followed, still record the LLM step
                if not replay_steps or replay_steps[-1].step != step_num:
                    replay_steps.append(ReplayStep(
                        step=step_num,
                        decision_type="tool_call",
                        tool_name="(no tools executed)",
                        timestamp_ms=s.started_at_ms,
                    ))

        elif s.step_type == "Error":
            step_num += 1
            replay_steps.append(ReplayStep(
                step=step_num,
                decision_type="final_answer",
                content=f"Error: {s.error_message or 'unknown error'}",
                success=False,
                timestamp_ms=s.started_at_ms,
            ))
            i += 1

        elif s.step_type == "Checkpoint":
            # Skip checkpoints in replay
            i += 1

        else:
            # Standalone ToolCall without preceding LlmCall (unusual)
            step_num += 1
            replay_steps.append(ReplayStep(
                step=step_num,
                decision_type="tool_call",
                tool_name=s.tool_name,
                success=s.tool_success if s.tool_success is not None else True,
                timestamp_ms=s.started_at_ms,
            ))
            i += 1

    return replay_steps


class ReplayEngine:
    """Replay agent runs from trace JSON files — zero LLM cost.

    Every LLM decision and tool result is loaded from trace files.
    Replay reconstructs the exact sequence without any API calls.

    Example::

        engine = ReplayEngine(trace_dir="./traces")
        trace = engine.replay("abc123")
        for step in trace.steps:
            print(f"Step {step.step}: {step.tool_name or step.content}")

        # Or with config:
        from lumen import LumenConfig
        engine = ReplayEngine(config=LumenConfig.auto())
    """

    def __init__(self, trace_dir: str = "./traces", *, config: Any = None):
        if config is not None:
            self.trace_dir = config.tracing.trace_dir
        else:
            self.trace_dir = trace_dir

    def replay(self, trace_id: str) -> ReplayTrace:
        """Replay an agent run by trace ID.

        Args:
            trace_id: The trace identifier from a previous agent run.

        Returns:
            ReplayTrace with the full step-by-step reconstruction.

        Raises:
            ValueError: If trace_id not found in trace directory.
        """
        traces = load_traces(self.trace_dir)
        target = None
        for t in traces:
            if t.trace_id == trace_id:
                target = t
                break

        if target is None:
            raise ValueError(
                f"Trace '{trace_id}' not found in {self.trace_dir}. "
                f"Available: {[t.trace_id for t in traces[:5]]}"
            )

        # Count LLM iterations
        llm_count = sum(1 for s in target.steps if s.step_type == "LlmCall")

        return ReplayTrace(
            trace_id=target.trace_id,
            steps=_build_replay_steps(target),
            original_cost_usd=_estimate_trace_cost(target),
            total_iterations=llm_count,
            success=target.status == "Completed",
        )

    def list_traces(self) -> list[str]:
        """List available trace IDs, newest first."""
        traces = load_traces(self.trace_dir)
        return [t.trace_id for t in traces]
