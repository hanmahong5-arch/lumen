"""Deterministic replay of agent runs from trace JSON files.

Loads trace data and reconstructs the step-by-step execution sequence,
grouping LLM decisions with their tool call results. Zero LLM cost.

Step shape mirrors Rust ``lumen-core::replay::build_replay_steps``:
one replay step per LLM iteration (multiple tools → one ``tool_call`` step
with ``tool_calls`` populated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumen.pricing import estimate_cost
from lumen.trace_reader import AgentTrace, load_traces


@dataclass
class ReplayToolCall:
    """One tool invocation inside a tool_call replay step (mirrors Rust)."""

    tool_name: str
    success: bool = True
    call_id: str = ""
    output_preview: str = ""


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
    duration_ms: int = 0
    tool_calls: list[ReplayToolCall] = field(default_factory=list)


@dataclass
class RecoveryInfo:
    """Crash-recovery provenance (mirrors Rust ``replay::RecoveryInfo``)."""

    parent_trace_id: str
    resumed_at_iteration: int | None = None


@dataclass
class ReplayTrace:
    """Full replay of an agent run."""

    trace_id: str = ""
    agent_name: str = ""
    steps: list[ReplayStep] = field(default_factory=list)
    original_cost_usd: float = 0.0
    total_iterations: int = 0
    success: bool = False
    total_duration_ms: int = 0
    recovery: RecoveryInfo | None = None


def _extract_recovery(trace: AgentTrace) -> RecoveryInfo | None:
    """Mirror Rust ``extract_recovery`` — parent_trace_id + optional resume iter."""
    if not trace.parent_trace_id:
        return None
    resumed: int | None = None
    if trace.steps:
        for key, val in trace.steps[0].metadata:
            if key == "resumed_at_iteration":
                try:
                    resumed = int(val)
                except ValueError:
                    resumed = None
                break
    return RecoveryInfo(
        parent_trace_id=trace.parent_trace_id,
        resumed_at_iteration=resumed,
    )


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


def _flush_tools(
    replay_steps: list[ReplayStep],
    step_num: int,
    pending: list[ReplayToolCall],
    pending_ts: int,
    pending_dur: int,
) -> int:
    """Append one aggregated tool_call step; return next step_num."""
    if not pending:
        return step_num
    step_num += 1
    first = pending[0]
    replay_steps.append(
        ReplayStep(
            step=step_num,
            decision_type="tool_call",
            tool_name=first.tool_name,
            success=all(t.success for t in pending),
            timestamp_ms=pending_ts,
            duration_ms=pending_dur,
            tool_calls=list(pending),
        )
    )
    return step_num


def _build_replay_steps(trace: AgentTrace) -> list[ReplayStep]:
    """Group trace steps into replay steps (Rust ``build_replay_steps``)."""
    replay_steps: list[ReplayStep] = []
    step_num = 0
    pending: list[ReplayToolCall] = []
    pending_ts = 0
    pending_dur = 0

    for s in trace.steps:
        if s.step_type == "LlmCall":
            step_num = _flush_tools(
                replay_steps, step_num, pending, pending_ts, pending_dur
            )
            pending = []
            pending_dur = 0

            finish = s.finish_reason or ""
            if finish in ("stop", "end_turn"):
                step_num += 1
                replay_steps.append(
                    ReplayStep(
                        step=step_num,
                        decision_type="final_answer",
                        content=f"Agent completed ({finish})",
                        timestamp_ms=s.started_at_ms,
                        duration_ms=s.duration_ms,
                    )
                )
            else:
                pending_ts = s.started_at_ms
                pending_dur += s.duration_ms

        elif s.step_type == "ToolCall":
            if pending_ts == 0:
                pending_ts = s.started_at_ms
            pending_dur += s.duration_ms
            ok = s.tool_success if s.tool_success is not None else True
            pending.append(
                ReplayToolCall(
                    tool_name=s.tool_name or "(unknown)",
                    success=ok,
                    call_id=f"step_{s.step_num}",
                    output_preview=(
                        f"ok ({s.duration_ms}ms)" if ok else "failed"
                    ),
                )
            )

        elif s.step_type == "Error":
            step_num = _flush_tools(
                replay_steps, step_num, pending, pending_ts, pending_dur
            )
            pending = []
            pending_dur = 0
            step_num += 1
            replay_steps.append(
                ReplayStep(
                    step=step_num,
                    decision_type="final_answer",
                    content=f"Error: {s.error_message or 'unknown error'}",
                    success=False,
                    timestamp_ms=s.started_at_ms,
                    duration_ms=s.duration_ms,
                )
            )

        elif s.step_type == "Checkpoint":
            continue

    step_num = _flush_tools(replay_steps, step_num, pending, pending_ts, pending_dur)
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

        Note:
            Mid-run filtering (``--from-step``) is a CLI concern; this API
            always returns the full reconstructed sequence.
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

        llm_count = sum(1 for s in target.steps if s.step_type == "LlmCall")
        steps = _build_replay_steps(target)
        total_dur = 0
        if target.completed_at_ms and target.started_at_ms:
            total_dur = max(0, target.completed_at_ms - target.started_at_ms)

        return ReplayTrace(
            trace_id=target.trace_id,
            agent_name=target.agent_id,
            steps=steps,
            original_cost_usd=_estimate_trace_cost(target),
            total_iterations=llm_count,
            success=target.status == "Completed",
            total_duration_ms=total_dur,
            recovery=_extract_recovery(target),
        )

    def list_traces(self) -> list[str]:
        """List available trace IDs, newest first."""
        traces = load_traces(self.trace_dir)
        return [t.trace_id for t in traces]
