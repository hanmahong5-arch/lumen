"""Trace comparison engine using Longest Common Subsequence.

Computes structural differences between two agent traces, enabling
regression debugging: "why did this run cost 10x more?"

Algorithm: LCS on step sequences using (step_type, model/tool_name)
as comparison keys. O(N*M) where N and M are step counts (typically
<100 steps per trace).

Example::

    from lumen._diff import compare_traces
    from lumen.trace_reader import load_traces

    traces = load_traces("./traces")
    diff = compare_traces(traces[0], traces[1])
    print(f"Cost delta: ${diff.cost_delta_usd:+.4f}")
    print(f"Added steps: {len(diff.added)}")
    print(f"Removed steps: {len(diff.removed)}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepSummary:
    """Lightweight step representation for diff output."""

    step_num: int
    step_type: str
    model_or_tool: str
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @staticmethod
    def from_step(step: dict[str, Any]) -> StepSummary:
        """Extract a StepSummary from a raw trace step dict."""
        step_type_data = step.get("step_type", {})
        step_type = "Unknown"
        model_or_tool = ""

        if isinstance(step_type_data, dict):
            if "LlmCall" in step_type_data:
                step_type = "LlmCall"
                model_or_tool = step_type_data["LlmCall"].get("model", "")
            elif "ToolCall" in step_type_data:
                step_type = "ToolCall"
                model_or_tool = step_type_data["ToolCall"].get("tool_name", "")
            elif "Error" in step_type_data:
                step_type = "Error"
                model_or_tool = step_type_data["Error"].get("message", "")[:50]

        tokens = step.get("tokens") or {}
        return StepSummary(
            step_num=step.get("step_num", 0),
            step_type=step_type,
            model_or_tool=model_or_tool,
            duration_ms=step.get("duration_ms", 0),
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0),
        )

    @property
    def _key(self) -> tuple[str, str]:
        """Comparison key for LCS matching."""
        return (self.step_type, self.model_or_tool)


@dataclass
class StepDelta:
    """A step that exists in both traces but with different metrics."""

    step_a: StepSummary
    step_b: StepSummary
    duration_delta_ms: int = 0
    token_delta: int = 0


@dataclass
class TraceDiff:
    """Structural diff between two agent traces."""

    trace_id_a: str
    trace_id_b: str
    added: list[StepSummary] = field(default_factory=list)
    removed: list[StepSummary] = field(default_factory=list)
    changed: list[StepDelta] = field(default_factory=list)
    unchanged: int = 0
    cost_delta_usd: float = 0.0
    token_delta: int = 0
    duration_delta_ms: int = 0
    step_count_a: int = 0
    step_count_b: int = 0


def compare_traces(trace_a: Any, trace_b: Any) -> TraceDiff:
    """Compare two traces and return structural differences.

    Accepts either AgentTrace objects (from trace_reader) or raw
    trace dicts. Uses LCS to align steps by (step_type, model/tool).

    Args:
        trace_a: The baseline trace.
        trace_b: The trace to compare against.

    Returns:
        TraceDiff with added, removed, and changed steps.
    """
    steps_a = _extract_steps(trace_a)
    steps_b = _extract_steps(trace_b)

    tid_a = _extract_id(trace_a)
    tid_b = _extract_id(trace_b)

    cost_a = _extract_cost(trace_a)
    cost_b = _extract_cost(trace_b)

    duration_a = _extract_duration(trace_a)
    duration_b = _extract_duration(trace_b)

    tokens_a = sum(s.prompt_tokens + s.completion_tokens for s in steps_a)
    tokens_b = sum(s.prompt_tokens + s.completion_tokens for s in steps_b)

    # LCS to find matching steps
    lcs_pairs = _lcs(steps_a, steps_b)

    # Build sets of matched indices
    matched_a = {i for i, _ in lcs_pairs}
    matched_b = {j for _, j in lcs_pairs}

    removed = [steps_a[i] for i in range(len(steps_a)) if i not in matched_a]
    added = [steps_b[j] for j in range(len(steps_b)) if j not in matched_b]

    changed: list[StepDelta] = []
    unchanged = 0
    for i, j in lcs_pairs:
        sa, sb = steps_a[i], steps_b[j]
        dur_diff = sb.duration_ms - sa.duration_ms
        tok_a = sa.prompt_tokens + sa.completion_tokens
        tok_b = sb.prompt_tokens + sb.completion_tokens
        tok_diff = tok_b - tok_a

        if dur_diff != 0 or tok_diff != 0:
            changed.append(StepDelta(
                step_a=sa,
                step_b=sb,
                duration_delta_ms=dur_diff,
                token_delta=tok_diff,
            ))
        else:
            unchanged += 1

    return TraceDiff(
        trace_id_a=tid_a,
        trace_id_b=tid_b,
        added=added,
        removed=removed,
        changed=changed,
        unchanged=unchanged,
        cost_delta_usd=cost_b - cost_a,
        token_delta=tokens_b - tokens_a,
        duration_delta_ms=duration_b - duration_a,
        step_count_a=len(steps_a),
        step_count_b=len(steps_b),
    )


# ── LCS algorithm ─────────────────────────────────────────────────────────

def _lcs(
    seq_a: list[StepSummary],
    seq_b: list[StepSummary],
) -> list[tuple[int, int]]:
    """Longest Common Subsequence returning matched (index_a, index_b) pairs.

    Comparison uses (step_type, model_or_tool) as the equality key.
    O(N*M) time and space.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return []

    # Build DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1]._key == seq_b[j - 1]._key:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find pairs
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if seq_a[i - 1]._key == seq_b[j - 1]._key:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_steps(trace: Any) -> list[StepSummary]:
    """Extract step summaries from AgentTrace or raw dict."""
    if isinstance(trace, dict):
        raw_steps = trace.get("steps", [])
    elif hasattr(trace, "steps"):
        raw_steps = trace.steps
    else:
        return []
    return [StepSummary.from_step(s) if isinstance(s, dict) else s for s in raw_steps]


def _extract_id(trace: Any) -> str:
    if isinstance(trace, dict):
        return trace.get("trace_id", "unknown")
    return getattr(trace, "trace_id", "unknown")


def _extract_cost(trace: Any) -> float:
    if isinstance(trace, dict):
        return float(trace.get("total_cost_usd", 0.0))
    return float(getattr(trace, "total_cost_usd", 0.0))


def _extract_duration(trace: Any) -> int:
    if isinstance(trace, dict):
        started = trace.get("started_at_ms", 0)
        completed = trace.get("completed_at_ms", 0)
        return completed - started
    started = getattr(trace, "started_at_ms", 0)
    completed = getattr(trace, "completed_at_ms", 0)
    return completed - started
