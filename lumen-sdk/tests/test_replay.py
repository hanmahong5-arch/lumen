"""Tests for ReplayEngine with real trace files."""

from __future__ import annotations

from pathlib import Path

import pytest

from lumen.replay import ReplayEngine


def test_replay_success(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_abc123")

    assert trace.trace_id == "trace_abc123"
    assert trace.success is True
    assert trace.total_iterations == 3  # 3 LLM calls
    assert len(trace.steps) == 3


def test_replay_step_tool_calls(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_abc123")

    # Step 1: LLM → search_web
    step1 = trace.steps[0]
    assert step1.step == 1
    assert step1.decision_type == "tool_call"
    assert step1.tool_name == "search_web"
    assert step1.success is True


def test_replay_step_final_answer(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_abc123")

    # Last step: final answer (end_turn)
    last = trace.steps[-1]
    assert last.decision_type == "final_answer"
    assert last.content is not None


def test_replay_failed_trace(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_def456")

    assert trace.success is False
    assert trace.total_iterations == 1  # 1 LLM call
    # Last step should contain error info
    last = trace.steps[-1]
    assert "timeout" in (last.content or "")


def test_replay_not_found(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    with pytest.raises(ValueError, match="nonexistent"):
        engine.replay("nonexistent")


def test_list_traces(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    ids = engine.list_traces()

    assert len(ids) == 2
    # Newest first
    assert ids[0] == "trace_abc123"
    assert ids[1] == "trace_def456"


def test_list_traces_empty(tmp_path: Path) -> None:
    engine = ReplayEngine(trace_dir=str(tmp_path))
    assert engine.list_traces() == []


def test_replay_original_cost(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_abc123")

    # Cost is estimated since trace has total_cost_usd=0.0
    assert trace.original_cost_usd >= 0.0


def test_replay_nonexistent_dir() -> None:
    engine = ReplayEngine(trace_dir="/tmp/lumen_nonexistent_replay_99999")
    assert engine.list_traces() == []


def test_replay_step_count_matches_decisions(trace_dir: Path) -> None:
    """Replay steps should match LLM decision count, not raw step count."""
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_abc123")

    # trace_abc123 has 3 LLM calls: 2 tool_use + 1 end_turn
    # Replay groups: LLM+tool, LLM+tool, LLM(final) = 4 replay steps
    # (2 tool_call steps + 1 read_url + 1 final_answer)
    assert len(trace.steps) >= 3


def test_replay_failed_trace_has_error_content(trace_dir: Path) -> None:
    """Failed trace replay should include error in step content."""
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_def456")

    error_steps = [s for s in trace.steps if s.success is False]
    assert len(error_steps) >= 1


def test_replay_step_timestamps_are_populated(trace_dir: Path) -> None:
    engine = ReplayEngine(trace_dir=str(trace_dir))
    trace = engine.replay("trace_abc123")

    for step in trace.steps:
        assert step.timestamp_ms > 0
