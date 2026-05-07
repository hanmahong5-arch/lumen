"""Tests for CostTracker with real trace files."""

from __future__ import annotations

from pathlib import Path

from lumen.cost import CostTracker


def test_cost_report_all_time(trace_dir: Path) -> None:
    tracker = CostTracker(trace_dir=str(trace_dir))
    report = tracker.report(last="10000d")

    assert report.total_runs == 2
    # Cost should be estimated from tokens (trace files have total_cost_usd=0.0)
    assert report.total_usd > 0.0
    assert report.total_prompt_tokens == 3500  # 2500 + 1000
    assert report.total_completion_tokens == 750  # 550 + 200


def test_cost_by_agent(trace_dir: Path) -> None:
    tracker = CostTracker(trace_dir=str(trace_dir))
    report = tracker.report(last="10000d")

    assert "research-agent" in report.by_agent
    assert "summary-agent" in report.by_agent
    assert report.by_agent["research-agent"] > 0.0
    assert report.by_agent["summary-agent"] > 0.0


def test_cost_by_model(trace_dir: Path) -> None:
    tracker = CostTracker(trace_dir=str(trace_dir))
    report = tracker.report(last="10000d")

    assert "claude-sonnet-4-6" in report.by_model
    assert "gpt-4o" in report.by_model


def test_cost_time_filter(trace_dir: Path) -> None:
    tracker = CostTracker(trace_dir=str(trace_dir))
    # Only last "1h" — both traces are from 2024, so should be filtered out
    report = tracker.report(last="1h")
    assert report.total_runs == 0


def test_cost_empty_dir(tmp_path: Path) -> None:
    tracker = CostTracker(trace_dir=str(tmp_path))
    report = tracker.report(last="24h")
    assert report.total_runs == 0
    assert report.total_usd == 0.0


def test_cost_nonexistent_dir() -> None:
    tracker = CostTracker(trace_dir="/tmp/lumen_nonexistent_99999")
    report = tracker.report(last="24h")
    assert report.total_runs == 0


def test_cost_anomaly_detection(trace_dir: Path) -> None:
    tracker = CostTracker(trace_dir=str(trace_dir))
    report = tracker.report(last="10000d")

    # With only 2 runs and different models, anomaly detection may or may not
    # flag depending on cost distribution. Just verify the field exists.
    assert isinstance(report.anomalies, list)


def test_cost_estimation_accuracy(trace_dir: Path) -> None:
    """Verify estimated costs use correct per-model pricing."""
    tracker = CostTracker(trace_dir=str(trace_dir))
    report = tracker.report(last="10000d")

    # trace_abc123: 3 LLM calls with claude-sonnet-4-6 ($3/$15 per 1K)
    # (500*3 + 100*15)/1000 + (800*3 + 150*15)/1000 + (1200*3 + 300*15)/1000
    claude_cost = (500 * 3 + 100 * 15 + 800 * 3 + 150 * 15 + 1200 * 3 + 300 * 15) / 1000.0
    # trace_def456: 1 LLM call with gpt-4o ($2.50/$10 per 1K)
    gpt_cost = (1000 * 2.5 + 200 * 10.0) / 1000.0

    expected_total = claude_cost + gpt_cost
    assert abs(report.total_usd - expected_total) < 0.01


def test_cost_report_duration_parsing() -> None:
    """Verify time window parsing for different suffixes."""
    from lumen.cost import _parse_duration_ms

    assert _parse_duration_ms("1h") == 3_600_000
    assert _parse_duration_ms("24h") == 86_400_000
    assert _parse_duration_ms("7d") == 7 * 86_400_000
    assert _parse_duration_ms("30m") == 30 * 60_000
    # Invalid input defaults to 24h
    assert _parse_duration_ms("invalid") == 86_400_000


def test_cost_multiple_traces_same_agent(tmp_path: Path) -> None:
    """Multiple traces from same agent should aggregate correctly."""
    import json
    import time

    now_ms = int(time.time() * 1000)
    for i in range(3):
        trace = {
            "trace_id": f"multi_{i}",
            "agent_id": "same-agent",
            "task_id": i,
            "steps": [
                {
                    "step_num": 0,
                    "step_type": {"LlmCall": {"model": "gpt-4o-mini", "finish_reason": "stop"}},
                    "started_at_ms": now_ms - 1000,
                    "duration_ms": 500,
                    "tokens": {"prompt_tokens": 100, "completion_tokens": 50},
                    "metadata": [],
                },
            ],
            "status": "Completed",
            "total_tokens": {"prompt_tokens": 100, "completion_tokens": 50},
            "total_cost_usd": 0.0,
            "started_at_ms": now_ms - 1000,
            "completed_at_ms": now_ms,
        }
        (tmp_path / f"multi_{i}.json").write_text(json.dumps(trace))

    tracker = CostTracker(trace_dir=str(tmp_path))
    report = tracker.report(last="1h")

    assert report.total_runs == 3
    assert "same-agent" in report.by_agent
    # gpt-4o-mini: $0.15/$0.60 per 1K
    single_cost = (100 * 0.15 + 50 * 0.60) / 1000.0
    assert abs(report.by_agent["same-agent"] - single_cost * 3) < 0.001
