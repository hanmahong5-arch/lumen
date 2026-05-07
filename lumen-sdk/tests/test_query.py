"""Comprehensive tests for the trace query and filtering API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from lumen.query import TraceQuery, TraceQueryResult, _parse_duration_ms
from lumen.trace_reader import AgentTrace, TokenUsage, TraceStep, load_traces


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_trace(
    trace_dir: Path,
    *,
    trace_id: str = "t001",
    agent_id: str = "test-agent",
    status: str | dict[str, str] = "Completed",
    cost_usd: float = 0.01,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    started_at_ms: int = 0,
    completed_at_ms: int | None = None,
    model: str = "gpt-4o",
    user_id: str = "",
    session_id: str = "",
    project: str = "",
    environment: str = "",
    steps: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a trace JSON file and return its path."""
    if started_at_ms == 0:
        started_at_ms = int(time.time() * 1000)
    if completed_at_ms is None:
        completed_at_ms = started_at_ms + 5000

    if steps is None:
        steps = [
            {
                "step_num": 0,
                "step_type": {"LlmCall": {"model": model, "finish_reason": "stop"}},
                "started_at_ms": started_at_ms,
                "duration_ms": 500,
                "tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "metadata": [],
            },
        ]

    trace_data: dict[str, Any] = {
        "trace_id": trace_id,
        "agent_id": agent_id,
        "task_id": 0,
        "steps": steps,
        "status": status,
        "total_tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "total_cost_usd": cost_usd,
        "started_at_ms": started_at_ms,
        "completed_at_ms": completed_at_ms,
    }

    if user_id:
        trace_data["user_id"] = user_id
    if session_id:
        trace_data["session_id"] = session_id
    if project:
        trace_data["project"] = project
    if environment:
        trace_data["environment"] = environment

    path = trace_dir / f"{trace_id}.json"
    with open(path, "w") as f:
        json.dump(trace_data, f)
    return path


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


@pytest.fixture()
def populated_dir(trace_dir: Path) -> Path:
    """Create a directory with diverse traces for query testing."""
    now = int(time.time() * 1000)

    # Agent A — 3 recent completed traces, different costs
    _write_trace(trace_dir, trace_id="a1", agent_id="research-agent", cost_usd=0.05,
                 started_at_ms=now - 1000, user_id="u1", session_id="s1",
                 project="proj-a", environment="production")
    _write_trace(trace_dir, trace_id="a2", agent_id="research-agent", cost_usd=0.10,
                 started_at_ms=now - 2000, user_id="u1", session_id="s1",
                 project="proj-a", environment="production")
    _write_trace(trace_dir, trace_id="a3", agent_id="research-agent", cost_usd=0.50,
                 started_at_ms=now - 3000, user_id="u2", session_id="s2",
                 project="proj-a", environment="staging")

    # Agent B — 2 traces, one failed
    _write_trace(trace_dir, trace_id="b1", agent_id="code-agent", cost_usd=0.02,
                 started_at_ms=now - 4000, model="claude-sonnet-4-6",
                 user_id="u1", project="proj-b")
    _write_trace(trace_dir, trace_id="b2", agent_id="code-agent", cost_usd=0.03,
                 status={"Failed": "timeout"}, started_at_ms=now - 5000,
                 model="claude-sonnet-4-6", user_id="u3")

    # Agent C — old trace (30 days ago)
    _write_trace(trace_dir, trace_id="c1", agent_id="old-agent", cost_usd=1.00,
                 started_at_ms=now - 30 * 86_400_000)

    # Agent D — trace with tool call and error steps
    tool_error_steps = [
        {
            "step_num": 0,
            "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "tool_use"}},
            "started_at_ms": now - 6000,
            "duration_ms": 300,
            "tokens": {"prompt_tokens": 200, "completion_tokens": 100},
            "metadata": [],
        },
        {
            "step_num": 1,
            "step_type": {"ToolCall": {"tool_name": "web_search", "success": True}},
            "started_at_ms": now - 5700,
            "duration_ms": 1000,
            "tokens": None,
            "metadata": [],
        },
        {
            "step_num": 2,
            "step_type": {"LlmCall": {"model": "gpt-4o-mini", "finish_reason": "stop"}},
            "started_at_ms": now - 4700,
            "duration_ms": 200,
            "tokens": {"prompt_tokens": 300, "completion_tokens": 50},
            "metadata": [],
        },
        {
            "step_num": 3,
            "step_type": {"Error": {"message": "Rate limited"}},
            "started_at_ms": now - 4500,
            "duration_ms": 0,
            "tokens": None,
            "metadata": [],
        },
    ]
    _write_trace(trace_dir, trace_id="d1", agent_id="multi-step-agent", cost_usd=0.08,
                 started_at_ms=now - 6000, steps=tool_error_steps,
                 prompt_tokens=500, completion_tokens=150,
                 status={"Failed": "Rate limited"})

    return trace_dir


# ── Duration parsing ─────────────────────────────────────────────────────────

class TestParseDurationMs:
    def test_days(self) -> None:
        assert _parse_duration_ms("7d") == 7 * 86_400_000

    def test_hours(self) -> None:
        assert _parse_duration_ms("24h") == 24 * 3_600_000

    def test_minutes(self) -> None:
        assert _parse_duration_ms("30m") == 30 * 60_000

    def test_seconds(self) -> None:
        assert _parse_duration_ms("90s") == 90 * 1_000

    def test_whitespace(self) -> None:
        assert _parse_duration_ms("  7d  ") == 7 * 86_400_000

    def test_invalid_defaults_24h(self) -> None:
        assert _parse_duration_ms("invalid") == 86_400_000

    def test_empty_defaults_24h(self) -> None:
        assert _parse_duration_ms("") == 86_400_000

    def test_bad_number(self) -> None:
        assert _parse_duration_ms("abcd") == 86_400_000


# ── TraceQuery basic ─────────────────────────────────────────────────────────

class TestTraceQueryBasic:
    def test_empty_dir(self, trace_dir: Path) -> None:
        result = TraceQuery(str(trace_dir)).execute()
        assert result.total_scanned == 0
        assert result.total_matched == 0
        assert len(result.traces) == 0

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        result = TraceQuery(str(tmp_path / "nope")).execute()
        assert len(result.traces) == 0

    def test_all_traces(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        assert result.total_scanned == 7
        assert result.total_matched == 7

    def test_query_time_recorded(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        assert result.query_time_ms >= 0


# ── Filter by agent ──────────────────────────────────────────────────────────

class TestTraceQueryAgent:
    def test_exact_match(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).agent("research-agent").execute()
        assert result.total_matched == 3
        assert all(t.agent_id == "research-agent" for t in result.traces)

    def test_no_match(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).agent("nonexistent").execute()
        assert result.total_matched == 0

    def test_contains(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).agent_contains("agent").execute()
        # All agents contain "agent"
        assert result.total_matched == 7

    def test_contains_partial(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).agent_contains("research").execute()
        assert result.total_matched == 3

    def test_contains_case_insensitive(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).agent_contains("RESEARCH").execute()
        assert result.total_matched == 3


# ── Filter by status ─────────────────────────────────────────────────────────

class TestTraceQueryStatus:
    def test_completed(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).succeeded().execute()
        assert all(t.status == "Completed" for t in result.traces)

    def test_failed(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).failed().execute()
        assert result.total_matched == 2  # b2 and d1
        assert all(t.status == "Failed" for t in result.traces)

    def test_explicit_status(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).status("Failed").execute()
        assert result.total_matched == 2


# ── Filter by user / session ─────────────────────────────────────────────────

class TestTraceQueryUserSession:
    def test_user_filter(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).user("u1").execute()
        assert result.total_matched == 3  # a1, a2, b1
        assert all(t.user_id == "u1" for t in result.traces)

    def test_session_filter(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).session("s1").execute()
        assert result.total_matched == 2  # a1, a2

    def test_user_and_session(self, populated_dir: Path) -> None:
        result = (
            TraceQuery(str(populated_dir))
            .user("u1")
            .session("s1")
            .execute()
        )
        assert result.total_matched == 2


# ── Filter by cost ───────────────────────────────────────────────────────────

class TestTraceQueryCost:
    def test_min_cost(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).min_cost(0.10).execute()
        assert all(t.total_cost_usd >= 0.10 for t in result.traces)

    def test_max_cost(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).max_cost(0.05).execute()
        assert all(t.total_cost_usd <= 0.05 for t in result.traces)

    def test_cost_range(self, populated_dir: Path) -> None:
        result = (
            TraceQuery(str(populated_dir))
            .min_cost(0.04)
            .max_cost(0.11)
            .execute()
        )
        assert all(0.04 <= t.total_cost_usd <= 0.11 for t in result.traces)


# ── Filter by steps ──────────────────────────────────────────────────────────

class TestTraceQuerySteps:
    def test_min_steps(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).min_steps(3).execute()
        assert all(len(t.steps) >= 3 for t in result.traces)

    def test_max_steps(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).max_steps(1).execute()
        assert all(len(t.steps) <= 1 for t in result.traces)


# ── Filter by model ──────────────────────────────────────────────────────────

class TestTraceQueryModel:
    def test_model_filter(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).model("claude-sonnet").execute()
        # b1 and b2 use claude-sonnet-4-6
        assert result.total_matched == 2

    def test_model_partial(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).model("gpt-4o").execute()
        # matches gpt-4o and gpt-4o-mini
        assert result.total_matched >= 4


# ── Filter by tool / error ───────────────────────────────────────────────────

class TestTraceQueryToolError:
    def test_has_tool(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).has_tool("web_search").execute()
        assert result.total_matched == 1  # d1
        assert result.traces[0].trace_id == "d1"

    def test_has_tool_case_insensitive(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).has_tool("WEB_SEARCH").execute()
        assert result.total_matched == 1

    def test_has_errors(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).has_errors().execute()
        assert result.total_matched == 1  # d1 has explicit Error step


# ── Filter by project / environment ──────────────────────────────────────────

class TestTraceQueryProjectEnv:
    def test_project_filter(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).project("proj-a").execute()
        assert result.total_matched == 3

    def test_environment_filter(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).environment("production").execute()
        assert result.total_matched == 2  # a1, a2


# ── Time-based filters ───────────────────────────────────────────────────────

class TestTraceQueryTime:
    def test_since_1h(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).since("1h").execute()
        # All except old-agent (30 days ago)
        assert result.total_matched == 6

    def test_since_1d(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).since("1d").execute()
        assert result.total_matched == 6

    def test_between(self, populated_dir: Path) -> None:
        now = int(time.time() * 1000)
        result = (
            TraceQuery(str(populated_dir))
            .between(now - 5000, now - 1000)
            .execute()
        )
        assert all(now - 5000 <= t.started_at_ms <= now - 1000 for t in result.traces)


# ── Custom filter ────────────────────────────────────────────────────────────

class TestTraceQueryCustom:
    def test_custom_predicate(self, populated_dir: Path) -> None:
        result = (
            TraceQuery(str(populated_dir))
            .custom(lambda t: t.total_cost_usd > 0.05 and t.status == "Completed")
            .execute()
        )
        assert all(t.total_cost_usd > 0.05 and t.status == "Completed" for t in result.traces)


# ── Sorting ──────────────────────────────────────────────────────────────────

class TestTraceQuerySort:
    def test_sort_by_cost_desc(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).sort_by("cost", desc=True).execute()
        costs = [t.total_cost_usd for t in result.traces]
        assert costs == sorted(costs, reverse=True)

    def test_sort_by_cost_asc(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).sort_by("cost", desc=False).execute()
        costs = [t.total_cost_usd for t in result.traces]
        assert costs == sorted(costs)

    def test_sort_by_time(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).sort_by("time", desc=True).execute()
        times = [t.started_at_ms for t in result.traces]
        assert times == sorted(times, reverse=True)

    def test_sort_by_steps(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).sort_by("steps", desc=True).execute()
        step_counts = [len(t.steps) for t in result.traces]
        assert step_counts == sorted(step_counts, reverse=True)

    def test_sort_by_tokens(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).sort_by("tokens", desc=True).execute()
        token_counts = [t.total_tokens.total() for t in result.traces]
        assert token_counts == sorted(token_counts, reverse=True)

    def test_invalid_sort_key_uses_time(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).sort_by("invalid_key").execute()
        times = [t.started_at_ms for t in result.traces]
        assert times == sorted(times, reverse=True)


# ── Pagination ───────────────────────────────────────────────────────────────

class TestTraceQueryPagination:
    def test_limit(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).limit(3).execute()
        assert len(result.traces) == 3
        assert result.total_matched == 7  # total still 7

    def test_offset(self, populated_dir: Path) -> None:
        all_result = TraceQuery(str(populated_dir)).execute()
        offset_result = TraceQuery(str(populated_dir)).offset(2).execute()
        assert len(offset_result.traces) == len(all_result.traces) - 2

    def test_limit_and_offset(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).offset(1).limit(2).execute()
        assert len(result.traces) == 2

    def test_limit_larger_than_results(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).limit(100).execute()
        assert len(result.traces) == 7

    def test_offset_larger_than_results(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).offset(100).execute()
        assert len(result.traces) == 0


# ── Convenience methods ──────────────────────────────────────────────────────

class TestTraceQueryConvenience:
    def test_count(self, populated_dir: Path) -> None:
        count = TraceQuery(str(populated_dir)).agent("research-agent").count()
        assert count == 3

    def test_first(self, populated_dir: Path) -> None:
        trace = TraceQuery(str(populated_dir)).agent("research-agent").first()
        assert trace is not None
        assert trace.agent_id == "research-agent"

    def test_first_no_match(self, populated_dir: Path) -> None:
        trace = TraceQuery(str(populated_dir)).agent("nonexistent").first()
        assert trace is None


# ── Chaining ─────────────────────────────────────────────────────────────────

class TestTraceQueryChaining:
    def test_complex_chain(self, populated_dir: Path) -> None:
        result = (
            TraceQuery(str(populated_dir))
            .agent("research-agent")
            .succeeded()
            .min_cost(0.05)
            .sort_by("cost", desc=True)
            .limit(2)
            .execute()
        )
        assert len(result.traces) <= 2
        assert all(t.agent_id == "research-agent" for t in result.traces)
        assert all(t.status == "Completed" for t in result.traces)
        assert all(t.total_cost_usd >= 0.05 for t in result.traces)


# ── TraceQueryResult ─────────────────────────────────────────────────────────

class TestTraceQueryResult:
    def test_total_cost(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        expected = sum(t.total_cost_usd for t in result.traces)
        assert result.total_cost_usd == pytest.approx(expected)

    def test_total_tokens(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        expected = sum(t.total_tokens.total() for t in result.traces)
        assert result.total_tokens == expected

    def test_success_rate(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        completed = sum(1 for t in result.traces if t.status == "Completed")
        expected = completed / len(result.traces)
        assert result.success_rate == pytest.approx(expected)

    def test_success_rate_empty(self, trace_dir: Path) -> None:
        result = TraceQuery(str(trace_dir)).execute()
        assert result.success_rate == 0.0

    def test_agents_property(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        agents = result.agents
        assert isinstance(agents, list)
        assert agents == sorted(agents)  # sorted
        assert "research-agent" in agents
        assert "code-agent" in agents

    def test_models_property(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        models = result.models
        assert isinstance(models, list)
        assert "gpt-4o" in models

    def test_group_by_agent(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        groups = result.group_by_agent()
        assert "research-agent" in groups
        assert len(groups["research-agent"]) == 3

    def test_group_by_status(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        groups = result.group_by_status()
        assert "Completed" in groups
        assert "Failed" in groups

    def test_group_by_user(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        groups = result.group_by_user()
        assert "u1" in groups
        assert "(anonymous)" in groups  # traces without user_id

    def test_group_by_session(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        groups = result.group_by_session()
        assert "s1" in groups
        assert "(no-session)" in groups

    def test_summary(self, populated_dir: Path) -> None:
        result = TraceQuery(str(populated_dir)).execute()
        summary = result.summary()
        assert "total_traces" in summary
        assert "total_cost_usd" in summary
        assert "success_rate" in summary
        assert "agents" in summary
        assert "models" in summary
        assert "query_time_ms" in summary
        assert summary["total_traces"] == 7


class TestIndexedQuery:
    """TraceQuery with use_index=True for accelerated lookups."""

    def test_indexed_agent_filter(self, trace_dir: Path) -> None:
        for i in range(5):
            _write_trace(trace_dir, trace_id=f"t{i}", agent_id=f"agent-{i % 2}",
                         started_at_ms=1000000 + i * 1000)
        result = TraceQuery(str(trace_dir), use_index=True).agent("agent-0").execute()
        assert result.total_matched == 3  # indices 0, 2, 4

    def test_indexed_status_filter(self, trace_dir: Path) -> None:
        _write_trace(trace_dir, trace_id="ok", status="Completed", started_at_ms=1000000)
        _write_trace(trace_dir, trace_id="fail", status={"Failed": "err"}, started_at_ms=1001000)
        result = TraceQuery(str(trace_dir), use_index=True).status("Failed").execute()
        assert result.total_matched == 1
        assert result.traces[0].trace_id == "fail"

    def test_indexed_project_filter(self, trace_dir: Path) -> None:
        _write_trace(trace_dir, trace_id="t1", project="alpha", started_at_ms=1000000)
        _write_trace(trace_dir, trace_id="t2", project="beta", started_at_ms=1001000)
        _write_trace(trace_dir, trace_id="t3", project="alpha", started_at_ms=1002000)
        result = TraceQuery(str(trace_dir), use_index=True).project("alpha").execute()
        assert result.total_matched == 2

    def test_indexed_combined_filters(self, trace_dir: Path) -> None:
        _write_trace(trace_dir, trace_id="t1", agent_id="a1", status="Completed", started_at_ms=1000000)
        _write_trace(trace_dir, trace_id="t2", agent_id="a1", status="Failed", started_at_ms=1001000,
                     steps=[{"step_num": 0, "step_type": {"Error": {"message": "err"}},
                             "started_at_ms": 1001000, "duration_ms": 0,
                             "tokens": None, "metadata": []}])
        _write_trace(trace_dir, trace_id="t3", agent_id="a2", status="Completed", started_at_ms=1002000)
        result = (
            TraceQuery(str(trace_dir), use_index=True)
            .agent("a1")
            .status("Completed")
            .execute()
        )
        assert result.total_matched == 1
        assert result.traces[0].trace_id == "t1"

    def test_indexed_matches_non_indexed(self, trace_dir: Path) -> None:
        """Indexed and non-indexed paths produce same results."""
        for i in range(10):
            _write_trace(trace_dir, trace_id=f"t{i}",
                         agent_id=f"agent-{i % 3}",
                         cost_usd=0.01 * (i + 1),
                         started_at_ms=1000000 + i * 1000)
        normal = TraceQuery(str(trace_dir)).agent("agent-1").execute()
        indexed = TraceQuery(str(trace_dir), use_index=True).agent("agent-1").execute()
        assert normal.total_matched == indexed.total_matched
        normal_ids = {t.trace_id for t in normal.traces}
        indexed_ids = {t.trace_id for t in indexed.traces}
        assert normal_ids == indexed_ids

    def test_indexed_no_hints_returns_all(self, trace_dir: Path) -> None:
        for i in range(3):
            _write_trace(trace_dir, trace_id=f"t{i}", started_at_ms=1000000 + i)
        result = TraceQuery(str(trace_dir), use_index=True).execute()
        assert result.total_matched == 3

    def test_indexed_with_limit(self, trace_dir: Path) -> None:
        for i in range(10):
            _write_trace(trace_dir, trace_id=f"t{i}", started_at_ms=1000000 + i * 1000)
        result = TraceQuery(str(trace_dir), use_index=True).limit(3).execute()
        assert len(result.traces) == 3
