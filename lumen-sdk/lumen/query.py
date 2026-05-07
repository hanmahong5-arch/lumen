"""Trace query and filtering API.

Provides a fluent query builder for searching and filtering traces
by agent, user, session, time range, status, cost, and model.

Usage::

    from lumen.query import TraceQuery

    results = (
        TraceQuery("./traces")
        .agent("research-agent")
        .status("Failed")
        .since("7d")
        .min_cost(0.10)
        .model("gpt-4o")
        .limit(20)
        .execute()
    )

    for t in results:
        print(f"{t.trace_id}: {t.agent_id} ${t.total_cost_usd:.4f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from lumen.trace_reader import AgentTrace, _parse_trace, load_traces


def _parse_duration_ms(s: str) -> int:
    """Parse '24h', '7d', '30m', '90s' into milliseconds."""
    s = s.strip()
    try:
        if s.endswith("d"):
            return int(s[:-1]) * 86_400_000
        if s.endswith("h"):
            return int(s[:-1]) * 3_600_000
        if s.endswith("m"):
            return int(s[:-1]) * 60_000
        if s.endswith("s"):
            return int(s[:-1]) * 1_000
    except ValueError:
        pass
    return 86_400_000  # default 24h


@dataclass
class TraceQueryResult:
    """Results from a trace query."""

    traces: list[AgentTrace]
    total_scanned: int
    total_matched: int
    query_time_ms: int

    @property
    def total_cost_usd(self) -> float:
        return sum(t.total_cost_usd for t in self.traces)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens.total() for t in self.traces)

    @property
    def success_rate(self) -> float:
        if not self.traces:
            return 0.0
        ok = sum(1 for t in self.traces if t.status == "Completed")
        return ok / len(self.traces)

    @property
    def agents(self) -> list[str]:
        return sorted({t.agent_id for t in self.traces})

    @property
    def models(self) -> list[str]:
        models: set[str] = set()
        for t in self.traces:
            for s in t.steps:
                if s.step_type == "LlmCall" and s.model:
                    models.add(s.model)
        return sorted(models)

    def group_by_agent(self) -> dict[str, list[AgentTrace]]:
        groups: dict[str, list[AgentTrace]] = {}
        for t in self.traces:
            groups.setdefault(t.agent_id, []).append(t)
        return groups

    def group_by_status(self) -> dict[str, list[AgentTrace]]:
        groups: dict[str, list[AgentTrace]] = {}
        for t in self.traces:
            groups.setdefault(t.status, []).append(t)
        return groups

    def group_by_user(self) -> dict[str, list[AgentTrace]]:
        groups: dict[str, list[AgentTrace]] = {}
        for t in self.traces:
            uid = t.user_id or "(anonymous)"
            groups.setdefault(uid, []).append(t)
        return groups

    def group_by_session(self) -> dict[str, list[AgentTrace]]:
        groups: dict[str, list[AgentTrace]] = {}
        for t in self.traces:
            sid = t.session_id or "(no-session)"
            groups.setdefault(sid, []).append(t)
        return groups

    def summary(self) -> dict[str, Any]:
        return {
            "total_traces": len(self.traces),
            "total_scanned": self.total_scanned,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "success_rate": self.success_rate,
            "agents": self.agents,
            "models": self.models,
            "query_time_ms": self.query_time_ms,
        }


class TraceQuery:
    """Fluent builder for querying and filtering traces.

    Example::

        results = (
            TraceQuery("./traces")
            .agent("research-agent")
            .since("7d")
            .status("Completed")
            .limit(50)
            .execute()
        )
    """

    def __init__(self, trace_dir: str = "./traces", *, use_index: bool = False) -> None:
        self._trace_dir = trace_dir
        self._use_index = use_index
        self._filters: list[Callable[[AgentTrace], bool]] = []
        self._limit: int | None = None
        self._offset: int = 0
        self._sort_key: str = "time"  # time | cost | tokens | duration
        self._sort_desc: bool = True

        # Index-accelerated filter hints (set by specific filter methods)
        self._idx_agent: str | None = None
        self._idx_status: str | None = None
        self._idx_user: str | None = None
        self._idx_session: str | None = None
        self._idx_project: str | None = None
        self._idx_since_ms: int = 0
        self._idx_until_ms: int = 0

    def agent(self, agent_id: str) -> TraceQuery:
        """Filter by agent ID (exact match)."""
        self._idx_agent = agent_id
        self._filters.append(lambda t: t.agent_id == agent_id)
        return self

    def agent_contains(self, substring: str) -> TraceQuery:
        """Filter by agent ID (substring match)."""
        sub = substring.lower()
        self._filters.append(lambda t: sub in t.agent_id.lower())
        return self

    def user(self, user_id: str) -> TraceQuery:
        """Filter by user ID."""
        self._idx_user = user_id
        self._filters.append(lambda t: t.user_id == user_id)
        return self

    def session(self, session_id: str) -> TraceQuery:
        """Filter by session ID."""
        self._idx_session = session_id
        self._filters.append(lambda t: t.session_id == session_id)
        return self

    def status(self, status: str) -> TraceQuery:
        """Filter by status (Completed, Failed, Running)."""
        self._idx_status = status
        self._filters.append(lambda t: t.status == status)
        return self

    def succeeded(self) -> TraceQuery:
        """Filter to only completed traces."""
        return self.status("Completed")

    def failed(self) -> TraceQuery:
        """Filter to only failed traces."""
        return self.status("Failed")

    def since(self, duration: str) -> TraceQuery:
        """Filter to traces started within the given duration (e.g., '7d', '24h')."""
        duration_ms = _parse_duration_ms(duration)
        cutoff = int(time.time() * 1000) - duration_ms
        self._idx_since_ms = cutoff
        self._filters.append(lambda t: t.started_at_ms >= cutoff)
        return self

    def between(self, start_ms: int, end_ms: int) -> TraceQuery:
        """Filter to traces started within a specific time range."""
        self._idx_since_ms = start_ms
        self._idx_until_ms = end_ms
        self._filters.append(
            lambda t: start_ms <= t.started_at_ms <= end_ms,
        )
        return self

    def min_cost(self, usd: float) -> TraceQuery:
        """Filter traces costing at least this much."""
        self._filters.append(lambda t: t.total_cost_usd >= usd)
        return self

    def max_cost(self, usd: float) -> TraceQuery:
        """Filter traces costing at most this much."""
        self._filters.append(lambda t: t.total_cost_usd <= usd)
        return self

    def min_steps(self, n: int) -> TraceQuery:
        """Filter traces with at least N steps."""
        self._filters.append(lambda t: len(t.steps) >= n)
        return self

    def max_steps(self, n: int) -> TraceQuery:
        """Filter traces with at most N steps."""
        self._filters.append(lambda t: len(t.steps) <= n)
        return self

    def model(self, model_name: str) -> TraceQuery:
        """Filter to traces that used a specific model."""
        name = model_name.lower()
        def _has_model(t: AgentTrace) -> bool:
            for s in t.steps:
                if s.step_type == "LlmCall" and s.model and name in s.model.lower():
                    return True
            return False
        self._filters.append(_has_model)
        return self

    def has_errors(self) -> TraceQuery:
        """Filter to traces containing error steps."""
        def _check(t: AgentTrace) -> bool:
            return any(s.step_type == "Error" for s in t.steps)
        self._filters.append(_check)
        return self

    def has_tool(self, tool_name: str) -> TraceQuery:
        """Filter to traces that invoked a specific tool."""
        name = tool_name.lower()
        def _check(t: AgentTrace) -> bool:
            for s in t.steps:
                if s.step_type == "ToolCall" and s.tool_name and name in s.tool_name.lower():
                    return True
            return False
        self._filters.append(_check)
        return self

    def project(self, project_name: str) -> TraceQuery:
        """Filter by project name."""
        self._idx_project = project_name
        self._filters.append(lambda t: t.project == project_name)
        return self

    def environment(self, env: str) -> TraceQuery:
        """Filter by environment."""
        self._filters.append(lambda t: t.environment == env)
        return self

    def custom(self, predicate: Callable[[AgentTrace], bool]) -> TraceQuery:
        """Add a custom filter predicate."""
        self._filters.append(predicate)
        return self

    def sort_by(self, key: str, desc: bool = True) -> TraceQuery:
        """Sort results. Keys: time, cost, tokens, duration, steps."""
        self._sort_key = key
        self._sort_desc = desc
        return self

    def limit(self, n: int) -> TraceQuery:
        """Limit results to N traces."""
        self._limit = n
        return self

    def offset(self, n: int) -> TraceQuery:
        """Skip first N results (pagination)."""
        self._offset = n
        return self

    def execute(self) -> TraceQueryResult:
        """Execute the query and return results.

        When ``use_index=True``, uses TraceIndex for O(log N) pre-filtering
        on agent, status, user, session, project, and time range. Only loads
        full trace JSON for candidate matches.
        """
        start_ms = int(time.time() * 1000)

        if self._use_index:
            return self._execute_indexed(start_ms)

        all_traces = load_traces(self._trace_dir)
        total_scanned = len(all_traces)

        # Apply filters
        matched = all_traces
        for f in self._filters:
            matched = [t for t in matched if f(t)]

        total_matched = len(matched)

        # Sort
        matched = self._apply_sort(matched)

        # Offset + Limit
        if self._offset > 0:
            matched = matched[self._offset:]
        if self._limit is not None:
            matched = matched[:self._limit]

        query_time = int(time.time() * 1000) - start_ms

        return TraceQueryResult(
            traces=matched,
            total_scanned=total_scanned,
            total_matched=total_matched,
            query_time_ms=query_time,
        )

    def _execute_indexed(self, start_ms: int) -> TraceQueryResult:
        """Index-accelerated query execution.

        Strategy:
          1. Build/refresh TraceIndex
          2. Use index hints to narrow candidates (O(1) or O(log N))
          3. Load full traces only for candidates
          4. Apply remaining filters on full traces
        """
        from lumen._trace_index import TraceIndex

        idx = TraceIndex(self._trace_dir)
        total_scanned = idx.build()

        # Phase 1: Narrow candidates using index (metadata-only, no disk reads)
        candidates = self._narrow_with_index(idx)

        # Phase 2: Load full traces and apply all filters
        matched: list[AgentTrace] = []
        for meta in candidates:
            full_json = idx.load_full(meta.trace_id)
            if full_json is None:
                continue
            trace = _parse_trace(full_json)
            if all(f(trace) for f in self._filters):
                matched.append(trace)

        total_matched = len(matched)

        # Sort, offset, limit
        matched = self._apply_sort(matched)
        if self._offset > 0:
            matched = matched[self._offset:]
        if self._limit is not None:
            matched = matched[:self._limit]

        query_time = int(time.time() * 1000) - start_ms

        return TraceQueryResult(
            traces=matched,
            total_scanned=total_scanned,
            total_matched=total_matched,
            query_time_ms=query_time,
        )

    def _narrow_with_index(self, idx: Any) -> list[Any]:
        """Use index hints to narrow candidate set before full load.

        Intersects results from multiple index lookups. Falls back to
        all traces if no hints are set.
        """
        from lumen._trace_index import TraceMeta

        candidate_sets: list[set[str]] = []

        if self._idx_agent is not None:
            ids = {m.trace_id for m in idx.by_agent(self._idx_agent)}
            candidate_sets.append(ids)

        if self._idx_status is not None:
            ids = {m.trace_id for m in idx.by_status(self._idx_status)}
            candidate_sets.append(ids)

        if self._idx_user is not None:
            ids = {m.trace_id for m in idx.by_user(self._idx_user)}
            candidate_sets.append(ids)

        if self._idx_session is not None:
            ids = {m.trace_id for m in idx.by_session(self._idx_session)}
            candidate_sets.append(ids)

        if self._idx_project is not None:
            ids = {m.trace_id for m in idx.by_project(self._idx_project)}
            candidate_sets.append(ids)

        # Time range narrowing
        if self._idx_since_ms > 0 or self._idx_until_ms > 0:
            time_results = idx.by_time_range(
                since_ms=self._idx_since_ms,
                until_ms=self._idx_until_ms,
            )
            ids = {m.trace_id for m in time_results}
            candidate_sets.append(ids)

        if not candidate_sets:
            # No index hints — return all
            return idx.by_time_range()

        # Intersect all candidate sets
        result_ids = candidate_sets[0]
        for s in candidate_sets[1:]:
            result_ids &= s

        # Return TraceMeta objects in time order
        return [m for m in idx.by_time_range() if m.trace_id in result_ids]

    def count(self) -> int:
        """Execute query and return only the count."""
        return self.execute().total_matched

    def first(self) -> AgentTrace | None:
        """Execute query and return the first match."""
        result = self.limit(1).execute()
        return result.traces[0] if result.traces else None

    def _apply_sort(self, traces: list[AgentTrace]) -> list[AgentTrace]:
        sort_fns: dict[str, Callable[[AgentTrace], Any]] = {
            "time": lambda t: t.started_at_ms,
            "cost": lambda t: t.total_cost_usd,
            "tokens": lambda t: t.total_tokens.total(),
            "duration": lambda t: (t.completed_at_ms or 0) - t.started_at_ms,
            "steps": lambda t: len(t.steps),
        }
        fn = sort_fns.get(self._sort_key, sort_fns["time"])
        return sorted(traces, key=fn, reverse=self._sort_desc)
