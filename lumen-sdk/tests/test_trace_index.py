"""Tests for TraceIndex — in-memory trace index with binary search and hash maps.

Covers: build/refresh, time-range queries (bisect), O(1) hash lookups,
load_full, most_expensive, agents list, total_cost, incremental refresh,
edge cases (empty dir, corrupt files, missing fields).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from lumen._trace_index import TraceMeta, TraceIndex


def _write_trace(
    trace_dir: Path,
    trace_id: str,
    agent_id: str = "test-agent",
    status: str = "Completed",
    cost: float = 0.01,
    started_at_ms: int = 1000000,
    completed_at_ms: int = 1001000,
    steps: int = 3,
    user_id: str = "",
    session_id: str = "",
    project: str = "",
    environment: str = "",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> Path:
    """Helper: write a minimal trace JSON file."""
    data = {
        "trace_id": trace_id,
        "agent_id": agent_id,
        "status": status,
        "total_cost_usd": cost,
        "started_at_ms": started_at_ms,
        "completed_at_ms": completed_at_ms,
        "steps": [{"step_num": i} for i in range(steps)],
        "total_tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    if user_id:
        data["user_id"] = user_id
    if session_id:
        data["session_id"] = session_id
    if project:
        data["project"] = project
    if environment:
        data["environment"] = environment

    path = trace_dir / f"{trace_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestTraceMetaDataclass:
    """TraceMeta frozen slotted dataclass."""

    def test_creation(self):
        m = TraceMeta(
            trace_id="t1",
            agent_id="a1",
            status="Completed",
            total_cost_usd=0.05,
            started_at_ms=1000,
            completed_at_ms=2000,
            step_count=3,
            total_prompt_tokens=100,
            total_completion_tokens=50,
        )
        assert m.trace_id == "t1"
        assert m.step_count == 3

    def test_frozen(self):
        m = TraceMeta(
            trace_id="t1", agent_id="a1", status="Completed",
            total_cost_usd=0.0, started_at_ms=0, completed_at_ms=0,
            step_count=0, total_prompt_tokens=0, total_completion_tokens=0,
        )
        with pytest.raises(AttributeError):
            m.trace_id = "t2"  # type: ignore[misc]

    def test_defaults(self):
        m = TraceMeta(
            trace_id="t1", agent_id="a1", status="Completed",
            total_cost_usd=0.0, started_at_ms=0, completed_at_ms=0,
            step_count=0, total_prompt_tokens=0, total_completion_tokens=0,
        )
        assert m.user_id == ""
        assert m.session_id == ""
        assert m.project == ""
        assert m.environment == ""
        assert m.file_path == ""


class TestTraceIndexBuild:
    """Building the full index from a trace directory."""

    def test_build_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            idx = TraceIndex(td)
            count = idx.build()
            assert count == 0
            assert len(idx) == 0

    def test_build_nonexistent_dir(self):
        idx = TraceIndex("/nonexistent/path/should/not/exist")
        assert idx.build() == 0

    def test_build_single_trace(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "trace-001")
            idx = TraceIndex(td)
            count = idx.build()
            assert count == 1
            assert len(idx) == 1

    def test_build_multiple_traces(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i in range(10):
                _write_trace(td_path, f"trace-{i:03d}", started_at_ms=1000 + i)
            idx = TraceIndex(td)
            count = idx.build()
            assert count == 10

    def test_build_skips_non_json(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "trace-001")
            (td_path / "notes.txt").write_text("not a trace")
            (td_path / ".hidden.json").write_text("{}")
            idx = TraceIndex(td)
            assert idx.build() == 1

    def test_build_skips_corrupt_json(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "trace-good")
            (td_path / "bad.json").write_text("not json at all {{{")
            idx = TraceIndex(td)
            assert idx.build() == 1

    def test_build_skips_missing_trace_id(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "good")
            no_id = td_path / "no-id.json"
            no_id.write_text(json.dumps({"agent_id": "a", "steps": []}))
            idx = TraceIndex(td)
            assert idx.build() == 1


class TestTraceIndexLookup:
    """O(1) hash-map lookups."""

    def _build_index(self, td_path: Path) -> TraceIndex:
        _write_trace(td_path, "t1", agent_id="agent-a", user_id="u1", session_id="s1",
                      project="proj-x", status="Completed", started_at_ms=1000)
        _write_trace(td_path, "t2", agent_id="agent-b", user_id="u1", session_id="s1",
                      project="proj-x", status="Failed", started_at_ms=2000)
        _write_trace(td_path, "t3", agent_id="agent-a", user_id="u2", session_id="s2",
                      project="proj-y", status="Completed", started_at_ms=3000)
        idx = TraceIndex(td_path)
        idx.build()
        return idx

    def test_get_by_id(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            m = idx.get("t1")
            assert m is not None
            assert m.agent_id == "agent-a"

    def test_get_missing_id(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            assert idx.get("nonexistent") is None

    def test_by_agent(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            agent_a = idx.by_agent("agent-a")
            assert len(agent_a) == 2
            assert all(m.agent_id == "agent-a" for m in agent_a)

    def test_by_agent_empty(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            assert idx.by_agent("no-such-agent") == []

    def test_by_status(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            failed = idx.by_status("Failed")
            assert len(failed) == 1
            assert failed[0].trace_id == "t2"

    def test_by_user(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            u1 = idx.by_user("u1")
            assert len(u1) == 2

    def test_by_session(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            s1 = idx.by_session("s1")
            assert len(s1) == 2

    def test_by_project(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_index(Path(td))
            px = idx.by_project("proj-x")
            assert len(px) == 2


class TestTraceIndexTimeRange:
    """O(log N) time-range queries via binary search."""

    def _build_time_index(self, td_path: Path) -> TraceIndex:
        for i in range(10):
            _write_trace(td_path, f"t{i}", started_at_ms=1000 + i * 100)
        idx = TraceIndex(td_path)
        idx.build()
        return idx

    def test_all_traces_no_filter(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_time_index(Path(td))
            result = idx.by_time_range()
            assert len(result) == 10

    def test_since_filter(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_time_index(Path(td))
            # started_at_ms: 1000, 1100, ..., 1900
            # since_ms=1500 → t5 (1500), t6, t7, t8, t9 = 5 traces
            result = idx.by_time_range(since_ms=1500)
            assert len(result) == 5

    def test_until_filter(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_time_index(Path(td))
            # until_ms=1200 → t0 (1000), t1 (1100), t2 (1200) = 3 traces
            result = idx.by_time_range(until_ms=1200)
            assert len(result) == 3

    def test_range_filter(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_time_index(Path(td))
            # since=1300, until=1600 → t3 (1300), t4 (1400), t5 (1500), t6 (1600) = 4
            result = idx.by_time_range(since_ms=1300, until_ms=1600)
            assert len(result) == 4

    def test_empty_range(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_time_index(Path(td))
            result = idx.by_time_range(since_ms=9999)
            assert len(result) == 0

    def test_sorted_order(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._build_time_index(Path(td))
            result = idx.by_time_range()
            times = [m.started_at_ms for m in result]
            assert times == sorted(times)


class TestTraceIndexAggregation:
    """most_expensive, agents, total_cost."""

    def test_most_expensive(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "cheap", cost=0.01)
            _write_trace(td_path, "medium", cost=0.50)
            _write_trace(td_path, "expensive", cost=5.00)
            idx = TraceIndex(td)
            idx.build()

            top = idx.most_expensive(2)
            assert len(top) == 2
            assert top[0].trace_id == "expensive"
            assert top[1].trace_id == "medium"

    def test_agents_list(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", agent_id="zebra")
            _write_trace(td_path, "t2", agent_id="alpha")
            _write_trace(td_path, "t3", agent_id="alpha")
            idx = TraceIndex(td)
            idx.build()
            assert idx.agents() == ["alpha", "zebra"]

    def test_total_cost(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", cost=0.10)
            _write_trace(td_path, "t2", cost=0.25)
            _write_trace(td_path, "t3", cost=0.15)
            idx = TraceIndex(td)
            idx.build()
            assert idx.total_cost() == pytest.approx(0.50)


class TestTraceIndexRefresh:
    """Incremental refresh picks up new files."""

    def test_refresh_adds_new(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", started_at_ms=1000)
            idx = TraceIndex(td)
            idx.build()
            assert len(idx) == 1

            # Add new trace file
            _write_trace(td_path, "t2", started_at_ms=2000)
            added = idx.refresh()
            assert added == 1
            assert len(idx) == 2

    def test_refresh_skips_known(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1")
            idx = TraceIndex(td)
            idx.build()

            added = idx.refresh()
            assert added == 0

    def test_refresh_maintains_sort(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t2", started_at_ms=2000)
            idx = TraceIndex(td)
            idx.build()

            # Add earlier trace
            _write_trace(td_path, "t1", started_at_ms=1000)
            idx.refresh()

            # Should be sorted: t1 before t2
            result = idx.by_time_range()
            assert result[0].trace_id == "t1"
            assert result[1].trace_id == "t2"

    def test_refresh_nonexistent_dir(self):
        idx = TraceIndex("/nonexistent")
        assert idx.refresh() == 0


class TestTraceIndexLoadFull:
    """Load complete trace JSON from disk."""

    def test_load_full_success(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", agent_id="a1")
            idx = TraceIndex(td)
            idx.build()

            full = idx.load_full("t1")
            assert full is not None
            assert full["trace_id"] == "t1"
            assert full["agent_id"] == "a1"
            assert "steps" in full

    def test_load_full_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            idx = TraceIndex(td)
            idx.build()
            assert idx.load_full("nonexistent") is None

    def test_load_full_deleted_file(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            path = _write_trace(td_path, "t1")
            idx = TraceIndex(td)
            idx.build()

            # Delete the file after indexing
            os.remove(path)
            assert idx.load_full("t1") is None


class TestTraceIndexStatusParsing:
    """Status field parsing: string, dict, and missing."""

    def test_string_status(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", status="Completed")
            idx = TraceIndex(td)
            idx.build()
            assert idx.get("t1").status == "Completed"

    def test_dict_failed_status(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data = {
                "trace_id": "t-fail",
                "agent_id": "a1",
                "status": {"Failed": "timeout occurred"},
                "steps": [],
                "total_tokens": {},
            }
            with open(td_path / "t-fail.json", "w") as f:
                json.dump(data, f)
            idx = TraceIndex(td)
            idx.build()
            m = idx.get("t-fail")
            assert m is not None
            assert m.status == "Failed"

    def test_missing_status_defaults_running(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data = {"trace_id": "t-no-status", "agent_id": "a1", "steps": []}
            with open(td_path / "t-no-status.json", "w") as f:
                json.dump(data, f)
            idx = TraceIndex(td)
            idx.build()
            m = idx.get("t-no-status")
            assert m is not None
            assert m.status == "Running"


class TestTraceIndexTokenParsing:
    """Token counts from total_tokens field."""

    def test_tokens_parsed(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", prompt_tokens=500, completion_tokens=200)
            idx = TraceIndex(td)
            idx.build()
            m = idx.get("t1")
            assert m.total_prompt_tokens == 500
            assert m.total_completion_tokens == 200

    def test_missing_tokens_defaults_zero(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data = {"trace_id": "t1", "agent_id": "a", "steps": []}
            with open(td_path / "t1.json", "w") as f:
                json.dump(data, f)
            idx = TraceIndex(td)
            idx.build()
            m = idx.get("t1")
            assert m.total_prompt_tokens == 0
            assert m.total_completion_tokens == 0


class TestTraceIndexRebuild:
    """build() clears and rebuilds from scratch."""

    def test_rebuild_clears_old_data(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1")
            idx = TraceIndex(td)
            idx.build()
            assert len(idx) == 1

            # Delete the file and rebuild
            os.remove(td_path / "t1.json")
            idx.build()
            assert len(idx) == 0


class TestTraceIndexLRUCache:
    """LRU cache for load_full() avoids repeated disk reads."""

    def test_load_full_caches_result(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1", agent_id="cached-agent")
            idx = TraceIndex(td, cache_size=10)
            idx.build()

            # First load — disk read
            data1 = idx.load_full("t1")
            assert data1 is not None
            assert data1["agent_id"] == "cached-agent"

            # Second load — cache hit (same object)
            data2 = idx.load_full("t1")
            assert data2 is data1  # same dict object, from cache

    def test_cache_eviction(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i in range(5):
                _write_trace(td_path, f"t{i}")
            idx = TraceIndex(td, cache_size=2)
            idx.build()

            # Load t0 and t1 — fills cache
            idx.load_full("t0")
            idx.load_full("t1")

            # Load t2 — evicts t0 (LRU)
            idx.load_full("t2")

            # t0 reload returns data (from disk, not cache)
            data = idx.load_full("t0")
            assert data is not None

    def test_rebuild_clears_cache(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _write_trace(td_path, "t1")
            idx = TraceIndex(td, cache_size=10)
            idx.build()
            idx.load_full("t1")  # populate cache

            # Rebuild clears everything including cache
            idx.build()
            # Cache should be empty, but load_full still works (disk read)
            data = idx.load_full("t1")
            assert data is not None
