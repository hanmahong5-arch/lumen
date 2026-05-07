"""Lightweight in-memory trace index with lazy loading.

Instead of loading and parsing every trace JSON file for each query,
this module maintains a compact index of trace metadata. Full trace
data is loaded on demand only when needed.

Data structure: sorted list of TraceMeta entries, keyed by started_at_ms.
Binary search enables O(log N) time-range queries. Hash maps provide
O(1) lookup by trace_id, agent_id, user_id, and status.

Memory: ~200 bytes per trace entry → 1M traces ≈ 200 MB index.
IO: One stat() + partial read per trace file during indexing.

Usage::

    index = TraceIndex("./traces")
    index.build()   # scan directory, build index

    # Fast queries via index
    recent = index.by_time_range(since_ms=cutoff)
    agent_traces = index.by_agent("research-agent")
    failed = index.by_status("Failed")

    # Load full trace only when needed
    trace = index.load_full("abc123")
"""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from lumen._lru_cache import LRUCache


@dataclass(slots=True, frozen=True)
class TraceMeta:
    """Compact trace metadata for indexing (no step data).

    Frozen + slotted for memory efficiency — ~200 bytes per entry.
    """

    trace_id: str
    agent_id: str
    status: str
    total_cost_usd: float
    started_at_ms: int
    completed_at_ms: int
    step_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    user_id: str = ""
    session_id: str = ""
    project: str = ""
    environment: str = ""
    file_path: str = ""


class TraceIndex:
    """In-memory index of trace metadata with O(1) and O(log N) queries.

    The index is built by scanning the trace directory once. Subsequent
    queries use in-memory data structures. Call ``refresh()`` to pick up
    new traces.
    """

    def __init__(self, trace_dir: str | Path, *, cache_size: int = 64) -> None:
        self._trace_dir = Path(trace_dir)

        # Primary: sorted by started_at_ms (for time-range queries)
        self._by_time: list[TraceMeta] = []

        # Secondary indices: hash maps for O(1) lookup
        self._by_id: dict[str, TraceMeta] = {}
        self._by_agent: dict[str, list[TraceMeta]] = {}
        self._by_status: dict[str, list[TraceMeta]] = {}
        self._by_user: dict[str, list[TraceMeta]] = {}
        self._by_session: dict[str, list[TraceMeta]] = {}
        self._by_project: dict[str, list[TraceMeta]] = {}

        # Track known files to enable incremental refresh
        self._known_files: set[str] = set()

        # LRU cache for full trace JSON — avoids re-reading hot traces
        self._full_cache: LRUCache[str, dict[str, Any]] = LRUCache(cache_size)

    def build(self) -> int:
        """Scan trace directory and build the full index.

        Returns the number of traces indexed.
        """
        self._clear()

        if not self._trace_dir.exists():
            return 0

        try:
            entries = os.listdir(self._trace_dir)
        except OSError:
            return 0

        for name in entries:
            if name.startswith(".") or not name.endswith(".json"):
                continue
            path = self._trace_dir / name
            meta = self._read_meta(path)
            if meta is not None:
                self._insert(meta)
                self._known_files.add(name)

        # Sort by time for binary search
        self._by_time.sort(key=lambda m: m.started_at_ms)
        return len(self._by_time)

    def refresh(self) -> int:
        """Incrementally index new traces (files not yet seen).

        Returns the number of new traces added.
        """
        if not self._trace_dir.exists():
            return 0

        try:
            entries = os.listdir(self._trace_dir)
        except OSError:
            return 0

        added = 0
        for name in entries:
            if name in self._known_files:
                continue
            if name.startswith(".") or not name.endswith(".json"):
                continue
            path = self._trace_dir / name
            meta = self._read_meta(path)
            if meta is not None:
                self._insert_sorted(meta)
                self._known_files.add(name)
                added += 1
        return added

    # ── Query methods ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._by_time)

    def get(self, trace_id: str) -> TraceMeta | None:
        """Lookup by trace ID. O(1)."""
        return self._by_id.get(trace_id)

    def by_agent(self, agent_id: str) -> list[TraceMeta]:
        """All traces for a given agent. O(1)."""
        return list(self._by_agent.get(agent_id, []))

    def by_status(self, status: str) -> list[TraceMeta]:
        """All traces with a given status. O(1)."""
        return list(self._by_status.get(status, []))

    def by_user(self, user_id: str) -> list[TraceMeta]:
        """All traces for a given user. O(1)."""
        return list(self._by_user.get(user_id, []))

    def by_session(self, session_id: str) -> list[TraceMeta]:
        """All traces in a given session. O(1)."""
        return list(self._by_session.get(session_id, []))

    def by_project(self, project: str) -> list[TraceMeta]:
        """All traces for a given project. O(1)."""
        return list(self._by_project.get(project, []))

    def by_time_range(
        self,
        since_ms: int = 0,
        until_ms: int = 0,
    ) -> list[TraceMeta]:
        """Traces within a time range. O(log N + K) via binary search.

        Args:
            since_ms: Start of range (inclusive). 0 = no lower bound.
            until_ms: End of range (inclusive). 0 = no upper bound.

        Returns:
            List of TraceMeta within the range, oldest first.
        """
        if not self._by_time:
            return []

        if since_ms <= 0 and until_ms <= 0:
            return list(self._by_time)

        # Binary search for start position
        lo = 0
        if since_ms > 0:
            lo = bisect.bisect_left(
                self._by_time, since_ms,
                key=lambda m: m.started_at_ms,
            )

        # Binary search for end position
        hi = len(self._by_time)
        if until_ms > 0:
            hi = bisect.bisect_right(
                self._by_time, until_ms,
                key=lambda m: m.started_at_ms,
            )

        return self._by_time[lo:hi]

    def most_expensive(self, n: int = 10) -> list[TraceMeta]:
        """Top N most expensive traces. O(N log N) via partial sort."""
        return sorted(self._by_time, key=lambda m: m.total_cost_usd, reverse=True)[:n]

    def agents(self) -> list[str]:
        """All unique agent IDs. O(1)."""
        return sorted(self._by_agent.keys())

    def total_cost(self) -> float:
        """Sum of all indexed trace costs. O(N)."""
        return sum(m.total_cost_usd for m in self._by_time)

    def load_full(self, trace_id: str) -> dict[str, Any] | None:
        """Load the complete trace JSON for a given trace ID.

        Uses an LRU cache to avoid re-reading hot traces from disk.
        Only loads from disk on cache miss.
        """
        # Check cache first — O(1)
        cached = self._full_cache.get(trace_id)
        if cached is not None:
            return cached

        meta = self._by_id.get(trace_id)
        if meta is None or not meta.file_path:
            return None
        try:
            with open(meta.file_path, encoding="utf-8") as f:
                data = json.load(f)
            self._full_cache.put(trace_id, data)
            return data
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    # ── Internal ──────────────────────────────────────────────────────

    def _clear(self) -> None:
        self._by_time.clear()
        self._by_id.clear()
        self._by_agent.clear()
        self._by_status.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._by_project.clear()
        self._known_files.clear()
        self._full_cache.clear()

    def _read_meta(self, path: Path) -> TraceMeta | None:
        """Read only metadata fields from a trace JSON file."""
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None

        trace_id = raw.get("trace_id")
        if not trace_id:
            return None

        # Parse status
        status_raw = raw.get("status", "Running")
        if isinstance(status_raw, str):
            status = status_raw
        elif isinstance(status_raw, dict) and "Failed" in status_raw:
            status = "Failed"
        else:
            status = "Running"

        total_tokens = raw.get("total_tokens", {})
        steps = raw.get("steps", [])

        return TraceMeta(
            trace_id=trace_id,
            agent_id=raw.get("agent_id", ""),
            status=status,
            total_cost_usd=raw.get("total_cost_usd", 0.0),
            started_at_ms=raw.get("started_at_ms", 0),
            completed_at_ms=raw.get("completed_at_ms", 0),
            step_count=len(steps),
            total_prompt_tokens=total_tokens.get("prompt_tokens", 0),
            total_completion_tokens=total_tokens.get("completion_tokens", 0),
            user_id=raw.get("user_id", ""),
            session_id=raw.get("session_id", ""),
            project=raw.get("project", ""),
            environment=raw.get("environment", ""),
            file_path=str(path),
        )

    def _insert(self, meta: TraceMeta) -> None:
        """Insert into all indices (unsorted — caller must sort _by_time)."""
        self._by_time.append(meta)
        self._by_id[meta.trace_id] = meta
        self._by_agent.setdefault(meta.agent_id, []).append(meta)
        self._by_status.setdefault(meta.status, []).append(meta)
        if meta.user_id:
            self._by_user.setdefault(meta.user_id, []).append(meta)
        if meta.session_id:
            self._by_session.setdefault(meta.session_id, []).append(meta)
        if meta.project:
            self._by_project.setdefault(meta.project, []).append(meta)

    def _insert_sorted(self, meta: TraceMeta) -> None:
        """Insert into all indices, maintaining time sort order."""
        pos = bisect.bisect_left(
            self._by_time, meta.started_at_ms,
            key=lambda m: m.started_at_ms,
        )
        self._by_time.insert(pos, meta)
        self._by_id[meta.trace_id] = meta
        self._by_agent.setdefault(meta.agent_id, []).append(meta)
        self._by_status.setdefault(meta.status, []).append(meta)
        if meta.user_id:
            self._by_user.setdefault(meta.user_id, []).append(meta)
        if meta.session_id:
            self._by_session.setdefault(meta.session_id, []).append(meta)
        if meta.project:
            self._by_project.setdefault(meta.project, []).append(meta)
