"""LangGraph integration: CheckpointSaver + Tracer.

Drop-in replacement for LangGraph's built-in SQLite/PostgreSQL checkpointers,
plus a callback handler for execution tracing.

Example::

    from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
    from langgraph.prebuilt import create_react_agent

    checkpointer = LumenCheckpointer(storage_dir="./checkpoints")
    tracer = LumenTracer(trace_dir="./traces")

    graph = create_react_agent(model, tools, checkpointer=checkpointer)
    result = graph.invoke(
        {"messages": [...]},
        config={"callbacks": [tracer]},
    )

    # Process crash → restart → resumes from last checkpoint.
    # Traces enable: lumen replay / lumen cost
    # Zero external services. Zero configuration.

Install with: ``pip install "lumen-ai[langgraph]"``
"""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

# Dynamically choose base class
_Base = BaseCheckpointSaver if _HAS_LANGGRAPH else object


class LumenCheckpointer(_Base):  # type: ignore[misc]
    """File-backed CheckpointSaver for LangGraph.

    Stores checkpoints as JSON files on disk with atomic writes (write → rename).
    Each thread gets its own file: ``{storage_dir}/{thread_id}.json``.

    Implements the 4 core methods required by ``BaseCheckpointSaver``:
    ``get_tuple``, ``list``, ``put``, ``put_writes``.

    Requires ``langgraph-checkpoint>=2.0``. Install with::

        pip install lumen-ai[langgraph]
    """

    def __init__(self, storage_dir: str = "./checkpoints") -> None:
        if _HAS_LANGGRAPH:
            super().__init__()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state: mirrors langgraph InMemorySaver's 3-layer pattern
        # Key format: (thread_id, checkpoint_ns, checkpoint_id)
        self._checkpoints: dict[tuple[str, str, str], dict[str, str]] = {}
        self._metadata: dict[tuple[str, str, str], dict[str, str]] = {}
        # Key: (thread_id, checkpoint_ns, channel, version)
        self._blobs: dict[tuple[str, str, str, str], dict[str, str]] = {}
        # Key: (thread_id, checkpoint_ns, checkpoint_id) -> list of (task_id, channel, blob)
        self._writes: dict[tuple[str, str, str], list[tuple[str, str, dict[str, str]]]] = {}

        self._load_from_disk()

    # ── Core methods (required by BaseCheckpointSaver) ──────────────────

    def get_tuple(self, config: dict[str, Any]) -> Any:
        """Fetch the checkpoint tuple for a config.

        If ``checkpoint_id`` is in config, returns that exact checkpoint.
        Otherwise returns the latest checkpoint for the thread.
        """
        if not _HAS_LANGGRAPH:
            raise RuntimeError("langgraph-checkpoint not installed. Run: pip install lumen-ai[langgraph]")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            key = (thread_id, checkpoint_ns, checkpoint_id)
            if key not in self._checkpoints:
                return None
        else:
            matching = [k for k in self._checkpoints if k[0] == thread_id and k[1] == checkpoint_ns]
            if not matching:
                return None
            key = max(matching, key=lambda k: k[2])

        # Deserialize checkpoint
        cp_ser = self._checkpoints[key]
        checkpoint = self.serde.loads_typed((cp_ser["type"], base64.b64decode(cp_ser["data"])))

        # Reassemble channel_values from blobs
        channel_values = {}
        for ch, ver in checkpoint.get("channel_versions", {}).items():
            blob_key = (thread_id, checkpoint_ns, ch, str(ver))
            if blob_key in self._blobs:
                blob_ser = self._blobs[blob_key]
                channel_values[ch] = self.serde.loads_typed(
                    (blob_ser["type"], base64.b64decode(blob_ser["data"]))
                )
        checkpoint["channel_values"] = channel_values

        # Deserialize metadata
        md_ser = self._metadata.get(key)
        metadata = {}
        if md_ser:
            metadata = self.serde.loads_typed((md_ser["type"], base64.b64decode(md_ser["data"])))

        # Parent config
        parent_config = None
        parent_id = cp_ser.get("parent_id")
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        # Pending writes
        pending_writes = None
        if key in self._writes:
            pending_writes = []
            for task_id, channel, blob_ser in self._writes[key]:
                value = self.serde.loads_typed((blob_ser["type"], base64.b64decode(blob_ser["data"])))
                pending_writes.append((task_id, channel, value))

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": key[2],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[Any]:
        """List checkpoints matching criteria, ordered by checkpoint_id DESC."""
        if not _HAS_LANGGRAPH:
            raise RuntimeError("langgraph-checkpoint not installed")

        if config is not None:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            keys = sorted(
                [k for k in self._checkpoints if k[0] == thread_id and k[1] == checkpoint_ns],
                key=lambda k: k[2],
                reverse=True,
            )
        else:
            keys = sorted(self._checkpoints.keys(), key=lambda k: k[2], reverse=True)

        if before:
            before_id = before["configurable"]["checkpoint_id"]
            keys = [k for k in keys if k[2] < before_id]

        count = 0
        for key in keys:
            if limit is not None and count >= limit:
                break
            result = self.get_tuple(
                {"configurable": {"thread_id": key[0], "checkpoint_ns": key[1], "checkpoint_id": key[2]}}
            )
            if result is None:
                continue
            if filter:
                if any(result.metadata.get(fk) != fv for fk, fv in filter.items()):
                    continue
            yield result
            count += 1

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a checkpoint with atomic file write."""
        if not _HAS_LANGGRAPH:
            raise RuntimeError("langgraph-checkpoint not installed")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Store channel_values as separate blobs (same pattern as InMemorySaver)
        channel_values = checkpoint.pop("channel_values", {})
        for ch, ver in new_versions.items():
            if ch in channel_values:
                type_str, data = self.serde.dumps_typed(channel_values[ch])
                blob_key = (thread_id, checkpoint_ns, ch, str(ver))
                self._blobs[blob_key] = {"type": type_str, "data": base64.b64encode(data).decode()}

        # Serialize and store checkpoint
        cp_type, cp_data = self.serde.dumps_typed(checkpoint)
        key = (thread_id, checkpoint_ns, checkpoint["id"])
        self._checkpoints[key] = {
            "type": cp_type,
            "data": base64.b64encode(cp_data).decode(),
            "parent_id": parent_checkpoint_id,
        }

        # Serialize and store metadata
        md_type, md_data = self.serde.dumps_typed(metadata)
        self._metadata[key] = {"type": md_type, "data": base64.b64encode(md_data).decode()}

        # Persist to disk (atomic write)
        self._save_thread(thread_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes for a checkpoint."""
        if not _HAS_LANGGRAPH:
            raise RuntimeError("langgraph-checkpoint not installed")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        key = (thread_id, checkpoint_ns, checkpoint_id)
        if key not in self._writes:
            self._writes[key] = []

        for channel, value in writes:
            type_str, data = self.serde.dumps_typed(value)
            self._writes[key].append((
                task_id,
                channel,
                {"type": type_str, "data": base64.b64encode(data).decode()},
            ))

        self._save_thread(thread_id)

    # ── Persistence ─────────────────────────────────────────────────────

    def _save_thread(self, thread_id: str) -> None:
        """Persist a thread's state to disk atomically."""
        data: dict[str, Any] = {"checkpoints": {}, "metadata": {}, "blobs": {}, "writes": {}}

        for key, val in self._checkpoints.items():
            if key[0] == thread_id:
                data["checkpoints"][f"{key[1]}|{key[2]}"] = val

        for key, val in self._metadata.items():
            if key[0] == thread_id:
                data["metadata"][f"{key[1]}|{key[2]}"] = val

        for key, val in self._blobs.items():
            if key[0] == thread_id:
                data["blobs"][f"{key[1]}|{key[2]}|{key[3]}"] = val

        for key, val in self._writes.items():
            if key[0] == thread_id:
                data["writes"][f"{key[1]}|{key[2]}"] = [
                    {"task_id": t, "channel": c, "blob": b} for t, c, b in val
                ]

        safe_tid = thread_id.replace("/", "_").replace("\\", "_")
        path = self.storage_dir / f"{safe_tid}.json"
        tmp_path = self.storage_dir / f".{safe_tid}.json.tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), default=str)
            os.replace(tmp_path, path)
        except OSError:
            pass  # checkpoint persistence is best-effort

    def _load_from_disk(self) -> None:
        """Load all thread state from disk."""
        if not self.storage_dir.exists():
            return

        for path in self.storage_dir.glob("*.json"):
            if path.name.startswith("."):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            tid = path.stem

            for sk, val in data.get("checkpoints", {}).items():
                parts = sk.split("|", 1)
                if len(parts) == 2:
                    self._checkpoints[(tid, parts[0], parts[1])] = val

            for sk, val in data.get("metadata", {}).items():
                parts = sk.split("|", 1)
                if len(parts) == 2:
                    self._metadata[(tid, parts[0], parts[1])] = val

            for sk, val in data.get("blobs", {}).items():
                parts = sk.split("|", 2)
                if len(parts) == 3:
                    self._blobs[(tid, parts[0], parts[1], parts[2])] = val

            for sk, val in data.get("writes", {}).items():
                parts = sk.split("|", 1)
                if len(parts) == 2:
                    try:
                        self._writes[(tid, parts[0], parts[1])] = [
                            (w["task_id"], w["channel"], w["blob"]) for w in val
                        ]
                    except (KeyError, TypeError):
                        continue


# Re-export LumenTracer for convenience:
#   from lumen.integrations.langgraph import LumenCheckpointer, LumenTracer
try:
    from lumen.integrations.langgraph_tracer import LumenTracer as LumenTracer
except ImportError:
    pass
