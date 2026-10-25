"""Tests for LumenCheckpointer resume affordances (no live LLM)."""

from __future__ import annotations

from pathlib import Path

from lumen.integrations.langgraph import LumenCheckpointer


def _seed(cp: LumenCheckpointer, thread_id: str, *checkpoint_ids: str) -> None:
    """Inject checkpoint keys without going through LangGraph serde."""
    for cid in checkpoint_ids:
        cp._checkpoints[(thread_id, "", cid)] = {
            "type": "msgpack",
            "data": "e30=",  # placeholder; resume helpers only need keys
            "parent_id": None,
        }


def test_list_threads_and_latest_config(tmp_path: Path) -> None:
    cp = LumenCheckpointer(storage_dir=str(tmp_path))
    assert cp.list_threads() == []
    assert cp.latest_config("missing") is None

    _seed(cp, "research-1", "ckpt-a", "ckpt-c", "ckpt-b")
    _seed(cp, "other", "z")

    assert cp.list_threads() == ["other", "research-1"]
    latest = cp.latest_config("research-1")
    assert latest is not None
    assert latest["configurable"]["thread_id"] == "research-1"
    assert latest["configurable"]["checkpoint_id"] == "ckpt-c"
    assert latest["configurable"]["checkpoint_ns"] == ""


def test_resume_hint_empty_and_populated(tmp_path: Path) -> None:
    cp = LumenCheckpointer(storage_dir=str(tmp_path))
    empty = cp.resume_hint("ghost")
    assert "No checkpoint" in empty
    assert "lumen demo" in empty

    _seed(cp, "t1", "id-9")
    hint = cp.resume_hint("t1")
    assert "id-9" in hint
    assert "graph.invoke" in hint
    assert "lumen export" in hint


def test_list_threads_survives_reload(tmp_path: Path) -> None:
    cp = LumenCheckpointer(storage_dir=str(tmp_path))
    _seed(cp, "persist-me", "c1")
    cp._save_thread("persist-me")

    reloaded = LumenCheckpointer(storage_dir=str(tmp_path))
    assert reloaded.list_threads() == ["persist-me"]
    cfg = reloaded.latest_config("persist-me")
    assert cfg is not None
    assert cfg["configurable"]["checkpoint_id"] == "c1"
