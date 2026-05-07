"""Tests for nested trace (span tree) support."""

import json
import os
import tempfile
import unittest

from lumen._context import TraceSession, get_active_session, set_active_session


class TestNestedTraces(unittest.TestCase):
    """Nested traces — parent-child span tree relationships."""

    def test_parent_trace_id_included_in_output(self):
        with tempfile.TemporaryDirectory() as td:
            session = TraceSession(
                name="child",
                trace_dir=td,
                parent_trace_id="parent-abc",
            )
            session.add_llm_step(
                model="gpt-4o", prompt_tokens=10, completion_tokens=5,
                finish_reason="stop", duration_ms=100, started_at_ms=1000,
            )
            session.finalize()

            # Read the trace file
            files = [f for f in os.listdir(td) if f.endswith(".json")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(td, files[0])) as f:
                data = json.load(f)
            self.assertEqual(data["parent_trace_id"], "parent-abc")

    def test_no_parent_when_top_level(self):
        with tempfile.TemporaryDirectory() as td:
            session = TraceSession(name="top", trace_dir=td)
            session.add_llm_step(
                model="gpt-4o", prompt_tokens=10, completion_tokens=5,
                finish_reason="stop", duration_ms=100, started_at_ms=1000,
            )
            session.finalize()

            files = [f for f in os.listdir(td) if f.endswith(".json")]
            with open(os.path.join(td, files[0])) as f:
                data = json.load(f)
            self.assertNotIn("parent_trace_id", data)

    def test_span_ids_on_steps(self):
        session = TraceSession(name="test", trace_dir="/dev/null")
        session.add_llm_step(
            model="gpt-4o", prompt_tokens=10, completion_tokens=5,
            finish_reason="stop", duration_ms=100, started_at_ms=1000,
        )
        session.add_tool_step(
            tool_name="search", success=True, duration_ms=50, started_at_ms=1100,
        )
        session.add_error_step("oops")

        for step in session.steps:
            self.assertIn("span_id", step)
            self.assertEqual(len(step["span_id"]), 8)

    def test_all_span_ids_unique(self):
        session = TraceSession(name="test", trace_dir="/dev/null")
        for i in range(20):
            session.add_llm_step(
                model="gpt-4o", prompt_tokens=10, completion_tokens=5,
                finish_reason="stop", duration_ms=100, started_at_ms=1000 + i,
            )
        ids = [s["span_id"] for s in session.steps]
        self.assertEqual(len(ids), len(set(ids)))


class TestNestedTraceContextManager(unittest.TestCase):
    """Integration: trace() context manager records parent_trace_id."""

    def test_nested_trace_via_context_manager(self):
        with tempfile.TemporaryDirectory() as td:
            from lumen.instrument import trace

            with trace("outer", config=None) as outer:
                outer.trace_dir = td
                with trace("inner", config=None) as inner:
                    inner.trace_dir = td
                    inner.add_llm_step(
                        model="gpt-4o", prompt_tokens=10, completion_tokens=5,
                        finish_reason="stop", duration_ms=100, started_at_ms=1000,
                    )

            # Read inner trace
            inner_path = os.path.join(td, f"{inner.trace_id}.json")
            if os.path.exists(inner_path):
                with open(inner_path) as f:
                    data = json.load(f)
                self.assertEqual(data.get("parent_trace_id"), outer.trace_id)

    def test_top_level_has_no_parent(self):
        # Ensure no active session before test
        set_active_session(None)
        with tempfile.TemporaryDirectory() as td:
            from lumen.instrument import trace

            with trace("top", config=None) as session:
                session.trace_dir = td
                session.add_llm_step(
                    model="gpt-4o", prompt_tokens=10, completion_tokens=5,
                    finish_reason="stop", duration_ms=100, started_at_ms=1000,
                )

            path = os.path.join(td, f"{session.trace_id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                self.assertNotIn("parent_trace_id", data)


if __name__ == "__main__":
    unittest.main()
