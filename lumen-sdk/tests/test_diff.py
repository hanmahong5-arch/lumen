"""Tests for the trace comparison / diff engine."""

import unittest

from lumen._diff import StepSummary, TraceDiff, compare_traces


def _make_trace(trace_id, steps, cost=0.0, started=0, completed=0):
    """Helper to build a trace dict."""
    return {
        "trace_id": trace_id,
        "steps": steps,
        "total_cost_usd": cost,
        "started_at_ms": started,
        "completed_at_ms": completed,
    }


def _llm_step(num, model="gpt-4o", tokens=None, duration_ms=100):
    tokens = tokens or {"prompt_tokens": 100, "completion_tokens": 50}
    return {
        "step_num": num,
        "step_type": {"LlmCall": {"model": model, "finish_reason": "stop"}},
        "started_at_ms": 0,
        "duration_ms": duration_ms,
        "tokens": tokens,
        "metadata": [],
    }


def _tool_step(num, tool_name="search", duration_ms=50):
    return {
        "step_num": num,
        "step_type": {"ToolCall": {"tool_name": tool_name, "success": True}},
        "started_at_ms": 0,
        "duration_ms": duration_ms,
        "tokens": None,
        "metadata": [],
    }


def _error_step(num, msg="something failed"):
    return {
        "step_num": num,
        "step_type": {"Error": {"message": msg}},
        "started_at_ms": 0,
        "duration_ms": 0,
        "tokens": None,
        "metadata": [],
    }


class TestCompareTraces(unittest.TestCase):
    """compare_traces — LCS-based structural diff."""

    def test_identical_traces(self):
        steps = [_llm_step(0), _tool_step(1), _llm_step(2)]
        a = _make_trace("a", steps, cost=0.03)
        b = _make_trace("b", steps, cost=0.03)
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.removed), 0)
        self.assertEqual(diff.unchanged, 3)
        self.assertAlmostEqual(diff.cost_delta_usd, 0.0)

    def test_added_steps(self):
        a = _make_trace("a", [_llm_step(0)])
        b = _make_trace("b", [_llm_step(0), _tool_step(1), _llm_step(2)])
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.added), 2)
        self.assertEqual(len(diff.removed), 0)
        self.assertEqual(diff.step_count_a, 1)
        self.assertEqual(diff.step_count_b, 3)

    def test_removed_steps(self):
        a = _make_trace("a", [_llm_step(0), _tool_step(1), _llm_step(2)])
        b = _make_trace("b", [_llm_step(0)])
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.removed), 2)
        self.assertEqual(len(diff.added), 0)

    def test_changed_steps(self):
        a = _make_trace("a", [_llm_step(0, duration_ms=100)])
        b = _make_trace("b", [_llm_step(0, duration_ms=500)])
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.changed), 1)
        self.assertEqual(diff.changed[0].duration_delta_ms, 400)

    def test_cost_delta(self):
        a = _make_trace("a", [_llm_step(0)], cost=0.01)
        b = _make_trace("b", [_llm_step(0)], cost=0.10)
        diff = compare_traces(a, b)
        self.assertAlmostEqual(diff.cost_delta_usd, 0.09, places=4)

    def test_duration_delta(self):
        a = _make_trace("a", [_llm_step(0)], started=1000, completed=2000)
        b = _make_trace("b", [_llm_step(0)], started=1000, completed=5000)
        diff = compare_traces(a, b)
        self.assertEqual(diff.duration_delta_ms, 3000)

    def test_empty_traces(self):
        a = _make_trace("a", [])
        b = _make_trace("b", [])
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.removed), 0)
        self.assertEqual(diff.unchanged, 0)

    def test_completely_different_steps(self):
        a = _make_trace("a", [_llm_step(0, model="gpt-4o")])
        b = _make_trace("b", [_tool_step(0, tool_name="calculator")])
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(len(diff.removed), 1)
        self.assertEqual(diff.unchanged, 0)

    def test_mixed_changes(self):
        a = _make_trace("a", [
            _llm_step(0, model="gpt-4o"),
            _tool_step(1, tool_name="search"),
            _llm_step(2, model="gpt-4o"),
        ])
        b = _make_trace("b", [
            _llm_step(0, model="gpt-4o"),
            _llm_step(1, model="claude-sonnet-4-6"),
            _tool_step(2, tool_name="search"),
            _llm_step(3, model="gpt-4o"),
        ])
        diff = compare_traces(a, b)
        # gpt-4o(0), search(1), gpt-4o(2) vs gpt-4o(0), claude(1), search(2), gpt-4o(3)
        # LCS matches: gpt-4o, search, gpt-4o = 3
        self.assertEqual(len(diff.added), 1)  # claude-sonnet-4-6 added
        self.assertEqual(len(diff.removed), 0)

    def test_error_steps_in_diff(self):
        a = _make_trace("a", [_llm_step(0), _error_step(1)])
        b = _make_trace("b", [_llm_step(0)])
        diff = compare_traces(a, b)
        self.assertEqual(len(diff.removed), 1)
        self.assertEqual(diff.removed[0].step_type, "Error")

    def test_token_delta(self):
        a = _make_trace("a", [_llm_step(0, tokens={"prompt_tokens": 100, "completion_tokens": 50})])
        b = _make_trace("b", [_llm_step(0, tokens={"prompt_tokens": 500, "completion_tokens": 200})])
        diff = compare_traces(a, b)
        self.assertEqual(diff.token_delta, 550)  # (700 - 150)


class TestStepSummary(unittest.TestCase):
    """StepSummary extraction from raw step dicts."""

    def test_from_llm_step(self):
        s = StepSummary.from_step(_llm_step(0, model="gpt-4o"))
        self.assertEqual(s.step_type, "LlmCall")
        self.assertEqual(s.model_or_tool, "gpt-4o")

    def test_from_tool_step(self):
        s = StepSummary.from_step(_tool_step(0, tool_name="calculator"))
        self.assertEqual(s.step_type, "ToolCall")
        self.assertEqual(s.model_or_tool, "calculator")

    def test_from_error_step(self):
        s = StepSummary.from_step(_error_step(0, "boom"))
        self.assertEqual(s.step_type, "Error")

    def test_from_empty_step(self):
        s = StepSummary.from_step({})
        self.assertEqual(s.step_type, "Unknown")


if __name__ == "__main__":
    unittest.main()
