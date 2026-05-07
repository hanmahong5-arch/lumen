"""Tests for the runtime metrics pipeline."""

import unittest

from lumen._metrics import RuntimeMetrics


class TestRuntimeMetrics(unittest.TestCase):
    """RuntimeMetrics — live observability from LLM calls."""

    def test_record_increments_counters(self):
        m = RuntimeMetrics()
        m.record(
            model="gpt-4o", agent_name="agent-1",
            prompt_tokens=100, completion_tokens=50,
            cost_usd=0.01, duration_ms=1000,
        )
        self.assertEqual(m.total_calls, 1)
        self.assertEqual(m.total_errors, 0)

    def test_record_multiple_models(self):
        m = RuntimeMetrics()
        m.record(model="gpt-4o", agent_name="a", prompt_tokens=100,
                 completion_tokens=50, cost_usd=0.01, duration_ms=1000)
        m.record(model="claude-sonnet-4-6", agent_name="a", prompt_tokens=200,
                 completion_tokens=100, cost_usd=0.02, duration_ms=2000)
        self.assertEqual(m.total_calls, 2)
        self.assertIn("gpt-4o", m.model_names())
        self.assertIn("claude-sonnet-4-6", m.model_names())

    def test_per_model_metrics(self):
        m = RuntimeMetrics()
        for i in range(10):
            m.record(model="gpt-4o", agent_name="test",
                     prompt_tokens=100, completion_tokens=50,
                     cost_usd=0.01, duration_ms=1000 + i * 100)

        mm = m.by_model("gpt-4o")
        self.assertIsNotNone(mm)
        self.assertEqual(mm.calls, 10)
        self.assertAlmostEqual(mm.cost.total, 0.10, places=4)
        self.assertGreater(mm.latency.mean, 0)

    def test_per_agent_metrics(self):
        m = RuntimeMetrics()
        m.record(model="gpt-4o", agent_name="research",
                 prompt_tokens=100, completion_tokens=50,
                 cost_usd=0.01, duration_ms=1000)
        m.record(model="gpt-4o", agent_name="summarizer",
                 prompt_tokens=50, completion_tokens=25,
                 cost_usd=0.005, duration_ms=500)

        self.assertIn("research", m.agent_names())
        self.assertIn("summarizer", m.agent_names())
        self.assertEqual(m.by_agent("research").calls, 1)
        self.assertEqual(m.by_agent("summarizer").calls, 1)

    def test_unknown_model_returns_none(self):
        m = RuntimeMetrics()
        self.assertIsNone(m.by_model("nonexistent"))

    def test_summary(self):
        m = RuntimeMetrics()
        m.record(model="gpt-4o", agent_name="test",
                 prompt_tokens=100, completion_tokens=50,
                 cost_usd=0.01, duration_ms=1000)

        s = m.summary()
        self.assertEqual(s["calls"], 1)
        self.assertEqual(s["errors"], 0)
        self.assertAlmostEqual(s["total_cost_usd"], 0.01, places=4)
        self.assertGreater(s["latency_p50_ms"], 0)
        self.assertIn("gpt-4o", s["models"])

    def test_record_error(self):
        m = RuntimeMetrics()
        m.record_error()
        m.record_error()
        self.assertEqual(m.total_errors, 2)

    def test_clear(self):
        m = RuntimeMetrics()
        m.record(model="gpt-4o", agent_name="test",
                 prompt_tokens=100, completion_tokens=50,
                 cost_usd=0.01, duration_ms=1000)
        m.clear()
        self.assertEqual(m.total_calls, 0)
        self.assertEqual(len(m.model_names()), 0)
        self.assertEqual(len(m.agent_names()), 0)

    def test_protocol_methods(self):
        """MetricsCollector protocol methods work."""
        m = RuntimeMetrics()
        m.record_latency(500.0)
        m.record_cost(0.05)
        m.record_tokens(100, 50)

        from lumen._protocols import MetricsCollector
        self.assertIsInstance(m, MetricsCollector)

    def test_thread_safety(self):
        """Concurrent recording doesn't crash."""
        import threading
        m = RuntimeMetrics()
        errors = []

        def worker(agent_id):
            try:
                for _ in range(100):
                    m.record(model="gpt-4o", agent_name=f"agent-{agent_id}",
                             prompt_tokens=10, completion_tokens=5,
                             cost_usd=0.001, duration_ms=100)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(m.total_calls, 1000)


if __name__ == "__main__":
    unittest.main()
