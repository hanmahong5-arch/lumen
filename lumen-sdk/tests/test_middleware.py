"""Tests for the trace processing middleware chain."""

import unittest
from unittest import mock

from lumen._middleware import (
    ContentStripMiddleware,
    MiddlewareChain,
    RedactionMiddleware,
    SizeLimitMiddleware,
    TimestampMiddleware,
)


def _sample_trace():
    return {
        "trace_id": "test-123",
        "steps": [
            {
                "step_num": 0,
                "step_type": {"LlmCall": {"model": "gpt-4o", "finish_reason": "stop"}},
                "metadata": [["input_messages", "Tell me about secret-key-abc"]],
                "duration_ms": 100,
            },
            {
                "step_num": 1,
                "step_type": {"Error": {"message": "API key sk-12345 is invalid"}},
                "metadata": [],
                "duration_ms": 0,
            },
        ],
        "status": "Failed",
        "started_at_ms": 1700000000000,
        "completed_at_ms": 1700000001000,
    }


class TestMiddlewareChain(unittest.TestCase):
    """MiddlewareChain — ordered processing pipeline."""

    def test_empty_chain_passes_through(self):
        chain = MiddlewareChain()
        data = {"trace_id": "t1"}
        result = chain.process(data)
        self.assertEqual(result, data)

    def test_chain_applies_in_order(self):
        calls = []

        class MW:
            def __init__(self, name):
                self.name = name

            def process(self, data):
                calls.append(self.name)
                data[self.name] = True
                return data

        chain = MiddlewareChain([MW("a"), MW("b"), MW("c")])
        result = chain.process({"trace_id": "t1"})
        self.assertEqual(calls, ["a", "b", "c"])
        self.assertTrue(result["a"])
        self.assertTrue(result["c"])

    def test_none_drops_trace(self):
        class Dropper:
            def process(self, data):
                return None

        class NeverCalled:
            def process(self, data):
                raise AssertionError("should not be called")

        chain = MiddlewareChain([Dropper(), NeverCalled()])
        result = chain.process({"trace_id": "t1"})
        self.assertIsNone(result)

    def test_exception_passes_through(self):
        class Faulty:
            def process(self, data):
                raise RuntimeError("boom")

        chain = MiddlewareChain([Faulty()])
        result = chain.process({"trace_id": "t1"})
        self.assertEqual(result["trace_id"], "t1")

    def test_add_middleware(self):
        chain = MiddlewareChain()

        class MW:
            def process(self, data):
                data["added"] = True
                return data

        chain.add(MW())
        result = chain.process({"trace_id": "t1"})
        self.assertTrue(result["added"])

    def test_len(self):
        chain = MiddlewareChain()
        self.assertEqual(len(chain), 0)


class TestRedactionMiddleware(unittest.TestCase):
    """RedactionMiddleware — PII redaction in trace steps."""

    def test_redacts_error_messages(self):
        from lumen.config import RedactionConfig
        rc = RedactionConfig(enabled=True, patterns=[r"sk-\w+"])
        rc.compile_patterns()

        mw = RedactionMiddleware(rc)
        data = _sample_trace()
        result = mw.process(data)

        error_msg = result["steps"][1]["step_type"]["Error"]["message"]
        self.assertNotIn("sk-12345", error_msg)
        self.assertIn("[REDACTED]", error_msg)

    def test_redacts_metadata_values(self):
        from lumen.config import RedactionConfig
        rc = RedactionConfig(enabled=True, patterns=[r"secret-key-\w+"])
        rc.compile_patterns()

        mw = RedactionMiddleware(rc)
        data = _sample_trace()
        result = mw.process(data)

        meta_val = result["steps"][0]["metadata"][0][1]
        self.assertNotIn("secret-key-abc", meta_val)

    def test_disabled_passes_through(self):
        from lumen.config import RedactionConfig
        rc = RedactionConfig(enabled=False)

        mw = RedactionMiddleware(rc)
        data = _sample_trace()
        result = mw.process(data)
        self.assertEqual(result["steps"][1]["step_type"]["Error"]["message"],
                         "API key sk-12345 is invalid")


class TestContentStripMiddleware(unittest.TestCase):
    """ContentStripMiddleware — strips verbose metadata."""

    def test_strips_metadata(self):
        mw = ContentStripMiddleware()
        data = _sample_trace()
        result = mw.process(data)
        meta = result["steps"][0]["metadata"]
        self.assertEqual(meta[0][1], "[stripped]")

    def test_empty_metadata_unchanged(self):
        mw = ContentStripMiddleware()
        data = {"trace_id": "t", "steps": [{"metadata": []}]}
        result = mw.process(data)
        self.assertEqual(result["steps"][0]["metadata"], [])


class TestSizeLimitMiddleware(unittest.TestCase):
    """SizeLimitMiddleware — trace size governor."""

    def test_small_trace_passes_through(self):
        mw = SizeLimitMiddleware(max_bytes=100_000)
        data = _sample_trace()
        result = mw.process(data)
        # Metadata should be preserved
        self.assertIn("input_messages", result["steps"][0]["metadata"][0][0])

    def test_oversized_trace_truncated(self):
        mw = SizeLimitMiddleware(max_bytes=10)  # tiny limit
        data = _sample_trace()
        result = mw.process(data)
        meta = result["steps"][0]["metadata"]
        self.assertEqual(meta[0][1], "[truncated:size_limit]")


class TestTimestampMiddleware(unittest.TestCase):
    """TimestampMiddleware — ISO 8601 human-readable timestamps."""

    def test_adds_iso_timestamps(self):
        mw = TimestampMiddleware()
        data = _sample_trace()
        result = mw.process(data)
        self.assertIn("started_at", result)
        self.assertIn("completed_at", result)
        self.assertIn("2023-", result["started_at"])  # 1700000000 is ~2023

    def test_missing_timestamps_no_error(self):
        mw = TimestampMiddleware()
        data = {"trace_id": "t1"}
        result = mw.process(data)
        self.assertNotIn("started_at", result)


if __name__ == "__main__":
    unittest.main()
