"""Tests for pluggable trace writer backends."""

import json
import os
import tempfile
import unittest
from unittest import mock

from lumen._writers import CompositeWriter, FileTraceWriter, NullWriter


class TestFileTraceWriter(unittest.TestCase):
    """FileTraceWriter — atomic JSON file writes."""

    def test_write_creates_json_file(self):
        with tempfile.TemporaryDirectory() as td:
            writer = FileTraceWriter(td)
            data = {"trace_id": "abc123", "steps": [], "status": "Completed"}
            writer.write_trace(data)

            path = os.path.join(td, "abc123.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["trace_id"], "abc123")

    def test_write_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as td:
            writer = FileTraceWriter(td)
            writer.write_trace({"trace_id": "t1", "status": "Running"})
            writer.write_trace({"trace_id": "t1", "status": "Completed"})

            with open(os.path.join(td, "t1.json")) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["status"], "Completed")

    def test_write_non_serializable_uses_default_str(self):
        with tempfile.TemporaryDirectory() as td:
            writer = FileTraceWriter(td)
            data = {"trace_id": "t2", "custom": object()}
            writer.write_trace(data)
            self.assertTrue(os.path.exists(os.path.join(td, "t2.json")))

    def test_write_bad_dir_warns(self):
        with mock.patch("lumen._writers.Path.mkdir", side_effect=PermissionError("denied")):
            writer = FileTraceWriter("/impossible/path")
            with self.assertWarns(UserWarning):
                writer.write_trace({"trace_id": "x"})

    def test_trace_dir_property(self):
        writer = FileTraceWriter("/my/traces")
        self.assertEqual(writer.trace_dir, "/my/traces")


class TestNullWriter(unittest.TestCase):
    """NullWriter discards all traces."""

    def test_write_does_nothing(self):
        writer = NullWriter()
        writer.write_trace({"trace_id": "dropped"})  # no error

    def test_implements_protocol(self):
        from lumen._protocols import TraceWriter
        writer = NullWriter()
        self.assertIsInstance(writer, TraceWriter)


class TestCompositeWriter(unittest.TestCase):
    """CompositeWriter fans out to multiple backends."""

    def test_calls_all_writers(self):
        calls = []

        class MockWriter:
            def __init__(self, name):
                self.name = name

            def write_trace(self, data):
                calls.append(self.name)

        writer = CompositeWriter([MockWriter("a"), MockWriter("b"), MockWriter("c")])
        writer.write_trace({"trace_id": "t1"})
        self.assertEqual(calls, ["a", "b", "c"])

    def test_one_failure_does_not_block_others(self):
        calls = []

        class GoodWriter:
            def write_trace(self, data):
                calls.append("good")

        class BadWriter:
            def write_trace(self, data):
                raise RuntimeError("disk full")

        writer = CompositeWriter([GoodWriter(), BadWriter(), GoodWriter()])
        writer.write_trace({"trace_id": "t1"})
        self.assertEqual(calls, ["good", "good"])

    def test_writers_property(self):
        w = CompositeWriter([NullWriter()])
        self.assertEqual(len(w.writers), 1)

    def test_empty_composite(self):
        writer = CompositeWriter([])
        writer.write_trace({"trace_id": "t1"})  # no error


class TestFileTraceWriterProtocol(unittest.TestCase):
    """Verify protocol conformance."""

    def test_file_writer_implements_protocol(self):
        from lumen._protocols import TraceWriter
        writer = FileTraceWriter()
        self.assertIsInstance(writer, TraceWriter)


if __name__ == "__main__":
    unittest.main()
