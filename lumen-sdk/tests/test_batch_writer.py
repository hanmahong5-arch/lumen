"""Tests for the batched trace writer pipeline."""

import threading
import time
import unittest

from lumen._batch_writer import BatchTraceWriter
from lumen._writers import NullWriter


class _RecordingWriter:
    """Test helper that records all traces written."""

    def __init__(self):
        self.traces = []
        self.lock = threading.Lock()

    def write_trace(self, data):
        with self.lock:
            self.traces.append(data)


class TestBatchTraceWriter(unittest.TestCase):
    """BatchTraceWriter — background buffered writes."""

    def test_add_and_flush(self):
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=10, flush_interval_sec=60)
        try:
            bw.add({"trace_id": "t1"})
            bw.add({"trace_id": "t2"})
            self.assertGreaterEqual(bw.pending, 0)  # may have been flushed

            bw.flush()
            # After flush, traces should be written
            self.assertEqual(len(rec.traces), 2)
            self.assertEqual(rec.traces[0]["trace_id"], "t1")
            self.assertEqual(rec.traces[1]["trace_id"], "t2")
        finally:
            bw.shutdown()

    def test_auto_flush_on_high_water_mark(self):
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=3, flush_interval_sec=60)
        try:
            for i in range(5):
                bw.add({"trace_id": f"t{i}"})
            # Give background thread time to process
            time.sleep(0.3)
            self.assertGreater(len(rec.traces), 0)
        finally:
            bw.shutdown()

    def test_shutdown_flushes_remaining(self):
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=100, flush_interval_sec=60)
        for i in range(5):
            bw.add({"trace_id": f"t{i}"})
        bw.shutdown()
        self.assertEqual(len(rec.traces), 5)

    def test_double_shutdown_safe(self):
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=10, flush_interval_sec=60)
        bw.shutdown()
        bw.shutdown()  # no error

    def test_add_after_shutdown_ignored(self):
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=10, flush_interval_sec=60)
        bw.shutdown()
        bw.add({"trace_id": "late"})
        self.assertEqual(len(rec.traces), 0)

    def test_write_trace_protocol(self):
        """write_trace() delegates to add() for protocol compatibility."""
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=10, flush_interval_sec=60)
        try:
            bw.write_trace({"trace_id": "proto"})
            bw.flush()
            self.assertEqual(len(rec.traces), 1)
        finally:
            bw.shutdown()

    def test_backend_error_does_not_crash(self):
        class BadWriter:
            def write_trace(self, data):
                raise RuntimeError("boom")

        bw = BatchTraceWriter(BadWriter(), buffer_size=10, flush_interval_sec=60)
        try:
            bw.add({"trace_id": "will_fail"})
            bw.flush()  # should not raise
        finally:
            bw.shutdown()

    def test_periodic_flush(self):
        rec = _RecordingWriter()
        bw = BatchTraceWriter(rec, buffer_size=100, flush_interval_sec=0.2)
        try:
            bw.add({"trace_id": "timed"})
            time.sleep(0.5)
            self.assertEqual(len(rec.traces), 1)
        finally:
            bw.shutdown()


if __name__ == "__main__":
    unittest.main()
