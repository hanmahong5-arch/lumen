"""Tests for RingBuffer and MetricsWindow — streaming metrics data structures.

Covers: circular buffer mechanics, overflow eviction, iteration, indexing,
MetricsWindow statistics (mean, variance, stddev, percentile via quickselect),
edge cases (single element, empty, full cycle), and numerical stability.
"""

from __future__ import annotations

import math

import pytest

from lumen._ring_buffer import MetricsWindow, RingBuffer


class TestRingBufferBasics:
    """Core ring buffer operations."""

    def test_empty_buffer(self):
        buf = RingBuffer[int](5)
        assert len(buf) == 0
        assert not buf
        assert buf.capacity == 5
        assert not buf.is_full

    def test_append_and_len(self):
        buf = RingBuffer[int](5)
        buf.append(10)
        buf.append(20)
        assert len(buf) == 2
        assert buf

    def test_capacity_constraint(self):
        with pytest.raises(ValueError, match="Capacity must be >= 1"):
            RingBuffer[int](0)

    def test_fill_to_capacity(self):
        buf = RingBuffer[int](3)
        buf.append(1)
        buf.append(2)
        buf.append(3)
        assert len(buf) == 3
        assert buf.is_full

    def test_overflow_evicts_oldest(self):
        buf = RingBuffer[int](3)
        for i in range(5):
            buf.append(i)
        # Should have [2, 3, 4]
        assert list(buf) == [2, 3, 4]
        assert len(buf) == 3

    def test_to_list(self):
        buf = RingBuffer[int](4)
        for i in range(7):
            buf.append(i)
        assert buf.to_list() == [3, 4, 5, 6]


class TestRingBufferIndexing:
    """__getitem__ with positive and negative indices."""

    def test_positive_index(self):
        buf = RingBuffer[str](5)
        for ch in "abcde":
            buf.append(ch)
        assert buf[0] == "a"  # oldest
        assert buf[4] == "e"  # newest

    def test_negative_index(self):
        buf = RingBuffer[int](5)
        for i in range(5):
            buf.append(i * 10)
        assert buf[-1] == 40  # newest
        assert buf[-5] == 0   # oldest

    def test_index_out_of_range(self):
        buf = RingBuffer[int](5)
        buf.append(1)
        with pytest.raises(IndexError):
            buf[5]
        with pytest.raises(IndexError):
            buf[-2]

    def test_index_after_wrap(self):
        buf = RingBuffer[int](3)
        for i in range(10):
            buf.append(i)
        # Contains [7, 8, 9]
        assert buf[0] == 7
        assert buf[1] == 8
        assert buf[2] == 9


class TestRingBufferIteration:
    """Iterator correctness with wrapping."""

    def test_iterate_partial(self):
        buf = RingBuffer[int](10)
        buf.append(1)
        buf.append(2)
        assert list(buf) == [1, 2]

    def test_iterate_full(self):
        buf = RingBuffer[int](3)
        buf.append(1)
        buf.append(2)
        buf.append(3)
        assert list(buf) == [1, 2, 3]

    def test_iterate_after_wrap(self):
        buf = RingBuffer[int](3)
        for i in range(6):
            buf.append(i)
        assert list(buf) == [3, 4, 5]

    def test_iterate_empty(self):
        buf = RingBuffer[int](5)
        assert list(buf) == []


class TestRingBufferPeek:
    """peek_newest and peek_oldest."""

    def test_peek_empty(self):
        buf = RingBuffer[int](5)
        assert buf.peek_newest() is None
        assert buf.peek_oldest() is None

    def test_peek_single(self):
        buf = RingBuffer[int](5)
        buf.append(42)
        assert buf.peek_newest() == 42
        assert buf.peek_oldest() == 42

    def test_peek_after_wrap(self):
        buf = RingBuffer[int](3)
        for i in range(7):  # [4, 5, 6]
            buf.append(i)
        assert buf.peek_oldest() == 4
        assert buf.peek_newest() == 6

    def test_clear(self):
        buf = RingBuffer[int](5)
        for i in range(3):
            buf.append(i)
        buf.clear()
        assert len(buf) == 0
        assert buf.peek_newest() is None


class TestMetricsWindowBasics:
    """MetricsWindow statistical computations."""

    def test_empty_window(self):
        w = MetricsWindow(100)
        assert w.count == 0
        assert w.mean == 0.0
        assert w.variance == 0.0
        assert w.stddev == 0.0
        assert w.total == 0.0

    def test_single_value(self):
        w = MetricsWindow(100)
        w.record(5.0)
        assert w.count == 1
        assert w.mean == 5.0
        assert w.variance == 0.0
        assert w.total == 5.0

    def test_mean_of_known_values(self):
        w = MetricsWindow(100)
        for v in [10.0, 20.0, 30.0]:
            w.record(v)
        assert w.mean == pytest.approx(20.0)

    def test_total_sum(self):
        w = MetricsWindow(100)
        for v in [1.5, 2.5, 3.0]:
            w.record(v)
        assert w.total == pytest.approx(7.0)


class TestMetricsWindowVariance:
    """Variance, stddev via running sum of squares."""

    def test_variance_known_values(self):
        # [1, 2, 3, 4, 5] → mean=3, var=2.0
        w = MetricsWindow(100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.record(v)
        assert w.mean == pytest.approx(3.0)
        assert w.variance == pytest.approx(2.0)
        assert w.stddev == pytest.approx(math.sqrt(2.0))

    def test_variance_identical_values(self):
        w = MetricsWindow(100)
        for _ in range(10):
            w.record(7.0)
        assert w.variance == pytest.approx(0.0)

    def test_variance_two_values(self):
        # [0, 10] → mean=5, var=25
        w = MetricsWindow(100)
        w.record(0.0)
        w.record(10.0)
        assert w.variance == pytest.approx(25.0)


class TestMetricsWindowMinMax:
    """Min and max over the sliding window."""

    def test_min_max_basic(self):
        w = MetricsWindow(100)
        for v in [5.0, 1.0, 9.0, 3.0]:
            w.record(v)
        assert w.min == 1.0
        assert w.max == 9.0

    def test_min_max_empty(self):
        w = MetricsWindow(100)
        assert w.min == 0.0
        assert w.max == 0.0

    def test_min_max_after_eviction(self):
        w = MetricsWindow(3)
        w.record(100.0)  # will be evicted
        w.record(1.0)
        w.record(2.0)
        w.record(3.0)    # evicts 100.0
        assert w.min == 1.0
        assert w.max == 3.0


class TestMetricsWindowPercentile:
    """Percentile computation via quickselect."""

    def test_p50_odd_count(self):
        w = MetricsWindow(100)
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            w.record(v)
        assert w.percentile(50) == pytest.approx(30.0)

    def test_p0_is_min(self):
        w = MetricsWindow(100)
        for v in [5.0, 3.0, 8.0, 1.0, 9.0]:
            w.record(v)
        assert w.percentile(0) == pytest.approx(1.0)

    def test_p100_is_max(self):
        w = MetricsWindow(100)
        for v in [5.0, 3.0, 8.0, 1.0, 9.0]:
            w.record(v)
        assert w.percentile(100) == pytest.approx(9.0)

    def test_percentile_empty(self):
        w = MetricsWindow(100)
        assert w.percentile(50) == 0.0

    def test_percentile_single(self):
        w = MetricsWindow(100)
        w.record(42.0)
        assert w.percentile(50) == 42.0
        assert w.percentile(99) == 42.0

    def test_p99_near_max(self):
        w = MetricsWindow(1000)
        for i in range(100):
            w.record(float(i))
        p99 = w.percentile(99)
        assert p99 >= 95.0  # should be near the top


class TestMetricsWindowEviction:
    """Statistics stay correct when old values are evicted."""

    def test_mean_after_eviction(self):
        w = MetricsWindow(3)
        w.record(100.0)
        w.record(200.0)
        w.record(300.0)
        assert w.mean == pytest.approx(200.0)

        # Evict 100, add 400 → [200, 300, 400] → mean=300
        w.record(400.0)
        assert w.mean == pytest.approx(300.0)

    def test_total_after_eviction(self):
        w = MetricsWindow(3)
        w.record(10.0)
        w.record(20.0)
        w.record(30.0)
        w.record(40.0)  # evicts 10
        assert w.total == pytest.approx(90.0)

    def test_count_capped_at_capacity(self):
        w = MetricsWindow(5)
        for i in range(100):
            w.record(float(i))
        assert w.count == 5

    def test_variance_after_complete_replacement(self):
        w = MetricsWindow(3)
        # Fill with varied data
        w.record(1.0)
        w.record(100.0)
        w.record(1000.0)
        # Replace entirely with identical values
        w.record(5.0)
        w.record(5.0)
        w.record(5.0)
        assert w.variance == pytest.approx(0.0, abs=1e-6)


class TestMetricsWindowRate:
    """Events-per-second calculation."""

    def test_rate_basic(self):
        w = MetricsWindow(100)
        for i in range(10):
            w.record(float(i))
        # 10 events in 2000ms = 5/s
        assert w.rate(2000) == pytest.approx(5.0)

    def test_rate_zero_window(self):
        w = MetricsWindow(100)
        w.record(1.0)
        assert w.rate(0) == 0.0

    def test_rate_negative_window(self):
        w = MetricsWindow(100)
        w.record(1.0)
        assert w.rate(-100) == 0.0


class TestMetricsWindowClear:
    """Reset all statistics."""

    def test_clear_resets_everything(self):
        w = MetricsWindow(100)
        for i in range(50):
            w.record(float(i))
        w.clear()
        assert w.count == 0
        assert w.mean == 0.0
        assert w.total == 0.0
        assert w.variance == 0.0


class TestMetricsWindowNumericalStability:
    """Guard against floating-point drift in running sums."""

    def test_many_small_values(self):
        w = MetricsWindow(1000)
        for _ in range(1000):
            w.record(0.001)
        assert w.mean == pytest.approx(0.001, rel=1e-6)
        assert w.total == pytest.approx(1.0, rel=1e-3)

    def test_large_values_no_overflow(self):
        w = MetricsWindow(100)
        for _ in range(100):
            w.record(1e15)
        assert w.mean == pytest.approx(1e15, rel=1e-6)

    def test_mixed_scale_values(self):
        w = MetricsWindow(100)
        w.record(0.0001)
        w.record(1000000.0)
        # Just verify no crash or NaN
        assert math.isfinite(w.mean)
        assert math.isfinite(w.variance)
        assert math.isfinite(w.stddev)
