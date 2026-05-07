"""Fixed-size ring buffer for streaming metrics aggregation.

Provides O(1) append and O(N) statistical queries over a sliding window
of the most recent N observations. Used for real-time latency tracking,
cost monitoring, and anomaly detection.

Data structures:
  - RingBuffer[T]: Generic circular buffer with O(1) append
  - MetricsWindow: Specialized float buffer with percentile, mean,
    variance, and rate calculations

Design:
  - Pre-allocated array avoids GC pressure from frequent appends
  - No locks — each MetricsWindow is owned by one thread/session
  - Percentile uses quickselect (O(N) average) not full sort (O(N log N))
"""

from __future__ import annotations

import math
import random
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-size circular buffer with O(1) append and O(1) index.

    When full, the oldest element is overwritten. Supports iteration
    from oldest to newest.

    Example::

        buf = RingBuffer(5)
        for i in range(8):
            buf.append(i)
        list(buf)  # [3, 4, 5, 6, 7] — oldest 0,1,2 evicted
    """

    __slots__ = ("_data", "_capacity", "_head", "_count")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self._data: list[T | None] = [None] * capacity
        self._capacity = capacity
        self._head = 0  # next write position
        self._count = 0

    def append(self, value: T) -> None:
        """Add a value, evicting the oldest if full. O(1)."""
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._count == self._capacity

    def __getitem__(self, index: int) -> T:
        """Access by index (0 = oldest). O(1)."""
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(f"index {index} out of range for size {self._count}")
        # oldest element starts at (head - count) mod capacity
        start = (self._head - self._count) % self._capacity
        actual = (start + index) % self._capacity
        return self._data[actual]  # type: ignore[return-value]

    def __iter__(self) -> Iterator[T]:
        """Iterate from oldest to newest. O(N)."""
        if self._count == 0:
            return
        start = (self._head - self._count) % self._capacity
        for i in range(self._count):
            yield self._data[(start + i) % self._capacity]  # type: ignore[misc]

    def to_list(self) -> list[T]:
        """Return all elements as a list, oldest first. O(N)."""
        return list(self)

    def clear(self) -> None:
        """Remove all elements. O(1)."""
        self._head = 0
        self._count = 0

    def peek_newest(self) -> T | None:
        """Return the most recently added element. O(1)."""
        if self._count == 0:
            return None
        idx = (self._head - 1) % self._capacity
        return self._data[idx]  # type: ignore[return-value]

    def peek_oldest(self) -> T | None:
        """Return the oldest element. O(1)."""
        if self._count == 0:
            return None
        return self[0]


class MetricsWindow:
    """Fixed-size sliding window of float observations with statistical queries.

    Provides O(1) append and O(N) aggregation for latency/cost monitoring.
    Uses quickselect for percentile computation (O(N) average).

    Example::

        window = MetricsWindow(1000)
        for latency in measurements:
            window.record(latency)

        print(f"p50={window.percentile(50):.1f}ms")
        print(f"p99={window.percentile(99):.1f}ms")
        print(f"mean={window.mean:.1f}ms")
    """

    __slots__ = ("_buf", "_sum", "_sum_sq")

    def __init__(self, capacity: int = 1000) -> None:
        self._buf = RingBuffer[float](capacity)
        self._sum = 0.0
        self._sum_sq = 0.0

    def record(self, value: float) -> None:
        """Record an observation. O(1)."""
        # If buffer is full, subtract the value being evicted
        if self._buf.is_full:
            evicted = self._buf.peek_oldest()
            if evicted is not None:
                self._sum -= evicted
                self._sum_sq -= evicted * evicted
        self._buf.append(value)
        self._sum += value
        self._sum_sq += value * value

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def mean(self) -> float:
        """Arithmetic mean. O(1)."""
        n = len(self._buf)
        return self._sum / n if n > 0 else 0.0

    @property
    def variance(self) -> float:
        """Population variance. O(1) via running sum of squares."""
        n = len(self._buf)
        if n < 2:
            return 0.0
        mean = self._sum / n
        return max(0.0, self._sum_sq / n - mean * mean)

    @property
    def stddev(self) -> float:
        """Population standard deviation. O(1)."""
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        """Minimum value. O(N)."""
        if not self._buf:
            return 0.0
        return min(self._buf)  # type: ignore[type-var]

    @property
    def max(self) -> float:
        """Maximum value. O(N)."""
        if not self._buf:
            return 0.0
        return max(self._buf)  # type: ignore[type-var]

    @property
    def total(self) -> float:
        """Sum of all values in window. O(1)."""
        return self._sum

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile using quickselect. O(N) average.

        Args:
            p: Percentile value (0-100).

        Returns:
            The p-th percentile value.
        """
        n = len(self._buf)
        if n == 0:
            return 0.0
        if n == 1:
            return self._buf[0]

        # Copy values for in-place partitioning
        values = self._buf.to_list()
        k = max(0, min(n - 1, int(p / 100.0 * (n - 1) + 0.5)))
        return _quickselect(values, k)

    def rate(self, window_ms: int) -> float:
        """Compute events per second over the given window.

        This is simply count / (window_ms / 1000). The caller provides
        the actual time window since MetricsWindow doesn't track timestamps.
        """
        if window_ms <= 0:
            return 0.0
        return len(self._buf) / (window_ms / 1000.0)

    def clear(self) -> None:
        """Reset all statistics."""
        self._buf.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


def _quickselect(arr: list[float], k: int) -> float:
    """Find the k-th smallest element. O(N) average, O(N^2) worst.

    Uses median-of-three pivot selection for better average performance.
    """
    if len(arr) == 1:
        return arr[0]

    lo, hi = 0, len(arr) - 1

    while lo < hi:
        # Median-of-three pivot
        mid = (lo + hi) // 2
        if arr[lo] > arr[mid]:
            arr[lo], arr[mid] = arr[mid], arr[lo]
        if arr[lo] > arr[hi]:
            arr[lo], arr[hi] = arr[hi], arr[lo]
        if arr[mid] > arr[hi]:
            arr[mid], arr[hi] = arr[hi], arr[mid]
        pivot = arr[mid]

        # Hoare partition
        i, j = lo, hi
        while True:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1

        if k <= j:
            hi = j
        else:
            lo = j + 1

    return arr[k]
